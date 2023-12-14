import json
import multiprocessing
from pathlib import Path
from typing import BinaryIO, Tuple, Optional, Mapping, Any, Union, Callable
from io import SEEK_CUR, SEEK_END, SEEK_SET
import numpy as np
from typing_extensions import Protocol
from importlib import import_module
from diplomat.predictors.fpe.sparse_storage import ForwardBackwardFrame
from diplomat.predictors.sfpe.avl_tree import BufferTree, NumpyTree, insert, nearest_pop, remove
import zlib


DIPLOMAT_STATE_HEADER = b"DPST"

DIPST_OFFSET_CHUNK = b"COFF"
DIPST_DATA_CHUNK = b"DATA"

DIPST_FRAME_HEADER = b"DFRM"
DIPST_METADATA_HEADER = b"DMET"

DIPST_END_CHUNK = b"DEND"

Offset = np.dtype("<u8")


class DummyLock:
    """
    Lock class that does nothing, this is used to disable locking functionality if a lock is not passed.
    """
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class SharedMemory(Protocol):
    buf: memoryview


class FPEMetadataEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if(isinstance(o, Path)):
            return str(o)

        to_json = getattr(o, "__tojson__", None)
        if(to_json is None):
            return super().default(o)
        d = to_json()

        if(not type(o).__module__.startswith("diplomat.")):
            if(not type(o).__module__.startswith("matplotlib.colors")):
                raise IOError("Can only write diplomat internal modules to disk!")

        return {
            "___name": type(o).__qualname__,
            "___module": type(o).__module__,
            "___data": d
        }


def reconstruct_from_json(data: Union[dict, list]) -> Union[dict, list]:
    iterator = data.items() if(isinstance(data, dict)) else enumerate(data)
    for k, val in iterator:
        if(isinstance(val, dict)):
            if("___name" in val):
                if(not val["___module"].startswith("diplomat.")):
                    if(not val["___module"].startswith("matplotlib.colors")):
                        raise IOError("Only internal diplomat modules can be stored!")
                mod = import_module(val["___module"])
                cls = mod
                for attr in val["___name"].split("."):
                    cls = getattr(cls, attr)

                data[k] = cls.__fromjson__(val["___data"])
            else:
                data[k] = reconstruct_from_json(val)
        elif(isinstance(val, list)):
            data[k] = reconstruct_from_json(val)

    return data


class DiplomatFPEState:

    INFINITY = np.iinfo(np.int64).max
    METADATA_GROW_SIZE = 2

    def __init__(
        self,
        file_obj: BinaryIO,
        frame_count: int = 0,
        compression_level: int = 6,
        float_type: str = "<f4",
        int_type: str = "<i4",
        immediate_mode: bool = False,
        lock: Optional[multiprocessing.RLock] = None,
        memory_backing: Optional[Union[SharedMemory, Callable[[int], SharedMemory]]] = None
    ):
        self._file_obj = file_obj
        self._compression_level = compression_level
        self._file_start = 0
        self._float_type = float_type
        self._int_type = int_type
        self._immediate_mode = immediate_mode
        self._lock = lock if(lock is not None) else DummyLock()
        self._from_pickle = False
        self._closed = False

        self._shared_mem = memory_backing
        self._frame_offsets = None
        if(frame_count > 0):
            if(self._shared_mem is None):
                self._frame_offsets = np.zeros((frame_count + 1, 2), dtype=Offset)
            else:
                if(callable(self._shared_mem)):
                    self._shared_mem = self._shared_mem(
                        (frame_count + 1) * Offset.itemsize * 2
                        + BufferTree.get_buffer_size(frame_count + 2) * 2
                    )
                self._frame_offsets = np.ndarray(
                    (frame_count + 1, 2), dtype=Offset, buffer=self._shared_mem.buf, order="C"
                )
                self._frame_offsets[:] = 0

        self._find_chunks(frame_count)

        offset1 = self._frame_offsets.nbytes
        offset2 = offset1 + BufferTree.get_buffer_size(self._frame_offsets.shape[0] + 1)

        self._free_space = (
            BufferTree(memory_backing.buf[offset1:offset2])
            if(memory_backing is not None) else
            NumpyTree(self._frame_offsets.shape[0] + 1)
        )
        self._free_space_offsets = (
            BufferTree(memory_backing.buf[offset2:])
            if(memory_backing is not None) else
            NumpyTree(self._frame_offsets.shape[0] + 1)
        )
        if(
            self._free_space.data.shape[0] <= self._frame_offsets.shape[0]
            or self._free_space_offsets.data.shape[0] <= self._frame_offsets.shape[0]
        ):
            raise ValueError("Free space buffer not large enough...")
        self._compute_free_space()

    def _find_chunks(self, frame_count: int):
        with self._lock:
            self._file_obj.seek(-12, SEEK_END)
            data = self._file_obj.read(12)

            if(data[:4] == DIPST_END_CHUNK):
                self._file_start = int.from_bytes(data[4:], "little", signed=False)

            self._file_obj.seek(self._file_start)
            dip_header = self._file_obj.read(4)

            if(len(dip_header) == 0 or dip_header != DIPLOMAT_STATE_HEADER):
                if(frame_count <= 0):
                    raise IOError("No ui state found in this file.")
                self._file_obj.seek(0, SEEK_END)
                self._file_start = self._file_obj.tell()
                self._write_new_header()

            self._load_offsets()
            if((frame_count > 0) and ((self._frame_offsets.shape[0] - 1) != frame_count)):
                raise ValueError("Loaded file doesn't have same frame count!")

    def _compute_free_space(self):
        with self._lock:
            # Sort the frame offsets...
            order = np.argsort(self._frame_offsets[:, 0])
            prior_offset, prior_size = self._data_offset(), 0

            for i, i2 in enumerate(order):
                offset, size = self._frame_offsets[i2]
                if(i + 1 >= len(order)):
                    offset, size = self._frame_offsets[i2]
                    data_offset = self._data_offset() - self._file_start
                    insert(self._free_space, self.INFINITY, int(max(offset + size, data_offset)))
                    insert(self._free_space_offsets, int(max(offset + size, data_offset)), self.INFINITY)
                    break

                if(size == 0):
                    continue

                if(prior_offset + prior_size < offset):
                    insert(self._free_space, offset - (prior_offset + prior_size), prior_offset + prior_size)
                    insert(self._free_space_offsets, prior_offset + prior_size, offset - (prior_offset + prior_size))

                prior_offset, prior_size = offset, size

    def _data_offset(self) -> int:
        return int(
            self._file_start
            + len(DIPLOMAT_STATE_HEADER)
            + len(DIPST_OFFSET_CHUNK)
            + Offset.itemsize
            + int(self._frame_offsets.nbytes)
            + len(DIPST_DATA_CHUNK)
        )

    def _write_new_header(self):
        with self._lock:
            self._file_obj.write(DIPLOMAT_STATE_HEADER)
            # Offset chunk...
            self._file_obj.write(DIPST_OFFSET_CHUNK)
            self._write_offsets()
            # Data chunk...
            self._file_obj.write(DIPST_DATA_CHUNK)

    def _load_offsets(self):
        with self._lock:
            self._file_obj.seek(self._file_start + 4)
            magic = self._file_obj.read(4)
            if(magic != DIPST_OFFSET_CHUNK):
                raise IOError("Corrupted offset chunk!")

            length = int.from_bytes(self._file_obj.read(Offset.itemsize), "little", signed=False)

            if(self._frame_offsets is None):
                if(self._shared_mem is not None):
                    if(callable(self._shared_mem)):
                        self._shared_mem = self._shared_mem(
                            (length + 1) * Offset.itemsize * 2
                            + BufferTree.get_buffer_size(length + 2) * 2
                        )
                    self._frame_offsets = np.ndarray(
                        (1 + length, 2), dtype=Offset, buffer=self._shared_mem.buf, order="C"
                    )
                else:
                    self._frame_offsets = np.zeros((1 + length, 2), dtype=Offset)

            self._frame_offsets[:] = np.frombuffer(
                self._file_obj.read((2 + length * 2) * Offset.itemsize), Offset
            ).reshape((length + 1, 2))

    def _write_offsets(self):
        with self._lock:
            self._file_obj.seek(self._file_start + len(DIPLOMAT_STATE_HEADER))
            magic = self._file_obj.read(len(DIPST_OFFSET_CHUNK))
            if(magic != DIPST_OFFSET_CHUNK):
                raise IOError("Corrupted offset chunk!")

            self._file_obj.write(
                int(self._frame_offsets.shape[0] - 1).to_bytes(Offset.itemsize, "little", signed=False)
            )  # Size
            self._file_obj.write(self._frame_offsets.astype(Offset).tobytes())

    def _add_free_space(
        self,
        offset: int,
        size: int
    ):
        if(size <= 0):
            return

        offset_below, size_below = nearest_pop(self._free_space_offsets, offset, size, left=True)
        offset_above, size_above = nearest_pop(self._free_space_offsets, offset, size, left=False)

        if(offset_below is not None):
            remove(self._free_space, size_below, offset_below)
            if((offset_below + size_below) >= offset):
                if(self.INFINITY in [size, size_below]):
                    size = self.INFINITY
                else:
                    size = int(max(offset + size, offset_below + size_below) - offset_below)
                offset = offset_below
            else:
                insert(self._free_space, size_below, offset_below)
                insert(self._free_space_offsets, offset_below, size_below)

        if(offset_above is not None):
            remove(self._free_space, size_above, offset_above)
            if(offset_above <= (offset + size)):
                if(self.INFINITY in [size, size_above]):
                    size = self.INFINITY
                else:
                    size = int(max(offset_above + size_above, offset + size) - offset)
            else:
                insert(self._free_space, size_above, offset_above)
                insert(self._free_space_offsets, offset_above, size_above)

        insert(self._free_space, size, offset)
        insert(self._free_space_offsets, offset, size)

    def _find_free_space(self, size_needed: int) -> Tuple[int, int]:
        size, offset = nearest_pop(self._free_space, size_needed, left=False)
        if(size is None):
            raise RuntimeError("No free space!")
        remove(self._free_space_offsets, offset, size)
        return offset, size

    def _write_chunk(self, index: int, chunk_type: bytes, data: bytes):
        with self._lock:
            full_data = chunk_type + data
            if(index > self._frame_offsets.shape[0]):
                raise ValueError("Growth not supported yet...")

            offset, size = self._frame_offsets[index]
            needed_size = len(full_data) if(index > 0) else int(1 << int(np.ceil(np.log2(len(full_data)))))

            if(needed_size > size or needed_size < size):
                self._add_free_space(offset, size)
                new_offset, available_size = self._find_free_space(needed_size)

                if(available_size == self.INFINITY):
                    self._add_free_space(new_offset + needed_size, self.INFINITY)
                elif(needed_size < available_size):
                    self._add_free_space(new_offset + needed_size, available_size - needed_size)

                self._frame_offsets[index] = (new_offset, needed_size)

            offset, size = self._frame_offsets[index]
            from multiprocessing.process import current_process
            p = current_process()
            with open("DEBUG.txt", "a") as f:
                print(f"PROCESS {p.name} PID {p.pid}, WRITE FRAME {index} to {offset} with size {size}", file=f)

            if(self._immediate_mode):
                self._write_offsets()
                self._write_end()

            self._file_obj.seek(int(self._file_start + offset))
            self._file_obj.write(full_data)
            if(not isinstance(self._lock, DummyLock)):
                self._file_obj.flush()

    def _load_chunk(self, index: int) -> Tuple[bytes, bytes]:
        with self._lock:
            if(index > self._frame_offsets.shape[0]):
                raise ValueError("Index out of bounds")

            header_type = DIPST_FRAME_HEADER if(index != 0) else DIPST_METADATA_HEADER
            offset, size = self._frame_offsets[index]

            from multiprocessing.process import current_process
            p = current_process()
            with open("DEBUG.txt", "a") as f:
                print(f"PROCESS {p.name} PID {p.pid}, READ FRAME {index} from {offset} with size {size}", file=f)

            if(size == 0):
                return (header_type, b"")

            self._file_obj.seek(int(self._file_start + offset))
            data = self._file_obj.read(size)

            if(data[:len(header_type)] != header_type):
                print(data)
                raise IOError(f"Found incorrect chunk type for chunk {index}, (offset {offset}, size {size}).")

            return (header_type, data[4:])

    def _space_coverage(self):
        from diplomat.predictors.sfpe.avl_tree import inorder_traversal
        free_space = inorder_traversal(self._free_space_offsets)
        end = free_space[-1, 0]
        arr = np.zeros(shape=end, dtype=np.uint8)

        for offset, size in free_space[:-1]:
            arr[offset:offset + size] += 1
        for offset, size in self._frame_offsets:
            arr[offset:offset + size] += 2

        offset_space = self._data_offset() - self._file_start
        return [np.sum(arr[offset_space:] == i) for i in range(4)]

    def _is_fully_covered_no_overlap(self):
        cov = self._space_coverage()
        if(cov[0] != 0 or cov[-1] != 0):
            raise ValueError(cov)
        else:
            print("COVERAGE:", cov)

    def _encode_meta_chunk(self, data: dict = None) -> bytes:
        if(data is None):
            data = {}

        return zlib.compress(json.dumps(data, cls=FPEMetadataEncoder).encode(), self._compression_level)

    def _decode_meta_chunk(self, data: bytes) -> dict:
        if(len(data) == 0):
            return {}
        return reconstruct_from_json(json.loads(zlib.decompress(data).decode()))

    def _encode_frame(self, frame: ForwardBackwardFrame) -> bytes:
        return zlib.compress(frame.to_bytes(self._float_type, self._int_type), self._compression_level)

    def _decode_frame(self, data: bytes) -> ForwardBackwardFrame:
        if(len(data) == 0):
            return ForwardBackwardFrame()
        return ForwardBackwardFrame().from_bytes(self._float_type, self._int_type, zlib.decompress(data))

    def _write_end(self):
        with self._lock:
            self._file_obj.seek(-12, SEEK_END)
            end_data = self._file_obj.read(12)
            if(end_data[:len(DIPST_END_CHUNK)] != DIPST_END_CHUNK):
                self._file_obj.write(DIPST_END_CHUNK)
                self._file_obj.write(self._file_start.to_bytes(8, "little", signed=False))

    def __getitem__(self, item: int) -> ForwardBackwardFrame:
        if(self._closed):
            raise ValueError("State object is closed!")
        if(item < 0):
            raise IndexError("Negative indexes not supported...")
        __, data = self._load_chunk(1 + item)
        return self._decode_frame(data)

    def __setitem__(self, item: int, value: ForwardBackwardFrame):
        if(self._closed):
            raise ValueError("State object is closed!")
        if(item < 0):
            raise IndexError("Negative indexes not supported...")
        data = self._encode_frame(value)
        self._write_chunk(1 + item, DIPST_FRAME_HEADER, data)

    def __len__(self) -> int:
        return (self._frame_offsets.shape[0] - 1)

    def get_metadata(self) -> dict:
        if(self._closed):
            raise ValueError("State object is closed!")
        __, data = self._load_chunk(0)
        return self._decode_meta_chunk(data)

    def set_metadata(self, data: Mapping):
        if(self._closed):
            raise ValueError("State object is closed!")
        data = self._encode_meta_chunk(dict(data))
        self._write_chunk(0, DIPST_METADATA_HEADER, data)

    def flush(self):
        with self._lock:
            if(self._closed):
                raise ValueError("State object is closed!")
            self._write_offsets()
            self._write_end()
            self._file_obj.flush()

    def close(self):
        with self._lock:
            if(self._closed):
                return
            self.flush()
            if(self._from_pickle):
                self._file_obj.close()
            self._closed = True

    @classmethod
    def get_shared_memory_size(cls, frame_count: int) -> int:
        return (((frame_count + 1) * 2) * Offset.itemsize) + BufferTree.get_buffer_size(frame_count + 2) * 2

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    def __getstate__(self):
        state = self.__dict__.copy()

        if(state["_shared_mem"] is None):
            raise RuntimeError("Attempting to pickle a diplomat file state without memory backing.")

        state["_file_obj"] = state["_file_obj"].name
        state["_frame_offsets"] = (state["_frame_offsets"].shape[0] - 1)
        state["_free_space"] = None
        state["_frame_space_offsets"] = None
        state["_from_pickle"] = True
        return state

    def __setstate__(self, state: dict):
        self.__dict__ = state
        frame_count = self._frame_offsets
        offset1 = (frame_count + 1) * Offset.itemsize * 2
        offset2 = offset1 + BufferTree.get_buffer_size(frame_count + 2)

        self._frame_offsets = np.ndarray(
            (frame_count + 1, 2), dtype=Offset, buffer=self._shared_mem.buf[:offset1], order="C"
        )
        self._free_space = BufferTree(self._shared_mem.buf[offset1:offset2])
        self._free_space_offsets = BufferTree(self._shared_mem.buf[offset2:])
        self._file_obj = open(str(self._file_obj), "r+b")
