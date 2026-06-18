import json
import zlib
from importlib import import_module
from io import SEEK_END
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Tuple, Union

import numpy as np

from diplomat.predictors.fpe.sparse_storage import ForwardBackwardFrame
from diplomat.predictors.sfpe.avl_tree import (
    NumpyTree,
    insert,
    nearest_pop,
    remove,
)

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


class OldFPEMetadataEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)

        if isinstance(o, np.generic):
            return o.tolist()

        to_json = getattr(o, "__tojson__", None)
        if to_json is None:
            print(o)
            return super().default(o)
        d = to_json()

        if not type(o).__module__.startswith("diplomat."):
            if not type(o).__module__.startswith("matplotlib.colors"):
                raise IOError("Can only write diplomat internal modules to disk!")

        return {
            "___name": type(o).__qualname__,
            "___module": type(o).__module__,
            "___data": d,
        }


def old_reconstruct_from_json(data: Union[dict, list]) -> Union[dict, list]:
    iterator = data.items() if (isinstance(data, dict)) else enumerate(data)
    for k, val in iterator:
        if isinstance(val, dict):
            if "___name" in val:
                if not val["___module"].startswith("diplomat."):
                    raise IOError("Only internal diplomat modules can be stored!")
                mod = import_module(val["___module"])
                cls = mod
                for attr in val["___name"].split("."):
                    cls = getattr(cls, attr)

                data[k] = cls.__fromjson__(val["___data"])
            else:
                data[k] = old_reconstruct_from_json(val)
        elif isinstance(val, list):
            data[k] = old_reconstruct_from_json(val)

    return data


class OldDiplomatFPEState:
    INFINITY = np.iinfo(np.int64).max
    METADATA_GROW_SIZE = 2

    def __init__(
        self,
        file_obj: BinaryIO,
        frame_count: int = 0,
        compression_level: int = 6,
        float_type: str = "<f4",
    ):
        self._file_obj = file_obj
        self._compression_level = compression_level
        self._file_start = 0
        self._float_type = float_type
        self._closed = False

        self._frame_offsets: np.ndarray = None
        if frame_count > 0:
            self._frame_offsets = np.zeros((frame_count + 1, 2), dtype=Offset)

        self._find_chunks(frame_count)

        if self._frame_offsets is None:
            raise RuntimeError("Should not be possible!")

        self._free_space = NumpyTree(self._frame_offsets.shape[0] + 1)
        self._free_space_offsets = NumpyTree(self._frame_offsets.shape[0] + 1)
        self._compute_free_space()

    def file_start(self) -> int:
        return self._file_start

    def _find_chunks(self, frame_count: int):
        self._file_obj.seek(-12, SEEK_END)
        data = self._file_obj.read(12)

        if data[:4] == DIPST_END_CHUNK:
            self._file_start = int.from_bytes(data[4:], "little", signed=False)

        self._file_obj.seek(self._file_start)
        dip_header = self._file_obj.read(4)

        if len(dip_header) == 0 or dip_header != DIPLOMAT_STATE_HEADER:
            if frame_count <= 0:
                raise IOError("No ui state found in this file.")
            self._file_obj.seek(0, SEEK_END)
            self._file_start = self._file_obj.tell()
            self._write_new_header()

        self._load_offsets()
        if (frame_count > 0) and ((self._frame_offsets.shape[0] - 1) != frame_count):
            raise ValueError("Loaded file doesn't have same frame count!")

    def _compute_free_space(self):
        # Sort the frame offsets...
        order = np.argsort(self._frame_offsets[:, 0])
        prior_offset, prior_size = self._data_offset(), 0

        for i, i2 in enumerate(order):
            offset, size = self._frame_offsets[i2]
            if i + 1 >= len(order):
                offset, size = self._frame_offsets[i2]
                data_offset = self._data_offset() - self._file_start
                insert(
                    self._free_space,
                    self.INFINITY,
                    int(max(offset + size, data_offset)),
                )
                insert(
                    self._free_space_offsets,
                    int(max(offset + size, data_offset)),
                    self.INFINITY,
                )
                break

            if size == 0:
                continue

            if prior_offset + prior_size < offset:
                insert(
                    self._free_space,
                    offset - (prior_offset + prior_size),
                    prior_offset + prior_size,
                )
                insert(
                    self._free_space_offsets,
                    prior_offset + prior_size,
                    offset - (prior_offset + prior_size),
                )

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
        self._file_obj.write(DIPLOMAT_STATE_HEADER)
        # Offset chunk...
        self._file_obj.write(DIPST_OFFSET_CHUNK)
        self._write_offsets()
        # Data chunk...
        self._file_obj.write(DIPST_DATA_CHUNK)

    def _load_offsets(self):
        self._file_obj.seek(self._file_start + 4)
        magic = self._file_obj.read(4)
        if magic != DIPST_OFFSET_CHUNK:
            raise IOError("Corrupted offset chunk!")

        length = int.from_bytes(
            self._file_obj.read(Offset.itemsize), "little", signed=False
        )

        if self._frame_offsets is None:
            self._frame_offsets = np.zeros((1 + length, 2), dtype=Offset)

        self._frame_offsets[:] = np.frombuffer(
            self._file_obj.read((2 + length * 2) * Offset.itemsize), Offset
        ).reshape((length + 1, 2))

        if self._file_obj.read(4) != DIPST_DATA_CHUNK:
            raise RuntimeError("File is missing data chunk!")

    def _write_offsets(self):
        self._file_obj.seek(self._file_start + len(DIPLOMAT_STATE_HEADER))
        magic = self._file_obj.read(len(DIPST_OFFSET_CHUNK))
        if magic != DIPST_OFFSET_CHUNK:
            raise IOError("Corrupted offset chunk!")

        self._file_obj.write(
            int(self._frame_offsets.shape[0] - 1).to_bytes(
                Offset.itemsize, "little", signed=False
            )
        )  # Size
        self._file_obj.write(self._frame_offsets.astype(Offset).tobytes())

    def _add_free_space(self, offset: int, size: int):
        if size <= 0:
            return

        offset_below, size_below = nearest_pop(
            self._free_space_offsets, offset, size, left=True
        )
        offset_above, size_above = nearest_pop(
            self._free_space_offsets, offset, size, left=False
        )

        if offset_below is not None:
            remove(self._free_space, size_below, offset_below)
            if (offset_below + size_below) >= offset:
                if self.INFINITY in [size, size_below]:
                    size = self.INFINITY
                else:
                    size = int(
                        max(offset + size, offset_below + size_below) - offset_below
                    )
                offset = offset_below
            else:
                insert(self._free_space, size_below, offset_below)
                insert(self._free_space_offsets, offset_below, size_below)

        if offset_above is not None:
            remove(self._free_space, size_above, offset_above)
            if offset_above <= (offset + size):
                if self.INFINITY in [size, size_above]:
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
        if size is None:
            raise RuntimeError("No free space!")
        remove(self._free_space_offsets, offset, size)
        return offset, size

    def _write_chunk(self, index: int, chunk_type: bytes, data: bytes):
        full_data = chunk_type + data
        if index > self._frame_offsets.shape[0]:
            raise ValueError("Growth not supported yet...")

        offset, size = self._frame_offsets[index]
        needed_size = (
            len(full_data)
            if (index > 0)
            else int(1 << int(np.ceil(np.log2(len(full_data)))))
        )

        if needed_size > size or needed_size < size:
            self._add_free_space(offset, size)
            new_offset, available_size = self._find_free_space(needed_size)

            if available_size == self.INFINITY:
                self._add_free_space(new_offset + needed_size, self.INFINITY)
            elif needed_size < available_size:
                self._add_free_space(
                    new_offset + needed_size, available_size - needed_size
                )

            self._frame_offsets[index] = (new_offset, needed_size)

        offset, size = self._frame_offsets[index]

        self._file_obj.seek(int(self._file_start + offset))
        self._file_obj.write(full_data)

    def _load_chunk(self, index: int) -> Tuple[bytes, bytes]:
        if index > self._frame_offsets.shape[0]:
            raise ValueError("Index out of bounds")

        header_type = DIPST_FRAME_HEADER if (index != 0) else DIPST_METADATA_HEADER
        offset, size = self._frame_offsets[index]

        if size == 0:
            return (header_type, b"")

        self._file_obj.seek(int(self._file_start + offset))
        data = self._file_obj.read(size)

        if data[: len(header_type)] != header_type:
            print(data)
            raise IOError(
                f"Found incorrect chunk type for chunk {index}, (offset {offset}, size {size})."
            )

        return (header_type, data[4:])

    def _space_coverage(self):
        from diplomat.predictors.sfpe.avl_tree import inorder_traversal

        free_space = inorder_traversal(self._free_space_offsets)
        end = free_space[-1, 0]
        arr = np.zeros(shape=end, dtype=np.uint8)

        for offset, size in free_space[:-1]:
            arr[offset : offset + size] += 1
        for offset, size in self._frame_offsets:
            arr[offset : offset + size] += 2

        offset_space = self._data_offset() - self._file_start
        return [np.sum(arr[offset_space:] == i) for i in range(4)]

    def _is_fully_covered_no_overlap(self):
        cov = self._space_coverage()
        if cov[0] != 0 or cov[-1] != 0:
            raise ValueError(cov)
        else:
            print("COVERAGE:", cov)

    def _encode_meta_chunk(self, data: dict = None) -> bytes:
        if data is None:
            data = {}
        try:
            return zlib.compress(
                json.dumps(data, cls=OldFPEMetadataEncoder).encode(),
                self._compression_level,
            )
        except TypeError as e:
            raise ValueError(f"Bad metadata object: {data}") from e

    def _decode_meta_chunk(self, data: bytes) -> dict:
        if len(data) == 0:
            return {}
        return old_reconstruct_from_json(json.loads(zlib.decompress(data).decode()))

    def _encode_frame(self, frame: ForwardBackwardFrame) -> bytes:
        return zlib.compress(frame.to_bytes(self._float_type), self._compression_level)

    def _decode_frame(self, data: bytes) -> ForwardBackwardFrame:
        if len(data) == 0:
            return ForwardBackwardFrame()
        return ForwardBackwardFrame().from_bytes(
            self._float_type, zlib.decompress(data)
        )

    def _write_end(self):
        self._file_obj.seek(-12, SEEK_END)
        end_data = self._file_obj.read(12)
        if end_data[: len(DIPST_END_CHUNK)] != DIPST_END_CHUNK:
            self._file_obj.write(DIPST_END_CHUNK)
            self._file_obj.write(self._file_start.to_bytes(8, "little", signed=False))

    def __getitem__(self, item: int) -> ForwardBackwardFrame:
        if self._closed:
            raise ValueError("State object is closed!")
        if item < 0:
            raise IndexError("Negative indexes not supported...")
        __, data = self._load_chunk(1 + item)
        return self._decode_frame(data)

    def __setitem__(self, item: int, value: ForwardBackwardFrame):
        if self._closed:
            raise ValueError("State object is closed!")
        if item < 0:
            raise IndexError("Negative indexes not supported...")
        try:
            data = self._encode_frame(value)
        except Exception as e:
            # Print frame data so we get more info about a failure...
            raise ValueError(
                f"Failed to encode frame data: {value} at index {item}."
            ) from e
        self._write_chunk(1 + item, DIPST_FRAME_HEADER, data)

    def __len__(self) -> int:
        return self._frame_offsets.shape[0] - 1

    def get_metadata(self) -> dict:
        if self._closed:
            raise ValueError("State object is closed!")
        __, data = self._load_chunk(0)
        return self._decode_meta_chunk(data)

    def set_metadata(self, data: Mapping):
        if self._closed:
            raise ValueError("State object is closed!")
        data = self._encode_meta_chunk(dict(data))
        self._write_chunk(0, DIPST_METADATA_HEADER, data)

    def flush(self):
        if self._closed:
            raise ValueError("State object is closed!")
        if ("r" not in self._file_obj.mode) or ("+" in self._file_obj.mode):
            self._write_offsets()
            self._write_end()
        self._file_obj.flush()

    def close(self):
        if self._closed:
            return
        self.flush()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    def __getstate__(self):
        raise RuntimeError(
            "Attempting to pickle an old diplomat state file, use the new file format..."
        )

    def __setstate__(self, state: dict): ...
