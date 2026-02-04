import dataclasses
import json
import mmap
import multiprocessing
import os
import random
import shutil
import time
import warnings
from pathlib import Path
from typing import BinaryIO, Tuple, Optional, Mapping, Any, Union, Callable
from io import SEEK_END
import numpy as np
from PIL.ImageChops import offset
from typing_extensions import Protocol
from importlib import import_module
from diplomat.predictors.fpe.sparse_storage import ForwardBackwardFrame
from diplomat.predictors.sfpe.avl_tree import (
    BufferTree,
    NumpyTree,
    insert,
    nearest_pop,
    remove, Tree,
)
import zlib
import string


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


def reconstruct_from_json(data: Union[dict, list]) -> Union[dict, list]:
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
                data[k] = reconstruct_from_json(val)
        elif isinstance(val, list):
            data[k] = reconstruct_from_json(val)

    return data


class DiplomatFPEState:

    INFINITY = np.iinfo(np.int64).max
    METADATA_GROW_SIZE = 2

    def __init__(
        self,
        file_obj: Union[BinaryIO],
        frame_count: int = 0,
        compression_level: int = 6,
        float_type: str = "<f4",
        lock: Optional[multiprocessing.RLock] = None,
    ):
        # Get file length...
        c_loc = file_obj.tell()
        file_obj.seek(0, os.SEEK_END)
        file_length = file_obj.tell()
        file_obj.seek(c_loc)

        if file_length == 0:
            if frame_count <= 0:
                raise ValueError("Trying to read an empty file!")
            memmap_alloc = self.get_minimum_file_size(frame_count)
        else:
            memmap_alloc = 0

        self._file_map = mmap.mmap(file_obj.fileno(), memmap_alloc)
        self._compression_level = compression_level
        self._file_start = 0
        self._float_type = float_type
        self._lock = lock if (lock is not None) else DummyLock()
        self._from_pickle = False
        self._closed = False
        self._frame_offsets: np.ndarray = None
        self._free_space: Tree = None
        self._free_space_offsets: Tree = None

        self._find_chunks(frame_count, file_length)

        if (
            self._free_space.data.shape[0] <= self._frame_offsets.shape[0]
            or self._free_space_offsets.data.shape[0] <= self._frame_offsets.shape[0]
        ):
            raise ValueError("Free space buffer not large enough...")
        self._compute_free_space()

    def _find_chunks(self, frame_count: int, file_size: int):
        with self._lock:
            data = self._file_map[-12:]

            if data[:4] == DIPST_END_CHUNK:
                self._file_start = int.from_bytes(data[4:], "little", signed=False)
            else:
                self._file_start = file_size

            dip_header = self._file_map[self._file_start:self._file_start + 4]

            if dip_header != DIPLOMAT_STATE_HEADER and file_size != 0:
                warnings.warn("DIPLOMAT found possibly corrupted file, attempting to recover...")
                header_loc = self._file_map.rfind(DIPLOMAT_STATE_HEADER + DIPST_OFFSET_CHUNK)
                if header_loc >= 0:
                    dip_header = DIPLOMAT_STATE_HEADER
                    self._file_start = header_loc

            if dip_header != DIPLOMAT_STATE_HEADER:
                if frame_count <= 0:
                    raise IOError("No ui state found in this file.")
                self._file_start = file_size
                new_size = self._file_start + self.get_minimum_file_size(frame_count)
                if new_size > self._file_map.size():
                    self._file_map.resize(new_size)
                self._init_new_header(frame_count)

            self._init_offset_structures()

            if (frame_count > 0) and (
                (self._frame_offsets.shape[0] - 1) != frame_count
            ):
                raise ValueError("Loaded file doesn't have same frame count!")

    def _compute_free_space(self):
        with self._lock:
            # Sort the frame offsets...
            offsets_flat = self._frame_offsets.reshape((-1, 3))
            order = np.argsort(offsets_flat[:, 0])
            prior_offset, prior_size = self._data_offset(), 0

            for i, i2 in enumerate(order):
                offset, size, _ = offsets_flat[i2]
                if i + 1 >= len(order):
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
            + self.get_shared_structure_size(self._frame_offsets.shape[0])
            + len(DIPST_DATA_CHUNK)
        )

    def _write(self, offset: int, data: bytes):
        self._file_map[offset:offset + len(data)] = data
        return offset + len(data)

    def _simple_read(self, offset: int, size: int):
        return self._read(offset, size)[0]

    def _read(self, offset: int, size: int):
        return self._file_map[offset:offset + size], offset + size

    def _init_new_header(self, frame_count: int):
        with self._lock:
            frame_bytes = frame_count.to_bytes(Offset.itemsize, "little", signed=False)
            empty_space = self.get_shared_structure_size(frame_count)
            offset = self._write(self._file_start, (
                DIPLOMAT_STATE_HEADER
                + DIPST_OFFSET_CHUNK
                + frame_bytes
                + bytes(empty_space)
                + DIPST_DATA_CHUNK
                + DIPST_END_CHUNK
                + self._file_start.to_bytes(8, "little", signed=False)
            ))
            # Data chunk...
            return offset

    def _init_offset_structures(self):
        with self._lock:
            header, struct_offset = self._read(self._file_start + len(DIPLOMAT_STATE_HEADER), 12)

            if header[:4] != DIPST_OFFSET_CHUNK:
                raise IOError("Corrupted offset chunk!")

            length = int.from_bytes(header[4:], "little", signed=False)

            self._frame_offsets = np.ndarray(
                (length + 1, 2, 3),
                Offset,
                self._file_map,
                struct_offset
            )
            tree1_offset = struct_offset + self._frame_offsets.nbytes
            tree_size = BufferTree.get_buffer_size((length + 1) * 2 + 1)
            self._free_space = BufferTree(
                self._file_map,
                tree1_offset,
                tree_size
            )
            tree2_offset = tree1_offset + tree_size
            self._free_space_offsets = BufferTree(
                self._file_map,
                tree2_offset,
                tree_size
            )

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

    def _select_frame(self, frames, select_old: bool = False) -> tuple[int, np.ndarray]:
        # Frame 2 is newer, swap them...
        select_second = (frames[1, 2] - frames[0, 2]).astype(np.int64) > 0
        select_second = int(not select_second if select_old else select_second)

        return select_second, frames[select_second]

    def _write_chunk(self, index: int, chunk_type: bytes, data: bytes):
        with self._lock:
            full_data = chunk_type + data
            if index > self._frame_offsets.shape[0]:
                raise ValueError("Growth not supported yet...")

            frame = self._frame_offsets[index]
            frame_version_idx, (offset, size, version) = self._select_frame(frame, True)

            needed_size = (
                len(full_data)
                if (index > 0)
                else int(1 << int(np.ceil(np.log2(len(full_data)))))
            )
            appending_to_end = False

            if needed_size > size or needed_size < size:
                self._add_free_space(offset, size)
                new_offset, available_size = self._find_free_space(needed_size)

                if available_size == self.INFINITY:
                    appending_to_end = True
                    total_file_size = self._file_start + new_offset + needed_size + 12
                    if self._file_map.size() < total_file_size:
                        self._file_map.resize(total_file_size)
                    self._add_free_space(new_offset + needed_size, self.INFINITY)
                elif needed_size < available_size:
                    self._add_free_space(
                        new_offset + needed_size, available_size - needed_size
                    )

                self._frame_offsets[index, frame_version_idx] = (
                    new_offset,
                    needed_size,
                    self._frame_offsets[index, int(not frame_version_idx), 2] + 1
                )

            offset, size, version = self._frame_offsets[index, frame_version_idx]
            if appending_to_end:
                full_data = (
                    full_data
                    + DIPST_END_CHUNK
                    + self._file_start.to_bytes(Offset.itemsize, "little", signed=False)
                )

            self._write(int(self._file_start + offset), full_data)

    def _load_chunk(self, index: int, use_fallback: bool = False) -> Tuple[bytes, int, bytes]:
        with self._lock:
            if index > self._frame_offsets.shape[0]:
                raise ValueError("Index out of bounds")

            header_type = DIPST_FRAME_HEADER if (index != 0) else DIPST_METADATA_HEADER
            frame = self._frame_offsets[index]

            frame_idx, (offset, size, version) = self._select_frame(frame, use_fallback)

            if size == 0:
                return (header_type, frame_idx, b"")

            data, _ = self._read(int(self._file_start + offset), size)

            if data[: len(header_type)] != header_type:
                raise IOError(
                    f"Found incorrect chunk type for chunk {index}, (offset {offset}, size {size}).\nData:\n{data}"
                )

            return (header_type, frame_idx, data[len(header_type):])

    def _robust_load_chunk(self, index: int, decoder: Callable[[bytes], Any]):
        with self._lock:
            try:
                __, __, data = self._load_chunk(index, False)
                data = decoder(data)
            except Exception:
                warnings.warn(f"Fallback to old data for chunk {index}")
                __, frame_idx, data = self._load_chunk(index, False)
                data = decoder(data)
                # Increment the versioning number...
                self._frame_offsets[index, int(not frame_idx), 2] = self._frame_offsets[index, frame_idx, 2] + 1

        return data

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
                json.dumps(data, cls=FPEMetadataEncoder).encode(),
                self._compression_level,
            )
        except TypeError as e:
            raise ValueError(f"Bad metadata object: {data}") from e

    def _decode_meta_chunk(self, data: bytes) -> dict:
        if len(data) == 0:
            return {}
        return reconstruct_from_json(json.loads(zlib.decompress(data).decode()))

    def _encode_frame(self, frame: ForwardBackwardFrame) -> bytes:
        return zlib.compress(frame.to_bytes(self._float_type), self._compression_level)

    def _decode_frame(self, data: bytes) -> ForwardBackwardFrame:
        if len(data) == 0:
            return ForwardBackwardFrame()
        return ForwardBackwardFrame().from_bytes(
            self._float_type, zlib.decompress(data)
        )

    def __getitem__(self, item: int) -> ForwardBackwardFrame:
        if self._closed:
            raise ValueError("State object is closed!")
        if item < 0:
            raise IndexError("Negative indexes not supported...")
        __, data = self._robust_load_chunk(1 + item, self._decode_frame)
        return data

    def __setitem__(self, item: int, value: ForwardBackwardFrame):
        if self._closed:
            raise ValueError("State object is closed!")
        if item < 0:
            raise IndexError("Negative indexes not supported...")
        try:
            data = self._encode_frame(value)
        except Exception as e:
            # Print frame data so we get more info about a failure...
            raise ValueError(f"Failed to encode frame data: {value} at index {item}.") from e
        self._write_chunk(1 + item, DIPST_FRAME_HEADER, data)

    def __len__(self) -> int:
        return self._frame_offsets.shape[0] - 1

    def get_metadata(self) -> dict:
        if self._closed:
            raise ValueError("State object is closed!")
        return self._robust_load_chunk(0, self._decode_meta_chunk)

    def set_metadata(self, data: Mapping):
        if self._closed:
            raise ValueError("State object is closed!")
        data = self._encode_meta_chunk(dict(data))
        self._write_chunk(0, DIPST_METADATA_HEADER, data)

    def flush(self):
        with self._lock:
            if self._closed:
                raise ValueError("State object is closed!")
            self._file_map.flush()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self.flush()
            self._closed = True

    @classmethod
    def get_shared_structure_size(cls, frame_count: int) -> int:
        return (((frame_count + 1) * 6) * Offset.itemsize) + BufferTree.get_buffer_size(
            ((frame_count + 1) * 2) + 1
        ) * 2

    @classmethod
    def get_minimum_file_size(cls, frame_count: int) -> int:
        return int(
            len(DIPLOMAT_STATE_HEADER)
            + len(DIPST_OFFSET_CHUNK)
            + Offset.itemsize
            + cls.get_shared_structure_size(frame_count)
            + len(DIPST_DATA_CHUNK)
            + len(DIPST_END_CHUNK)
            + Offset.itemsize
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def closed(self) -> bool:
        return self._closed
