import json
import multiprocessing
from typing import BinaryIO, Tuple, Optional, Mapping
from io import SEEK_CUR, SEEK_END, SEEK_SET
import numpy as np

from diplomat.predictors.fpe.sparse_storage import ForwardBackwardFrame
import zlib


DIPLOMAT_STATE_HEADER = b"DPST"

DIPST_OFFSET_CHUNK = b"COFF"
DIPST_DATA_CHUNK = b"DATA"

DIPST_FRAME_HEADER = b"DFRM"
DIPST_METADATA_HEADER = b"DMET"

DIPST_END_CHUNK = b"DEND"

Offset = np.dtype("<u8")


class DiplomatFPEState:
    def __init__(
        self,
        file_obj: BinaryIO,
        frame_count: int = 0,
        compression_level: int = 6,
        float_type: str = "<f4",
        int_type: str = "<u4",
        immediate_mode: bool = False
    ):
        self._file_obj = file_obj
        self._compression_level = compression_level
        self._frame_offsets = np.zeros((frame_count + 1, 2), dtype=Offset) if(frame_count > 0) else None
        self._file_start = 0
        self._float_type = float_type
        self._int_type = int_type
        self._free_space = {}
        self._immediate_mode = immediate_mode

        self._find_chunks(frame_count)
        self._compute_free_space()

    def _find_chunks(self, frame_count: int):
        self._file_obj.seek(-8, SEEK_END)
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
        if((self._frame_offsets.shape[0] - 1) != frame_count):
            raise ValueError("Loaded file doesn't have same frame count!")

    def _compute_free_space(self):
        # Sort the frame offsets...
        order = np.argsort(self._frame_offsets[:, 0])

        for i, i2 in enumerate(order):
            offset, size = self._frame_offsets[i2]
            if(i + 1 >= len(order)):
                offset, size = self._frame_offsets[i2]
                self._free_space[i2] = (int(max(offset, self._data_offset()) + size), np.inf)
                break

            offset2, size2 = self._frame_offsets[i2 + 1]
            if(offset + size < offset2):
                self._free_space[i2] = ((offset + size, offset2 - (offset + size)))

    def _data_offset(self) -> int:
        return int(
            self._file_start
            + len(DIPLOMAT_STATE_HEADER)
            + len(DIPST_OFFSET_CHUNK)
            + int(self._frame_offsets.size * Offset.itemsize)
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
        if(magic != DIPST_OFFSET_CHUNK):
            raise IOError("Corrupted offset chunk!")

        length = int.from_bytes(self._file_obj.read(Offset.itemsize), "little", signed=False)
        self._frame_offsets = np.copy(np.frombuffer(
            self._file_obj.read((2 + length * 2) * Offset.itemsize), Offset
        ).reshape((length + 1, 2)))

    def _write_offsets(self):
        self._file_obj.seek(self._file_start + 4)
        magic = self._file_obj.read(4)
        if(magic != DIPST_OFFSET_CHUNK):
            raise IOError("Corrupted offset chunk!")

        self._file_obj.write(int(self._frame_offsets.shape[0] - 1).to_bytes(8, "little", signed=False))  # Size
        self._file_obj.write(self._frame_offsets.astype(Offset).tobytes())

    def _find_free_space(
        self,
        size_needed: int,
        preset_position: Optional[Tuple[int, int, int]] = None
    ) -> Tuple[int, int, int]:
        for prior_frame, (offset, size) in sorted(self._free_space.items(), key=lambda k: k[1][0]):
            if(preset_position is not None and offset > preset_position[1]):
                return preset_position
            if(size == -1 or size_needed <= size):
                return (prior_frame, offset, size)

    def _write_chunk(self, index: int, chunk_type: bytes, data: bytes):
        full_data = chunk_type + data
        if(index > self._frame_offsets.shape[0]):
            raise ValueError("Growth not supported yet...")

        offset, size = self._frame_offsets[index]
        needed_size = len(full_data) if(index > 0) else int(2 ** np.ceil(np.log2(len(full_data))))

        if(needed_size > size):
            after_idx, new_offset, available_size = self._find_free_space(needed_size)

            if(after_idx == index):
                # Were needing to move the item at the end of the file, we can move the offset down to take the space
                # it's already sitting in, and just have it 'grow' out.
                new_offset = self._frame_offsets[index, 0]

            del self._free_space[after_idx]
            if(available_size == np.inf):
                self._free_space[index] = (new_offset + needed_size, np.inf)
            elif(needed_size < available_size):
                self._free_space[index] = (new_offset + needed_size, available_size - needed_size)

            self._frame_offsets[index] = (new_offset, needed_size)
        elif(needed_size < size):
            after_idx, new_offset, available_size = self._find_free_space(needed_size, (index, offset, size))
            if(after_idx in self._free_space):
                del self._free_space[after_idx]
            self._free_space[index] = (new_offset + needed_size, available_size - needed_size)
            self._frame_offsets[index] = (new_offset, needed_size)

        offset, size = self._frame_offsets[index]

        if(self._immediate_mode):
            self._write_offsets()
            self._write_end()

        self._file_obj.seek(offset)
        self._file_obj.write(full_data)

    def _load_chunk(self, index: int) -> Tuple[bytes, bytes]:
        if(index > self._frame_offsets.shape[0]):
            raise ValueError("Index out of bounds")

        header_type = DIPST_FRAME_HEADER if(index != 0) else DIPST_METADATA_HEADER
        offset, size = self._frame_offsets[index]

        if(size == 0):
            return (header_type, b"")

        self._file_obj.seek(self._file_start + offset)
        data = self._file_obj.read(size)

        if(data[:len(header_type)] != header_type):
            raise IOError(f"Found incorrect chunk type for chunk {index}.")

        return (header_type, data[4:])

    def _encode_meta_chunk(self, data: dict = None) -> bytes:
        if(data is None):
            data = {}

        return zlib.compress(json.dumps(data).encode(), self._compression_level)

    def _decode_meta_chunk(self, data: bytes) -> dict:
        if(len(data) == 0):
            return {}
        return json.loads(zlib.decompress(data).decode())

    def _encode_frame(self, frame: ForwardBackwardFrame) -> bytes:
        return zlib.compress(frame.to_bytes(self._float_type, self._int_type), self._compression_level)

    def _decode_frame(self, data: bytes) -> ForwardBackwardFrame:
        if(len(data) == 0):
            return ForwardBackwardFrame()
        return ForwardBackwardFrame().from_bytes(self._float_type, self._int_type, zlib.decompress(data))

    def _write_end(self):
        self._file_obj.seek(0, SEEK_END)
        self._file_obj.write(DIPST_END_CHUNK)
        self._file_obj.write(self._file_start.to_bytes(8, "little", signed=False))

    def __getitem__(self, item: int) -> ForwardBackwardFrame:
        if(item < 0):
            raise IndexError("Negative indexes not supported...")
        __, data = self._load_chunk(1 + item)
        return self._decode_frame(data)

    def __setitem__(self, item: int, value: ForwardBackwardFrame):
        if(item < 0):
            raise IndexError("Negative indexes not supported...")
        data = self._encode_frame(value)
        self._write_chunk(item, DIPST_FRAME_HEADER, data)

    def __len__(self) -> int:
        return self._frame_offsets.shape[0]

    def get_metadata(self) -> dict:
        __, data = self._load_chunk(0)
        return self._decode_meta_chunk(data)

    def set_metadata(self, data: Mapping):
        data = self._encode_meta_chunk(data)
        self._write_chunk(0, DIPST_METADATA_HEADER, data)

    def flush(self):
        self._write_offsets()
        self._write_end()

    def close(self):
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

