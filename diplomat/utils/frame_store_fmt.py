"""
Contains 2 Utility Classes for reading and writing the DeepLabCut Frame Store format. The format allows for processing
videos using DeepLabCut and then running predictions on the probability map data later. Below is a specification for
the DeepLabCut Frame Store format.

::

    DIPLOMAT FRAME STORE BINARY FORMAT (All multi-byte fields are in little-endian format)
    ['DLFS'] -> DipLomat (or Deep Learning) Frame Store - 4 Bytes (file magic)

    Header:
        ['DLFH'] -> Diplomat Header
        [num_frames] - the number of frames. 8 Bytes (long unsigned integer)
        [num_bp] - number of body parts contained per frame. 4 Bytes (unsigned integer)
        [frame_height] - The height of a frame. 4 Bytes (unsigned integer)
        [frame_width] - The width of a frame. 4 Bytes (unsigned integer)
        [frame_rate] - The frame rate, in frames per second. 8 Bytes (double float).
        [stride] - The original video upscaling multiplier relative to current frame size. 8 Bytes (double float)
        [orig_video_height] - The original video height. 4 Bytes (unsigned integer)
        [orig_video_width] - The original video width. 4 Bytes (unsigned integer)
        [crop_y1] - The y offset of the cropped box, set to max value to indicate no cropping... 4 Bytes (unsigned integer)
        [crop_x1] - The x offset of the cropped box, set to max value to indicate no cropping... 4 Bytes (unsigned integer)

    Bodypart Names:
        ['DBPN'] -> Diplomat Body Part Names
        (num_bp entries):
            [bp_len] - The length of the name of the bodypart. 2 Bytes (unsigned short)
            [DATA of length bp_len] - UTF8 Encoded name of the bodypart.

    Frame Lookup Chunk:
        ['FLUP'] -> Frame LookUP
        (num_frames entries):
            [frame_offset_ptr] -> The offset of frame i into the FDAT chunk, excluding the chunk signature. 8 Bytes (unsigned long)

    Frame data block:
        ['FDAT'] -> Frame DATa
        Now the data (num_frames entries):
            Each sub-frame entry (num_bp entries):

                Single Byte: 000000[offsets_included][sparse_fmt]:
                    [sparse_fmt]- Single bit, whether we are using the sparse format. See difference in storage below:
                    [offsets_included] - Single bit, whether we have offset data included. See difference in storage below:
                [data_length] - The length of the compressed/uncompressed frame data, 8 Bytes (long unsigned integer)

                DATA (The below is compressed in the zlib format and must be uncompressed first). Based on 'sparse_fmt' flag:

                    If it is false, frames are stored as 4 byte float arrays, row-by-row, as below (x, y order below):
                        prob(1, 1), prob(2, 1), prob(3, 1), ....., prob(x, 1)
                        prob(1, 2), prob(2, 2), prob(3, 2), ....., prob(x, 2)
                        .....................................................
                        prob(1, y), prob(2, y), prob(3, y), ....., prob(x, y)
                    Length of the above data will be frame height * frame width...
                    if [offsets_included] == 1:
                        Then 2 more maps equivalent to the above store the offset within the map when converting back
                        to video:
                            off_y(1, 1), off_y(2, 1), off_y(3, 1), ....., off_y(x, 1)
                            off_y(1, 2), off_y(2, 2), off_y(3, 2), ....., off_y(x, 2)
                            .........................................................
                            off_y(1, y), off_y(2, y), off_y(3, y), ....., off_y(x, y)

                            off_x(1, 1), off_x(2, 1), off_x(3, 1), ....., off_x(x, 1)
                            off_x(1, 2), off_x(2, 2), off_x(3, 2), ....., off_x(x, 2)
                            .........................................................
                            off_x(1, y), off_x(2, y), off_x(3, y), ....., off_x(x, y)
                    Otherwise frames are stored in the format below.

                    Sparse Frame Format (num_bp entries):
                        [num_entries] - Number of sparse entries in the frame, 8 bytes, unsigned integer.
                        [arr y] - list of 4 byte unsigned integers of length num_entries. Stores y coordinates of probabilities.
                        [arr x] - list of 4 byte unsigned integers of length num_entries. Stores x coordinates of probabilities.
                        [probs] - list of 4 byte floats, Stores probabilities specified at x and y coordinates above.
                        if [offsets_included] == 1:
                            [off y] - list of 4 byte floats, stores y offset within the block of pixels.
                            [off x] - list of 4 byte floats, stores x offset within the block of pixels.

    Frame Terminating/Ending Chunk:
        This chunk is optional, but when included allows this file to be placed onto the end of another file
        (in most cases this is a video file). Readers should first check for the above header, and if they don't see "DLFS",
        then seek to the end of the file and search for a DLFE chunk.

        ['DLFE'] -> Deep Learning Frame End
        [file_size] -> The size of the file, excluding this chunk (or the whole file size -12 bytes). (long unsigned integer, 8 bytes)


"""
import os
from io import BytesIO
from typing import Tuple
from diplomat.utils.frame_store_api import *
import numpy as np
import zlib


class DLFSConstants:
    """
    Class stores some constants for the Diplomat Frame Store format.
    """

    # The frame must become 1/3 or less its original size when sparsified to save space over the entire frame format,
    # so we check for this by dividing the original frame size by the sparse frame size and checking to see if it is
    # greater than this factor below...
    MIN_SPARSE_SAVING_FACTOR = 3
    # Magic...
    FILE_MAGIC = b"DLFS"
    # Chunk names...
    HEADER_CHUNK_MAGIC = b"DLFH"
    # The header length, including the 'DLFH' magic
    HEADER_LENGTH = 56
    BP_NAME_CHUNK_MAGIC = b"DBPN"
    FRAME_LOOKUP_MAGIC = b"FLUP"
    FRAME_DATA_CHUNK_MAGIC = b"FDAT"
    FRAME_END_CHUNK_MAGIC = b"DLFE"


class DLFSReader(FrameReader):
    """
    A DeepLabCut Frame Store Reader. Allows for reading ".dlfs" files.
    """

    # See spec above, describing each of these types in order...
    HEADER_DATA_TYPES = [
        luint64,
        luint32,
        luint32,
        luint32,
        ldouble,
        ldouble,
        luint32,
        luint32,
        luint32,
        luint32,
    ]
    HEADER_OFFSETS = np.cumsum([4] + [dtype.itemsize for dtype in HEADER_DATA_TYPES])[
        :-1
    ]

    def _assert_true(self, assertion: bool, error_msg: str):
        """
        Private method, if the assertion is false, throws a ValueError.
        """
        if not assertion:
            raise ValueError(error_msg)

    def _find_header(self, file: BinaryIO):
        """
        Private method, find the start of a dlfs file.
        """
        if(file.read(4) == DLFSConstants.FILE_MAGIC):
            return

        file.seek(-12, os.SEEK_END)
        tail = file.read(12)

        self._assert_true(
            tail[:4] == DLFSConstants.FRAME_END_CHUNK_MAGIC,
            "File is not of the DIPLOMAT Frame Store Format!"
        )

        offset = int(from_bytes(tail[4:], luint64))
        file.seek(-(offset + 12), os.SEEK_CUR)

        self._assert_true(
            file.read(4) == DLFSConstants.FILE_MAGIC,
            "File is not of the DIPLOMAT Frame Store Format!"
        )

    def __init__(self, file: BinaryIO):
        """
        Create a new DeepLabCut Frame Store Reader.

        :param file: The binary file object to read a frame store from, file opened with 'rb'.
        """
        super().__init__(file)

        # Search for the start of the file...
        self._find_header(file)

        # Check for valid header...
        header_bytes = file.read(DLFSConstants.HEADER_LENGTH)
        self._assert_true(
            header_bytes[0:4] == DLFSConstants.HEADER_CHUNK_MAGIC,
            "First Chunk must be the Header ('DLFH')!",
        )

        # Read the header into a DLFS header...
        parsed_data = [
            from_bytes(header_bytes[off : (off + dtype.itemsize)], dtype)
            for off, dtype in zip(self.HEADER_OFFSETS, self.HEADER_DATA_TYPES)
        ]
        self._header = DLFSHeader(parsed_data[0], *parsed_data[2:])
        body_parts = [None] * parsed_data[1]

        # Make sure cropping offsets land within the video if they are not None
        if self._header.crop_offset_y is not None:
            crop_end = self._header.crop_offset_y + (
                self._header.frame_height * self._header.stride
            )
            self._assert_true(
                crop_end < self._header.orig_video_height,
                "Cropping box in DLFS file is invalid!",
            )
        if self._header.crop_offset_x is not None:
            crop_end = self._header.crop_offset_x + (
                self._header.frame_width * self._header.stride
            )
            self._assert_true(
                crop_end < self._header.orig_video_width,
                "Cropping box in DLFS file is invalid!",
            )

        # Read the body part chunk...
        self._assert_true(
            file.read(4) == DLFSConstants.BP_NAME_CHUNK_MAGIC,
            "Body part chunk must come second!",
        )
        for i in range(len(body_parts)):
            length = from_bytes(file.read(2), luint16)
            body_parts[i] = file.read(int(length)).decode("utf-8")
        # Add the list of body parts to the header...
        self._header.bodypart_names = body_parts

        # Verify frame lookup chunk is there, and store its file offset.
        self._assert_true(
            file.read(4) == DLFSConstants.FRAME_LOOKUP_MAGIC,
            "Frame lookup chunk must come 3rd!"
        )
        self._flup_offset = file.tell()
        file.read(8 * self._header.number_of_frames)

        # Now we assert that we have reached the frame data chunk
        self._assert_true(
            file.read(4) == DLFSConstants.FRAME_DATA_CHUNK_MAGIC,
            f"Frame data chunk not found!",
        )
        self._fdat_offset = file.tell()

        self._file = file
        self._frames_processed = 0

    def get_header(self) -> DLFSHeader:
        """
        Get the header of this Diplomat frame store file.

        :return: A DLFSHeader object, contains important metadata info from this frame store.
        """
        return DLFSHeader(*self._header.to_list())

    def has_next(self, num_frames: int = 1) -> bool:
        """
        Checks if this frame store object at least num_frames more frames to be read.

        :param num_frames: An Integer, The number of frames to check the availability of, defaults to 1.
        :return: True if there are at least num_frames more frames to be read, otherwise False.
        """
        return (self._frames_processed + num_frames) <= self._header.number_of_frames

    @classmethod
    def _parse_flag_byte(cls, byte: luint8) -> Tuple[bool, bool]:
        """ Returns if it is of the sparse format, followed by if it includes offset data... """
        return ((byte & 1) == 1, ((byte >> 1) & 1) == 1)

    @classmethod
    def _take_array(
        cls, data: bytes, dtype: np.dtype, count: int
    ) -> Tuple[bytes, np.ndarray]:
        """ Reads a numpy array from the byte array, returning the leftover data and the array. """
        if count <= 0:
            raise ValueError("Can't have a negative amount of entries....")
        return (
            data[dtype.itemsize * count :],
            np.frombuffer(data, dtype=dtype, count=count),
        )

    @classmethod
    def _init_offset_data(cls, track_data: TrackingData):
        if track_data.get_offset_map() is None:
            # If tracking data is currently None, we need to create an empty array to store all data.
            shape = (
                track_data.get_frame_count(),
                track_data.get_frame_height(),
                track_data.get_frame_width(),
                track_data.get_bodypart_count(),
                2,
            )
            track_data.set_offset_map(np.zeros(shape, dtype=lfloat))

    def tell_frame(self) -> int:
        """
        Get the current frame the frame reader is on.

        :return: An integer, being the current frame the DLFSReader will read next.
        """
        return self._frames_processed

    def seek_frame(self, frame_idx: int):
        """
        Make this frame reader seek to the given frame index.

        :param frame_idx: An integer, the frame index to have this frame reader seek to. Must land within the valid
                          frame range for this file, being 0 to frame_count - 1.
        """
        if(not (0 <= frame_idx < self._header.number_of_frames)):
            raise IndexError(f"The provided frame index does not land within "
                             f"the valid range. (0 to {self._header.number_of_frames})")

        self._file.seek(self._flup_offset + (frame_idx * 8))
        data_offset = int(from_bytes(self._file.read(8), luint64))
        self._file.seek(self._fdat_offset + data_offset)
        self._frames_processed = frame_idx

    def read_frames(self, num_frames: int = 1) -> TrackingData:
        """
        Read the next num_frames frames from this frame store object and returns a TrackingData object.

        :param num_frames: The number of frames to read from the frame store.
        :return: A TrackingData object storing all frame info that was stored in this Diplomat Frame Store....

        :raises: An EOFError if more frames were requested then were available.
        """
        if not self.has_next(num_frames):
            frames_left = self._header.number_of_frames - self._frames_processed
            raise EOFError(
                f"Only '{frames_left}' were available, and '{num_frames}' were requested."
            )

        self._frames_processed += num_frames
        __, frame_h, frame_w, __, stride = self._header.to_list()[:5]
        bp_lst = self._header.bodypart_names

        track_data = TrackingData.empty_tracking_data(
            num_frames, len(bp_lst), frame_w, frame_h, stride
        )

        for frame_idx in range(track_data.get_frame_count()):
            for bp_idx in range(track_data.get_bodypart_count()):
                sparse_fmt_flag, has_offsets_flag = self._parse_flag_byte(
                    from_bytes(self._file.read(1), luint8)
                )
                data = zlib.decompress(
                    self._file.read(int(from_bytes(self._file.read(8), luint64)))
                )

                if sparse_fmt_flag:
                    entry_len = int(from_bytes(data[:8], luint64))

                    if(entry_len == 0):
                        # If the length is 0 there is no data, continue with following frames.
                        continue

                    data = data[8:]
                    data, sparse_y = self._take_array(
                        data, dtype=luint32, count=entry_len
                    )
                    data, sparse_x = self._take_array(
                        data, dtype=luint32, count=entry_len
                    )
                    data, probs = self._take_array(data, dtype=lfloat, count=entry_len)

                    if (
                        has_offsets_flag
                    ):  # If offset flag is set to true, load in offset data....
                        self._init_offset_data(track_data)

                        data, off_y = self._take_array(
                            data, dtype=lfloat, count=entry_len
                        )
                        data, off_x = self._take_array(
                            data, dtype=lfloat, count=entry_len
                        )
                        track_data.get_offset_map()[
                            frame_idx, sparse_y, sparse_x, bp_idx, 1
                        ] = off_y
                        track_data.get_offset_map()[
                            frame_idx, sparse_y, sparse_x, bp_idx, 0
                        ] = off_x

                    track_data.get_prob_table(frame_idx, bp_idx)[
                        sparse_y, sparse_x
                    ] = probs  # Set probability data...
                else:
                    data, probs = self._take_array(
                        data, dtype=lfloat, count=frame_w * frame_h
                    )
                    probs = np.reshape(probs, (frame_h, frame_w))

                    if has_offsets_flag:
                        self._init_offset_data(track_data)

                        data, off_y = self._take_array(
                            data, dtype=lfloat, count=frame_h * frame_w
                        )
                        data, off_x = self._take_array(
                            data, dtype=lfloat, count=frame_h * frame_w
                        )
                        off_y = np.reshape(off_y, (frame_h, frame_w))
                        off_x = np.reshape(off_x, (frame_h, frame_w))

                        track_data.get_offset_map()[frame_idx, :, :, bp_idx, 1] = off_y
                        track_data.get_offset_map()[frame_idx, :, :, bp_idx, 0] = off_x

                    track_data.get_prob_table(frame_idx, bp_idx)[:] = probs

        return track_data

    def close(self):
        """
        Close this frame reader, cleaning up any resources used during reading from the file. Does not close the passed
        file handle!
        """
        pass


class DLFSWriter(FrameWriter):
    """
    A DeepLabCut Frame Store Writer. Allows for writing ".dlfs" files.
    """
    def __init__(
        self,
        file: BinaryIO,
        header: DLFSHeader,
        threshold: Optional[float] = 1e-6,
        compression_level: int = 6,
    ):
        """
        Create a new DeepLabCut Frame Store Writer.

        :param file: The file to write to, a file opened in 'wb' mode.
        :param header: The DLFSHeader, with all properties filled out.
        :param threshold: A float between 0 and 1, the threshold at which to filter out any probabilities which fall
                          below it. The default value is 1e-6, and it can be set to None to force all frames to be
                          stored in the non-sparse format.
        :param compression_level: The compression of the data. 0 is no compression, 9 is max compression but is slow.
                                  The default is 6.
        """
        super().__init__(file, header, threshold)

        self._out_file = file
        self._file_start_offset = file.tell()
        self._header = header
        self._threshold = (
            threshold if (threshold is None or 0 <= threshold <= 1) else 1e-6
        )
        self._compression_level = (
            compression_level if (0 <= compression_level <= 9) else 6
        )
        self._current_frame = 0

        # Write the file magic...
        self._out_file.write(DLFSConstants.FILE_MAGIC)
        # Now we write the header:
        self._out_file.write(DLFSConstants.HEADER_CHUNK_MAGIC)
        self._out_file.write(
            to_bytes(header.number_of_frames, luint64)
        )  # The frame count
        self._out_file.write(
            to_bytes(len(header.bodypart_names), luint32)
        )  # The body part count
        self._out_file.write(
            to_bytes(header.frame_height, luint32)
        )  # The height of each frame
        self._out_file.write(
            to_bytes(header.frame_width, luint32)
        )  # The width of each frame
        self._out_file.write(
            to_bytes(header.frame_rate, ldouble)
        )  # The frames per second
        self._out_file.write(
            to_bytes(header.stride, ldouble)
        )  # The video up-scaling factor
        # Original video height and width.
        self._out_file.write(to_bytes(header.orig_video_height, luint32))
        self._out_file.write(to_bytes(header.orig_video_width, luint32))
        # The cropping (y, x) offset, or the max integer values if there is no cropping box...
        max_val = np.iinfo(luint32).max
        self._out_file.write(
            to_bytes(
                max_val if (header.crop_offset_y is None) else header.crop_offset_y,
                luint32,
            )
        )
        self._out_file.write(
            to_bytes(
                max_val if (header.crop_offset_x is None) else header.crop_offset_x,
                luint32,
            )
        )

        # Now we write the body part name chunk:
        self._out_file.write(DLFSConstants.BP_NAME_CHUNK_MAGIC)
        for bodypart in header.bodypart_names:
            body_bytes = bodypart.encode("utf-8")
            self._out_file.write(to_bytes(len(body_bytes), luint16))
            self._out_file.write(body_bytes)

        # The frame lookup chunk...
        self._out_file.write(DLFSConstants.FRAME_LOOKUP_MAGIC)
        self._flup_offset = self._out_file.tell()
        self._out_file.write(bytes(header.number_of_frames * 8))
        self._frame_offsets = np.zeros(header.number_of_frames, dtype=luint64)

        # Finish by writing the begining of the frame data chunk:
        self._out_file.write(DLFSConstants.FRAME_DATA_CHUNK_MAGIC)
        # Useful for figuring out how far frames are into the frame data chunk...
        self._fdat_offset = self._out_file.tell()

    def get_header(self) -> DLFSHeader:
        """
        Get the header of this DLFSWriter.

        :returns: The DLFSHeader of the frame store writer, which contains file metadata.
        """
        return DLFSHeader(*self._header.to_list())

    def _write_flup_data(self):
        """ Writes out current frame offset data into frame lookup chunk area of the file. """
        loc = self._out_file.tell()
        self._out_file.seek(self._flup_offset)
        self._out_file.write(self._frame_offsets.tobytes("C"))
        self._out_file.seek(loc)

    def _write_end_chunk(self):
        """ Writes out the final chunk of the file. """
        loc = self._out_file.tell()
        self._out_file.write(DLFSConstants.FRAME_END_CHUNK_MAGIC)
        self._out_file.write(to_bytes(loc - self._file_start_offset, luint64))

    def tell_frame(self) -> int:
        """
        Get the current frame the frame writer is on.

        :return: An integer, being the current frame the DLFSWriter will write out next.
        """
        return self._current_frame

    def write_data(self, data: TrackingData):
        """
        Write the following frames to the file.

        :param data: A TrackingData object, which contains frame data.
        """
        # Some checks to make sure tracking data parameters match those set in the header:
        self._current_frame += data.get_frame_count()
        if self._current_frame > self._header.number_of_frames:
            raise ValueError(
                f"Data Overflow! '{self._header.number_of_frames}' frames expected, tried to write "
                f"'{self._current_frame + 1}' frames."
            )

        if data.get_bodypart_count() != len(self._header.bodypart_names):
            raise ValueError(
                f"'{data.get_bodypart_count()}' body parts does not match the "
                f"'{len(self._header.bodypart_names)}' body parts specified in the header."
            )

        if(data.get_frame_width() != self._header.frame_width or data.get_frame_height() != self._header.frame_height):
            raise ValueError("Frame dimensions don't match ones specified in header!")

        for frm_idx in range(data.get_frame_count()):
            # Add this frame's offset to the offset list...
            idx = self._current_frame - data.get_frame_count() + frm_idx
            self._frame_offsets[idx] = self._out_file.tell() - self._fdat_offset

            for bp in range(data.get_bodypart_count()):
                frame = data.get_prob_table(frm_idx, bp)
                offset_table = data.get_offset_map()

                if offset_table is not None:
                    off_y = offset_table[frm_idx, :, :, bp, 1]
                    off_x = offset_table[frm_idx, :, :, bp, 0]
                else:
                    off_y = None
                    off_x = None

                if self._threshold is not None:
                    # Sparsify the data by removing everything below the threshold...
                    sparse_y, sparse_x = np.nonzero(frame > self._threshold)
                    probs = frame[(sparse_y, sparse_x)]

                    # Check if we managed to strip out at least 2/3rds of the data, and if so write the frame using the
                    # sparse format. Otherwise it is actually more memory efficient to just store the entire frame...
                    if len(frame.flat) >= (
                        len(sparse_y) * DLFSConstants.MIN_SPARSE_SAVING_FACTOR
                    ):
                        # Sparse indicator flag and the offsets included flag...
                        self._out_file.write(
                            to_bytes(True | ((offset_table is not None) << 1), luint8)
                        )
                        # COMPRESSED DATA:
                        buffer = BytesIO()
                        buffer.write(
                            to_bytes(len(sparse_y), luint64)
                        )  # The length of the sparse data entries.
                        buffer.write(
                            sparse_y.astype(luint32).tobytes("C")
                        )  # Y coord data
                        buffer.write(
                            sparse_x.astype(luint32).tobytes("C")
                        )  # X coord data
                        buffer.write(probs.astype(lfloat).tobytes("C"))  # Probabilities
                        if (
                            offset_table is not None
                        ):  # If offset table exists, write y offsets and then x offsets.
                            buffer.write(
                                off_y[(sparse_y, sparse_x)].astype(lfloat).tobytes("C")
                            )
                            buffer.write(
                                off_x[(sparse_y, sparse_x)].astype(lfloat).tobytes("C")
                            )
                        # Compress the sparse data and write it's length, followed by itself....
                        comp_data = zlib.compress(
                            buffer.getvalue(), self._compression_level
                        )
                        self._out_file.write(to_bytes(len(comp_data), luint64))
                        self._out_file.write(comp_data)

                        continue
                # If sparse optimization mode is off or the sparse format wasted more space, just write the entire
                # frame...
                self._out_file.write(
                    to_bytes(False | ((offset_table is not None) << 1), luint8)
                )

                buffer = BytesIO()
                buffer.write(
                    frame.astype(lfloat).tobytes("C")
                )  # The probability frame...
                if offset_table is not None:  # Y, then X offset data if it exists...
                    buffer.write(off_y.astype(lfloat).tobytes("C"))
                    buffer.write(off_x.astype(lfloat).tobytes("C"))

                comp_data = zlib.compress(buffer.getvalue(), self._compression_level)
                self._out_file.write(to_bytes(len(comp_data), luint64))
                self._out_file.write(comp_data)

        if(self._current_frame >= self._header.number_of_frames):
            # We have reached the end, dump the flup chunk
            self._write_flup_data()
            self._write_end_chunk()

    def close(self):
        """
        Close this frame writer, cleaning up any resources used during writing to the file. Does not close the passed
        file handle!
        """
        # If the file was only partially written, write the frame offset chunk for the frames that were written.
        if(self._current_frame < self._header.number_of_frames):
            # We have reached the end, dump the flup chunk
            self._write_flup_data()
            self._write_end_chunk()