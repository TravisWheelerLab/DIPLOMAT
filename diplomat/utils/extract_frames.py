"""
Provides utility functions for quickly extracting frames from diplomat frame store files, and also printing frame data to the terminal
for debugging and display purposes.
"""
from typing import BinaryIO, Sequence, Callable, Optional, Generator, Union, Tuple, NamedTuple, List
from diplomat.processing import TrackingData
from diplomat.utils import frame_store_fmt
from io import BytesIO
import base64
import numpy as np
import cv2


def extract_frames(
    dlfs_in: BinaryIO,
    dlfs_out: BinaryIO,
    frames: Sequence[int],
    threshold: float = 1e-6,
    compression_lvl: int = 6,
    on_frames: Optional[Callable[[TrackingData], None]] = None
):
    """
    Extract frames from a DIPLOMAT frame store and store them in another frame store.

    :param dlfs_in: A binary file, the input DeepLabCut frame store file. Frames will be extracted from this file.
    :param dlfs_out: A binary file, the file to write a DeepLabCut frame store to. This will contain the extracted
                      frames.
    :param frames: A sequence of integers, the indexes of the frames to extract, in order of extraction.
    :param threshold: A float between 0 and 1, inclusive. The frame sparsification threshold, any probabilities below
                      the threshold are ignored. If set to 0, sparsification of frames is disabled. Default is 1e6.
    :param compression_lvl: An integer between 0 and 9, inclusive. The compression level if using the .dlfs format.
                            Default is 6.
    :param on_frames: A function that accepts a TrackingData for each frame. Useful for pretty printing/doing something
                      with the data during rewrite. Defaults to None, meaning it is not called.
    """
    file_reader = frame_store_fmt.DLFSReader(dlfs_in)

    header = file_reader.get_header()
    header.number_of_frames = len(frames)
    args = (dlfs_out, header, threshold, compression_lvl)
    file_writer = frame_store_fmt.DLFSWriter(*args)

    for idx in frames:
        file_reader.seek_frame(idx)
        frm = file_reader.read_frames(1)

        if(on_frames is not None):
            on_frames(frm)

        file_writer.write_data(frm)

    file_writer.close()
    file_reader.close()


def get_terminal_size(fallback: Tuple[int, int] = (80, 24)):
    """
    Get the size of the terminal.

    :param fallback: The fallback size to use if one can't be found.

    :return: A tuple of two ints, the size of the terminal in characters and lines.
    """
    import os

    for file_desc in range(3):
        try:
            return os.get_terminal_size(file_desc)
        except OSError:
            pass

    return fallback


class BorderStyle(NamedTuple):
    """
    A named tuple, for representing a border style for the pretty print methods, mostly used internally...
    """
    top_left: str
    top_right: str
    bottom_right: str
    bottom_left: str
    top: str
    right: str
    bottom: str
    left: str
    info_func: Callable[[int, int], str] = lambda w, h: f"{w}x{h}"


class FrameStringFormats:
    """
    Some pre-provided frame string format fonts, used to style frames when dumping them to the console.
    """
    REGULAR = (" ░▒▓█", 2, BorderStyle("╔", "╗", "╝", "╚", "═", "║", "═", "║"))
    REGULAR_COMPACT = (" ░▒▓█", 1, BorderStyle("╔", "╗", "╝", "╚", "═", "║", "═", "║"))
    REGULAR_NO_SPACE = ("░▒▓█", 2, BorderStyle("╔", "╗", "╝", "╚", "═", "║", "═", "║"))
    DIGITS = ("0123456789", 1, BorderStyle("╔", "╗", "╝", "╚", "═", "║", "═", "║"))
    DIGITS_FUN = ('🯰🯱🯲🯳🯴🯵🯶🯷🯸🯹', 1, None)
    SQUARES = ("▢◫▥▩▣", 1, BorderStyle("◎", "◎", "◎", "◎", "▭", "◑", "▭", "◐", lambda w, h: ""))


def pretty_print_frame(
    data: TrackingData,
    frame_idx: int = 0,
    body_part: int = 0,
    dynamic_sz: bool = True,
    size_up: bool = False,
    interpol: int = cv2.INTER_CUBIC,
    format_type: Tuple[str, int, tuple] = FrameStringFormats.REGULAR
):
    """
    Print a DeepLabCut Probability Frame.

    :param data: The probability frame, a TrackingData object.
    :param frame_idx: Frame index to print, defaults to 0.
    :param body_part: Body part index to print, defaults to 0.
    :param dynamic_sz: Determines if the frame is resized to fit the window if it is to big....
    :param size_up: Determines if the frame should not only be downsized, but also upsized if the terminal provides
                    extra room.
    :param interpol: The interpolation method if a size is specified. Defaults to cv2.INTER_CUBIC, but any cv2
                     interpolation value works.
    :param format_type: The format or 'font' to use for the pretty printed string. A tuple, containing a sequence of
                        strings being the displayed characters at given magnitudes, and an integer being the number
                        of times to repeat the characters when displaying them. ('abcd', 2 with 0 becomes aa)
    """
    if(dynamic_sz):
        print(pretty_frame_string(data, frame_idx, body_part, get_terminal_size()[0], size_up, interpol, format_type))
    else:
        print(pretty_frame_string(data, frame_idx, body_part, 0, size_up, interpol, format_type))


def pretty_frame_string(
    data: TrackingData,
    frame_idx: int = 0,
    body_part: int = 0,
    width_limit: int = 0,
    size_up: bool = False,
    interpol: int = cv2.INTER_CUBIC,
    format_type: Tuple[str, int, tuple] = FrameStringFormats.REGULAR
) -> str:
    """
    Return a DeepLabCut Probability Frame in a pretty string for printing to the terminal.

    :param data: The probability frame, a TrackingData object.
    :param frame_idx: Frame index to print, defaults to 0.
    :param body_part: Body part index to print, defaults to 0.
    :param width_limit: Determines the max with the string can be when printing. Defaults to 0, meaning there is no
                        width limit.
    :param size_up: Determines if the frame should not only be downsized, but also upsized if the width provided
                    gives extra room...
    :param interpol: The interpolation method if a size is specified. Defaults to cv2.INTER_CUBIC, but any cv2
                     interpolation value works.
    :param format_type: The format or 'font' to use for the pretty printed string. A length 3 tuple, containing a
                        sequence of characters being the displayed characters at given magnitudes, an integer being
                        the number of times to repeat the characters when displaying them. ('abcd', 2 with 0 becomes
                        aa), and an optional tuple of 8 strings representing characters for drawing a border (If None,
                        no border is drawn, tuple order is top-left, top right, bottom-right, bottom-left, top, right,
                        bottom, left).

    :returns: A pretty string of the frame, that can be printed to the console.
    """
    chars, char_rep_amt, border = format_type

    if(border is not None):
        border = BorderStyle(*border)

    frame = data.get_prob_table(frame_idx, body_part)
    # Make range 0-1...
    max_val = np.nanmax(frame)
    if(max_val != 0):
        frame = frame / max_val

    if(width_limit >= (char_rep_amt * 2)):
        new_w = (width_limit / char_rep_amt) - 1
        new_w = min(frame.shape[1], new_w) if(not size_up) else new_w

        sized_f = cv2.resize(frame, (int(new_w), int(frame.shape[0] * (new_w / frame.shape[1]))),
                             interpolation=interpol)
        max_val = np.nanmax(sized_f)
        if(max_val != 0):
            sized_f = sized_f / max_val
    else:
        sized_f = frame

    res = bytearray()

    if(border is not None):
        dim_str = border.info_func(frame.shape[1], frame.shape[0])

        res += (
            f"{border.top_left}"
            f"{dim_str}{border.top * ((sized_f.shape[1] * char_rep_amt) - len(dim_str))}"
            f"{border.top_right}\n"
        ).encode()

    for y in range(sized_f.shape[0]):
        if(border is not None):
            res += border.left.encode()

        for x in range(sized_f.shape[1]):
            res += (chars[int(sized_f[y, x] * (len(chars) - 0.5))] * char_rep_amt).encode()

        if(border is not None):
            res += f"{border.right}\n".encode()
        else:
            res += "\n".encode()

    if(border is not None):
        res += (
            f"{border.bottom_left}"
            f"{border.bottom * (sized_f.shape[1] * char_rep_amt)}"
            f"{border.bottom_right}"
        ).encode()

    return res.decode()


def extract_n_pack(
    dlfs_in: BinaryIO,
    frames: Sequence[int],
    threshold: float = 1e-6,
    compression_lvl: int = 6,
    on_frames: Optional[Callable[[TrackingData], None]] = None
) -> bytes:
    """
    Extract frames from a DLC Frame Store file and pack them into a base64 encoded byte string.

    :param dlfs_in: A binary file, the input DeepLabCut frame store file. Frames will be extracted from this file.
    :param frames: A sequence of integers, the indexes of the frames to extract, in order of extraction.
    :param threshold: A float between 0 and 1, inclusive. The frame sparsification threshold, any probabilities below
                      the threshold are ignored. If set to 0, sparsification of frames is disabled. Default is 1e6.
    :param compression_lvl: An integer between 0 and 9, inclusive. The compression level if using the .dlfs format.
                            Default is 6.
    :param on_frames: A function that accepts a TrackingData for each frame. Useful for pretty printing/doing something
                      with the data during rewrite. Defaults to None, meaning it is not called.

    :returns: A bytes object, being a base64 encoded DLFS or HDF5 frame store file.
    """
    out = BytesIO()
    # Dump into the in memory file...
    extract_frames(dlfs_in, out, frames, threshold, compression_lvl, on_frames)

    return base64.encodebytes(out.getvalue())


def unpack_frame_string(
    frame_string: bytes,
    frames_per_iter: int = 0
) -> Tuple[List[str], Union[TrackingData, Generator[TrackingData, None, None]]]:
    """
    Unpack a frame store string into a tracking data object for access to the original probability frame data.

    :param frame_string: A bytes object containing the base64 encoded frame store file.
    :param frames_per_iter: Number of frames to return in each TrackingData object generated. If this value is set to
                            0 or less, this function returns a single TrackingData object storing all frames instead of
                            returning a generator.

    :returns: A tuple containing:
               - A list of strings (body parts) and,
               - A single TrackingData object if frames_per_iter <= 0,
                 or a Generator of TrackingData objects if frames_per_iter > 0.
    """
    f = BytesIO(base64.decodebytes(frame_string))

    reader = frame_store_fmt.DLFSReader(f)

    if(frames_per_iter <= 0):
        return (reader.get_header().bodypart_names, reader.read_frames(reader.get_header().number_of_frames))
    else:
        return (reader.get_header().bodypart_names, _unpack_frame_string_gen(reader, frames_per_iter))


def _unpack_frame_string_gen(reader: frame_store_fmt.DLFSReader, frames_per_iter: int = 0):
    while(reader.has_next(frames_per_iter)):
        yield reader.read_frames(frames_per_iter)

    extra = reader.get_header().number_of_frames - (reader.tell_frame() + 1)
    if(extra > 0):
        yield reader.read_frames(extra)

    return None
