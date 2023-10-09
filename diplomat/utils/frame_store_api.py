"""
Provides a generic API and some core data structures for frame store formats, or files which store model outputs on disk.
"""
from typing import Any, Optional, Iterator, List, BinaryIO, MutableMapping
from abc import ABC, abstractmethod
import numpy as np
from diplomat.processing import TrackingData

# REQUIRED DATA TYPES: (With little endian encoding...)
luint8 = np.dtype(np.uint8).newbyteorder("<")
luint16 = np.dtype(np.uint16).newbyteorder("<")
luint32 = np.dtype(np.uint32).newbyteorder("<")
luint64 = np.dtype(np.uint64).newbyteorder("<")
ldouble = np.dtype(np.float64).newbyteorder("<")
lfloat = np.dtype(np.float32).newbyteorder("<")


def to_bytes(obj: Any, dtype: np.dtype) -> bytes:
    """
    Converts an object to bytes.

    :param obj: The object to convert to bytes.
    :param dtype: The numpy data type to interpret the object as when converting to bytes.
    :return: A bytes object, representing the object obj as type dtype.
    """
    return dtype.type(obj).tobytes()


def from_bytes(data: bytes, dtype: np.dtype) -> Any:
    """
    Converts bytes to a single object depending on the passed data type.

    :param data: The bytes to convert to an object
    :param dtype: The numpy data type to convert the bytes to.
    :return: An object of the specified data type passed to this method.
    """
    return np.frombuffer(data, dtype=dtype)[0]


def string_list(lister: list):
    """
    Casts object to a list of strings, enforcing type...

    :param lister: The list
    :return: A list of strings

    :raises: ValueError if the list doesn't contain strings...
    """
    lister = list(lister)

    for item in lister:
        if not isinstance(item, str):
            raise ValueError("Must be a list of strings!")

    return lister


def non_max_int32(val: luint32) -> Optional[int]:
    """
    Casts an object to a non-max integer, being None if it is the maximum value.

    :param val: The value to cast...
    :return: An integer, or None if the value equals the max possible integer.
    """
    if val is None:
        return None

    val = int(val)

    if (val == np.iinfo(luint32).max) or (val < 0):
        return None
    else:
        return val


class DLFSHeader(MutableMapping):
    """
    Stores some basic info about a frame store...

    Below are the fields in order, their names, types and default values:
        ("number_of_frames", int, 0),
        ("frame_height", int, 0),
        ("frame_width", int, 0),
        ("frame_rate", float, 0),
        ("stride", float, 0),
        ("orig_video_height", int, 0),
        ("orig_video_width", int, 0),
        ("crop_offset_y", int or None if no cropping, None),
        ("crop_offset_x", int or None if no cropping, None),
        ("bodypart_names", list of strings, []),
    """
    SUPPORTED_FIELDS = [
        ("number_of_frames", int, 0),
        ("frame_height", int, 0),
        ("frame_width", int, 0),
        ("frame_rate", float, 0),
        ("stride", float, 0),
        ("orig_video_height", int, 0),
        ("orig_video_width", int, 0),
        ("crop_offset_y", non_max_int32, None),
        ("crop_offset_x", non_max_int32, None),
        ("bodypart_names", string_list, []),
    ]

    GET_VAR_CAST = {name: var_cast for name, var_cast, __ in SUPPORTED_FIELDS}
    GET_IDX = {name: idx for idx, (name, __, __) in enumerate(SUPPORTED_FIELDS)}

    def __init__(self, *args, **kwargs):
        """
        Make a new DLFSHeader. Supports tuple style construction and also supports setting the fields using
        keyword arguments. Look at the class documentation for all the fields.
        """
        # Make the fields.
        self._values = {}
        for name, var_caster, def_value in self.SUPPORTED_FIELDS:
            self._values[name] = def_value

        for new_val, (key, var_caster, __) in zip(args, self.SUPPORTED_FIELDS):
            self[key] = new_val

        for key, new_val in kwargs.items():
            if key in self._values:
                self[key] = new_val

    def __getattr__(self, item):
        if item == "_values":
            return self.__dict__[item]
        return self._values[item]

    def __setattr__(self, key, value):
        if key == "_values":
            self.__dict__["_values"] = value
            return
        self.__dict__["_values"][key] = self.GET_VAR_CAST[key](value)

    def _key_check(self, key):
        if(key not in self.GET_VAR_CAST):
            raise ValueError("Not a supported key!")

    def __setitem__(self, key, value):
        """
        Set the value of the specified header property.
        """
        self._key_check(key)
        self._values[key] = self.GET_VAR_CAST[key](value)

    def __delitem__(self, key) -> Any:
        """
        Clear the specified header property to its default value.
        """
        self._key_check(key)
        self._values[key] = self.GET_VAR_CAST[key](self.SUPPORTED_FIELDS[self.GET_IDX[key]][2])

    def __getitem__(self, key) -> Any:
        """
        Get the specified header property, returning its current value.
        """
        self._key_check(key)
        return self._values[key]

    def __len__(self) -> int:
        """
        Get the length of this header (Should always be 10).
        """
        return len(self._values)

    def __iter__(self) -> Iterator:
        """
        Iterate the keys of the header in order.
        """
        # Using () returns a generator, not using extra memory...
        return (name for name, __, __ in self.SUPPORTED_FIELDS)

    def __str__(self):
        return str(self._values)

    def to_list(self) -> List[Any]:
        return [self._values[key] for key, __, __ in self.SUPPORTED_FIELDS]


class FrameReader(ABC):
    """
    The frame reader API. Allows for reading frames from a diplomat frame store format to
    :py:class:`~diplomat.processing.track_data.TrackingData` object.
    """
    @abstractmethod
    def __init__(self, file: BinaryIO):
        """
        Construct a frame read frame reader.

        :param file: The file to read frames from.
        """
        pass

    @abstractmethod
    def get_header(self) -> DLFSHeader:
        """
        Get the header of this frame store.

        :returns: A DLFSHeader object, which contains important metadata for this frame store.
        """
        pass

    @abstractmethod
    def has_next(self, num_frames: int = 1) -> bool:
        """
        Checks to see if there are more frames available for reading.

        :param num_frames: An integer, the number of frames to check for. Defaults to 1 frame.

        :returns: A boolean, True if there are at least num_frames frames available for reading from the file.
                  Otherwise, this method returns False.
        """
        pass

    @abstractmethod
    def tell_frame(self) -> int:
        """
        Get the current frame this frame reader is on.

        :returns: An integer, being the index of the frame that the frame reader will be reading next.
        """
        pass

    def seek_frame(self, frame_idx: int):
        """
        Seek to the specified frame in the frame store object. Implementors of the FrameReader class are not required
        to support this method, and the default implementation of this method throws a NotImplementedError.

        :param frame_idx: The frame index that the frame reader will move to, an integer.
        """
        raise NotImplementedError("Seeking functionality is not supported for this implementation of FrameReader!")

    @abstractmethod
    def read_frames(self, num_frames: int = 1) -> TrackingData:
        """
        Read frames from the frame store.

        :param num_frames: The number of frames to read from the frame store, and integer. Defaults to 1.

        :returns: A DeepLabCut TrackingData object, which will contain all of the probability frames for num_frames
                  frames.

        :throws: ValueError if the frame reader reaches the end of the file and the number of frames requested is
                 greater than the number of frames available in the frame store.
        """
        pass

    # Adds with statement support so user does not have to call close manually...
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @abstractmethod
    def close(self):
        """
        Close this frame reader. This does not close the file handler that this frame reader is utilizing, simply the
        frame reader itself.
        """
        pass


class FrameWriter(ABC):
    """
    The frame writer API. Allows for writing frames in the form of
    :py:class:`~diplomat.processing.track_data.TrackingData` objects to a diplomat frame store format.
    """
    @abstractmethod
    def __init__(
        self,
        file: BinaryIO,
        header: DLFSHeader,
        threshold: Optional[float] = 1e-6,
    ):
        """
        Create a new frame writer.

        :param file: A binary frame object, the file to write the frames to.
        :param header: The DLFSHeader for this frame store, contains important metadata.
        :param threshold: The minimum threshold for keeping probabilities. If set to None, this indicates to the frame
                          writer that the probability frames should be stored in a non-sparse way. Defaults to
                          1e-6.
        """
        pass

    @abstractmethod
    def get_header(self) -> DLFSHeader:
        """
        Get the header of this frame writer.

        :returns: A DLFSHeader object, which contains important metadata for this frame writer.
        """
        pass

    @abstractmethod
    def tell_frame(self) -> int:
        """
        Get the current frame this frame writer is on.

        :returns: An integer, being the index of the frame that the frame writer will be writing next.
        """
        pass

    @abstractmethod
    def write_data(self, data: TrackingData):
        """
        Write data to the file using this frame writer.

        :param data: A TrackingData object, being the frames to write to the file.

        :throws: ValueError if there is an attempt to write more frames than the total number of frames specified
                 in the DLFSHeader passed when this frame writer was created.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close this frame writer. This does not close the file handler that this frame writer is utilizing, simply the
        frame writer itself.
        """
        pass

    # Adds with statement support so user does not have to call close manually...
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

