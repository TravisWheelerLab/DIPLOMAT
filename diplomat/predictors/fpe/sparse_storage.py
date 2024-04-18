import dataclasses
from collections import UserDict
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple, Union, List, Optional, Dict, Any, TypeVar, Iterator
from typing_extensions import Protocol

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
from numpy import ndarray
from diplomat.processing import TrackingData
from diplomat.utils.extract_frames import pretty_frame_string, FrameStringFormats, BorderStyle

# Represents a valid numpy indexing type.
Indexer = Union[slice, int, List[int], Tuple[int], None]

T = TypeVar("T")


class SettableSequence(Protocol[T]):
    def __getitem__(self, item: Indexer) -> T:
        pass

    def __setitem__(self, key: Indexer, value: T):
        pass

    def __len__(self) -> int:
        pass

    def __iter__(self) -> Iterator:
        pass


class SparseModes(IntEnum):
    """
    An enum for encoding the different ways of making frames sparse. See SparseTrackingData.sparsify to see what each
    of these modes does.
    """
    IGNORE_OFFSETS: int = 0
    OFFSET_DOMINATION: int = 1
    OFFSET_COMBINATION: int = 2
    OFFSET_SUMMATION: int = 3


class SparseTrackingData:
    """
    Represents sparse tracking data. Includes probabilities, offsets, and x/y coordinates in the probability map.
    """
    SparseModes = SparseModes
    __slots__ = ["_coords", "_offsets_probs", "_coords_access", "_off_access", "_prob_access"]

    class NumpyAccessor:
        """
        A numpy accessor class, allows for controlled access to a numpy array (Disables direct assignment).
        """

        def __init__(self, src_arr: ndarray, post_index: Tuple[Indexer] = None):
            self._src_arr = src_arr
            self._post_idx = post_index if (post_index is not None) else tuple()

        def __getitem__(self, *index: Indexer):
            if (self._post_idx is None):
                return self._src_arr[index]
            return self._src_arr[index + self._post_idx]

        def __setitem__(self, *index: Indexer, value: ndarray):
            self._src_arr[index, self._post_idx] = value

        def __str__(self):
            return self.__repr__()

        def __repr__(self):
            return f"{type(self).__name__}({self[:]})"

    def __init__(self):
        """
        Makes a new tracking data with all empty fields.
        """
        self._coords = None
        self._offsets_probs = None
        self._coords_access = None
        self._off_access = None
        self._prob_access = None

    @property
    def coords(self):
        """
        The coordinates of this SparceTrackingData in [y, x] order.
        """
        return self._coords_access

    @property
    def probs(self):
        """
        The probabilities of this SparceTrackingData.
        """
        return self._prob_access

    @property
    def offsets(self):
        """
        The offsets of this SparceTrackingData.
        """
        return self._off_access

    def _set_data(self, coords: ndarray, offsets: ndarray, probs: ndarray):
        # If no coordinates are passed, set everything to None as there is no data.
        if (coords is None or len(coords) == 0):
            self._coords = None
            self._coords_access = None
            self._offsets_probs = None
            self._off_access = None
            self._prob_access = None
            return
        # Combine offsets and probabilities into one numpy array...
        offsets_plus_probs = np.concatenate((np.transpose(np.expand_dims(probs, axis=0)), offsets), axis=1)

        if ((len(offsets_plus_probs.shape) != 2) and offsets_plus_probs.shape[1] == 3):
            raise ValueError("Offset/Probability array must be n by 3...")

        if (len(coords.shape) != 2 or coords.shape[0] != offsets_plus_probs.shape[0] or coords.shape[1] != 2):
            raise ValueError("Coordinate array must be n by 2...")

        # Set the arrays to the data and make new numpy accessors which point to the new numpy arrays
        self._coords = coords
        self._coords_access = self.NumpyAccessor(self._coords)
        self._offsets_probs = offsets_plus_probs
        self._off_access = self.NumpyAccessor(self._offsets_probs, (slice(1, 3),))
        self._prob_access = self.NumpyAccessor(self._offsets_probs, (0,))

    def pack(self, y_coords: Optional[ndarray], x_coords: Optional[ndarray], probs: Optional[ndarray],
             x_offset: Optional[ndarray], y_offset: Optional[ndarray]) -> "SparseTrackingData":
        """
        Pack the passed data into this SparceTrackingData object.

        :param y_coords: Y coordinate value of each probability within the probability map.
        :param x_coords: X coordinate value of each probability within the probability map.
        :param probs: The probabilities of the probability frame.
        :param x_offset: The x offset coordinate values within the original video.
        :param y_offset: The y offset coordinate values within the original video.

        :returns: This SparseTrackingData Object.
        """
        if(y_coords is None):
            self._coords = None
            self._coords_access = None
            self._offsets_probs = None
            self._off_access = None
            self._prob_access = None
            return self

        self._set_data(np.transpose((y_coords, x_coords)), np.transpose((x_offset, y_offset)), probs)
        return self

    # noinspection PyTypeChecker
    def unpack(self) -> Tuple[
        Optional[ndarray], Optional[ndarray], Optional[ndarray], Optional[ndarray], Optional[ndarray]]:
        """
        Return all fields of this SparseTrackingData

        :return: A tuple of 1 dimensional numpy arrays, being: (y_coord, x_coord, probs, x_offset, y_offset)
        """
        if (self._coords is None):
            return (None, None, None, None, None)

        return tuple(np.transpose(self._coords)) + tuple(np.transpose(self._offsets_probs))

    def duplicate(self) -> "SparseTrackingData":
        """
        Creates a copy this SparseTrackingData, returning it.

        :return: A new SparseTrackingData, which is identical to the current SparseTrackingData.
        """
        new_sparse_data = SparseTrackingData()

        if(self._coords is None):
            return new_sparse_data

        new_sparse_data.pack(*(np.copy(arr) for arr in self.unpack()))
        return new_sparse_data

    def shallow_duplicate(self) -> "SparseTrackingData":
        """
        Creates a shallow copy this SparseTrackingData, returning it.

        :return: A new SparseTrackingData, which is identical to the current SparseTrackingData.
        """
        new_sparse_data = SparseTrackingData()

        if(self._coords is None):
            return new_sparse_data

        new_sparse_data.pack(*(arr for arr in self.unpack()))
        return new_sparse_data

    def desparsify(self, orig_width: int, orig_height: int, orig_stride: int) -> TrackingData:
        """
        Desparsifies this SparseTrackingData object.

        :param orig_width: The original width of the tracking data object, will be the frame width of the newly
                           constructed TrackingData object.
        :param orig_height: The original height of the tracking data object, will be the frame height of the newly
                           constructed TrackingData object.
        :param orig_stride: The stride of the newly constructed tracking object (upscaling factor to scale to the size
                            of the original video).

        :return: A new TrackingData object, with data matching the original SparseTrackingData object. Returns an empty
                 TrackingData if this object has no data stored yet...
        """
        y, x, probs, x_off, y_off = self.unpack()

        new_td = TrackingData.empty_tracking_data(1, 1, orig_width, orig_height, orig_stride)
        new_td.set_offset_map(np.zeros((1, orig_height, orig_width, 1, 2), dtype=np.float32))

        if(y is None):
            return new_td

        new_td.get_source_map()[0, y, x, 0] = probs
        new_td.get_offset_map()[0, y, x, 0, 0] = x_off
        new_td.get_offset_map()[0, y, x, 0, 1] = y_off

        return new_td

    @classmethod
    def sparsify(
        cls,
        track_data: TrackingData,
        frame: int,
        bodypart: int,
        threshold: float,
        max_cell_count: Optional[int] = None,
        mode: SparseModes = SparseModes.IGNORE_OFFSETS
    ) -> "SparseTrackingData":
        """
        Sparsify the TrackingData.

        :param track_data: The TrackingData to sparsify
        :param frame: The frame of the TrackingData to sparsify
        :param bodypart: The bodypart of the TrackingData to sparsify
        :param threshold: The threshold to use when scarifying. All values below the threshold are removed.
        :param max_cell_count: The maximum number of cells allowed in this sparsified frame. Defaults to None, so no
                               limiting is done. If the number of cells is larger than this value, the top-k cells
                               will be pulls so k matches this value.
        :param mode: The mode to utilize when making the data sparse. The following modes currently exist:
            - SparseModes.IGNORE_OFFSETS: Ignores offsets, placing values based on the grid cell the value is stored in.
                                          This is the default mode.
            - SparseModes.OFFSET_DOMINATION: Add on offsets to the initial grid cell to determine what cell the data
                                             actually landed in. If multiple cells point to the same location, select
                                             the maximum of them.
            - SparseModes.OFFSET_COMBINATION: Add on offsets to the initial grid cell to determine what cell the data
                                              actually landed in. If multiple cells point to the same location, use the
                                              average of their values.
            - SparseModes.OFFSET_SUMMATION: Same as OFFSET_DOMINATION, except cell probabilities are determined by
                                            adding all the cells that point to a cell and then normalizing this sum
                                            array.

        :return: A new SparseTrackingData object containing the data of the TrackingData object.
        """
        new_sparse_data = cls()

        width, height = track_data.get_frame_width(), track_data.get_frame_height()
        y, x = np.nonzero(track_data.get_prob_table(frame, bodypart) > threshold)

        if(len(y) == 0):
            new_sparse_data.pack(None, None, None, None, None)
            return new_sparse_data

        if (track_data.get_offset_map() is None):
            x_off, y_off = np.zeros(x.shape, dtype=np.float32), np.zeros(y.shape, dtype=np.float32)
        else:
            x_off, y_off = np.transpose(track_data.get_offset_map()[frame, y, x, bodypart])

        probs = track_data.get_prob_table(frame, bodypart)[y, x]

        if(mode != SparseModes.IGNORE_OFFSETS):
            full_x = x + 0.5 + x_off / track_data.get_down_scaling()
            full_y = y + 0.5 + y_off / track_data.get_down_scaling()
            true_x = full_x.astype(int)
            true_y = full_y.astype(int)

            # Update x_off and y_off to be in the same square as true_x and true_y...
            x_off = ((full_x % 1) - 0.5) * track_data.get_down_scaling()
            y_off = ((full_y % 1) - 0.5) * track_data.get_down_scaling()

            if(mode == SparseModes.OFFSET_COMBINATION):
                # Combination, we'll use np.unique to get the unique values, and then bincount to get the averages...
                comb_coords = np.stack((true_x, true_y))
                __, indexes, inverse, counts = np.unique(comb_coords, return_index=True, return_inverse=True, return_counts=True, axis=-1)

                inv_counts = (1 / counts[inverse])
                x = true_x[indexes]
                y = true_y[indexes]
                probs = np.bincount(inverse, inv_counts * probs)
                x_off = np.bincount(inverse, inv_counts * x_off)
                y_off = np.bincount(inverse, inv_counts * y_off)
            elif(mode == SparseModes.OFFSET_SUMMATION):
                # Mode is offset summation, we use domination for offsets, summation for probabilities...
                ordered_coords = np.lexsort([-probs, true_y, true_x])

                true_x = true_x[ordered_coords]
                true_y = true_y[ordered_coords]

                changes = (true_x[:-1] != true_x[1:]) | (true_y[:-1] != true_y[1:])
                unique_locs = np.concatenate([[True], changes])
                ids = np.cumsum(np.concatenate([[False], changes]))

                x = true_x[unique_locs]
                y = true_y[unique_locs]
                probs = np.bincount(ids, probs[ordered_coords])
                probs /= np.sum(probs)
                x_off = x_off[ordered_coords][unique_locs]
                y_off = y_off[ordered_coords][unique_locs]
            else:
                # Mode is offset domination, only keep maximums...
                # We include -probs, as that sorts makes sure the first unique value is always the one with the
                # highest probability...
                ordered_coords = np.lexsort([-probs, true_y, true_x])

                true_x = true_x[ordered_coords]
                true_y = true_y[ordered_coords]

                unique_locs = np.concatenate([[True], (true_x[:-1] != true_x[1:]) | (true_y[:-1] != true_y[1:])])

                x = true_x[unique_locs]
                y = true_y[unique_locs]
                probs = probs[ordered_coords][unique_locs]
                x_off = x_off[ordered_coords][unique_locs]
                y_off = y_off[ordered_coords][unique_locs]

            # Remove locations that point outside the video frame...
            in_frame = ((x < width) & (x >= 0)) & ((y < height) & (y >= 0))
            x = x[in_frame]
            y = y[in_frame]
            probs = probs[in_frame]
            x_off = x_off[in_frame]
            y_off = y_off[in_frame]

        if((max_cell_count is not None) and (len(probs) > max_cell_count)):
            top_k = np.argpartition(probs, -max_cell_count)[-max_cell_count:]
            x = x[top_k]
            y = y[top_k]
            probs = probs[top_k]
            x_off = x_off[top_k]
            y_off = y_off[top_k]

        new_sparse_data.pack(y, x, probs, x_off, y_off)

        return new_sparse_data

    def to_bytes(self, float_dtype: str, int_dtype: str) -> bytes:
        if(float_dtype.lstrip("<>") not in ["f2", "f4", "f8"]):
            raise ValueError("Invalid float datatype!")
        if(int_dtype.lstrip("<>") not in ["u2", "u4", "u8", "i2", "i4", "i8"]):
            raise ValueError("Invalid integer datatype!")

        if(self._offsets_probs is None):
            # Null is encoded as all 1's
            return np.asarray(0, dtype="<u4").tobytes()

        length = self._offsets_probs.shape[0]
        enc_length = np.asarray(length, dtype="<u4").tobytes()

        return b"".join([
            enc_length,
            self._coords.astype(int_dtype).tobytes(),
            self._offsets_probs.astype(float_dtype).tobytes()
        ])

    def from_bytes_include_length(
        self,
        float_dtype: str,
        int_dtype: str,
        data: Union[bytes, memoryview]
    ) -> Tuple["SparseTrackingData", int]:
        if(float_dtype.lstrip("<>") not in ["f2", "f4", "f8"]):
            raise ValueError("Invalid float datatype!")
        if(int_dtype.lstrip("<>") not in ["u2", "u4", "u8", "i2", "i4", "i8"]):
            raise ValueError("Invalid integer datatype!")

        length = np.frombuffer(data, "<u4", 1)[0]
        if(length == 0):
            self._coords = None
            self._offsets_probs = None
            self._off_access = None
            self._coords_access = None
            self._prob_access = None
            return self, 4

        float_offset = 4 + length * 2 * np.dtype(int_dtype).itemsize

        # Calculate the expected size of the data based on `length` and `int_dtype`
        int_dtype_size = np.dtype(int_dtype).itemsize  # Size of int_dtype in bytes
        expected_data_size = length * 2 * int_dtype_size + 4  # +4 for the offset

        # Check if the actual data buffer is smaller than expected
        if len(data) < expected_data_size:
            print(f"Error: The data buffer length {len(data)} is smaller than the expected size {expected_data_size}.")

        # Proceed with the original line, wrapped in a try-except block for safety
        try:
            self._coords = np.frombuffer(data, np.dtype(int_dtype), length * 2, 4).reshape((length, 2))
        except ValueError as e:
            print(f"Encountered an error when trying to reshape the data: {e}")


        #self._coords = np.frombuffer(data, np.dtype(int_dtype), length * 2, 4).reshape((length, 2))
        self._coords_access = self.NumpyAccessor(self._coords)
        self._offsets_probs = np.frombuffer(data, np.dtype(float_dtype), length * 3, float_offset).reshape((length, 3))
        self._off_access = self.NumpyAccessor(self._offsets_probs, (slice(1, 3),))
        self._prob_access = self.NumpyAccessor(self._offsets_probs, (0,))

        return self, float_offset + length * 3 * np.dtype(float_dtype).itemsize

    def from_bytes(self, float_dtype: str, int_dtype: str, data: bytes) -> "SparseTrackingData":
        return self.from_bytes_include_length(float_dtype, int_dtype, data)[0]


    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"SparseTrackingData(coords={self.coords}, probs={self.probs}, offsets={self.offsets})"


# Improves memory performance by using slots instead of a dictionary to store attributes...
def add_slots(cls):
    # Need to create a new class, since we can't set __slots__
    #  after a class has been created.

    # Make sure __slots__ isn't already set.
    if '__slots__' in cls.__dict__:
        raise TypeError(f'{cls.__name__} already specifies __slots__')

    # Create a new dict for our new class.
    cls_dict = dict(cls.__dict__)
    field_names = tuple(f.name for f in dataclasses.fields(cls))
    cls_dict['__slots__'] = field_names
    for field_name in field_names:
        # Remove our attributes, if present. They'll still be
        #  available in _MARKER.
        cls_dict.pop(field_name, None)
    # Remove __dict__ itself.
    cls_dict.pop('__dict__', None)
    # And finally create the class.
    qualname = getattr(cls, '__qualname__', None)
    cls = type(cls)(cls.__name__, cls.__bases__, cls_dict)
    if qualname is not None:
        cls.__qualname__ = qualname
    return cls


@add_slots
@dataclass
class ForwardBackwardFrame:
    """
    Represents a single sparsified frame for Forward Backward. Contains the following attributes...

    orig_data: The original source data, a sparse tracking data object.
    src_data: The "source" data also, but this data can be modified by passes.
    frame_probs: Numpy array of floats, the in-frame post forward backward probabilities.
    occluded_coords: Numpy Nx2 array of integers, the occluded state post forward backward coordinate locations (x, y).
    occluded_probs: Numpy array of floats, the occluded state post forward backward probabilities.
    ignore_clustering: Boolean,
    """
    orig_data: Optional[SparseTrackingData] = None
    src_data: Optional[SparseTrackingData] = None
    frame_probs: Optional[np.ndarray] = None
    occluded_coords: Optional[np.ndarray] = None
    occluded_probs: Optional[np.ndarray] = None
    ignore_clustering: bool = False  # Has user edited this cell? If so, disable clustering for this frame...
    disable_occluded: bool = False  # If user has edited this frame, shouldn't try using the hidden state...
    enter_state: float = 0  # State used if no data is found in-frame...

    def copy(self) -> "ForwardBackwardFrame":
        """
        Copy the Forward Backward Frame, return a new one...
        """
        return type(self)(**dataclasses.asdict(self))

    def as_tuple(self) -> tuple:
        """
        Convert this Forward Backward Frame to a tuple, each property being stored in-order.
        """
        return dataclasses.astuple(self)

    def as_dict(self) -> dict:
        """
        Convert this Forward Backward Frame to a dictionary, with the same property names and values...
        """
        return dataclasses.asdict(self)

    def entry_to_sparse_track(
        self,
        field: Literal['frame', 'occluded'] = "frame"
    ) -> SparseTrackingData:
        """
        Convert an entry in this ForwardBackwardFrame to a SparseTrackingData
        for display, or saving...

        :param field: The field of this ForwardBackwardFrame to convert to
                      a sparse track. Default to the frame probabilities,
                      or 'frame'.

        :returns: SparseTrackingData object, being the tracking data for the
                  selected field.
        """
        if(field == "frame"):
            if(self.frame_probs is None):
                raise ValueError("No frame probabilities!")
            unpack_data = self.src_data.unpack()
            return SparseTrackingData().pack(
                *unpack_data[:2], self.frame_probs, *unpack_data[3:]
            )
        elif(field == "occluded"):
            if (self.occluded_probs is None or self.occluded_coords is None):
                raise ValueError("No occluded probabilities!")
            return SparseTrackingData().pack(
                *tuple(reversed(self.occluded_coords.T)), self.occluded_probs,
                np.zeros(len(self.occluded_probs)),
                np.zeros(len(self.occluded_probs))
            )
        else:
            raise ValueError(f"Invalid field value. ('{field}')")

    def to_fancy_string(
        self,
        w: int,
        h: int,
        width_limit: int = 80,
        format_type: Tuple[str, int, BorderStyle] = FrameStringFormats.REGULAR_COMPACT
    ) -> str:
        string_list = [f"{type(self).__name__}("]

        for name, value in self.as_dict().items():
            data = None

            if(name.endswith("coords")):
                string_list.append(f"{name}=...")
                continue

            if(name.endswith("probs")):
                try:
                    data = self.entry_to_sparse_track(name.split("_")[0])
                except ValueError:
                    continue
            elif(name.endswith("data")):
                data = value

            if(data is not None):
                string_list.append(sparse_to_string(
                    data, w, h, width_limit=width_limit,
                    format_type=format_type,
                    _extra_info=f"{name}="
                ))
            else:
                string_list.append(f"{name} = {value}")

        string_list.append(")")

        return "\n".join(string_list)

    @staticmethod
    def _save_arr(arr: Optional[np.ndarray], dtype: str) -> bytes:
        if(arr is None):
            return np.asarray([0], dtype=dtype).tobytes()
        return b"".join([
            np.asarray(len(arr.shape), dtype="<u4").tobytes(),
            np.asarray(arr.shape, dtype="<u4").tobytes(),
            arr.astype(dtype).tobytes()
        ])

    @staticmethod
    def _save_sparse_track(std: Optional[SparseTrackingData], float_dtype: str, int_dtype: str) -> bytes:
        if(std is None):
            return np.asarray([0xFFFFFFFF], dtype="<u4").tobytes()
        return std.to_bytes(float_dtype, int_dtype)

    def to_bytes(self, float_dtype: str, int_dtype: str) -> bytes:
        byte_list = [
            np.asarray([self.ignore_clustering, self.disable_occluded], dtype="?").tobytes(),
            np.asarray([self.enter_state], dtype=float_dtype).tobytes(),
            self.orig_data.to_bytes(float_dtype, int_dtype),
            self.src_data.to_bytes(float_dtype, int_dtype),
            self._save_arr(self.frame_probs, float_dtype),
            self._save_arr(self.occluded_coords, int_dtype),
            self._save_arr(self.occluded_probs, float_dtype)
        ]

        return b"".join(byte_list)

    @staticmethod
    def _load_array(data: bytes, dtype: str, offset: int) -> Tuple[Optional[np.ndarray], int]:
        offset = int(offset)
        shape_size = np.frombuffer(data, "<u4", 1, offset)[0]

        if(shape_size == 0):
            return (None, 4)

        if(shape_size > 2):
            raise ValueError("???")

        shape = tuple(np.frombuffer(data, "<u4", shape_size, offset + 4))
        size = int(np.prod(shape))
        data = np.frombuffer(data, dtype, size, offset + 4 * (shape_size + 1)).reshape(shape)
        return data, 4 * (shape_size + 1) + size * np.dtype(dtype).itemsize

    @staticmethod
    def _load_sparse_track(
        data: bytes,
        float_dtype: str,
        int_dtype: str,
        offset: int
    ) -> Tuple[Optional[SparseTrackingData], int]:
        size_indicator = np.frombuffer(data, "<u4", 1, offset)[0]
        if(size_indicator == 0xFFFFFFFF):
            return (None, 4)

        return SparseTrackingData().from_bytes_include_length(float_dtype, int_dtype, memoryview(data)[offset:])

    def from_bytes(self, float_dtype: str, int_dtype: str, data: bytes) -> "ForwardBackwardFrame":
        self.ignore_clustering, self.disable_occluded = np.frombuffer(data, np.dtype("?"), 2)
        self.enter_state = np.frombuffer(data, float_dtype, 1, 2)[0]

        offset = 2 + np.dtype(float_dtype).itemsize
        self.orig_data, size = self._load_sparse_track(data, float_dtype, int_dtype, offset)
        offset += size
        self.src_data, size = self._load_sparse_track(data, float_dtype, int_dtype, offset)
        offset += size

        self.frame_probs, size = self._load_array(data, float_dtype, offset)
        offset += size
        self.occluded_coords, size = self._load_array(data, int_dtype, offset)
        offset += size
        self.occluded_probs, size = self._load_array(data, float_dtype, offset)
        offset += size

        return self


def sparse_to_string(
    data: SparseTrackingData,
    w: int,
    h: int,
    **kwargs
) -> str:
    """
    Convert a SparseTrackingData to a string.
    """
    fmt = kwargs.get("format_type", FrameStringFormats.REGULAR)
    extra_info = kwargs.pop("_extra_info", "")
    fmt = (*fmt[:2], BorderStyle(*fmt[2][:8], lambda w, h: f"{extra_info}{type(data).__name__}({w}x{h})"))
    kwargs["format_type"] = fmt
    data2 = data.desparsify(w, h, 8)
    return pretty_frame_string(data2, 0, 0, **kwargs)


class AttributeDict(UserDict):
    """
    An attribute dictionary. Dictionary keys can be accessed as properties or attributes, with the '.' operator.

    Example:
        attr_dict["key"] = "value"
        attr_dict.key <--- Returns "value"
    """
    def __getattr__(self, item):
        if(item.startswith("_") or item == "data"):
            return super().__getattribute__(item)
        return super().__getitem__(item)

    def __setattr__(self, key, value):
        if(key.startswith("_") or key == "data"):
            return super().__setattr__(key, value)
        return super().__setitem__(key, value)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(f'{key}={value}' for key, value in self.items())})"


class ForwardBackwardData:
    """
    Represents all forward backward frames/data. This is a 2D list of "ForwardBackwardFrame"s, indexed by frame, then
    body part. It also supports storing metadata like skeleton in the metadata attribute.
    """

    def __init__(self, num_frames: int, num_bp: int):
        """
        Create a new ForwardBackwardData list/object.

        :param num_frames: Number of frames to allocate space for.
        :param num_bp: Number of body parts to allocate space for.
        """
        self._metadata = AttributeDict()
        self._num_bps = num_bp
        self.allow_pickle = True

        self._frames = [
            [
                ForwardBackwardFrame(None, None, None, None, None) for __ in range(num_bp)
            ] for __ in range(num_frames)
        ]

    @property
    def frames(self) -> SettableSequence[SettableSequence[ForwardBackwardFrame]]:
        """
        Get/Set the frames of this ForwardBackwardData, a 2D list of ForwardBackwardFrame. Indexing is frame, then body
        part.
        """
        return self._frames

    @frames.setter
    def frames(self, frames: SettableSequence[SettableSequence[ForwardBackwardFrame]]):
        """
        Frames setter...
        """
        self._frames = frames
        self._num_bps = len(frames[0])

    @property
    def num_bodyparts(self) -> int:
        """
        Read-Only: Get the number of body parts stored in this forward backward data object...
        """
        return self._num_bps

    @property
    def num_frames(self) -> int:
        """
        Read-Only: Get the number of frames stored in this forward backward data object...
        """
        return len(self._frames)

    @property
    def metadata(self) -> AttributeDict:
        """
        Get/Set the metadata of this forward backward data object. This property can be set to any mapping/dictionary
        type of string to values, but always returns an AttributeDict to allow dot operator access to properties.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, meta: Dict[str, Any]):
        """
        Metadata setter...
        """
        self._metadata = AttributeDict(meta)

    def copy(self) -> "ForwardBackwardData":
        """
        Copy this ForwardBackwardData, returning a new one.
        """
        new_fbd = type(self)(0, 0)
        new_fbd.frames = [[data for data in frame] for frame in self.frames]
        new_fbd.metadata = self.metadata.copy()

        return new_fbd

    def __reduce__(self, *args, **kwargs):
        if(not self.allow_pickle):
            raise ValueError("Not allowed to pickle this ForwardBackwardData object.")
        return super().__reduce__(*args, **kwargs)
