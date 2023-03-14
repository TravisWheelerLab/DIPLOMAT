"""
Provides the :py:class:`~diplomat.processing.pose.Pose` class, used for storing final predicted body part locations.
"""

from typing import Optional, Union, Tuple
from numpy import ndarray
import numpy as np


class Pose:
    """
    Class defines the Poses for given amount of frames and body parts... Note that pose has no concept of multiple
    predictions for body part, but rather simply expects the multiple predictions to be stored side-by-side as
    multiple body parts. Also, it should be noted that data is stored in terms of original video coordinates, not
    probability source map indexes.
    """
    def __init__(self, x: ndarray, y: ndarray, prob: ndarray):
        """
        Create a new Pose object, or batch of poses for frames.

        :param x: All x video coordinates for these poses, in ndarray indexing format frame -> body part -> x-value
        :param y: All y video coordinates for these poses, in ndarray indexing format frame -> body part -> y-value
        :param prob: All probabilities for these poses, in ndarray indexing format frame -> body part -> p-value
        """
        self._data = np.zeros((x.shape[0], x.shape[1] * 3))

        self.set_all_x(x)
        self.set_all_y(y)
        self.set_all_prob(prob)

    # Helper Constructor methods...

    @classmethod
    def empty_pose(cls, frame_count: int, part_count: int) -> "Pose":
        """
        Returns an empty pose object, or a pose object with numpy arrays full of zeros. It will have space for
        "frame_count" frames and "part_count" body parts.

        :param frame_count: The amount of frames to allocate space for in the underlying array, an Integer.
        :param part_count: The amount of body parts to allocate space for in the underlying array, an Integer.
        :return: A new Pose object.
        """
        return cls(
            np.zeros((frame_count, part_count)),
            np.zeros((frame_count, part_count)),
            np.zeros((frame_count, part_count)),
        )

    # Helper Methods

    def _fix_index(
        self, index: Union[int, slice], value_offset: int
    ) -> Union[int, slice]:
        """
        Fixes slice or integer indexing received by user for body part to fit the actual way it is stored.
        PRIVATE METHOD! Should not be used outside this class, for internal index correction!

        :param index: An integer or slice representing indexing
        :param value_offset: An integer representing the offset of the desired values in stored data
        :return: Slice or integer, being the fixed indexing to actually get the body parts
        """
        if isinstance(index, (int, np.integer)):
            # Since all data is all stored together, multiply by 3 and add the offset...
            return (index * 3) + value_offset
        elif isinstance(index, slice):
            # Normalize the slice and adjust the indexes.
            start, end, step = index.indices(self._data.shape[1] // 3)
            return slice((start * 3) + value_offset, (end * 3) + value_offset, step * 3)
        else:
            raise ValueError(
                f"Index is not of type slice or integer! It is type '{type(index)}'!"
            )

    # Represents point data, is a tuple of x, y data where x and y are numpy arrays or integers...
    PointData = Tuple[Union[int, ndarray], Union[int, ndarray]]
    # Represents point data, is a tuple of x, y data where x and y are numpy arrays or floats...
    FloatPointData = Tuple[Union[float, ndarray], Union[float, ndarray]]
    # Represents and Index, either an integer or a slice
    Index = Union[int, slice]

    def set_at(
        self,
        frame: Index,
        bodypart: Index,
        scmap_coord: PointData,
        offset: Optional[FloatPointData],
        prob: Union[float, ndarray],
        down_scale: int,
    ):
        """
        Set the probability data at a given location or locations to the specified data.

        :param frame: The index of the frame or frames to set, an integer or a slice.
        :param bodypart: The index of the bodypart or bodyparts to set, integer or a slice
        :param scmap_coord: The source map index to set this Pose's location to, specifically the index directly
                            selected from the downscaled source map stored in the TrackingData object. It is a tuple of
                            two integer or numpy arrays representing x and y coordinates...
        :param offset: The offset of the source map point once scaled to fit the video. This data should be collected
                       using get_offset_map in the TrackingData object. Is a tuple of x and y floating point
                       coordinates, or numpy arrays of floating point coordinates.
        :param prob: The probabilities to be set in this Pose object, between 0 and 1. Is a numpy array
                     of floating point numbers or a single floating point number.
        :param down_scale: The downscale factor of the original source map relative to the video, an integer.
                                  this is typically collected from the method TrackingData.get_down_scaling().
                                  Ex. Value of 8 means TrackingData probability map is 1/8th the size of the original
                                  video.
        :return: Nothing...
        """
        offset = (0, 0) if (offset is None) else offset

        scmap_x = scmap_coord[0] * down_scale + (0.5 * down_scale) + offset[0]
        scmap_y = scmap_coord[1] * down_scale + (0.5 * down_scale) + offset[1]

        self.set_x_at(frame, bodypart, scmap_x)
        self.set_y_at(frame, bodypart, scmap_y)
        self.set_prob_at(frame, bodypart, prob)

    # Setter Methods
    def set_all_x(self, x: ndarray):
        """
        Set the x video coordinates of this batch of Poses.

        :param x: A ndarray with the same dimensions as this Pose object, providing all x video coordinates...
        """
        self._data[:, 0::3] = x

    def set_all_y(self, y: ndarray):
        """
        Sets the y video coordinates of this batch of Poses.

        :param y: An ndarray with same dimensions as this pose object, providing all y video coordinates...
        """
        self._data[:, 1::3] = y

    def set_all_prob(self, probs: ndarray):
        """
        Set the probability values of this batch of Poses

        :param probs: An ndarray with same dimensions as this Pose object, providing all probability values for given
                      x, y video coordinates...
        """
        self._data[:, 2::3] = probs

    def set_x_at(
        self, frame: Union[int, slice], bodypart: Union[int, slice], values: ndarray
    ):
        """
        Set the x video coordinates for specific body parts or frames.

        :param frame: The frame index, can be a slice or integer
        :param bodypart: The body part index, can be a slice or integer
        :param values: The values to set this Pose's x video coordinates to, as a numpy array...
        """
        self._data[frame, self._fix_index(bodypart, 0)] = values

    def set_y_at(
        self, frame: Union[int, slice], bodypart: Union[int, slice], values: ndarray
    ):
        """
        Set the y video coordinates for specific body parts or frames.

        :param frame: The frame index, can be a slice or integer
        :param bodypart: The body part index, can be a slice or integer
        :param values: The values to set this Pose's y video coordinates to, as a numpy array...
        """
        self._data[frame, self._fix_index(bodypart, 1)] = values

    def set_prob_at(
        self, frame: Union[int, slice], bodypart: Union[int, slice], values: ndarray
    ):
        """
        Set the probability values for specific body parts or frames.

        :param frame: The frame index, can be a slice or integer
        :param bodypart: The body part index, can be a slice or integer
        :param values: The values to set this pose's probabilities to, as a numpy array...
        """
        self._data[frame, self._fix_index(bodypart, 2)] = values

    # Getter Methods

    def get_all(self) -> ndarray:
        """
        Returns all data combined into a numpy array. Note method is mostly useful to DLC, not Predictor
        plugins.

        :return: A numpy array with indexing of the dimensions: [frame -> x, y or prob every 3-slots].
        """
        return self._data

    def get_all_x(self) -> ndarray:
        """
        Returns x video coordinates for all frames and body parts.

        :return: The x video coordinates for all frames and body parts...
        """
        return self._data[:, 0::3]

    def get_all_y(self) -> ndarray:
        """
        Returns y video coordinates for all frames and body parts.

        :return: The y video coordinates for all frames and body parts...
        """
        return self._data[:, 1::3]

    def get_all_prob(self) -> ndarray:
        """
        Returns probability data for all frames and body parts...

        :return: The probability data for all frames and body parts...
        """
        return self._data[:, 2::3]

    def get_x_at(
        self, frame: Union[int, slice], bodypart: Union[int, slice]
    ) -> ndarray:
        """
        Get the x video coordinates for specific body parts or frames.

        :param frame: The frame index, can be a slice or integer
        :param bodypart: The body part index, can be a slice or integer
        :returns: The x video coordinates for the given frames, in the form of a numpy array...
        """
        return self._data[frame, self._fix_index(bodypart, 0)]

    def get_y_at(
        self, frame: Union[int, slice], bodypart: Union[int, slice]
    ) -> ndarray:
        """
        Get the y video coordinates for specific body parts or frames.

        :param frame: The frame index, can be a slice or integer
        :param bodypart: The body part index, can be a slice or integer
        :returns: The y video coordinates for the given frames, in the form of a numpy array...
        """
        return self._data[frame, self._fix_index(bodypart, 1)]

    def get_prob_at(
        self, frame: Union[int, slice], bodypart: Union[int, slice]
    ) -> ndarray:
        """
        Get the probability values for specific body parts or frames.

        :param frame: The frame index, can be a slice or integer
        :param bodypart: The body part index, can be a slice or integer
        :returns: The probability values for the given frames, in the form of a numpy array...
        """
        return self._data[frame, self._fix_index(bodypart, 2)]

    def get_frame_count(self) -> int:
        """
        Returns the amount of frames in this pose object

        :return: An integer, being the amount of total frames stored in this pose
        """
        return self._data.shape[0]

    def get_bodypart_count(self) -> int:
        """
        Gets the amount of body parts per frame in this pose object

        :return: The amount of body parts per frame, as an integer.
        """
        return self._data.shape[1] // 3
