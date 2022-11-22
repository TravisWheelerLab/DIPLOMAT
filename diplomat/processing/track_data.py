"""
Provides the :py:class:`~diplomat.processing.track_data.TrackingData` class, used for storing model outputs and passing them to predictors.
"""

from typing import Optional, Tuple, Union, Sequence
import numpy as np
from numpy import ndarray
from diplomat.processing.pose import Pose


class TrackingData:
    """
    Represents tracking data received from the DeepLabCut neural network. Includes a source map of probabilities,
    the predicted location offsets within the source map, stride info and ect. Also provides many convenience methods
    for working with and getting info from the DLC neural network data.
    """

    """ The default image down scaling used by DeepLabCut """
    DEFAULT_SCALE: int = 8

    def __init__(
        self,
        scmap: ndarray,
        locref: Optional[ndarray] = None,
        stride: float = DEFAULT_SCALE,
    ):
        """
        Create an new tracking data object to store DLC neural network data for one frame or a batch of frames.

        :param scmap: The probability maps produced by the neural network, a 4-dimensional numpy array containing the
                      dimensions: [frame, y location, x location, body part].
        :param locref: The "offsets" produced by DeepLabCut neural network, stored in a 5-dimensional numpy array
                       containing the dimensions:
                       [frame, y location, x location, bodypart, 0 for x offset or 1 for y offset]
        :param stride: Float which stores the down scaling of the probability map relative to the size of the original
                       video. This value defaults to 8, meaning the original video is 8 times the size of the
                       probability map.
        """
        # If scmap received is only 3-dimension, it is of only 1 frame, so add the batch dimension so it works better.
        if len(scmap.shape) == 3:
            self._scmap = np.expand_dims(scmap, axis=0)
        else:
            self._scmap = scmap

        if (locref is not None) and (len(locref.shape) == 3):
            self._locref = np.expand_dims(locref, axis=0)
        else:
            self._locref = locref

        self._scaling = stride

    @classmethod
    def empty_tracking_data(
        cls,
        frame_amt: int,
        part_count: int,
        width: int,
        height: int,
        stride: float = DEFAULT_SCALE,
        allocate_offsets: bool = False
    ) -> "TrackingData":
        """
        Create a new empty tracking data object with space allocated to fit the specified sizes of data.

        :param frame_amt: The amount of probability map frames to allocate space for, an Integer.
        :param part_count: The amount of body parts per frame to allocate space for, an Integer.
        :param width: The width of each probability frame, an Integer.
        :param height: The height of each probability frame, an Integer.
        :param stride: The downscaling of the probability frame relative to the original video, defaults to 8,
                       meaning the original video is 8 times the size of the probability map.
        :param allocate_offsets: A boolean, determines if the offset array is setup as an array of zeros. Default value
                                 is False, meaning the offsets array is just set to None.
        :return: A tracking data object full of zeroes.
        """
        return cls(
            np.zeros((frame_amt, height, width, part_count), dtype="float32"),
            np.zeros((frame_amt, height, width, part_count, 2), dtype="float32") if(allocate_offsets) else None,
            stride,
        )

    def get_source_map(self) -> ndarray:
        """
        Gets the raw probability source map of this tracking data.

        :return: A numpy array representing the source probability map of this tracking data. It is a 4-dimensional
                array containing the dimensions: [frame, y location, x location, body part] -> probability
        """
        return self._scmap

    def get_offset_map(self) -> Optional[ndarray]:
        """
        Returns the offset prediction map representing offset predictions for each location in the probability map.

        :return: A numpy array representing the predicted offsets for each location within the probability map, or
                 None if this TrackingData doesn't have offset predictions...
                 Indexing: [frame, y location, x location, bodypart, 0 for x offset or 1 for y offset]
        """
        return self._locref

    def get_down_scaling(self) -> float:
        """
        Get the down scaling performed on this source map, as an integer.

        :return: An integer representing the downscaling of the probability map compared to the original video file, in
                 terms of what the dimensions of the probability map need to multiplied by to match the dimensions of
                 the original video file.
        """
        return self._scaling

    def set_source_map(self, scmap: ndarray):
        """
        Set the raw probability map of this tracking data.

        :param scmap: A numpy array representing the probability map of this tracking data. It is a 4-dimensional
                      array containing the dimensions: [frame, y location, x location, body part]
        """
        self._scmap = scmap

    def set_offset_map(self, locref: Optional[ndarray]):
        """
        Set the offset prediction map representing offset predictions for each location in the probability map.

        :param locref: A numpy array representing the predicted offsets within the probability map. Can also
                       be set to None to indicate this TrackingData doesn't have or support higher precision
                       predictions.
                       Dimensions: [frame, y location, x location, bodypart, 0 for x offset or 1 for y offset]
        """
        self._locref = locref

    def set_down_scaling(self, scale: float):
        """
        Set the down scaling performed on the probability map, as an integer

        :param scale: An float representing the downscaling of the probability map compared to the original video file, in
                 terms of what the dimensions of the probability map need to multiplied by to match the dimensions of
                 the original video file.
        """
        self._scaling = scale

    def get_max_scmap_points(self, num_max: int = 1) -> Tuple[ndarray, ndarray]:
        """
        Get the locations with the max probabilities for each frame in this TrackingData.

        :param num_max: Specifies the number of maximums to grab for each body part from each frame. Defaults to 1.

        :return: A tuple of numpy arrays, the first numpy array being the y index for the max of each frame, the second
                 being the x index for the max of each frame.
                 Dimensions: [1 for x and 0 for y index, frame, body part] -> index
        """
        batchsize, ny, nx, num_joints = self._scmap.shape
        scmap_flat = self._scmap.reshape((batchsize, nx * ny, num_joints))

        if num_max <= 1:
            scmap_top = np.argmax(scmap_flat, axis=1)
        else:
            # Grab top values
            scmap_top = np.argpartition(scmap_flat, -num_max, axis=1)[:, -num_max:]
            for ix in range(batchsize):
                # Sort predictions for each body part from highest to least...
                vals = scmap_flat[ix, scmap_top[ix], np.arange(num_joints)]
                arg = np.argsort(-vals, axis=0)
                scmap_top[ix] = scmap_top[ix, arg, np.arange(num_joints)]
            # Flatten out the map so arrangement is:
            # [frame] -> [joint 1 prediction 1, joint 1 prediction 2, ... , joint 2 prediction 1, ... ]
            # Note this mimics single prediction format...
            scmap_top = scmap_top.swapaxes(1, 2).reshape(
                batchsize, num_max * num_joints
            )

        # Convert to x, y locations....
        return np.unravel_index(scmap_top, (ny, nx))

    def get_max_of_frame(
        self, frame: int, num_outputs: int = 1
    ) -> Tuple[ndarray, ndarray]:
        """
        Get the locations of the highest probabilities for a single frame in the array.

        :param frame: The index of the frame to get the maximum of, in form of an integer.
        :param num_outputs: Specifies the number of maximums to grab for each body part from the frame. Defaults to 1.
        :return: A tuple of numpy arrays, the first numpy array being the y index of the max probability for each body
                 part in the frame, the second being the x index of the max probability for each body part in the frame
                 Indexing: [1 for x or 0 for y index] -> [bodypart 1, bodypart 1 prediction 2, ..., bodypart 2, ...]
        """
        y_dim, x_dim, num_joints = self._scmap.shape[1:4]
        scmap_flat = self._scmap[frame].reshape((y_dim * x_dim, num_joints))

        if num_outputs <= 1:
            # When num_outputs is 1, we just grab the single maximum...
            flat_max = np.argmax(scmap_flat, axis=0)
        else:
            # When num_outputs greater then 1, use partition to get multiple maximums...
            scmap_top = np.argpartition(scmap_flat, -num_outputs, axis=0)[-num_outputs:]
            vals = scmap_flat[scmap_top, np.arange(num_joints)]
            arg = np.argsort(-vals, axis=0)
            flat_max = (
                scmap_top[arg, np.arange(num_joints)]
                .swapaxes(1, 2)
                .reshape(num_outputs * num_joints)
            )

        return np.unravel_index(flat_max, dims=(y_dim, x_dim))

    def get_poses_for(self, points: Tuple[ndarray, ndarray]):
        """
        Return a pose object for the "maximum" predicted indexes passed in.

        :param points: A tuple of 2 numpy arrays, one representing the y indexes for each frame and body part,
                       the other being the x indexes represented the same way. (Note 'get_max_scmap_points' returns
                       maximum predictions in this exact format).
        :return: The Pose object representing all predicted maximum locations for selected points...

        NOTE: This method detects when multiple predictions(num_outputs > 1) have been made and will still work
              correctly...
        """
        y, x = points
        # Create new numpy array to store probabilities, x offsets, and y offsets...
        probs = np.zeros(x.shape)

        # Get the number of predicted values for each joint for the passed maximums... We will divide the body part
        # index by this value in order to get the correct body part in this source map...
        num_outputs = x.shape[1] / self.get_bodypart_count()

        x_offsets = np.zeros(x.shape)
        y_offsets = np.zeros(y.shape)

        # Iterate the frame and body part indexes in x and y, we just use x since both are the same size
        for frame in range(x.shape[0]):
            for bp in range(x.shape[1]):
                probs[frame, bp] = self._scmap[
                    frame, y[frame, bp], x[frame, bp], int(bp // num_outputs)
                ]
                # Locref is frame -> y -> x -> bodypart -> relative coordinate pair offset. if it is None, just keep
                # all offsets as 0.
                if self._locref is not None:
                    x_offsets[frame, bp], y_offsets[frame, bp] = self._locref[
                        frame, y[frame, bp], x[frame, bp], int(bp // num_outputs)
                    ]

        # Now apply offsets to x and y to get actual x and y coordinates...
        # Done by multiplying by scale, centering in the middle of the "scale square" and then adding extra offset
        x = x.astype("float32") * self._scaling + (0.5 * self._scaling) + x_offsets
        y = y.astype("float32") * self._scaling + (0.5 * self._scaling) + y_offsets

        return Pose(x, y, probs)

    @staticmethod
    def _get_count_of(val: Union[int, slice, Sequence[int]], length: int) -> int:
        """ Internal private method to get length of an index selection(as in how many indexes it selects...) """
        if isinstance(val, Sequence):
            return len(val)
        elif isinstance(val, slice):
            start, stop, step = val.indices(length)
            return len(range(start, stop - 1, step))
        elif isinstance(val, int):
            return 1
        else:
            raise ValueError("Value is not a slice, integer, or list...")

    def get_prob_table(
        self,
        frame: Union[int, slice, Sequence[int]],
        bodypart: Union[int, slice, Sequence[int]],
    ) -> ndarray:
        """
        Get the probability map for a selection of frames and body parts or a single frame and body part.

        :param frame: The frame index, as an integer or slice.
        :param bodypart: The body part index, as an integer or slice.
        :return: The probability map(s) for a single frame or selection of frames based on indexes, as a numpy array...
                 Dimensions: [frame, body part, y location, x location] -> probability
        """
        # Compute amount of frames and body parts selected....
        frame_count = self._get_count_of(frame, self.get_frame_count())
        part_count = self._get_count_of(bodypart, self.get_bodypart_count())

        # Return the frames, reshaped to be more "frame like"...
        slicer = self._scmap[frame, :, :, bodypart]

        # If the part_count is greater then one, move bodypart dimension back 2 in the dimensions
        if part_count > 1 and frame_count > 1:
            return np.transpose(slicer, (0, 3, 1, 2))
        elif part_count > 1:
            return np.transpose(slicer, (2, 0, 1))
        # Otherwise just return the slice...
        else:
            return slicer

    def set_prob_table(
        self,
        frame: Union[int, slice, Sequence[int]],
        bodypart: Union[int, slice, Sequence[int]],
        values: ndarray,
    ):
        """
        Set the probability table for a selection of frames and body parts or a single frame and body part.

        :param frame: The frame index, as an integer or slice.
        :param bodypart: The body part index, as an integer or slice.
        :param values: The probability map(s) to set in this TrackingData object based on the frame and body parts
                       specified, as a numpy array...
                       Dimensions of values: [frame, body part, y location, x location] -> probability
        """
        # Compute amount of frames and body parts selected....
        frame_count = self._get_count_of(frame, self.get_frame_count())
        part_count = self._get_count_of(bodypart, self.get_bodypart_count())

        # If multiple body parts were selected, rearrange dimensions to match those used by the scmap...
        if part_count > 1 and frame_count > 1:
            values = np.transpose(self._scmap[frame, :, :, bodypart], (0, 2, 3, 1))
        elif part_count > 1:
            values = np.transpose(self._scmap[frame, :, :, bodypart], (1, 2, 0))

        # Set the frames, resizing the array to fit
        self._scmap[frame, :, :, bodypart] = values

    def get_frame_count(self) -> int:
        """
        Get the number of frames stored in this TrackingData object.

        :return: The number of frames stored in this tracking data object.
        """
        return self._scmap.shape[0]

    def get_bodypart_count(self) -> int:
        """
        Get the number of body parts stored in this TrackingData object per frame.

        :return: The number of body parts per frame as an integer.
        """
        return self._scmap.shape[3]

    def get_frame_width(self) -> int:
        """
        Return the width of each probability map in this TrackingData object.

        :return: The width of each probability map as an integer.
        """
        return self._scmap.shape[2]

    def get_frame_height(self) -> int:
        """
        Return the height of each probability map in this TrackingData object.

        :return: The height of each probability map as an integer.
        """
        return self._scmap.shape[1]

    # Used for setting single poses....
    def set_pose_at(
        self,
        frame: int,
        bodypart: int,
        scmap_x: int,
        scmap_y: int,
        pose_object: Pose,
        output_num: int = 0,
    ):
        """
        Set a pose in the specified Pose object to the specified x and y coordinate for a provided body part and frame.
        This method will use data from this TrackingData object to correctly set the information in the Pose object.

        :param frame: The specified frame to copy from this TrackingData to the Pose object and set.
        :param bodypart: The specified body part to copy from this TrackingData to the pose object and set
        :param scmap_x: The x index of this TrackingData to set the Pose prediction to.
        :param scmap_y: The y index of this TrackingData to set the Pose prediction to.
        :param pose_object: The pose object to be modified/copied to.
        :param output_num: The output number to set in the pose object (Which prediction for this bodypart?).
                           Should only be needed if num_outputs > 1. Defaults to 0, meaning the first prediction.
        :return: Nothing, changes stored in pose_object...
        """
        # Get probability...
        prob = self._scmap[frame, scmap_y, scmap_x, bodypart]
        # Default offsets to 0
        off_x, off_y = 0, 0

        # If we are actually using locref, set offsets to it
        if self._locref is not None:
            off_y, off_x = self._locref[frame, scmap_y, scmap_x, bodypart]

        # Compute actual x and y values in the video...
        scmap_x = float(scmap_x) * self._scaling + (0.5 * self._scaling) + off_x
        scmap_y = float(scmap_y) * self._scaling + (0.5 * self._scaling) + off_y

        # Set values...
        pose_object.set_x_at(frame, bodypart + output_num, scmap_x)
        pose_object.set_y_at(frame, bodypart + output_num, scmap_y)
        pose_object.set_prob_at(frame, bodypart + output_num, prob)

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"frames={self.get_frame_count()}, "
            f"parts={self.get_bodypart_count()}, "
            f"width={self.get_frame_width()}, "
            f"height={self.get_frame_height()}, "
            f"has_offset_map={self.get_offset_map() is not None}, "
            f"downscaling={self.get_down_scaling()}"
            f")"
        )