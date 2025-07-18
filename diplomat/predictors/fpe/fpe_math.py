from typing import Optional, Callable, Tuple, Iterable, Union
import numpy as np


float_like = Union[float, np.ndarray]


def gaussian_formula(
    prior_x: float_like,
    x: float_like,
    prior_y: float_like,
    y: float_like,
    std: float,
    amplitude: float,
    lowest_value: float = 0,
    in_log_space: bool = False,
) -> float_like:
    """
    Compute a value on a 2D Gaussian curve.

    :param prior_x: The prior x value. Center of the gaussian bell.
    :param x: The current x value. Value to evaluate on the curve.
    :param prior_y: The prior y value. Center of the gaussian bell.
    :param y: The current y value. Value to evaluate on the curve.
    :param std: The standard deviation of the 2D gaussian, in both dimensions.
    :param amplitude: The amplitude, or height of the gaussian curve.
    :param lowest_value: The lowest value the gaussian curve can reach (achieved by clipping the curve). Defaults to 0.
    :param in_log_space: Boolean, if true returns the log-probability instead of the probability (in log base 2 space).

    :returns: A float, the 2D gaussian evaluated with the above parameters.
    """
    inner_x_delta = ((prior_x - x) ** 2) / (2 * std * std)
    inner_y_delta = ((prior_y - y) ** 2) / (2 * std * std)
    if not in_log_space:
        return np.maximum(
            amplitude * np.exp(-(inner_x_delta + inner_y_delta)), lowest_value
        )
    else:
        return np.maximum(
            np.log2(amplitude) - (inner_x_delta + inner_y_delta) * np.log(np.e),
            np.log2(lowest_value),
        )


def skeleton_formula(
    x: float_like,
    y: float_like,
    peak_dist_out: float,
    peak_std: float,
    peak_amplitude: float,
    trough_amplitude: float,
    in_log_space: bool = False,
) -> float_like:
    """
    Compute a location on the 2D skeletal transition curve with given parameters. The equation takes the form:

    S(x) = {
        max(trough_amp, peak_amplitude * G(x: peak_dist_out, peak_std))  if x < dist_out
        peak_amplitude * G(x: dist_out, peak_std)                        otherwise
    }

    Where G(x: mu, sigma) is a normal distribution without normalization, or a peak of 1 (gaussian curve), centered at
    mu with a standard deviation of sigma. This is in 1 dimension, to generalize to two dimensions, the Euclidean
    distance to x, y from 0, 0 is used as the input to the 1D version of the function.

    :param x: The x location to evaluate the skeletal transition function at.
    :param y: The y location to evaluate the skeletal transition function at.
    :param peak_dist_out: The location of the peak or max value of the curve on all sides from 0, 0.
    :param peak_std: The standard deviation of the gaussian centered at the peak.
    :param peak_amplitude: The amplitude of the peek.
    :param trough_amplitude: The amplitude of the trough, or the curve value at 0, 0.
    :param in_log_space: Boolean, if True (defaults to False) return log-probabilities (base 2) instead of regular
                         probabilities.

    :returns: A float, the skeletal transition function evaluated at the provided location.
    """
    d0 = np.sqrt(x * x + y * y)
    if not in_log_space:
        g = peak_amplitude * np.exp(-((d0 - peak_dist_out) ** 2) / (2 * peak_std**2))
        return np.where(d0 < peak_dist_out, np.maximum(g, trough_amplitude), g)
    else:
        g = np.log2(peak_amplitude) + np.log2(np.e) * (
            -((d0 - peak_dist_out) ** 2) / (2 * peak_std**2)
        )
        return np.where(d0 < peak_dist_out, np.maximum(g, np.log2(trough_amplitude)), g)


def old_skeleton_formula(
    x: float_like,
    y: float_like,
    peak_dist_out: float,
    peak_amplitude: float,
    trough_amplitude: float,
    in_log_space: bool = False,
) -> float_like:
    """
    Compute a location on the 2D skeletal transition curve with given parameters. The equation takes the form:

    S(x) = c * e ^ (ax^2 - bx^4)

    In 1 dimension. To generalize to two dimensions, the euclidean distance to x, y from 0, 0 is used as the
    input to the 1D version of the function.

    :param x: The x location to evaluate the skeletal transition function at.
    :param y: The y location to evaluate the skeletal transition function at.
    :param peak_dist_out: The location of the peak or max value of the curve on all sides from 0, 0.
    :param peak_amplitude: The amplitude of the peek.
    :param trough_amplitude: The amplitude of the trough, or the curve value at 0, 0.
    :param in_log_space: Boolean, if True (defaults to False) return log-probabilities (base 2) instead of regular
                         probabilities.

    :returns: A float, the skeletal transition function evaluated at the provided location.
    """
    # Compute a, b, and c.
    a = (2 / (peak_dist_out**2)) * np.log(peak_amplitude / trough_amplitude)
    b = (1 / (peak_dist_out**4)) * np.log(peak_amplitude / trough_amplitude)
    c = trough_amplitude

    # To use 1D formula...
    x_y_out = x**2 + y**2

    if not in_log_space:
        return c * np.exp((a * x_y_out) - (b * x_y_out**2))
    else:
        return np.log2(c) + ((a * x_y_out) - (b * x_y_out**2)) * np.log2(np.e)


def get_func_table(
    width: int,
    height: int,
    fill_func: Callable[[float_like, float_like], float_like],
    flatten_radius: Optional[float] = None,
) -> np.ndarray:
    """
    Create a precomputed table of values for a given 2D function.

    :param width: The width of the 2D array or table.
    :param height: The height of the 2D array or table.
    :param fill_func: A function that accepts two argument (x and y) and returns a values for the given locations.
    :param flatten_radius: Optional. The radius in which to average the values and then replace all the values with
                           the average. Has the effect of flattening the values within the radius from 0, 0.

    :returns: A 2D numpy array of floats, with the function evaluated at each index. (Indexing is x then y).
    """
    x, y = np.ogrid[0:width, 0:height]
    table = fill_func(x, y)

    if flatten_radius is not None:
        flat_locs = (width * width + height * height) < flatten_radius**2

        if np.sum(flat_locs) > 0:
            table[flat_locs] = np.mean(table[flat_locs])

    return table


def gaussian_table(
    width: int,
    height: int,
    std: float,
    amplitude: float,
    lowest_value: float = 0,
    flatten_radius: Optional[float] = None,
    square_dists: bool = False,
    in_log_space: bool = False,
) -> np.ndarray:
    """
    Creates a pre-computed 2D gaussian table.

    :param width: The width of the table, or 2D numpy array.
    :param height: The height of the table, or 2D numpy array.
    :param std: The standard deviation of the gaussian curve used.
    :param amplitude: The amplitude of the gaussian table.
    :param lowest_value: The lowest value the gaussian curve can get down to.
    :param flatten_radius: The radius out from the peak the gaussian curve to flatten.
    :param square_dists: Boolean, if true square the distance values before putting them through the Gaussian curve.
    :param in_log_space: Boolean, if true return log-probs (base 2) instead of probabilities

    :return: A 2D numpy array of floats, containing a 2D gaussian curve. (Indexing is x then y).
    """
    dist_func = (lambda v: v) if (not square_dists) else (lambda v: v * v)
    g = lambda x, y: gaussian_formula(
        0, dist_func(x), 0, dist_func(y), std, amplitude, lowest_value, in_log_space
    )
    return get_func_table(width, height, g, flatten_radius)


def normalize_all(arrays: Iterable[np.ndarray]) -> Tuple[np.ndarray, ...]:
    """
    Normalize a collection of numpy arrays all together. The sums of all the values in all the arrays will equal 1.

    :param arrays: An iterable of numpy arrays to normalize together.

    :returns: A tuple of numpy arrays the same length as the ones passed in, that have all been normalized together.
    """
    if not isinstance(arrays, (tuple, list)):
        # To be able to iterate it twice...
        arrays = list(arrays)
    total = sum(np.sum(array) for array in arrays)
    return tuple(array / total for array in arrays)


# Type for a transition function....
Probs = np.ndarray
Coords = Tuple[np.ndarray, np.ndarray]
TransitionFunction = Callable[
    [int, Probs, Coords, float, int, Probs, Coords, float], np.ndarray
]


def table_transition(
    prior_coords: Coords, current_coords: Coords, lookup_table: np.ndarray
) -> np.ndarray:
    """
    Compute transition probabilities from a transition probability table.

    :param prior_coords: The prior frame coordinates, 2xN numpy array (x, y).
    :param current_coords: The current frame coordinates, 2xN numpy array (x, y).
    :param lookup_table: The 2D probability lookup table ([delta x, delta y] -> prob). A numpy array.

    :return: A 2D array containing all the transition probabilities for going from any pixel in prior to
             any pixel in current.
    """
    px, py = prior_coords
    cx, cy = current_coords

    cx, cy, px, py = [
        v.astype(np.int64) if not np.issubdtype(cx.dtype, np.integer) else v
        for v in (cx, cy, px, py)
    ]

    delta_x = np.abs(np.expand_dims(cx, 1) - np.expand_dims(px, 0))
    delta_y = np.abs(np.expand_dims(cy, 1) - np.expand_dims(py, 0))

    return lookup_table[delta_y.flatten(), delta_x.flatten()].reshape(delta_y.shape)


def __trans(tbl, x, y):
    return tbl[y.flatten(), x.flatten()].reshape(y.shape)


def __rescale(coords, input_scale, dest_scale):
    mult = input_scale / dest_scale
    return [v * mult for v in coords]


def table_transition_interpolate(
    prior_coords: Coords,
    prior_scale: float,
    current_coords: Coords,
    current_scale: float,
    lookup_table: np.ndarray,
    lookup_scale: float,
):
    """
    Compute transition probabilities from a transition probability table. Unlike table_transition, this method
    supports floats (in-between) coordinated, which it resolves using bi-linear interpolation.

    :param prior_coords: The prior frame coordinates, 2xN numpy array (x, y).
    :param prior_scale: Downscaling factor for the prior data coordinates.
    :param current_coords: The current frame coordinates, 2xN numpy array (x, y).
    :param current_scale: Downscaling factor for the current data coordinates.
    :param lookup_table: The 2D probability lookup table ([delta x, delta y] -> prob). A numpy array.
    :param lookup_scale: Size of each cell in the lookup table...

    :return: A 2D array containing all the transition probabilities for going from any pixel in prior to
             any pixel in current, indexed by current first, prior second...
    """

    px, py = __rescale(prior_coords, prior_scale, lookup_scale)
    cx, cy = __rescale(current_coords, current_scale, lookup_scale)

    delta_x = np.abs(np.expand_dims(cx, 1) - np.expand_dims(px, 0))
    delta_y = np.abs(np.expand_dims(cy, 1) - np.expand_dims(py, 0))

    lx = delta_x.astype(np.int64)
    ly = delta_y.astype(np.int64)
    rx = delta_x - lx
    ry = delta_y - ly

    lyp1 = np.clip(ly + 1, 0, lookup_table.shape[0] - 1)
    lxp1 = np.clip(lx + 1, 0, lookup_table.shape[1] - 1)

    top_interp = (
        __trans(lookup_table, lx, ly) * (1 - rx) + __trans(lookup_table, lxp1, ly) * rx
    )
    bottom_interp = (
        __trans(lookup_table, lx, lyp1) * (1 - rx)
        + __trans(lookup_table, lxp1, lyp1) * rx
    )
    return top_interp * (1 - ry) + bottom_interp * ry
