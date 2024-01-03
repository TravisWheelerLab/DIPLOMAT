from typing import Optional, Callable, Tuple, Iterable
import numpy as np


def gaussian_formula(
    prior_x: float,
    x: float,
    prior_y: float,
    y: float,
    std: float,
    amplitude: float,
    lowest_value: float = 0,
    in_log_space: bool = False
) -> float:
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
    inner_x_delta = ((prior_x - x) ** 2) / (2 * std ** 2)
    inner_y_delta = ((prior_y - y) ** 2) / (2 * std ** 2)
    if(not in_log_space):
        return np.maximum(amplitude * np.exp(-(inner_x_delta + inner_y_delta)), lowest_value)
    else:
        return np.maximum(np.log2(amplitude) - (inner_x_delta + inner_y_delta) * np.log(np.e), np.log2(lowest_value))


def skeleton_formula(
    x: float,
    y: float,
    peak_dist_out: float,
    peak_amplitude: float,
    trough_amplitude: float,
    in_log_space: bool = False
) -> float:
    """
    Compute a location on the 2D skeletal transition curve with given parameters. The equation takes the form:

    S(x) = c * e ^ (ax^2 - bx^4)

    where . In 1 dimension. To generalize to two dimensions, the euclidean distance to x, y from 0, 0 is used as the
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
    a = (2 / (peak_dist_out ** 2)) * np.log(peak_amplitude / trough_amplitude)
    b = (1 / (peak_dist_out ** 4)) * np.log(peak_amplitude / trough_amplitude)
    c = trough_amplitude

    # To use 1D formula...
    x_y_out = x ** 2 + y ** 2

    if(not in_log_space):
        return c * np.exp((a * x_y_out) - (b * x_y_out ** 2))
    else:
        return np.log2(c) + ((a * x_y_out) - (b * x_y_out ** 2)) * np.log2(np.e)


def get_func_table(
    width: int,
    height: int,
    fill_func: Callable[[int, int], float],
    flatten_radius: Optional[float] = None
) -> np.ndarray:
    """
    Create a precomputed table of values for a given 2D function.

    :param width: The width of the 2D array or table.
    :param height: The height of the 2D array or table.
    :param fill_func: A function that accepts two argument (x and y) and returns a values for the given location.
    :param flatten_radius: Optional. The radius in which to average the values and then replace all the values with
                           the average. Has the effect of flattening the values within the radius from 0, 0.

    :returns: A 2D numpy array of floats, with the function evaluated at each index. (Indexing is x then y).
    """
    table = np.zeros((width, height), dtype=np.float32)
    tot_sum = 0
    count = 0

    for x in range(width):
        for y in range(height):
            table[x, y] = fill_func(x, y)

            if((flatten_radius is not None) and ((x ** 2 + y ** 2) < flatten_radius)):
                tot_sum += table[x, y]
                count += 1

    if(count != 0):
        substitute = tot_sum / count
        for x in range(width):
            for y in range(height):
                if((x ** 2 + y ** 2) < flatten_radius):
                    table[x, y] = substitute

    return table


def gaussian_table(
    width: int,
    height: int,
    std: float,
    amplitude: float,
    lowest_value: float = 0,
    flatten_radius: Optional[float] = None,
    square_dists: bool = False,
    in_log_space: bool = False
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
    dist_func = (lambda v: v) if(not square_dists) else (lambda v: v * v)
    g = lambda x, y: gaussian_formula(0, dist_func(x), 0, dist_func(y), std, amplitude, lowest_value, in_log_space)
    return get_func_table(width, height, g, flatten_radius)


def normalize_all(arrays: Iterable[np.ndarray]) -> Tuple[np.ndarray, ...]:
    """
    Normalize a collection of numpy arrays all together. The sums of all the values in all the arrays will equal 1.

    :param arrays: An iterable of numpy arrays to normalize together.

    :returns: A tuple of numpy arrays the same length as the ones passed in, that have all been normalized together.
    """
    if(not isinstance(arrays, (tuple, list))):
        # To be able to iterate it twice...
        arrays = list(arrays)
    total = sum(np.sum(array) for array in arrays)
    return tuple(array / total for array in arrays)


# Type for a transition function....
Probs = np.ndarray
Coords = Tuple[np.ndarray, np.ndarray]
TransitionFunction = Callable[[Probs, Coords, Probs, Coords], np.ndarray]


def table_transition(prior_coords: Coords, current_coords: Coords, lookup_table: np.ndarray) -> np.ndarray:
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

    delta_x = np.abs(np.expand_dims(cx, 1) - np.expand_dims(px, 0))
    delta_y = np.abs(np.expand_dims(cy, 1) - np.expand_dims(py, 0))

    return lookup_table[delta_y.flatten(), delta_x.flatten()].reshape(delta_y.shape)