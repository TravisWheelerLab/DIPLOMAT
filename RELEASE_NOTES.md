Changes for this version of DIPLOMAT:
 - Fixed a bug in assignment that induced very high rates of swapping between bodies. The Hungarian algorithm is now implemented by linear_sum_assignment from scipy.optimize, defined locally due to dependency issues. This reverts a previous change that implemented the Hungarian algorithm with numba routines.
