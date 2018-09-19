
cimport cython
import numpy as np
cimport numpy as np

from libc.math cimport erf, sqrt, pi, abs

cdef double SQRT2 = sqrt(2)
cdef double SQRTPIHALF = sqrt(pi * 0.5)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def gauss_weighted_avg(double a, double b, double amp, double stddev,
                       double mean):
    '''
    Calculate the weighted average of a Gaussian between a and b.
    '''

    cdef double erf_term

    erf_term = erf((mean - a) / (SQRT2 * stddev)) - \
        erf((mean - b) / (SQRT2 * stddev))

    cdef double weight_avg = (amp * stddev / (b - a)) * erf_term * SQRTPIHALF

    return weight_avg


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def gauss_model_discrete(np.ndarray[np.float64_t, ndim=1] vels, double amp,
                         double mean, double stddev):

    # Assume the channels are equally spaced.
    cdef double half_chan_width = np.abs(vels[1] - vels[0]) / 2.

    cdef np.ndarray[np.float64_t, ndim=1] vals = np.zeros_like(vels)

    cdef int i

    for i, vel in enumerate(vels):
        vals[i] = gauss_weighted_avg(vel - half_chan_width, vel + half_chan_width,
                                     amp, stddev, mean)

    return vals

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def sample_at_channels(np.ndarray[np.float64_t, ndim=1] vels,
                       np.ndarray[np.float64_t, ndim=1] upsamp_vels,
                       np.ndarray[np.float64_t, ndim=1] values):
    '''
    Re-sample values at a lower spectral resolution given by vels as the weighted average.
    '''

    cdef np.ndarray[np.float64_t, ndim=1] spec = np.zeros_like(vels)

    cdef double half_chan_width = (vels[1] - vels[0]) / 2.
    cdef double chan_width = abs(vels[1] - vels[0])

    cdef int i = 0

    cdef np.ndarray[np.uint8_t, ndim = 1, cast=True] bin_mask

    for vel in vels:

        bin_mask = np.logical_and(upsamp_vels >= vel - half_chan_width,
                                  upsamp_vels <= vel + half_chan_width)

        spec[i] = values[bin_mask].mean()

        i += 1

    return spec
