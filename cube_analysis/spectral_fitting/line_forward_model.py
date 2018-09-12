
import numpy as np
from astropy.convolution import convolve_fft
from scipy.optimize import curve_fit

from .discrete_sampler import sample_at_channels


def convolve_and_sample(vels, model, kernel=None, upsamp_fraction=4):
    '''
    Re-sample a model over finite bin sizes. If a kernel is provided, the
    model samples are first  convolved by the kernel prior to sampling.

    Parameters
    ----------
    vels : `~numpy.ndarray`
        Spectral bin centres.
    model : `~astropy.modeling.Models.Model`
        Model to be fit.
    kernel : function, optional
        The line response function. Must take spectral location and
        the channel width as inputs and return the kernel values.
    upsamp_fraction : int, optional
        Number of points to oversample the model (and kernel) prior
        to discretizing the model values in the given spectral bins.

    Returns
    -------
    spec : `~numpy.ndarray`
        Model samples (and convolved) to the given spectral bins.
    '''

    vels = vels.astype(float)

    chan_diff = np.diff(vels[:2])[0]
    if chan_diff > 0:
        min_vel = vels.min()
        max_vel = vels.max()
    else:
        min_vel = vels.max()
        max_vel = vels.min()

    # Upsample the given velocities for the convolution
    upsamp_vels = np.arange(min_vel, max_vel,
                            chan_diff / upsamp_fraction)

    chan_width = np.abs(chan_diff)

    gauss_mod = model(upsamp_vels)

    if kernel is not None:
        resp_mod = kernel(upsamp_vels, chan_width)

        gauss_resp_mod = convolve_fft(gauss_mod, resp_mod)
    else:
        gauss_resp_mod = gauss_mod

    # Now sample over the range of finite channels
    spec = sample_at_channels(vels, upsamp_vels, gauss_resp_mod)

    return spec
