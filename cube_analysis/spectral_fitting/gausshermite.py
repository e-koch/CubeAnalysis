
import numpy as np
from lmfit import minimize, Parameters
from spectral_cube import OneDSpectrum


def gaussfunc_gh(paramsin, x):
    '''
    Code adapted from http://research.endlessfernweh.com/curve-fitting/
    '''

    try:
        amp = paramsin['amp'].value
        center = paramsin['center'].value
        sig = paramsin['sig'].value
        skew = paramsin['skew'].value
        kurt = paramsin['kurt'].value
    except:
        # When given just an array or list
        amp, center, sig, skew, kurt = paramsin

    c1 = -np.sqrt(3)
    c2 = -np.sqrt(6)
    c3 = 2 / np.sqrt(3)
    c4 = np.sqrt(6) / 3.
    c5 = np.sqrt(6) / 4.

    term1 = (x - center) / sig

    gaustot_gh = amp * np.exp(-.5 * term1 ** 2) * \
        (1 + skew * (c1 * term1 + c3 * term1 ** 3) +
         kurt * (c5 + c2 * term1**2 + c4 * term1**4))
    return gaustot_gh


def herm_gauss_peak(y_data, x_data=None, chanwidth=1):
    '''
    Fits a Hermite-Gaussian model and returns the position of the peak.

    Code adapted from http://research.endlessfernweh.com/curve-fitting/
    '''

    # When passing a OneDSpectrum, extract the info directly
    if isinstance(y_data, OneDSpectrum):
        x_data = y_data.spectral_axis.value
        chanwidth = np.diff(x_data[:2])[0]
        y_data = y_data.value.copy()

    if x_data is None:
        minim = 0
        maxim = len(y_data)
        x_data = np.arange(maxim)
    else:
        minim = np.min(x_data)
        maxim = np.max(x_data)

    p_gh = Parameters()
    p_gh.add('amp', value=y_data.max(), vary=True)
    p_gh.add('center', value=x_data[y_data.argmax()], min=minim, max=maxim)
    p_gh.add('sig', value=30 * chanwidth, min=chanwidth, max=None)
    p_gh.add('skew', value=0, vary=True, min=None, max=None)
    p_gh.add('kurt', value=0, vary=True, min=None, max=None)

    def gausserr_gh(p, x, y):
        return gaussfunc_gh(p, x) - y

    fitout_gh = minimize(gausserr_gh, p_gh, args=(x_data, y_data))

    fit_gh = gaussfunc_gh(fitout_gh.params, x_data)

    # For debugging only
    verbose = False
    if verbose:
        import matplotlib.pyplot as p
        p.plot(x_data, fit_gh)

    return x_data[fit_gh.argmax()], 0
