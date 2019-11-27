
'''
Routines for generating Gaussian-convolved noise in 1D and 2D.
Useful for generating correlated additive noise to images and spectra.

These may find a permanent home in TurbuStat eventually.
'''


import numpy as np
from astropy.utils import NumpyRNGContext

# from turbustat.simulator.gen_field import make_extended


def gaussian_beam(f, beam_gauss_width):
    '''
    Fourier transform of a Gaussian beam. NOT the power spectrum (multiply exp
    argument by 2 for power spectrum).

    Parameters
    ----------
    f : np.ndarray
        Frequencies to evaluate beam at.
    beam_gauss_width : float
        Beam size. Should be the Gaussian rms, not FWHM.
    '''
    return np.exp(-f**2 * np.pi**2 * 2 * beam_gauss_width**2)


def gauss_correlated_noise_1D(size, sigma, beam_gauss_width,
                              randomseed=327485749):
    '''
    Generate correlated Gaussian noise with sigma, smoothed by a
    Gaussian kernel.
    '''

    # Making a real signal. Only need real part of FFT
    freqs = np.fft.rfftfreq(size)

    with NumpyRNGContext(randomseed):

        angles = np.random.uniform(0, 2 * np.pi,
                                   size=freqs.size)

        noise = np.cos(angles) + 1j * np.sin(angles)

    corr_field = np.fft.irfft(noise *
                              gaussian_beam(freqs, beam_gauss_width))

    norm = (np.sqrt(np.sum(corr_field**2)) / np.sqrt(corr_field.size)) / sigma

    corr_field /= norm

    return corr_field


def gauss_correlated_noise_2D(shape, sigma, beam_gauss_width,
                              randomseed=327485749):
    '''
    Generate correlated Gaussian noise with sigma, smoothed by a
    Gaussian kernel.
    '''

    # Making a real signal. Only need real part of FFT
    freqs_yy, freqs_xx = np.meshgrid(np.fft.fftfreq(shape[0]),
                                     np.fft.rfftfreq(shape[1]), indexing="ij")

    freqs = np.sqrt(freqs_yy**2 + freqs_xx**2)
    # freqs[freqs == 0.] = np.NaN
    # freqs[freqs == 0.] = 1.

    imsize = shape[0]

    Np1 = (imsize - 1) // 2 if imsize % 2 != 0 else imsize // 2

    with NumpyRNGContext(randomseed):

        angles = np.random.uniform(0, 2 * np.pi,
                                   size=freqs.shape)

    noise = np.cos(angles) + 1j * np.sin(angles)

    if imsize % 2 == 0:
        noise[1:Np1, 0] = np.conj(noise[imsize:Np1:-1, 0])
        noise[1:Np1, -1] = np.conj(noise[imsize:Np1:-1, -1])
        noise[Np1, 0] = noise[Np1, 0].real + 1j * 0.0
        noise[Np1, -1] = noise[Np1, -1].real + 1j * 0.0

    else:
        noise[1:Np1 + 1, 0] = np.conj(noise[imsize:Np1:-1, 0])
        noise[1:Np1 + 1, -1] = np.conj(noise[imsize:Np1:-1, -1])

    # Zero freq components must have no imaginary part to be own conjugate
    noise[0, -1] = noise[0, -1].real + 1j * 0.0
    noise[0, 0] = noise[0, 0].real + 1j * 0.0

    corr_field = np.fft.irfft2(noise *
                               gaussian_beam(freqs, beam_gauss_width))

    norm = (np.sqrt(np.sum(corr_field**2)) / np.sqrt(corr_field.size)) / sigma

    corr_field /= norm

    return corr_field
