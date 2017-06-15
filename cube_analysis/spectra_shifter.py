
from astropy.io import fits
from spectral_cube import SpectralCube, VaryingResolutionSpectralCube
from spectral_cube.lower_dimensional_structures import OneDSpectrum
import astropy.units as u
import numpy as np
import os
from astropy.utils.console import ProgressBar
from itertools import izip, repeat

from .io_utils import create_huge_fits


def fourier_shift(x, shift, axis=0):
    '''
    Shift a spectrum by a given number of pixels.

    Parameters
    ----------
    x : np.ndarray
        Array to be shifted
    shift : int or float
        Number of pixels to shift.
    axis : int, optional
        Axis to shift along.

    Returns
    -------
    x2 : np.ndarray
        Shifted array.
    '''
    mask = ~np.isfinite(x)
    nonan = x.copy()
    nonan[mask] = 0.0

    nonan_shift = _shifter(nonan, shift, axis)
    mask_shift = _shifter(mask, shift, axis) > 0.5

    nonan_shift[mask_shift] = np.NaN

    return nonan_shift


def _shifter(x, shift, axis):
    '''
    Helper function for `~fourier_shift`.
    '''
    ftx = np.fft.fft(x, axis=axis)
    m = np.fft.fftfreq(x.shape[axis])
    m_shape = [1] * len(x.shape)
    m_shape[axis] = m.shape[0]
    m = m.reshape(m_shape)
    phase = np.exp(-2 * np.pi * m * 1j * shift)
    x2 = np.real(np.fft.ifft(ftx * phase, axis=axis))
    return x2


def spectrum_shifter(spectrum, v0, vcent):
    '''
    Shift the central velocity of a spectrum by the difference if v0 and vcent.

    Parameters
    ----------
    spectrum : `~spectral_cube.lower_dimensional_objects.OneDSpectrum`
        1D spectrum to shift.
    v0 : `~astropy.units.Quantity`
        Velocity to shift spectrum to.
    vcent : `~astropy.units.Quantity`
        Initial center velocity of the spectrum.
    '''

    vdiff = np.abs(spectrum.spectral_axis[1] - spectrum.spectral_axis[0])
    vel_unit = vdiff.unit

    pix_shift = (vcent.to(vel_unit) - v0.to(vel_unit)) / vdiff

    shifted = fourier_shift(spectrum, pix_shift)

    if hasattr(spectrum, "beams"):
        beams = spectrum.beams
    else:
        beams = None

    return OneDSpectrum(shifted, unit=spectrum.unit, wcs=spectrum.wcs,
                        meta=spectrum.meta, spectral_unit=vel_unit,
                        beams=beams)


def mulproc_spectrum_shifter(inputs):
    y, x, spec, vcent, v0 = inputs

    return spectrum_shifter(spec, v0, vcent), y, x


def cube_shifter(cube, velocity_surface, v0=None, save_shifted=False,
                 save_name=None, xy_posns=None, pool=None,
                 return_spectra=True, chunk_size=20000, is_mask=False,
                 verbose=False):
    '''
    Shift spectra in a cube according to a given velocity surface (peak
    velocity, centroid, rotation model, etc.).
    '''

    if not save_shifted and not return_spectra:
        raise Exception("One of 'save_shifted' or 'return_spectra' must be "
                        "enabled.")

    if not np.isfinite(velocity_surface).any():
        raise Exception("velocity_surface contains no finite values.")

    if xy_posns is None:
        # Only compute where a shift can be found
        xy_posns = np.where(np.isfinite(velocity_surface))

    if v0 is None:
        # Set to near the center velocity of the cube if not given.
        v0 = cube.spectral_axis[cube.shape[0] // 2]
    else:
        if not isinstance(v0, u.Quantity):
            raise u.UnitsError("v0 must be a quantity.")
        spec_unit = cube.spectral_axis.unit
        if not v0.unit.is_equivalent(spec_unit):
            raise u.UnitsError("v0 must have units equivalent to the cube's"
                               " spectral unit ().".format(spec_unit))

    # Adjust the header to have velocities centered at v0.
    new_header = cube.header.copy()
    new_header["CRVAL3"] = new_header["CRVAL3"] - v0.to(u.m / u.s).value

    if save_shifted:

        if is_mask:
            dtype = 'int16'
        else:
            dtype = cube[:, 0, 0].dtype

        output_fits = create_huge_fits(save_name, new_header, dtype=dtype,
                                       return_hdu=True, fill_nan=not is_mask)

    if return_spectra:
        all_shifted_spectra = []
        out_posns = []

    # Create chunks of spectra for read-out.
    nchunks = 1 + xy_posns[0].size // chunk_size

    if verbose:
        iterat = ProgressBar(xrange(nchunks))
    else:
        iterat = xrange(nchunks)

    for i in iterat:

        y_posns = xy_posns[0][i * chunk_size:(i + 1) * chunk_size]
        x_posns = xy_posns[1][i * chunk_size:(i + 1) * chunk_size]

        gen = ((y, x, cube[:, y, x], velocity_surface[y, x], v0) for y, x in
               izip(y_posns, x_posns))

        if pool is not None:
            shifted_spectra = pool.map(mulproc_spectrum_shifter, gen)
        else:
            shifted_spectra = map(mulproc_spectrum_shifter, gen)

        if save_shifted:
            for out in shifted_spectra:
                if is_mask:
                    spec = (out[0].value > 0.5).astype(np.int)
                else:
                    spec = out[0].value
                output_fits[0].data[:, out[1], out[2]] = spec

            output_fits.flush()

        if return_spectra:
            all_shifted_spectra.extend([out[0] for out in shifted_spectra])
            out_posns.extend([out[1:] for out in shifted_spectra])

    if save_shifted:

        output_fits.flush()
        output_fits.close()

        # Append the beam table onto the output file.
        if isinstance(cube, VaryingResolutionSpectralCube):
            from spectral_cube.cube_utils import beams_to_bintable
            output_fits = fits.open(save_name, mode='append')
            output_fits.append(beams_to_bintable(cube.beams))
            output_fits.flush()
            output_fits.close()

    if return_spectra:
        return all_shifted_spectra, out_posns
