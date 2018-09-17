
from astropy.io import fits
from spectral_cube import VaryingResolutionSpectralCube
from spectral_cube.lower_dimensional_structures import OneDSpectrum
import astropy.units as u
from astropy import log
import numpy as np

import sys
if sys.version_info < (3,0):
    from itertools import izip as zip

from .io_utils import create_huge_fits
from .progressbar import _map_context, ProgressBar
from .feather_cubes import get_channel_chunks


def fourier_shift(x, shift, axis=0, add_pad=False, pad_size=None):
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

    # Optionally pad the edges
    if add_pad:
        if pad_size is None:
            # Pad by the size of the shift
            pad = np.ceil(shift).astype(int)

            # Determine edge to pad whether it is a positive or negative shift
            pad_size = (pad, 0) if shift > 0 else (0, pad)
        else:
            assert len(pad_size)

        pad_nonan = np.pad(nonan, pad_size, mode='constant',
                           constant_values=(0))
        pad_mask = np.pad(mask, pad_size, mode='constant',
                          constant_values=(0))
    else:
        pad_nonan = nonan
        pad_mask = mask

    nonan_shift = _shifter(pad_nonan, shift, axis)
    mask_shift = _shifter(pad_mask, shift, axis) > 0.5

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


def _spectrum_shifter(inputs):
    y, x, spec, shift, add_pad, pad_size = inputs

    return fourier_shift(spec, shift, add_pad=add_pad, pad_size=pad_size), y, x


def cube_shifter(cube, velocity_surface, v0=None, save_shifted=False,
                 save_name=None, xy_posns=None, num_cores=1,
                 return_spectra=True, chunk_size=20000, is_mask=False,
                 verbose=False, pad_edges=True):
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

    # Calculate the pixel shifts that will be applied.
    vdiff = np.abs(np.diff(cube.spectral_axis[:2])[0])
    vel_unit = vdiff.unit

    pix_shifts = ((velocity_surface.to(vel_unit) -
                   v0.to(vel_unit)) / vdiff).value[xy_posns]

    # May a header copy so we can start altering
    new_header = cube.header.copy()

    if pad_edges:
        # Enables padding the whole cube such that no spectrum will wrap around
        # This is critical if a low-SB component is far off of the bright
        # component that the velocity surface is derived from.

        # Find max +/- pixel shifts, rounding up to the nearest integer
        max_pos_shift = np.ceil(pix_shifts.max()).astype(int)
        max_neg_shift = np.ceil(pix_shifts.min()).astype(int)

        # The total pixel size of the new spectral axis
        num_vel_pix = cube.spectral_axis.size + max_pos_shift - max_neg_shift
        new_header['NAXIS3'] = num_vel_pix

        # Adjust CRPIX in header
        new_header['CRPIX3'] += max_pos_shift

        pad_size = (max_pos_shift, -max_neg_shift)

    else:
        pad_size = None

    # Adjust the header to have velocities centered at v0.
    new_header["CRVAL3"] = new_header["CRVAL3"] - v0.to(u.m / u.s).value

    if save_shifted:

        if is_mask:
            dtype = 'int16'
        else:
            dtype = cube[:, 0, 0].dtype

        create_huge_fits(save_name, new_header, dtype=dtype,
                         return_hdu=False, fill_nan=not is_mask,
                         verbose=verbose)

    if return_spectra:
        all_shifted_spectra = []
        out_posns = []

    n_chunks = len(xy_posns[0]) / chunk_size

    # Create chunks of spectra for read-out.
    for i, chunk in enumerate(get_channel_chunks(len(xy_posns[0]),
                                                 chunk_size)):

        log.info("On chunk {0} of {1}".format(i + 1, n_chunks))

        gen = [(y, x, cube.unmasked_data[:, y, x], shift, pad_edges, pad_size)
               for y, x, shift in
               zip(xy_posns[0][chunk], xy_posns[1][chunk], pix_shifts[chunk])]

        with _map_context(num_cores, verbose=verbose,
                          num_jobs=len(chunk)) as map:

            shifted_spectra = map(_spectrum_shifter, gen)

        if save_shifted:

            output_fits = fits.open(save_name, mode='update')
            log.info("Writing chunk to file")

            if verbose:
                iterat = ProgressBar(shifted_spectra)
            else:
                iterat = shifted_spectra

            for out in iterat:
                if is_mask:
                    spec = (out[0] > 0.5).astype(np.int)
                else:
                    spec = out[0]
                output_fits[0].data[:, out[1], out[2]] = spec

            output_fits.flush()
            output_fits.close()

        if return_spectra:
            all_shifted_spectra.extend([out[0] for out in shifted_spectra])
            out_posns.extend([out[1:] for out in shifted_spectra])

    if save_shifted:

        # output_fits.flush()
        # output_fits.close()

        # Append the beam table onto the output file.
        if isinstance(cube, VaryingResolutionSpectralCube):
            from spectral_cube.cube_utils import largest_beam
            output_fits = fits.open(save_name, mode='update')
            output_fits[0].header.update(largest_beam(cube.beams).to_header_keywords())
            output_fits.flush()
            output_fits.close()

    if return_spectra:
        if hasattr(cube, "beams"):
            beams = cube.beams
        else:
            beams = None

        all_shifted_oned_spectra = \
            [OneDSpectrum(shifted, unit=cube.unit,
                          wcs=cube[:, 0, 0].wcs,
                          meta=cube[:, 0, 0].meta, spectral_unit=vel_unit,
                          beams=beams) for shifted in all_shifted_spectra]

        return all_shifted_oned_spectra, out_posns
