
'''
Routines for feathering large cubes together using uvcombine.

Some of these *should* be moved into uvcombine in good time.
'''

from spectral_cube import SpectralCube
from uvcombine.uvcombine import feather_simple, feather_compare
from astropy import log
import astropy.units as u
from astropy.io import fits

import numpy as np


from .io_utils import create_huge_fits
from .progressbar import _map_context


def feather_cube(cube_hi, cube_lo, verbose=True, save_feather=True,
                 save_name=None, num_cores=1, chunk=100,
                 restfreq=1.42040575177 * u.GHz,
                 weights=None,
                 feather_kwargs={}):
    '''
    Feather two cubes together. The spectral axis of the cubes *must match*.
    This function is specifically for cube too large to fit in memory. The
    feathered planes are saved directly to a new FITS file.
    '''

    # Ensure the spectral axes are the same.
    if not np.allclose(cube_hi.spectral_axis.to(cube_lo._spectral_unit).value,
                       cube_lo.spectral_axis.value):
        raise Warning("The spectral axes do not match. Spectrally regrid the "
                      "cubes to be the same before feathering.")

    if save_feather:
        if save_name is None:
            raise TypeError("save_name must be given the save the shifted "
                            "cube.")

        create_huge_fits(save_name, cube_hi.header,
                         return_hdu=False)
    else:
        newcube = np.empty(cube_hi.shape, dtype=cube_hi[:1, 0, 0].dtype)

    num_chans = cube_hi.shape[0]
    chunked_channels = get_channel_chunks(num_chans, chunk)

    freq_axis = cube_hi.with_spectral_unit(u.Hz, velocity_convention='radio',
                                           rest_value=restfreq).spectral_axis

    if weights is not None:
        feather_kwargs['weights'] = weights

    for i, chunk_chans in enumerate(chunked_channels):

        log.info("On chunk {0} of {1}".format(i + 1, len(chunked_channels)))

        changen = ((chan, cube_hi[chan:chan + 1], cube_lo[chan:chan + 1],
                    freq_axis[chan], feather_kwargs) for chan in chunk_chans)

        with _map_context(num_cores, verbose=verbose,
                          num_jobs=len(chunk_chans)) as map:
            log.info("Feathering")
            output = map(_feather, changen)

        if save_feather:
            output_hdu = fits.open(save_name, mode='update')

        for chan, shifted_arr in output:
            if save_feather:
                output_hdu[0].data[chan] = shifted_arr.real
            else:
                newcube[chan] = shifted_arr.real

        if save_feather:
            output_hdu.flush()
            output_hdu.close()

    if save_feather:

        # Append the beam table, if needed
        if hasattr(cube_hi, 'beams'):
            from spectral_cube.cube_utils import beams_to_bintable
            output_hdu.append(beams_to_bintable(cube_hi.beams))

        output_hdu.close()
    else:
        return SpectralCube(newcube, wcs=cube_hi.wcs,
                            header=cube_hi.header, meta=cube_hi.meta)


def _feather(args):
    '''
    Feather 2D images together.
    '''
    chan, plane_hi, plane_lo, freq, kwargs = args

    # TODO: When Jy/beam is supported in SpectralCube, just let uvcombine
    # match the units. BUT we also need Slice objects to retain a 1 element
    # spectral axis!
    if plane_hi.unit != plane_lo.unit:
        if plane_hi.unit == u.K:
            if hasattr(plane_hi, 'beams'):
                beam = plane_hi.beams[0]
            else:
                beam = plane_hi.beam

            plane_hi = plane_hi.to(u.Jy, plane_hi.beam.jtok_equiv(freq)) / u.beam

        if plane_lo.unit == u.K:
            if hasattr(plane_lo, 'beams'):
                beam = plane_lo.beams[0]
            else:
                beam = plane_lo.beam

            plane_lo = plane_lo.to(u.Jy, beam.jtok_equiv(freq)) / u.beam

    if hasattr(plane_hi, 'hdu'):
        plane_hi_hdu = plane_hi.hdu
    else:
        plane_hi_hdu = plane_hi.hdulist[0]
        plane_hi_hdu.header.update(plane_hi.beams[0].to_header_keywords())

    if hasattr(plane_lo, 'hdu'):
        plane_lo_hdu = plane_lo.hdu
    else:
        plane_lo_hdu = plane_lo.hdulist[0]
        plane_lo_hdu.header.update(plane_lo.beams[0].to_header_keywords())

    feathered = feather_simple(plane_hi_hdu, plane_lo_hdu, **kwargs)

    # Expect that the interferometer image will cover a smaller region.
    # Cut to that region.
    feathered[np.isnan(plane_hi[0])] = np.NaN

    return chan, feathered


def feather_compare_cube(cube_hi, cube_lo, LAS, lowresfwhm=None,
                         restfreq=1.42040575177 * u.GHz,
                         verbose=True, num_cores=1, chunk=100,
                         weights=None):
    '''
    Record the ratios of the flux in the overlap region between the cubes.

    **Assumes velocity axes use the radio convention!**

    '''

    # Ensure the spectral axes are the same.
    if not np.allclose(cube_hi.spectral_axis.to(cube_lo._spectral_unit).value,
                       cube_lo.spectral_axis.value):
        raise Warning("The spectral axes do not match. Spectrally regrid the "
                      "cubes to be the same before feathering.")

    num_chans = cube_hi.shape[0]
    chunked_channels = get_channel_chunks(num_chans, chunk)

    radii = []
    ratios = []
    highres_pts = []
    lowres_pts = []

    if lowresfwhm is None:
        lowresfwhm = cube_lo.beam.major.to(u.arcsec)
    else:
        lowresfwhm = lowresfwhm.to(u.arcsec)

    freq_axis = cube_hi.with_spectral_unit(u.Hz, velocity_convention='radio',
                                           rest_value=restfreq).spectral_axis

    for i, chunk_chans in enumerate(chunked_channels):

        log.info("On chunk {0} of {1}".format(i + 1, len(chunked_channels)))

        changen = ((chan, cube_hi[chan], cube_lo[chan],
                    LAS, lowresfwhm, freq_axis[chan],
                    weights) for chan in chunk_chans)

        with _map_context(num_cores, verbose=verbose,
                          num_jobs=len(chunk_chans)) as map:
            output = map(_compare, changen)

        chan_out = np.array([out[0] for out in output])

        for jj in chan_out.argsort():

            radii.append(output[jj][1][0])
            ratios.append(output[jj][1][1])
            highres_pts.append(output[jj][1][2])
            lowres_pts.append(output[jj][1][3])

    return radii, ratios, highres_pts, lowres_pts


def _compare(args):

    chan, plane_hi, plane_lo, LAS, lowresfwhm, freq, weights = args

    out = feather_compare(plane_hi.to(u.K, freq=freq).hdu,
                          plane_lo.hdu,
                          return_samples=True, doplot=False,
                          LAS=LAS, SAS=lowresfwhm,
                          lowresfwhm=lowresfwhm,
                          weights=weights)

    return chan, out


def flux_recovery(cube_hi, cube_lo, frequency=1.42040575177 * u.GHz,
                  verbose=True, doplot=False,
                  num_cores=1, chunk=100, mask=None):
    '''
    Calculate the fraction of flux recovered between the high-resolution
    (interferometer) cube and the low-resolution (single-dish) cube.
    '''

    # Ensure the spectral axes are the same.
    if not np.allclose(cube_hi.spectral_axis.to(cube_lo._spectral_unit).value,
                       cube_lo.spectral_axis.value):
        raise Warning("The spectral axes do not match. Spectrally regrid the "
                      "cubes to be the same before feathering.")

    if mask is not None:
        if mask.shape != cube_hi[0].shape:
            raise ValueError("mask must have the same shape as the "
                             "high-resolution cube.")

    num_chans = cube_hi.shape[0]
    chunked_channels = get_channel_chunks(num_chans, chunk)

    total_hires = np.empty(num_chans) * u.Jy
    total_lores = np.empty(num_chans) * u.Jy

    for i, chunk_chans in enumerate(chunked_channels):

        log.info("On chunk {0} of {1}".format(i + 1, len(chunked_channels)))

        changen = ((chan, cube_hi[chan], cube_lo[chan],
                    frequency, mask) for chan in chunk_chans)

        with _map_context(num_cores, verbose=verbose)as map:
            output = map(_totalplanes, changen)

        chan_out = np.array([out[0] for out in output])

        for jj in chan_out.argsort():

            total_hires[chan_out[jj]] = output[jj][1]
            total_lores[chan_out[jj]] = output[jj][2]

    if doplot:
        import matplotlib.pyplot as plt
        plt.subplot(121)
        plt.plot(total_hires.value, label='High-res')
        plt.plot(total_lores.value, label='Low-res')
        plt.ylabel("Total Intensity (Jy)")
        plt.legend()
        plt.subplot(122)
        plt.plot(total_hires.value / total_lores.value)
        plt.ylabel("Fraction of recovered")
        plt.xlabel("Channel")

    return total_hires, total_lores


def _totalplanes(args):

    chan, plane_hi, plane_lo, freq, mask = args

    if mask is None:
        mask = (slice(None),) * 2

    if not plane_hi.wcs.wcs.compare(plane_lo.wcs.wcs):
        # Reproject the low resolution onto the high

        plane_lo = plane_lo.reproject(plane_hi.header)

    # !! Update to Jy/bm (or equiv) when spectral-cube can handle it
    if not plane_hi.unit.is_equivalent(u.Jy / u.beam):
        plane_hi = plane_hi.to(u.Jy, plane_hi.beam.jtok_equiv(freq))

    if not plane_lo.unit.is_equivalent(u.Jy / u.beam):
        plane_lo = plane_lo.to(u.Jy, plane_lo.beam.jtok_equiv(freq))

    total_hi = np.nansum(plane_hi[mask].value) * u.Jy
    total_lo = np.nansum(plane_lo[mask].value) * u.Jy

    # Convert from Jy/beam to actual Jy
    total_hi *= (1 / plane_hi.beam.sr.to(u.deg**2)) * \
        (plane_hi.header["CDELT2"] * u.deg)**2
    total_lo *= (1 / plane_lo.beam.sr.to(u.deg**2)) * \
        (plane_lo.header["CDELT2"] * u.deg)**2

    return chan, total_hi, total_lo


def get_channel_chunks(num_chans, chunk):
    '''
    Parameters
    ----------
    num_chans : int
        Number of channels
    chunk : int
        Size of chunks

    Returns
    -------
    chunked_channels : list of np.ndarray
        List of channels in chunks of the given size.
    '''
    channels = np.arange(num_chans)
    chunked_channels = \
        np.array_split(channels,
                       [chunk * i for i in range(num_chans // chunk)])
    if chunked_channels[-1].size == 0:
        chunked_channels = chunked_channels[:-1]
    if chunked_channels[0].size == 0:
        chunked_channels = chunked_channels[1:]

    return chunked_channels
