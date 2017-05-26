
'''
Routines for feathering large cubes together using uvcombine.

Some of these *should* be moved into uvcombine in good time.
'''

from spectral_cube import SpectralCube
from uvcombine.uvcombine import feather_simple, feather_compare
from astropy import log
import astropy.units as u

import numpy as np


from .io_utils import create_huge_fits
from .progressbar import _map_context, ProgressBar


def feather_cube(cube_hi, cube_lo, verbose=True, save_feather=True,
                 save_name=None, num_cores=1, chunk=100,
                 frequency=1.42040575177 * u.GHz,
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

        output_hdu = create_huge_fits(save_name, cube_hi.header,
                                      return_hdu=True)
    else:
        newcube = np.empty(cube_hi.shape, dtype=cube_hi[:1, 0, 0].dtype)

    num_chans = cube_hi.shape[0]
    channels = np.arange(num_chans)
    chunked_channels = \
        np.array_split(channels,
                       [chunk * i for i in xrange(num_chans / chunk)])
    if chunked_channels[-1].size == 0:
        chunked_channels = chunked_channels[:-1]

    for i, chunk_chans in enumerate(chunked_channels):

        log.info("On chunk {0} of {1}".format(i + 1, len(chunked_channels)))

        changen = ((chan, cube_hi[chan:chan + 1], cube_lo[chan:chan + 1],
                    frequency, feather_kwargs) for chan in chunk_chans)

        with _map_context(num_cores, verbose=verbose)as map:
            log.info("Feathering")
            output = map(_feather, changen)

        for chan, shifted_arr in output:
            if save_feather:
                output_hdu[0].data[chan] = shifted_arr.real
            else:
                newcube[chan] = shifted_arr.real

        if save_feather:
            output_hdu.flush()

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
    if plane_hi.unit == u.K:
        if hasattr(plane_hi, 'beams'):
            beam = plane_hi.beams[0]
        else:
            beam = plane_hi.beam

        plane_hi = plane_hi.to(u.Jy, plane_hi.beam.jtok_equiv(freq))
    plane_hi._unit = u.Jy / u.beam

    if plane_lo.unit == u.K:
        if hasattr(plane_lo, 'beams'):
            beam = plane_lo.beams[0]
        else:
            beam = plane_lo.beam

        plane_lo = plane_lo.to(u.Jy, beam.jtok_equiv(freq))
    plane_lo._unit = u.Jy / u.beam

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


def feather_compare_cube(cube_hi, cube_lo, LAS, verbose=True,
                         num_cores=1, chunk=100):
    '''
    Record the ratios of the flux in the overlap region between the cubes.
    '''

    # Ensure the spectral axes are the same.
    if not np.allclose(cube_hi.spectral_axis.to(cube_lo._spectral_unit).value,
                       cube_lo.spectral_axis.value):
        raise Warning("The spectral axes do not match. Spectrally regrid the "
                      "cubes to be the same before feathering.")

    num_chans = cube_hi.shape[0]
    channels = np.arange(num_chans)
    chunked_channels = \
        np.array_split(channels,
                       [chunk * i for i in xrange(num_chans / chunk)])
    if chunked_channels[-1].size == 0:
        chunked_channels = chunked_channels[:-1]

    radii = []
    ratios = []

    for i, chunk_chans in enumerate(chunked_channels):

        log.info("On chunk {0} of {1}".format(i + 1, len(chunked_channels)))

        changen = ((chan, cube_hi[chan], cube_lo[chan],
                    LAS) for chan in chunk_chans)

        with _map_context(num_cores, verbose=verbose)as map:
            output = map(_compare, changen)

        chan_out = np.array([out[0] for out in output])

        for jj in chan_out.argsort():

            radii.append(output[jj][0])
            ratios.append(output[jj][1])

    return radii, ratios


def _compare(args):

    chan, plane_hi, plane_lo, LAS = args

    hi_beam = plane_hi.beam
    lo_beam = plane_lo.beam

    out = feather_compare(plane_hi.to(u.K, hi_beam.jtok_equiv(1.42040575177 * u.GHz)).hdu,
                          plane_lo.hdu,
                          return_ratios=True, doplot=False,
                          LAS=LAS, SAS=lo_beam.major.to(u.arcsec),
                          lowresfwhm=lo_beam.major.to(u.arcsec))

    return chan, out
