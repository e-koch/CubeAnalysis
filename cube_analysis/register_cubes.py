
'''
Check for positional offsets between the VLA and SD datasets.
'''

from image_registration import register_images
from spectral_cube import Projection, SpectralCube
from uvcombine.uvcombine import match_flux_units

from astropy import log
import astropy.units as u
import numpy as np


from .progressbar import _map_context
from .io_utils import create_huge_fits
from .spectra_shifter import fourier_shift


def cube_registration(cube, target_cube, verbose=True, num_cores=1,
                      **kwargs):
    '''
    Use the DFT method in image_registration to find the offset for each
    plane. Returns the offset for each plane, which can be used to find the
    average offset between the cubes.
    '''

    # We're doing this per-channel, so both cubes need to have the same
    # spectral axis
    if not np.allclose(cube.spectral_axis.value,
                       target_cube.spectral_axis.to(cube.spectral_axis.unit).value):
        raise Warning("The spectral axes do not match. Spectrally regrid the "
                      "cubes to be the same before feathering.")

    cubegen = ((i, cube[i], target_cube[i]) for i in xrange(cube.shape[0]))

    with _map_context(num_cores, verbose=verbose,
                      num_jobs=cube.shape[0]) as map:

        offsets = [x for x in map(_register, cubegen)]

    channels = np.array([x[0] for x in offsets])
    offsets = np.array([x[1] for x in offsets])

    return offsets[channels]


def _register(args):
    '''
    Match units, convolve, and regrid, then register the channels.
    '''

    chan, plane, targ_plane = args

    # TODO: Should be using match_flux_units, but the Slices don't have
    # WCS spectral information anymore. That's an issue whenever the
    # units as Jy/beam.

    # header = plane.header.copy()
    # if header['BUNIT'] == "Jy":
    #     header.update({"BUNIT": "Jy/beam"})
    # targ_header = targ_plane.header.copy()
    # if targ_header['BUNIT'] == "Jy":
    #     targ_header.update({"BUNIT": "Jy/beam"})

    # plane_match = match_flux_units(plane.value, header,
    #                                targ_header)

    # Convert back to a projection for the next steps
    # new_plane = Projection(plane_match.value, unit=plane_match.unit,
    #                        wcs=plane.wcs, header=plane.header)

    # Hard-wired for my HI VLA to SD comparisons right now.

    hi_freq = 1.42040575177 * u.GHz

    if plane.unit == u.K and targ_plane.unit != u.K:
        targ_plane = targ_plane.to(u.K, targ_plane.beam.jtok_equiv(hi_freq))
    elif plane.unit != u.K and targ_plane.unit == u.K:
        plane = plane.to(u.K, plane.beam.jtok_equiv(hi_freq))

    # Convolve whichever has the larger beam
    if targ_plane.beam != plane.beam:
        try:
            plane = plane.convolve_to(targ_plane.beam)
        except ValueError:
            targ_plane = targ_plane.convolve_to(plane.beam)

    targ_plane = targ_plane.reproject(plane.header)

    # Now register
    # Assume that the image are reasonably close -- 20% of the smallest
    # spatial axis
    maxoff = min(targ_plane.shape) * 0.2
    # Upsample by the ratio between the image sizes.
    usfac = min(targ_plane.shape) / float(min(plane.shape))
    if usfac < 1:
        usfac = 1 / usfac
    # register_image return dx, dy. To be consistent throughout this package,
    # swap the output so it is dy, dx
    output = register_images(targ_plane.value, plane.value, zeromean=True,
                             nthreads=1, maxoff=maxoff, usfac=int(usfac))[::-1]

    return chan, output


def spatial_shift_cube(cube, dy, dx, verbose=True, save_shifted=True,
                       save_name=None, num_cores=1, chunk=100):
    '''
    Shift a SpectralCube by dy and dx in the spatial dimensions.
    '''

    if save_shifted:
        if save_name is None:
            raise TypeError("save_name must be given the save the shifted "
                            "cube.")

        output_hdu = create_huge_fits(save_name, cube.header, return_hdu=True)
    else:
        newcube = np.empty(cube.shape, dtype=cube[:1, 0, 0].dtype)
        newmask = np.empty(cube.shape, dtype=bool)

    channels = np.arange(cube.shape[0])
    chunked_channels = np.array_split(channels, [chunk])
    if chunked_channels[-1].size == 0:
        chunked_channels = chunked_channels[:-1]

    for i, chunk_chans in enumerate(chunked_channels):

        log.info("On chunk {0} of {1}".format(i + 1, len(chunked_channels)))

        changen = ((chan, cube[chan], dy, dx) for chan in chunk_chans)

        if not save_shifted:
            maskgen = ((chan,
                        cube.mask.include(view=(slice(chan, chan + 1))),
                        dy, dx) for chan in chunk_chans)

        with _map_context(num_cores, verbose=verbose)as map:
            log.info("Shifting array")
            output = map(_shifter, changen)

            if not save_shifted:
                log.info("Shifting mask")
                output_mask = map(_shifter, maskgen)

        for chan, shifted_arr in output:
            if save_shifted:
                output_hdu[0].data[chan] = shifted_arr
            else:
                newcube[chan] = shifted_arr

        if not save_shifted:
            for chan, shifted_mask in output_mask:
                newmask[chan] = shifted_mask > 0.5

        if save_shifted:
            output_hdu.flush()

    if save_shifted:
        output_hdu.close()
    else:
        return SpectralCube(newcube, wcs=cube.wcs,
                            header=cube.header, meta=cube.meta,
                            mask=cube.mask)


def _shifter(args):

    chan, arr, y_shift, x_shift = args

    if y_shift != 0:
        arr_shift = fourier_shift(arr, y_shift, axis=0)
    else:
        arr_shift = arr

    if x_shift != 0:
        arr_shift = fourier_shift(arr_shift, x_shift, axis=1)

    return chan, arr_shift
