
'''
Spatially deproject a cube
'''

import numpy as np
import scipy.ndimage as nd
import astropy.units as u
from astropy.io import fits
from warnings import warn
from astropy import log
from galaxies import Galaxy

from .feather_cubes import get_channel_chunks
from .progressbar import _map_context
from .io_utils import create_huge_fits


def deproject(image, header, gal, conv_circ_beam=False, inc_correction=True):
    '''
    Calculate the deprojection angles for the given image and reflect
    the changes in the header. Optionally smooth prior to the deprojection
    so the final image has a circular beam in the object's frame.
    '''

    if not isinstance(gal, Galaxy):
        raise TypeError("gal must be a Galaxy class.")

    inc = gal.inclination
    pa = gal.position_angle

    image_copy = image.copy()
    mask = np.isfinite(image)
    image_copy[~mask] = 0.0

    # First rotate to have the PA at 0 along the y axis.
    rot = nd.rotate(image_copy, pa.to(u.deg).value - 180)
    rot_mask = nd.rotate(mask.astype(float), pa.to(u.deg).value - 180)
    # Now scale the x axis to correct for inclination
    deproj = nd.zoom(rot, (1., 1. / np.cos(inc).value))
    deproj_mask = nd.zoom(rot_mask, (1., 1. / np.cos(inc).value))

    # Correct values by cos(inc)
    if inc_correction:
        deproj = deproj * np.cos(inc)

    deproj[deproj_mask < 1e-5] = np.NaN

    return deproj


def _deproject(args):

    chan, proj, gal = args

    return chan, deproject(proj.value, proj.header, gal)


def deproject_cube(cube, gal, save_name=None, num_cores=1,
                   chunk=50):
    '''
    Separately deproject each channel in a cube.
    '''

    save_cube = True if save_name is not None else False

    if not isinstance(gal, Galaxy):
        raise TypeError("gal must be a Galaxy class.")

    num_chans = cube.shape[0]
    chunked_channels = get_channel_chunks(num_chans, chunk)

    for i, chunk_chans in enumerate(chunked_channels):

        log.info("On chunk {0} of {1}".format(i + 1, len(chunked_channels)))

        changen = ((chan, cube[chan], gal) for chan in chunk_chans)

        with _map_context(num_cores, verbose=True,
                          num_jobs=len(chunk_chans)) as map:
            output = map(_deproject, changen)

        for j, (chan, dep_arr) in enumerate(output):

            dep_arr = dep_arr[nd.find_objects(np.isfinite(dep_arr))[0]]

            if i == 0 and j == 0:
                dep_shape = (cube.shape[0],) + dep_arr.shape
                if save_cube:
                    hdr = cube.header.copy()

                    hdr['NAXIS3'] = dep_shape[0]
                    hdr['NAXIS2'] = dep_shape[1]
                    hdr['NAXIS1'] = dep_shape[2]

                    output_hdu = create_huge_fits(save_name, hdr,
                                                  return_hdu=True)
                else:
                    dep_cube = np.empty(dep_shape)

            if save_cube:
                output_hdu[0].data[chan] = dep_arr
            else:
                dep_cube[chan] = dep_arr

        if save_cube:
            output_hdu.flush()

    warn("Assigning the original header to the deprojected cube. CHECK "
         "CAREFULLY")

    if save_cube:
        output_hdu.close()
    else:
        return fits.PrimaryHDU(dep_cube, cube.header)
