
from astropy.io import fits
import numpy as np
from astropy import log
from astropy.utils.console import ProgressBar
import os


def create_huge_fits(filename, header, shape=None, verbose=True,
                     return_hdu=False, dtype=None, fill_nan=False):
    '''
    Create empty FITS files too large to fit into memory.
    '''

    if os.path.exists(filename):
        raise IOError("{} already exists.".format(filename))

    # Add shape info to the header
    if "NAXIS" not in header and shape is None:
        raise TypeError("shape must be given when the header does not have "
                        "shape information ('NAXIS').")

    if shape is None:
        shape = (header["NAXIS3"], header["NAXIS2"], header["NAXIS1"])

    if dtype is None:
        dtype = fits.BITPIX2DTYPE[header['BITPIX']]
    else:
        # Make sure the header has the right BITPIX
        header['BITPIX'] = fits.DTYPE2BITPIX[dtype]

    # Iterate over the smallest axis
    min_axis = np.array(shape).argmin()
    plane_shape = [sh for i, sh in enumerate(shape) if i != min_axis]
    fill_plane = np.zeros(plane_shape, dtype=dtype)
    if fill_nan:
        fill_plane *= np.NaN

    if verbose:
        iterat = ProgressBar(shape[min_axis])
    else:
        iterat = xrange(shape[min_axis])

    output_fits = fits.StreamingHDU(filename, header)

    for chan in iterat:
        output_fits.write(fill_plane)

    output_fits.close()

    if return_hdu:
        output_fits = fits.open(filename, mode='update')
        return output_fits


def save_to_huge_fits(filename, cube, verbose=True, overwrite=False,
                      chunk=100):
    '''
    Save a huge SpectralCube by streaming the cube per channel.
    '''

    if os.path.exists(filename):
        if overwrite:
            raise IOError("{} already exists. Delete the file or enable "
                          "'overwrite'.".format(filename))
        output_fits = fits.open(filename, mode='update')
    else:
        output_fits = create_huge_fits(filename, cube.header, verbose=verbose,
                                       dtype=cube[:, 0, 0].dtype,
                                       return_hdu=True)

    for chan in xrange(cube.shape[0]):
        plane = cube[chan]
        if hasattr(plane, 'unit'):
            plane = plane.value

        output_fits[0][chan, :, :] = plane

        if chan % chunk == 0:
            output_fits.flush()

    output_fits.flush()
