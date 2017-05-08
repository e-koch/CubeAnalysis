
from astropy.io import fits
import numpy as np
from astropy import log
from astropy.utils.console import ProgressBar


def create_huge_fits(filename, header, shape=None, verbose=True,
                     return_hdu=False):
    '''
    Create empty FITS files too large to fit into memory.
    '''

    # Add shape info to the header
    if "NAXIS" not in header and shape is None:
        raise TypeError("shape must be given when the header does not have "
                        "shape information ('NAXIS').")
    output_fits = fits.StreamingHDU(filename, header)

    # Not covering all possible dtypes.
    if header["BITPIX"] == -64:
        dtype = np.float64
    if header["BITPIX"] == -32:
        dtype = np.float32
    else:
        log.info("Data type given in the header assumed to be a float.")
        dtype = np.float

    # Iterate over the smallest axis
    min_axis = np.array(shape).argmin()
    plane_shape = [sh for i, sh in enumerate(shape) if i != min_axis]
    fill_plane = np.zeros(plane_shape, dtype=dtype) * np.NaN

    if verbose:
        iterat = ProgressBar(shape[min_axis])
    else:
        iterat = xrange(shape[min_axis])

    for chan in iterat:
        output_fits.write(fill_plane)
        if verbose:
            iterat.update()

    output_fits.close()

    if return_hdu:
        output_fits = fits.open(filename, mode='update')
        return output_fits
