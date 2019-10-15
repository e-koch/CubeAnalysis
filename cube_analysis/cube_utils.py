
from spectral_cube import SpectralCube, VaryingResolutionSpectralCube
from astropy.io import fits
import astropy.units as u
import os

from .io_utils import create_huge_fits
from .progressbar import ProgressBar


def convert_K(cube_name, output_folder, is_huge=True, verbose=False):
    '''
    Convert a larger-than-memory cube from Jy/beam to K
    '''

    # Load the header from the file

    hdr = fits.getheader(cube_name)

    # Only need to change BUNIT
    new_hdr = hdr.copy()
    new_hdr['BUNIT'] = 'K'

    spec_shape = hdr['NAXIS3']

    # Append K to the cube name
    cube_K_name = os.path.join(output_folder,
                               f"{cube_name.rstrip('.fits')}_K.fits")

    if is_huge:

        create_huge_fits(cube_K_name, new_hdr, verbose=verbose)

        if verbose:
            pbar = ProgressBar(spec_shape)

        for chan in range(spec_shape):

            cube = SpectralCube.read(cube_name, mode='denywrite')
            cube_K_hdu = fits.open(cube_K_name, mode='update')

            cube_K_hdu[0].data[chan] = cube[chan:chan + 1].to(u.K).unitless_filled_data[:]

            cube_K_hdu.flush()
            cube_K_hdu.close()

            del cube

            if verbose:
                pbar.update(chan + 1)

        # Append a beams table.
        orig_cube = fits.open(cube_name, mode='denywrite')
        if len(orig_cube) == 2:
            cube_K_hdu = fits.open(cube_K_name, mode='update')
            cube_K_hdu.append(orig_cube[1])
            cube_K_hdu.flush()
            cube_K_hdu.close()
        orig_cube.close()

    else:
        cube = SpectralCube.read(cube_name)
        cube_K = cube.to(u.K)
        cube_K.write(cube_K_name)

