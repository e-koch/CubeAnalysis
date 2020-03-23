
from spectral_cube import SpectralCube, VaryingResolutionSpectralCube
from astropy.io import fits
import astropy.units as u
import os
import numpy as np
from astropy import log

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


def spectral_interpolate(cube_name, output_name,
                         nchan,
                         chunk=10000,
                         verbose=False):
    '''
    Interpolate to a new integer channel width set by `nchan`.

    .. warning:: Does not spectrally smooth.

    '''

    assert isinstance(nchan, int)

    cube = SpectralCube.read(cube_name)

    if hasattr(cube, 'beams'):
        com_beam = cube.beams.common_beam()
    else:
        com_beam = cube.beam

    spat_shape = cube.shape[1:]

    vels = cube.spectral_axis.to(u.km / u.s)

    # The channels are already fairly correlated, so
    # not much point in spectral smoothing first.
    unit = cube.spectral_axis.unit
    newchan_width = nchan * (vels[1] - vels[0]).to(unit).value
    spec_axis = np.arange(cube.spectral_axis[0].value,
                          cube.spectral_axis[-1].value,
                          newchan_width) * unit
    assert spec_axis.size > 0

    del cube

    # Make the new empty cube
    hdr = fits.getheader(cube_name)

    # Only need to change BUNIT
    new_hdr = hdr.copy()
    new_hdr['NAXIS3'] = spec_axis.size
    new_hdr['CDELT3'] = newchan_width.value

    create_huge_fits(output_name, new_hdr, verbose=verbose)

    nchunk = np.product(spat_shape) // chunk + 1

    for ii, (yy, xx) in enumerate(np.ndindex(spat_shape)):

        if ii % chunk == 0:
            log.info("On chunk {0} of {1}".format(ii // chunk, nchunk))

        cube = SpectralCube.read(cube_name)

        spec = cube[:, yy, xx].spectral_interpolate(spec_axis)

        hdu = fits.open(output_name, mode='update')

        hdu[0].data[:, yy, xx] = spec.unitless_filled_data[:].value

        hdu.flush()
        hdu.close()

        del cube

    # Since we're overwriting this cube, we have to remove the
    # beams table and write the beam to the header
    hdu = fits.open(cube_name, mode='update')

    if len(hdu) == 2:
        del hdu[1]

    hdu[0].header.update(com_beam.to_header_keywords())

    if "CASAMBM" in hdu[0].header:
        del hdu[0].header['CASAMBM']

    hdu.flush()
    hdu.close()
