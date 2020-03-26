
from spectral_cube import SpectralCube, VaryingResolutionSpectralCube, OneDSpectrum
from astropy.io import fits
import astropy.units as u
import os
import numpy as np
from astropy import log
from tqdm import tqdm

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
                         spatial_chunk=(256, 256),
                         verbose=False):
    '''
    Interpolate to a new integer channel width set by `nchan`,
    splitting a cube into spatial chunks to avoid memory issues.

    .. warning:: Does not spectrally smooth.

    '''

    assert isinstance(nchan, int)

    cube = SpectralCube.read(cube_name)

    if hasattr(cube, 'beams'):
        com_beam = cube.beams.common_beam()
        has_beams = True
    else:
        com_beam = cube.beam
        has_beams = False

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
    new_hdr['CDELT3'] = newchan_width

    create_huge_fits(output_name, new_hdr, verbose=verbose)

    nchunk_shape = [int(np.ceil(shp / chk)) for shp, chk in
                    zip(spat_shape, spatial_chunk)]

    # nchunk = np.product(spat_shape) // chunk + 1
    nchunk = np.product(nchunk_shape)

    # for ii, (yy, xx) in enumerate(np.ndindex(spat_shape)):
    for ych, xch in tqdm(np.ndindex(tuple(nchunk_shape)),
                         ascii=True,
                         desc="Spec. interp. chunks",
                         total=nchunk):

        ymin = spatial_chunk[0] * ych
        ymax = min(spatial_chunk[0] * (ych + 1), spat_shape[0])

        xmin = spatial_chunk[1] * xch
        xmax = min(spatial_chunk[1] * (xch + 1), spat_shape[1])

        spat_slice = (slice(None),
                      slice(ymin, ymax),
                      slice(xmin, xmax))

        cube = SpectralCube.read(cube_name)

        subcube = cube[spat_shape]

        if has_beams:
            subcube = subcube.convolve_to(com_beam)

        subcube_interp = subcube.spectral_interpolate(spec_axis)

        hdu = fits.open(output_name, mode='update')

        hdu[0].data[spat_slice] = subcube_interp.unitless_filled_data[:]

        hdu.flush()
        hdu.close()

        del cube, subcube, subcube_interp

    if has_beams:
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

        del hdu
