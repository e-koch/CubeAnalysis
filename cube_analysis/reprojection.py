
from spectral_cube import SpectralCube
from astropy.utils.console import ProgressBar
from astropy import logger
import os
from itertools import repeat
import numpy as np

from .io_utils import create_huge_fits, save_to_huge_fits


def reproject_cube(cubename, targ_cubename, output_cubename,
                   output_folder="",
                   common_beam=False, save_spectral=True,
                   is_huge=True, chunk=100):
    '''
    Reproject one cube to match another.
    '''
    # Load the non-pb masked cube
    targ_cube = SpectralCube.read(targ_cubename)

    cube = SpectralCube.read(cubename)

    # Before doing the time-consuming stuff, make sure there are beams
    if common_beam:
        if hasattr(targ_cube, 'beams'):
            beams = cube.beams
        elif hasattr(targ_cube, 'beam'):
            beams = repeat(cube.beam)
        else:
            raise AttributeError("The target cube does not have an associated "
                                 "beam. `common_beam` requires a beam object "
                                 "for both cubes.")

        if not hasattr(cube, 'beam'):
            raise AttributeError("The cube does not have an associated "
                                 "beam. `common_beam` requires a beam object "
                                 "for both cubes.")

    # Spectrally interpolate
    logger.info("Spectral interpolation")
    cube = cube.spectral_interpolate(targ_cube.spectral_axis)

    # Make sure the spectral axes are the same (and not reversed).
    if not np.allclose(cube.spectral_axis.value,
                       targ_cube.spectral_axis.value):
        raise Warning("The spectral axes do not match.")

    # Write out the spectrally interpolated cube
    if save_spectral:
        logger.info("Saving the spectrally interpolated cube.")
        spec_savename = \
            "{}_spectralregrid.fits".format(os.path.splitext(output_cubename)[0])
        spec_savename = os.path.join(output_folder, spec_savename)
        if is_huge:
            save_to_huge_fits(spec_savename, cube, verbose=True,
                              overwrite=False)
        else:
            cube.write(spec_savename)

    # Make the reprojected header
    new_header = cube.header.copy()
    new_header["NAXIS"] = 3
    new_header["NAXIS1"] = targ_cube.shape[2]
    new_header["NAXIS2"] = targ_cube.shape[1]
    new_header["NAXIS3"] = targ_cube.shape[0]
    kwarg_skip = ['TELESCOP', 'BUNIT', 'INSTRUME']
    for key in cube.header:
        if key == 'HISTORY' or key == 'COMMENT':
            continue
        if key in targ_cube.header:
            if "NAXIS" in key:
                continue
            if key in kwarg_skip:
                continue
            new_header[key] = targ_cube.header[key]
    new_header.update(cube.beam.to_header_keywords())
    new_header["BITPIX"] = targ_cube.header["BITPIX"]

    # Build up the reprojected cube per channel
    save_name = os.path.join(output_folder, output_cubename)

    logger.info("Creating new FITS file.")
    output_fits = create_huge_fits(save_name, new_header, dtype=None,
                                   return_hdu=True)

    targ_header = targ_cube[0].header
    targ_dtype = targ_cube[:1, 0, 0].dtype
    logger.info("Reprojecting and writing.")
    for chan, beam in zip(ProgressBar(cube.shape[0]), beams):
        proj = cube[chan].reproject(targ_header).value.astype(targ_dtype)
        if common_beam:
            proj = proj.convolve_to(beam)

        output_fits[0].data[chan] = proj
        if chan % chunk == 0:
            output_fits.flush()
    output_fits.close()
