
from spectral_cube import SpectralCube
from spectral_cube.cube_utils import largest_beam
from astropy.utils.console import ProgressBar
from astropy import log
from astropy.io import fits
import os
from itertools import repeat
import numpy as np

from .io_utils import create_huge_fits, save_to_huge_fits


def reproject_cube(cubename, targ_cubename, output_cubename,
                   output_folder="", reproject_type='all',
                   common_beam=False, save_spectral=True,
                   is_huge=True, chunk=100, verbose=True):
    '''
    Reproject one cube to match another.
    '''

    allowed_types = ['all', 'spatial']

    if reproject_type not in allowed_types:
        raise TypeError("reproject_type must be 'all' or 'spatial'.")

    # Load the non-pb masked cube
    targ_cube = SpectralCube.read(targ_cubename)

    cube = SpectralCube.read(cubename)

    # Before doing the time-consuming stuff, make sure there are beams
    if common_beam:
        if hasattr(targ_cube, 'beams'):
            if reproject_type == 'all':
                beams = targ_cube.beams
            else:
                beams = repeat(largest_beam(targ_cube.beams))
        elif hasattr(targ_cube, 'beam'):
            beams = repeat(targ_cube.beam)
        else:
            raise AttributeError("The target cube does not have an associated "
                                 "beam. `common_beam` requires a beam object "
                                 "for both cubes.")

        if not hasattr(cube, 'beam'):
            raise AttributeError("The cube does not have an associated "
                                 "beam. `common_beam` requires a beam object "
                                 "for both cubes.")
    else:
        beams = repeat(None)

    # Spectrally interpolate
    if reproject_type == 'all':
        if verbose:
            log.info("Spectral interpolation")

        cube = cube.spectral_interpolate(targ_cube.spectral_axis)

        # Make sure the spectral axes are the same (and not reversed).
        if not np.allclose(cube.spectral_axis.value,
                           targ_cube.spectral_axis.value):
            raise Warning("The spectral axes do not match.")

        # Write out the spectrally interpolated cube
        if save_spectral:
            if verbose:
                log.info("Saving the spectrally interpolated cube.")
            spec_savename = \
                "{}_spectralregrid.fits".format(os.path.splitext(output_cubename)[0])
            spec_savename = os.path.join(output_folder, spec_savename)
            if is_huge:
                save_to_huge_fits(spec_savename, cube, verbose=verbose,
                                  overwrite=False)
            else:
                cube.write(spec_savename)

    # Make the reprojected header
    new_header = cube._nowcs_header.copy()

    if reproject_type == 'all':
        new_header.update(targ_cube.wcs.to_header())
    else:
        new_header.update(cube[:, :1, :1].wcs.to_header())
        new_header.update(targ_cube.wcs.celestial.to_header())

    new_header["NAXIS"] = 3
    new_header["NAXIS1"] = targ_cube.shape[2]
    new_header["NAXIS2"] = targ_cube.shape[1]
    if reproject_type == 'all':
        new_header["NAXIS3"] = targ_cube.shape[0]
    else:
        new_header['NAXIS3'] = cube.shape[0]
    kwarg_skip = ["OBSGEO-X", "OBSGEO-Y", "OBSGEO-Z", "RESTFRQ"]
    for key in kwarg_skip:
        if key in new_header:
            del new_header[key]

    # If there is one beam for all channels, attach to the header
    if common_beam and isinstance(beams, repeat):
            new_header.update(beams.next().to_header_keywords())

    # Build up the reprojected cube per channel
    save_name = os.path.join(output_folder, output_cubename)

    if verbose:
        log.info("Creating new FITS file.")
    output_fits = create_huge_fits(save_name, new_header, dtype=None,
                                   return_hdu=True)

    targ_header = targ_cube[0].header
    targ_dtype = targ_cube[:1, 0, 0].dtype

    chan_iter = zip(range(cube.shape[0]), beams)
    if verbose:
        log.info("Reprojecting and writing.")
        chan_iter = ProgressBar(chan_iter)

    for chan, beam in chan_iter:
        proj = cube[chan]

        if common_beam:
            proj = proj.convolve_to(beam)

        proj = proj.reproject(targ_header).value.astype(targ_dtype)

        output_fits[0].data[chan] = proj
        if chan % chunk == 0:
            output_fits.flush()
    output_fits.close()

    # If there was a table of beams, be sure to append this.
    if common_beam and isinstance(beams, list):
            if verbose:
                log.info("Appending beam table to FITS file.")
            from spectral_cube.cube_utils import beams_to_bintable
            output_fits = fits.open(save_name, mode='append')
            output_fits.append(beams_to_bintable(beams))
            output_fits.flush()
            output_fits.close()
