
from spectral_cube import SpectralCube
from spectral_cube.lower_dimensional_structures import Projection
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy import log
from astropy.convolution import Gaussian1DKernel
from astropy.utils.console import ProgressBar
import os
import glob

from .feather_cubes import get_channel_chunks
from .progressbar import _map_context


def _peak_velocity(args):
    '''
    Return the velocity at the peak of a spectrum.
    '''

    spec, kern = args

    if kern is None:
        return spec.spectral_axis[np.argmax(spec.value)]
    else:
        smooth_spec = spec.spectral_smooth(kern)
        argmax = np.argmax(smooth_spec.value)

        return spec.spectral_axis[argmax]


def find_peakvelocity(cube_name, mask_name=None, chunk_size=1e4,
                      smooth_size=None,
                      # in_memory=False, num_cores=1,
                      spectral_slice=slice(None),
                      verbose=False):
    '''
    Calculate the peak velocity surface of a spectral cube
    '''

    # Open the cube to get some properties for the peak velocity
    # array
    cube_hdu = fits.open(cube_name, mode='denywrite')
    shape = cube_hdu[0].shape
    spat_wcs = WCS(cube_hdu[0].header).celestial
    vel_unit = u.Unit(cube_hdu[0].header['CUNIT3'])

    cube_hdu.close()
    del cube_hdu

    peakvels = Projection(np.zeros(shape[1:]) * np.NaN,
                          wcs=spat_wcs,
                          unit=vel_unit)

    # Now read in the source mask

    if mask_name is not None:
        source_mask = fits.getdata(mask_name)
        source_mask_spatial = source_mask.sum(0) > 0
        posns = np.where(source_mask_spatial)
    else:
        posns = np.indices(shape[1:])

    chunk_size = int(chunk_size)
    chunk_idx = get_channel_chunks(posns[0].size, chunk_size)

    if smooth_size is not None:
        kern = Gaussian1DKernel(smooth_size)
    else:
        kern = None

    for i, chunk in enumerate(chunk_idx):

        log.info("On chunk {0} of {1}".format(i + 1, len(chunk_idx)))

        y_posn = posns[0][chunk]
        x_posn = posns[1][chunk]

        if verbose:
            pbar = ProgressBar(y_posn.size)

        cube = SpectralCube.read(cube_name)
        if mask_name is not None:
            cube = cube.with_mask(source_mask)

        cube = cube[spectral_slice]

        for j, (y, x) in enumerate(zip(y_posn, x_posn)):

            peakvels[y, x] = _peak_velocity((cube[:, y, x], kern))

            if verbose:
                pbar.update(j + 1)

        del cube

        # if in_memory:
        #     gener = [(cube[:, y, x], kern)
        #              for y, x in zip(y_posn, x_posn)]
        # else:
        #     gener = ((cube[:, y, x], kern)
        #              for y, x in zip(y_posn, x_posn))

        # with _map_context(num_cores, verbose=verbose) as map:
        #     output = map(_peak_velocity, gener)

        # del gener

        # for out, y, x in zip(output, y_posn, x_posn):
        #     peakvels[y, x] = out

    # peakvels[peakvels == 0.0 * u.m / u.s] = np.NaN * u.m / u.s
    # Make sure there are no garbage points outside of the cube spectral range
    cube = SpectralCube.read(cube_name)[spectral_slice]

    peakvels[peakvels < cube.spectral_extrema[0]] = np.NaN * u.m / u.s
    peakvels[peakvels > cube.spectral_extrema[1]] = np.NaN * u.m / u.s

    del cube

    return peakvels


def find_peakvelocity_cube(cube, smooth_size=None,
                           pb_mask=None,
                           num_cores=1, verbose=False,
                           how='cube',
                           spectral_slice=slice(None)):
    '''
    Make peak velocity map with cube operations.
    '''

    if smooth_size is not None:
        kern = Gaussian1DKernel(smooth_size)
        parallel = True if num_cores > 1 else False
        smooth_cube = cube.spectral_smooth(kern,
                                           parallel=parallel,
                                           num_cores=num_cores)

    else:
        smooth_cube = cube

    argmax_plane = smooth_cube[spectral_slice].argmax(axis=0, how=how)

    peakvels = cube[spectral_slice].spectral_axis[argmax_plane]

    if pb_mask is not None:
        peakvels[~pb_mask] = np.NaN

    return peakvels


def make_moments(cube_name, mask_name, output_folder, freq=None,
                 num_cores=1, verbose=False, chunk_size=1e4,
                 in_memory=False, smooth_size=None,
                 how='slice', make_peakvels=True,
                 spectral_slice=slice(None)):
    '''
    Create the moment arrays.
    '''

    cube = SpectralCube.read(cube_name)

    # Load in source mask
    source_mask = fits.getdata(mask_name)
    source_mask = source_mask.astype(np.bool)

    cube = cube.with_mask(source_mask)

    # Now create the moment 1 and save it. Make a linewidth one too.

    cube_base_name = os.path.split(cube_name)[-1]

    log.info(f"Making moment 0 from cube {cube_base_name}")
    moment0 = cube.moment0[spectral_slice](how=how)
    moment0_name = "{}.mom0.fits".format(cube_base_name.rstrip(".fits"))
    moment0.write(os.path.join(output_folder, moment0_name),
                  overwrite=True)

    log.info(f"Making moment 1 from cube {cube_base_name}")
    moment1 = cube.moment1[spectral_slice](how=how).astype(np.float32)
    moment1[moment1 < cube.spectral_extrema[0]] = np.NaN * u.m / u.s
    moment1[moment1 > cube.spectral_extrema[1]] = np.NaN * u.m / u.s

    moment1_name = "{}.mom1.fits".format(cube_base_name.rstrip(".fits"))
    moment1.header["BITPIX"] = -32
    moment1.write(os.path.join(output_folder, moment1_name),
                  overwrite=True)

    log.info(f"Making line width from cube {cube_base_name}")
    linewidth = cube[spectral_slice].linewidth_sigma(how=how)
    lwidth_name = "{}.lwidth.fits".format(cube_base_name.rstrip(".fits"))
    linewidth.write(os.path.join(output_folder, lwidth_name),
                    overwrite=True)

    # Skewness
    log.info(f"Making skewness from cube {cube_base_name}")
    mom3 = cube[spectral_slice].moment(order=3, axis=0, how=how)

    # Normalize third moment by the linewidth to get the skewness
    skew = mom3 / linewidth ** 3
    skew_name = "{}.skewness.fits".format(cube_base_name.rstrip(".fits"))
    skew.write(os.path.join(output_folder, skew_name),
               overwrite=True)

    # Kurtosis: Uncorrected
    log.info(f"Making kurtosis from cube {cube_base_name}")
    mom4 = cube[spectral_slice].moment(order=4, axis=0, how=how)
    # Normalize third moment by the linewidth to get the skewness
    # And subtract 3 to correct for Gaussian kurtosis of 3.
    kurt = (mom4 / linewidth ** 4) - 3
    kurt_name = "{}.kurtosis.fits".format(cube_base_name.rstrip(".fits"))
    kurt.write(os.path.join(output_folder, kurt_name),
               overwrite=True)

    # Peak temperature map. And convert to K
    if in_memory:
        cube.allow_huge_operations = True

    log.info(f"Making peak temperature from cube {cube_base_name}")

    maxima = cube[spectral_slice].max(axis=0, how=how)
    if freq is not None:
        if not cube.unit.is_equivalent(u.K):
            if hasattr(cube, 'beams'):
                peak_temps = maxima * cube.beams.largest_beam().jtok(freq)
            elif hasattr(cube, 'beam'):
                peak_temps = maxima * cube.beam.jtok(freq)
            else:
                log.info("No beam object found. Cannot convert to K.")
    else:
        peak_temps = maxima

    peaktemps_name = "{}.peaktemps.fits".format(cube_base_name.rstrip(".fits"))
    peak_temps.write(os.path.join(output_folder, peaktemps_name),
                     overwrite=True)

    log.info(f"Making peak velocity from cube {cube_base_name}")
    if make_peakvels:
        if in_memory:
            peakvels = find_peakvelocity_cube(cube[spectral_slice],
                                              smooth_size=smooth_size,
                                              how=how, num_cores=num_cores,
                                              spectral_slice=spectral_slice)
        else:
            peakvels = find_peakvelocity(cube_name, mask_name,
                                         chunk_size=chunk_size,
                                         smooth_size=smooth_size,
                                         in_memory=in_memory,
                                         num_cores=num_cores,
                                         spectral_slice=spectral_slice,
                                         verbose=verbose)

        peakvels = peakvels.astype(np.float32)
        peakvels.header["BITPIX"] = -32
        peakvels_name = \
            "{}.peakvels.fits".format(cube_base_name.rstrip(".fits"))
        peakvels.write(os.path.join(output_folder, peakvels_name),
                       overwrite=True)


def find_moment_names(path):
    '''
    Given a path, make global variables of the moment names.
    '''

    search_dict = {"Moment0": "mom0", "Moment1": "mom1", "LWidth": "lwidth",
                   "Skewness": "skewness", "Kurtosis": "kurtosis",
                   "PeakTemp": "peaktemps", "PeakVels": "peakvels"}

    found_dict = {}

    for filename in glob.glob(os.path.join(path, "*.fits")):

        for key in search_dict:
            if search_dict[key] in filename:
                found_dict[key] = filename
                search_dict.pop(key)
                break

    return found_dict
