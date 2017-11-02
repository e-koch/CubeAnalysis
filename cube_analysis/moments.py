
from spectral_cube import SpectralCube
from spectral_cube.lower_dimensional_structures import Projection
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy import log
from astropy.convolution import Gaussian1DKernel
from itertools import izip
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


def find_peakvelocity(cube, source_mask=None, chunk_size=1e4, smooth_size=None,
                      in_memory=False, num_cores=1, verbose=False):
    '''
    Calculate the peak velocity surface of a spectral cube
    '''

    peakvels = Projection(np.zeros(cube.shape[1:]) * np.NaN,
                          wcs=cube.wcs.celestial,
                          unit=cube.spectral_axis.unit)

    if source_mask is not None:
        posns = np.where(source_mask.sum(0) > 0)
    else:
        posns = np.indices(cube[0].shape)

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

        if in_memory:
            gener = [(cube[:, y, x], kern)
                     for y, x in izip(y_posn, x_posn)]
        else:
            gener = ((cube[:, y, x], kern)
                     for y, x in izip(y_posn, x_posn))

        with _map_context(num_cores, verbose=verbose) as map:
            output = map(_peak_velocity, gener)

        for out, y, x in izip(output, y_posn, x_posn):
            peakvels[y, x] = out

    # peakvels[peakvels == 0.0 * u.m / u.s] = np.NaN * u.m / u.s
    # Make sure there are no garbage points outside of the cube spectral range
    peakvels[peakvels < cube.spectral_extrema[0]] = np.NaN * u.m / u.s
    peakvels[peakvels > cube.spectral_extrema[1]] = np.NaN * u.m / u.s

    return peakvels


def make_moments(cube_name, mask_name, output_folder, freq=None,
                 num_cores=1, verbose=False, chunk_size=1e4,
                 in_memory=False, smooth_size=None):
    '''
    Create the moment arrays.
    '''

    cube = SpectralCube.read(cube_name)

    # Load in source mask
    source_mask = fits.getdata(mask_name)
    source_mask = source_mask.astype(np.bool)

    cube = cube.with_mask(source_mask)

    # Now create the moment 1 and save it. Make a linewidth one too.

    moment0 = cube.moment0()
    moment0_name = "{}.mom0.fits".format(cube_name.rstrip(".fits"))
    moment0.write(os.path.join(output_folder, moment0_name),
                  overwrite=True)

    moment1 = cube.moment1().astype(np.float32)
    moment1[moment1 < cube.spectral_extrema[0]] = np.NaN * u.m / u.s
    moment1[moment1 > cube.spectral_extrema[1]] = np.NaN * u.m / u.s

    moment1_name = "{}.mom1.fits".format(cube_name.rstrip(".fits"))
    moment1.header["BITPIX"] = -32
    moment1.write(os.path.join(moment1_name),
                  overwrite=True)

    linewidth = cube.linewidth_sigma()
    lwidth_name = "{}.lwidth.fits".format(cube_name.rstrip(".fits"))
    linewidth.write(os.path.join(lwidth_name),
                    overwrite=True)

    # Higher moments
    # Not that +/- chan size now matters for skewness. But spectral_cube only
    # takes the abs value of the channel size. Check whether the spectral axis
    # is increasing or decreasing
    if np.diff(cube.spectral_axis[:2].value)[0] < 0:
        mult_fact = -1.
    else:
        mult_fact = 1.

    # Skewness
    mom3 = cube.moment(order=3, axis=0)

    # Normalize third moment by the linewidth to get the skewness
    skew = mult_fact * mom3 / linewidth ** 3
    skew_name = "{}.skewness.fits".format(cube_name.rstrip(".fits"))
    skew.write(os.path.join(skew_name),
               overwrite=True)

    # Kurtosis: Uncorrected
    mom4 = cube.moment(order=4, axis=0)
    # Normalize third moment by the linewidth to get the skewness
    # And subtract 3 to correct for Gaussian kurtosis of 3.
    kurt = (mom4 / linewidth ** 4) - 3
    kurt_name = "{}.kurtosis.fits".format(cube_name.rstrip(".fits"))
    kurt.write(kurt_name,
               overwrite=True)

    # Peak temperature map. And convert to K
    maxima = cube.max(axis=0)
    if freq is not None:
        if hasattr(cube, 'beams'):
            peak_temps = maxima * cube.beams.largest_beam().jtok(freq)
        elif hasattr(cube, 'beam'):
            peak_temps = maxima * cube.beam.jtok(freq)
        else:
            log.info("No beam object found. Cannot convert to K.")
    else:
        peak_temps = maxima

    peaktemps_name = "{}.peaktemps.fits".format(cube_name.rstrip(".fits"))
    peak_temps.write(peaktemps_name, overwrite=True)

    peakvels = find_peakvelocity(cube, source_mask, chunk_size=chunk_size,
                                 smooth_size=smooth_size, in_memory=in_memory,
                                 num_cores=num_cores, verbose=verbose)

    peakvels = peakvels.astype(np.float32)
    peakvels.header["BITPIX"] = -32
    peakvels_name = "{}.peakvels.fits".format(cube_name.rstrip(".fits"))
    peakvels.write(peakvels_name, overwrite=True)


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
