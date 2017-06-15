
from astropy.io import fits
from spectral_cube import SpectralCube, VaryingResolutionSpectralCube
from spectral_cube.cube_utils import largest_beam
from astropy.utils.console import ProgressBar
import os
import astropy.units as u
from astropy import log
from astropy.convolution import Box1DKernel
from signal_id import Noise
from scipy import ndimage as nd
from astropy.wcs.utils import proj_plane_pixel_scales
import skimage.morphology as mo
import numpy as np
from radio_beam import Beam
from itertools import groupby, chain
from operator import itemgetter
import matplotlib.pyplot as p

from .io_utils import save_to_huge_fits


def pb_masking(cube_name, pb_file, pb_lim, output_folder):
    '''
    Use the PB coverage map to mask the cleaned cube.
    '''

    cube = SpectralCube.read(cube_name)
    pbcov = fits.open(pb_file)[0]

    if (pb_lim <= 0) or (pb_lim >= 1):
        raise ValueError("pb_lim most be between 0 and 1.")

    masked_cube = cube.with_mask(pbcov.data > pb_lim)

    masked_cube = masked_cube.minimal_subcube()

    masked_name =  \
        "{0}.pbcov_gt_{1}_masked.fits".format(cube_name.rstrip(".fits"),
                                              pb_lim)

    # TODO: Make option for plane-by-plane write out for really large cubes.

    masked_cube.write(os.path.join(masked_name), overwrite=True)


def common_beam_convolve(cube_name, output_folder, is_huge=False):
    '''
    Convolve a VaryingResolutionSpectralCube to have a common beam size,
    defined by the largest channel beam.
    '''

    cube = SpectralCube.read(cube_name)

    if not isinstance(cube, VaryingResolutionSpectralCube):
        log.info("The cube already has a common beam size. "
                 "Skipping operation.")
        return

    cube = cube.convolve_to(largest_beam(cube.beams))

    # Remove path to the cube
    filename = os.path.split(cube_name)[1]
    save_name = os.path.join(output_folder, filename)

    if is_huge:
        save_to_huge_fits(save_name, cube, overwrite=True)
    else:
        cube.write(save_name, overwrite=True)


def signal_masking(cube_name, output_folder, method='ppv_connectivity',
                   save_cube=False, is_huge=False, **algorithm_kwargs):
    '''
    Run a signal masking algorithm and save the resulting mask
    '''

    cube = SpectralCube.read(cube_name)

    if method == "ppv_connectivity":
        masked_cube, mask = ppv_connectivity_masking(cube, **algorithm_kwargs)
    elif method == "ppv_dilation":
        if 'noise_map' not in algorithm_kwargs:
            raise ValueError("Must specify an RMS map as 'noise_map'.")
        else:
            noise_map = algorithm_kwargs['noise_map']
            algorithm_kwargs.pop('noise_map')

        if isinstance(noise_map, basestring):
            noise_map = fits.open(noise_map)[0].data
        elif isinstance(noise_map, np.ndarray):
            pass
        else:
            raise TypeError("noise_map must be a file name or an array. Found"
                            " type {}".format(type(noise_map)))

        masked_cube, mask = ppv_dilation_masking(cube, noise_map,
                                                 **algorithm_kwargs)
    else:
        raise ValueError("method must be 'spectral_spatial' or "
                         "'ppv_dilation'.")

    # TODO: Make option for plane-by-plane write out for really large cubes.

    new_header = cube.header.copy()
    new_header["BUNIT"] = ""
    new_header["BITPIX"] = 8

    mask_name = \
        "{}_source_mask.fits".format(cube_name.rstrip(".fits"))

    save_name = os.path.join(output_folder, mask_name)

    if is_huge:
        save_to_huge_fits(save_name, mask.astype('>i2'), overwrite=True)
    else:
        mask_hdu = fits.PrimaryHDU(mask.astype('>i2'), header=new_header)
        mask_hdu.writeto(save_name, overwrite=True)


def ppv_connectivity_masking(cube, smooth_chans=31, min_chan=10, peak_snr=5.,
                             min_snr=2, edge_thresh=1, verbose=False,
                             noise_map=None):
    '''
    Create a robust signal mask by requiring spatial and spectral
    connectivity.
    '''

    pixscale = proj_plane_pixel_scales(cube.wcs)[0]

    # # Want to smooth the mask edges
    mask = cube.mask.include().copy()

    # Set smoothing parameters and # consecutive channels.
    smooth_chans = int(round_up_to_odd(smooth_chans))

    # consecutive channels to be real emission.
    num_chans = min_chan

    # Smooth the cube, then create a noise model
    if smooth_chans is not None:
        if isinstance(cube, VaryingResolutionSpectralCube):
            raise TypeError("The resolution of this cube varies."
                            "Convolve to a common "
                            "beam size before spectrally smoothing.")

        spec_kernel = Box1DKernel(smooth_chans)
        smooth_cube = cube.spectral_smooth(spec_kernel)
    else:
        smooth_cube = cube

    if noise_map is None:
        log.info("No noise map given. Using Noise to estimate the spatial"
                 " noise.")
        noise = Noise(smooth_cube)
        noise.estimate_noise(spectral_flat=True)
        noise.get_scale_cube()

        snr = noise.snr.copy()

    else:
        try:
            snr = cube.filled_data[:] / noise_map
        except Exception as e:
            print(e)
            print("You may have to allow for huge cube operations.")

    snr[np.isnan(snr)] = 0.0

    posns = np.where(snr.max(axis=0) >= min_snr)

    # Blank the spectra for which none are above min_snr
    bad_pos = np.where(snr.max(axis=0) < min_snr)
    mask[:, bad_pos[0], bad_pos[1]] = False

    for i, j in ProgressBar(zip(*posns)):

        # Look for all pixels above min_snr
        good_posns = np.where(snr[:, i, j] > min_snr)[0]

        # Reject if the total is less than connectivity requirement
        if good_posns.size < num_chans:
            mask[:, i, j] = False
            continue

        # Find connected pixels
        sequences = []
        for k, g in groupby(enumerate(good_posns), lambda (i, x): i - x):
            sequences.append(map(itemgetter(1), g))

        # Check length and peak. Require a minimum of 3 pixels above the noise
        # to grow from.
        sequences = [seq for seq in sequences if len(seq) >= 3 and
                     np.nanmax(snr[:, i, j][seq]) >= peak_snr]

        # Continue if no good sequences found
        if len(sequences) == 0:
            mask[:, i, j] = False
            continue

        # Now take each valid sequence and expand the edges until the smoothed
        # spectrum approaches zero.
        edges = [[seq[0], seq[-1]] for seq in sequences]
        for n, edge in enumerate(edges):
            # Lower side
            if n == 0:
                start_posn = edge[0]
                stop_posn = 0
            else:
                start_posn = edge[0] - edges[n - 1][0]
                stop_posn = edges[n - 1][0]

            for pt in np.arange(start_posn, stop_posn, -1):
                # if smoothed[pt] <= mad * edge_thresh:
                if snr[:, i, j][pt] <= edge_thresh:
                    break

                sequences[n].insert(0, pt)

            # Upper side
            start_posn = edge[1]
            if n == len(edges) - 1:
                stop_posn = cube.shape[0]
            else:
                stop_posn = edges[n + 1][0]

            for pt in np.arange(start_posn, stop_posn, 1):
                # if smoothed[pt] <= mad * edge_thresh:
                if snr[:, i, j][pt] <= edge_thresh:
                    break

                sequences[n].insert(0, pt)

        # Final check for the min peak level and ensure all meet the
        # spectral connectivity requirement
        sequences = [seq for seq in sequences if len(seq) >= num_chans and
                     np.nanmax(snr[:, i, j][seq]) >= peak_snr]

        if len(sequences) == 0:
            mask[:, i, j] = False
            continue

        bad_posns = \
            list(set(np.arange(cube.shape[0])) - set(list(chain(*sequences))))

        mask[:, i, j][bad_posns] = False

        if verbose:
            p.subplot(121)
            p.plot(cube.spectral_axis.value, snr[:, i, j])
            min_val = cube.spectral_axis.value[np.where(mask[:, i, j])[0][-1]]
            max_val = cube.spectral_axis.value[np.where(mask[:, i, j])[0][0]]
            p.vlines(min_val, 0,
                     np.nanmax(snr[:, i, j]))
            p.vlines(max_val, 0,
                     np.nanmax(snr[:, i, j]))
            p.plot(cube.spectral_axis.value,
                   snr[:, i, j] * mask[:, i, j], 'bD')

            p.subplot(122)
            p.plot(cube.spectral_axis.value, cube[:, i, j], label='Cube')
            p.plot(cube.spectral_axis.value, smooth_cube[:, i, j],
                   label='Smooth Cube')
            p.axvline(min_val)
            p.axvline(max_val)
            p.plot(cube.spectral_axis.value,
                   smooth_cube[:, i, j] * mask[:, i, j], 'bD')
            p.draw()
            raw_input("Next spectrum?")
            p.clf()

    # initial_mask = mask.copy()

    # Now set the spatial connectivity requirements.

    kernel = cube.beam.as_tophat_kernel(pixscale)
    # kernel = Beam(major=0.75 * cube.beam.major, minor=0.75 * cube.beam.minor,
    #               pa=cube.beam.pa).as_tophat_kernel(pixscale)
    kernel_pix = (kernel.array > 0).sum()

    # Avoid edge effects in closing by padding by 1 in each axis
    mask = np.pad(mask, ((0, 0), (1, 1), (1, 1)), 'constant',
                  constant_values=False)

    for i in ProgressBar(mask.shape[0]):
        mask[i] = nd.binary_opening(mask[i], kernel)
        mask[i] = nd.binary_closing(mask[i], kernel)
        mask[i] = mo.remove_small_objects(mask[i], min_size=kernel_pix,
                                          connectivity=2)
        mask[i] = mo.remove_small_holes(mask[i], min_size=kernel_pix,
                                        connectivity=2)

    # Remove padding
    mask = mask[:, 1:-1, 1:-1]

    # Each region must contain a point above the peak_snr
    labels, num = nd.label(mask, np.ones((3, 3, 3)))
    for n in range(1, num + 1):
        pts = np.where(labels == n)
        if np.nanmax(snr[pts]) < peak_snr:
            mask[pts] = False

    masked_cube = cube.with_mask(mask)

    return masked_cube, mask


def ppv_dilation_masking(cube, noise_map, min_sig=3, max_sig=5, min_pix=27):
    '''
    Find connected regions above 3 sigma that contain a pixel at least above
    5 sigma, and contains some minimum number of pixels.
    '''

    mask_low = (cube > min_sig * noise_map * cube.unit).include()
    mask_high = (cube > max_sig * noise_map * cube.unit).include()

    mask_low = mo.remove_small_objects(mask_low, min_size=min_pix,
                                       connectivity=2)

    # Remove all regions that do not contain a 5 sigma pixel
    kernel = np.ones((3, 3, 3))
    labels, num = nd.label(mask_low, kernel)

    for i in xrange(1, num + 1):
        pix = np.where(labels == i)
        if np.any(mask_high[pix]):
            continue
        mask_low[pix] = False

    return cube.with_mask(mask_low), mask_low


def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1
