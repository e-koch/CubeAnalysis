
from astropy.io import fits
from spectral_cube import SpectralCube, VaryingResolutionSpectralCube
from spectral_cube.cube_utils import largest_beam
import os
from astropy import log
import astropy.units as u
from astropy.convolution import Box1DKernel, convolve
from scipy import ndimage as nd
from astropy.wcs.utils import proj_plane_pixel_scales
import skimage.morphology as mo
import numpy as np
from itertools import groupby, chain
from operator import itemgetter
from astropy.stats import mad_std
from scipy.signal import medfilt
import matplotlib.pyplot as p

try:
    from signal_id import Noise
    SIGNAL_ID_INSTALL = True
except ImportError:
    SIGNAL_ID_INSTALL = False

from .io_utils import create_huge_fits
from .progressbar import ProgressBar


def pb_masking(cube_name, pb_file, pb_lim, output_folder, is_huge=True):
    '''
    Use the PB coverage map to mask the cleaned cube.
    '''

    cube = SpectralCube.read(cube_name)
    pbcov = fits.open(pb_file, mode='denywrite')[0]

    if (pb_lim <= 0) or (pb_lim >= 1):
        raise ValueError("pb_lim most be between 0 and 1.")

    # Assume that the pbcov is constant across channels when a cube is
    # given
    if len(pbcov.shape) == 3:
        pb_slice = (slice(0, 1), slice(None), slice(None))
    elif len(pbcov.shape) == 2:
        pb_slice = (slice(None), slice(None))
    else:
        raise ValueError("pb_file must be a 2D or 3D array.")

    pbcov_plane = pbcov.data[pb_slice].squeeze()
    if pbcov_plane.shape != cube.shape[1:]:
        # Try slicing down the pbcov to the minimal shape (cut-out empty
        # regions).
        pbcov_plane = pbcov_plane[nd.find_objects(pbcov_plane > 0)[0]]
        assert pbcov_plane.shape == cube.shape[1:]

    masked_cube = cube.with_mask(pbcov_plane > pb_lim)

    masked_name =  \
        "{0}.pbcov_gt_{1}_masked.fits".format(cube_name.rstrip(".fits"),
                                              pb_lim)

    if is_huge:
        # Set out the shape from the first couple of channels

        min_shape = masked_cube[:2].minimal_subcube().shape

        # Create the FITS file, then write out per plane
        new_header = masked_cube.header.copy()
        new_header['NAXIS2'] = min_shape[1]
        new_header['NAXIS1'] = min_shape[2]

        create_huge_fits(masked_name, new_header)

        # Get the slice needed
        spat_slice = nd.find_objects(pbcov_plane > pb_lim)[0]

        for chan in ProgressBar(range(cube.shape[0])):

            orig_cube = fits.open(cube_name, mode='denywrite')
            mask_cube_hdu = fits.open(masked_name, mode='update')

            mask_cube_hdu[0].data[chan] = orig_cube[0].data[chan][spat_slice]

            mask_cube_hdu.flush()
            mask_cube_hdu.close()
            orig_cube.close()

        orig_cube = fits.open(cube_name, mode='denywrite')
        if len(orig_cube) == 2:
            mask_cube_hdu = fits.open(masked_name, mode='update')
            mask_cube_hdu.append(orig_cube[1])
            mask_cube_hdu.flush()
            mask_cube_hdu.close()
        orig_cube.close()

    else:
        masked_cube = masked_cube.minimal_subcube()
        masked_cube.write(os.path.join(masked_name), overwrite=True)


def common_beam_convolve(cube_name, output_name, is_huge=False, chunk=10,
                         **kwargs):
    '''
    Convolve a VaryingResolutionSpectralCube to have a common beam size,
    defined by the largest channel beam.
    '''

    cube = SpectralCube.read(cube_name)

    if not isinstance(cube, VaryingResolutionSpectralCube):
        log.info("The cube already has a common beam size. "
                 "Skipping operation.")
        return

    com_beam = cube.beams.common_beam(**kwargs)

    spec_axis = np.arange(cube.shape[0], dtype=int)

    spec_axis_chunks = \
        np.array_split(spec_axis,
                       [chunk * i for i in
                        range(1, int(np.ceil(len(spec_axis) / chunk)))])

    if len(spec_axis_chunks[0]) == 0:
        spec_axis_chunks = spec_axis_chunks[1:]

    del cube

    for ii, spec_chunk in enumerate(spec_axis_chunks):

        log.info("On chunk {0} of {1}".format(ii + 1, len(spec_axis_chunks)))

        cube = SpectralCube.read(cube_name)

        planes = []

        for chan in ProgressBar(spec_chunk):

            planes.append(cube[chan].convolve_to(com_beam,
                                                 convolve=convolve).value)

        del cube

        hdu = fits.open(cube_name, mode='update')

        for chan, plane in zip(spec_chunk, planes):
            hdu[0].data[chan] = plane

        hdu.flush()
        hdu.close()

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

    # if is_huge:
    #     save_to_huge_fits(output_name, cube, overwrite=True)
    # else:
    #     cube.write(output_name, overwrite=True)


def signal_masking(cube_name, output_folder, method='ppv_connectivity',
                   save_cube=False, is_huge=False, **algorithm_kwargs):
    '''
    Run a signal masking algorithm and save the resulting mask
    '''

    # cube = SpectralCube.read(cube_name)

    mask_name = \
        "{}_source_mask.fits".format(cube_name.rstrip(".fits"))

    mask_name = os.path.join(output_folder, mask_name)

    if method == "ppv_connectivity":
        if is_huge:
            # Big cubes need to use the per-spectrum based approach
            ppv_connectivity_perspec_masking(cube_name, mask_name, **algorithm_kwargs)
        else:
            raise NotImplementedError("Need to update this routine.")

            # Small cubes can be done quicker by using cube-operations
            masked_cube, mask = \
                ppv_connectivity_masking(cube, **algorithm_kwargs)
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

        raise NotImplementedError("Need to update this routine.")

        masked_cube, mask = ppv_dilation_masking(cube, noise_map,
                                                 **algorithm_kwargs)
    else:
        raise ValueError("method must be 'spectral_spatial' or "
                         "'ppv_dilation'.")

    # TODO: Make option for plane-by-plane write out for really large cubes.

    # new_header = cube.header.copy()
    # new_header["BUNIT"] = ""
    # new_header["BITPIX"] = 8


    # if is_huge:
    #     save_to_huge_fits(save_name, mask.astype('>i2'), overwrite=True)
    # else:
    #     mask_hdu = fits.PrimaryHDU(mask.astype('>i2'), header=new_header)
    #     mask_hdu.writeto(save_name, overwrite=True)


def ppv_connectivity_masking(cube, mask_name, smooth_chans=31, min_chan=10,
                             peak_snr=5., min_snr=2,
                             edge_thresh=1, show_plots=False,
                             noise_map=None, verbose=False,
                             spatial_kernel='beam'):
    '''
    Create a robust signal mask by requiring spatial and spectral
    connectivity.
    '''

    pixscale = proj_plane_pixel_scales(cube.wcs)[0]

    # # Want to smooth the mask edges
    mask = cube.mask.include().copy()

    # consecutive channels to be real emission.
    num_chans = min_chan

    # Smooth the cube, then create a noise model
    if smooth_chans is not None:
        # Set smoothing parameters and # consecutive channels.
        smooth_chans = int(round_up_to_odd(smooth_chans))

        if isinstance(cube, VaryingResolutionSpectralCube):
            raise TypeError("The resolution of this cube varies."
                            "Convolve to a common "
                            "beam size before spectrally smoothing.")

        spec_kernel = Box1DKernel(smooth_chans)
        smooth_cube = cube.spectral_smooth(spec_kernel)
    else:
        smooth_cube = cube

    if noise_map is None:
        if not SIGNAL_ID_INSTALL:
            raise ImportError("signal-id needs to be installed for noise map"
                              " estimation.")
        log.info("No noise map given. Using Noise to estimate the spatial"
                 " noise.")
        noise = Noise(smooth_cube)
        noise.estimate_noise(spectral_flat=True)
        noise.get_scale_cube()

        snr = noise.snr.copy()

    else:
        # The noise map needs the same units as the cube
        if not hasattr(noise_map, 'unit'):
            raise TypeError("noise_map must be an astropy.units.Quantity.")
        elif not noise_map.unit.is_equivalent(cube.unit):
            raise u.UnitsError("noise_map ({0}) must have equivalent unit to "
                               "the cube ({1})".format(noise_map.unit,
                                                       cube.unit))

        try:
            snr = (cube.filled_data[:] / noise_map).to(u.dimensionless_unscaled).value
        except Exception as e:
            print(e)
            print("You may have to allow for huge cube operations.")

    snr[np.isnan(snr)] = 0.0

    posns = np.where(snr.max(axis=0) >= min_snr)

    # Blank the spectra for which none are above min_snr
    bad_pos = np.where(snr.max(axis=0) < min_snr)
    mask[:, bad_pos[0], bad_pos[1]] = False

    iter = zip(*posns)
    if verbose:
        iter = ProgressBar(iter)

    for i, j in iter:

        sequences = _get_mask_edges(snr[:, i, j], min_snr, peak_snr,
                                    edge_thresh, num_chans)

        if len(sequences) == 0:
            mask[:, i, j] = False
            continue

        bad_posns = \
            list(set(np.arange(cube.shape[0])) - set(list(chain(*sequences))))

        mask[:, i, j][bad_posns] = False

        if show_plots:
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
    if spatial_kernel is "beam":
        if hasattr(cube, 'beams'):
            kernel = cube.beams.common_beam().as_tophat_kernel(pixscale)
        elif hasattr(cube, 'beam'):
            kernel = cube.beam.as_tophat_kernel(pixscale)
        else:
            raise AttributeError("cube doesn't have 'beam' or 'beams'?")
        # kernel = Beam(major=0.75 * cube.beam.major,
        #               minor=0.75 * cube.beam.minor,
        #               pa=cube.beam.pa).as_tophat_kernel(pixscale)
        kernel_pix = (kernel.array > 0).sum()
    else:
        if isinstance(spatial_kernel, np.ndarray):
            kernel = spatial_kernel
            kernel_pix = (kernel > 0).sum()
        else:
            kernel = None

    if kernel is not None:

        mask = fits.open(mask_name)[0]

        if verbose:
            iter = ProgressBar(mask.shape[0])
        else:
            iter = range(mask.shape[0])

        for i in iter:

            mask_hdu = fits.open(mask_name, mode='update')

            mask = mask_hdu[0].data[i] > 0

            # Avoid edge effects in closing by padding by 1 in each axis
            mask_i = np.pad(mask, ((1, 1), (1, 1)), 'constant',
                            constant_values=False)

            mask_i = nd.binary_opening(mask_i, kernel)
            mask_i = nd.binary_closing(mask_i, kernel)
            mask_i = mo.remove_small_objects(mask_i, min_size=kernel_pix,
                                             connectivity=2)
            mask_i = mo.remove_small_holes(mask_i, area_threshold=kernel_pix,
                                           connectivity=2)

            # Remove padding
            mask[i] = mask[1:-1, 1:-1].astype(mask_hdu.astype(int))

            mask.flush()
            mask.close()

    # Each region must contain a point above the peak_snr
    labels, num = nd.label(mask, np.ones((3, 3, 3)))
    for n in range(1, num + 1):
        pts = np.where(labels == n)
        if np.nanmax(snr[pts]) < peak_snr:
            mask[pts] = False

    # masked_cube = cube.with_mask(mask)

    # return masked_cube, mask


def ppv_connectivity_perspec_masking(cube_name, mask_name, smooth_chans=31,
                                     min_chan=10,
                                     peak_snr=5., min_snr=2, edge_thresh=1,
                                     show_plots=False, pb_map_name=None,
                                     noise_spectrum=None,
                                     verbose=False, spatial_kernel='beam',
                                     chunk=200000):
    '''
    Uses the same approach as `~ppv_connectivity_masking`, but avoids doing
    any operations with the full cube. This will take longer, but will use
    far less memory.
    '''

    cube = SpectralCube.read(cube_name)

    # Load in the pb map. If it's a cube, take the first channel and assume
    # The coverage is constant in the spectral dimension
    if pb_map_name is not None:
        pb_hdu = fits.open(pb_map_name, mode='denywrite')[0]
        if len(pb_hdu.shape) == 3:
            pb_map = pb_hdu.data[0].copy()
        elif len(pb_hdu.shape) == 2:
            pb_map = pb_hdu.data.copy()
        else:
            raise ValueError("pb_map must have 2 or 3 dimensions.")

        del pb_hdu
    else:
        pb_map = np.ones(cube.shape[1:])

    new_header = cube.header.copy()
    new_header["BUNIT"] = ""
    new_header["BITPIX"] = 8

    create_huge_fits(mask_name, new_header, shape=cube.shape)

    # mask_hdu = fits.PrimaryHDU(mask.astype('>i2'), header=new_header)
    # mask_hdu.writeto(save_name, overwrite=True)

    pixscale = proj_plane_pixel_scales(cube.wcs)[0]

    # Set smoothing parameters and # consecutive channels.
    smooth_chans = int(round_up_to_odd(smooth_chans))

    # consecutive channels to be real emission.
    num_chans = min_chan

    # Look for places where there is the minimum number of channels

    # Want to smooth the mask edges
    summed_mask = np.zeros(cube.shape[1:], dtype=int)
    for chan in range(cube.shape[0]):
        summed_mask += cube.mask[chan].include()
    posns = np.where(summed_mask >= num_chans)

    # Save some memory
    del cube

    # Blank the spectra for which none are above min_snr
    # bad_pos = np.where(summed_mask < num_chans)
    # mask[:, bad_pos[0], bad_pos[1]] = False

    # Create chunks of the positions
    yposn_chunks = np.array_split(posns[0],
                                  [chunk * i for i in
                                   range(1,
                                         int(np.ceil(len(posns[0]) / chunk)))])

    xposn_chunks = np.array_split(posns[1],
                                  [chunk * i for i in
                                   range(1,
                                         int(np.ceil(len(posns[0]) / chunk)))])

    for k in range(len(yposn_chunks)):
        log.info("On {0} of {1}".format(k, len(yposn_chunks) + 1))

        y_chunk = yposn_chunks[k]
        x_chunk = xposn_chunks[k]

        if show_plots:
            cube = SpectralCube.read(cube_name)
            specs = [cube[0].data[:, i, j] for i, j in zip(y_chunk, x_chunk)]

        else:
            cube = fits.open(cube_name, mode='denywrite')

            specs = [cube[0].data[:, i, j] for i, j in zip(y_chunk, x_chunk)]
            pb_vals = [pb_map[i, j] for i, j in zip(y_chunk, x_chunk)]

            cube.close()
            del cube

        if verbose:
            iter = ProgressBar(len(y_chunk))
        else:
            iter = range(len(y_chunk))

        masks = []

        for ii in iter:

            spectrum = specs[ii]

            pb_val = pb_vals[ii]

            mask_spec = np.zeros_like(spectrum, dtype=bool)

            # Assume for now that the noise level is ~ constant across the
            # channels. This is fine for HI, but not for, e.g., CO(1-0).
            smoothed = medfilt(spectrum, smooth_chans)

            mad = sigma_rob(smoothed, thresh=min_snr, iterations=5)

            snr = spectrum / (mad / pb_val)

            if np.nanmax(snr) < peak_snr:
                masks.append(mask_spec)
                continue

            sequences = _get_mask_edges(snr, min_snr, peak_snr, edge_thresh,
                                        num_chans)

            if len(sequences) == 0:
                masks.append(mask_spec)
                continue
            else:
                good_posns = list(chain(*sequences))

                mask_spec[good_posns] = True

                # masks[ii] = mask_spec
                masks.append(mask_spec)

            if show_plots and mask_spec.any():
                p.subplot(121)
                p.plot(cube.spectral_axis.value, snr)
                min_val = cube.spectral_axis.value[np.where(mask_spec)[0][-1]]
                max_val = cube.spectral_axis.value[np.where(mask_spec)[0][0]]
                p.vlines(min_val, 0,
                         np.nanmax(snr))
                p.vlines(max_val, 0,
                         np.nanmax(snr))
                p.plot(cube.spectral_axis.value,
                       snr * mask_spec, 'bD')

                p.subplot(122)
                p.plot(cube.spectral_axis.value, spectrum, label='Cube')
                p.plot(cube.spectral_axis.value, smoothed,
                       label='Smooth Cube')
                p.axvline(min_val)
                p.axvline(max_val)
                p.plot(cube.spectral_axis.value,
                       smoothed * mask_spec, 'bD')
                p.draw()
                raw_input("Next spectrum?")
                p.clf()

        mask = fits.open(mask_name, mode='update')
        for i, j, mask_spec in zip(y_chunk, x_chunk, masks):
            mask[0].data[:, i, j] = mask_spec.astype(">i2")

        mask.flush()
        mask.close()

        del mask

    cube = SpectralCube.read(cube_name)

    # Now set the spatial connectivity requirements.
    if spatial_kernel is "beam":
        if hasattr(cube, 'beams'):
            kernel = cube.beams.largest_beam().as_tophat_kernel(pixscale)
        elif hasattr(cube, 'beam'):
            kernel = cube.beam.as_tophat_kernel(pixscale)
        else:
            raise AttributeError("cube doesn't have 'beam' or 'beams'?")
        kernel_pix = (kernel.array > 0).sum()
        kernel = kernel.array > 0

    else:
        if isinstance(spatial_kernel, np.ndarray):
            kernel = spatial_kernel
            kernel_pix = (kernel > 0).sum()
        else:
            kernel = None

    if kernel is not None:

        mask = fits.open(mask_name)[0]

        if verbose:
            iter = ProgressBar(mask.shape[0])
        else:
            iter = range(mask.shape[0])

        del mask

        for i in iter:

            mask_hdu = fits.open(mask_name, mode='update')

            mask = mask_hdu[0].data[i] > 0

            # Avoid edge effects in closing by padding by 1 in each axis
            mask_i = np.pad(mask, ((1, 1), (1, 1)), 'constant',
                            constant_values=False)

            mask_i = nd.binary_opening(mask_i, kernel)
            mask_i = nd.binary_closing(mask_i, kernel)
            mask_i = mo.remove_small_objects(mask_i, min_size=kernel_pix,
                                             connectivity=2)
            mask_i = mo.remove_small_holes(mask_i, area_threshold=kernel_pix,
                                           connectivity=2)

            # Remove padding
            mask_hdu[0].data[i] = mask_i[1:-1, 1:-1].astype(">i2")

            mask_hdu.flush()
            mask_hdu.close()

    # Each region must contain a point above the peak_snr
    # mask = mask_hdu[0].data > 0
    # labels, num = nd.label(mask, np.ones((3, 3, 3)))
    # for n in range(1, num + 1):
    #     pts = np.where(labels == n)
    #     if np.nanmax(snr[pts]) < peak_snr:
    #         mask[pts] = False

    # masked_cube = cube.with_mask(mask)

    # return masked_cube, mask


def _get_mask_edges(snr, min_snr, peak_snr, edge_thresh, num_chans,
                    min_chans=3):
    '''
    '''

    good_posns = np.where(snr > min_snr)[0]

    # Reject if the total is less than connectivity requirement
    if good_posns.size < num_chans:
        return []

    sequences = []
    for k, g in groupby(enumerate(good_posns), lambda i: i[0] - i[1]):
        sequences.append(list(map(itemgetter(1), g)))

    # Check length and peak. Require a minimum of 3 pixels above the noise
    # to grow from.
    sequences = [seq for seq in sequences if len(seq) >= min_chans and
                 np.nanmax(snr[seq]) >= peak_snr]

    # Continue if no good sequences found
    if len(sequences) == 0:
        # Return empty list
        return sequences

    # Now take each valid sequence and expand the edges until the smoothed
    # spectrum approaches zero.
    edges = [[seq[0], seq[-1]] for seq in sequences]
    for n, edge in enumerate(edges):
        # Lower side
        if n == 0:
            start_posn = edge[0]
            stop_posn = 0
        else:
            start_posn = edge[0] - edges[n - 1][-1]
            stop_posn = edges[n - 1][-1]

        for pt in np.arange(start_posn - 1, stop_posn, -1):
            # if smoothed[pt] <= mad * edge_thresh:
            if snr[pt] <= edge_thresh:
                break

            sequences[n].insert(0, pt)

        # Upper side
        start_posn = edge[1]
        if n == len(edges) - 1:
            stop_posn = snr.shape[0]
        else:
            stop_posn = edges[n + 1][0]

        for pt in np.arange(start_posn + 1, stop_posn, 1):
            # if smoothed[pt] <= mad * edge_thresh:
            if snr[pt] <= edge_thresh:
                break

            sequences[n].insert(0, pt)

    # Final check for the min peak level and ensure all meet the
    # spectral connectivity requirement
    sequences = [seq for seq in sequences if len(seq) >= num_chans and
                 np.nanmax(snr[seq]) >= peak_snr]

    for i in range(len(sequences)):
        sequences[i].sort()

    return sequences


def ppv_dilation_masking(cube, noise_map, min_sig=3, max_sig=5, min_pix=27,
                         verbose=False):
    '''
    Find connected regions above 3 sigma that contain a pixel at least above
    5 sigma, and contains some minimum number of pixels.
    '''

    if not hasattr(noise_map, "unit"):
        noise_map = noise_map.copy() * cube.unit

    mask_low = (cube > min_sig * noise_map).include()
    mask_high = (cube > max_sig * noise_map).include()

    mask_low = mo.remove_small_objects(mask_low, min_size=min_pix,
                                       connectivity=2)

    # Remove all regions that do not contain a 5 sigma pixel
    kernel = np.ones((3, 3, 3))
    labels, num = nd.label(mask_low, kernel)

    iter = range(1, num + 1)
    if verbose:
        iter = ProgressBar(iter)

    for i in iter:
        pix = np.where(labels == i)
        if np.any(mask_high[pix]):
            continue
        mask_low[pix] = False

    return cube.with_mask(mask_low), mask_low


def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1


def sigma_rob(data, iterations=1, thresh=3.0, axis=None):
    """
    Iterative m.a.d. based sigma with positive outlier rejection.
    """
    noise = mad_std(data, axis=axis)
    for _ in range(iterations):
        ind = (np.abs(data) <= thresh * noise).nonzero()
        noise = mad_std(data[ind], axis=axis)
    return noise
