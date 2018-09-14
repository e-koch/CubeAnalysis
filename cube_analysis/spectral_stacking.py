
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from astropy import log

import sys
if sys.version_info < (3, 0):
    from itertools import izip as zip


from .progressbar import _map_context
from .feather_cubes import get_channel_chunks


def total_profile(cube, spatial_mask=None, how='slice'):
    '''
    Create the total profile over a region in a given spatial mask.
    '''

    if spatial_mask is not None:
        total_spec = cube.with_mask(spatial_mask).sum(axis=(1, 2), how=how)
    else:
        total_spec = cube.sum(axis=(1, 2), how=how)

    # Set NaNs to 0
    total_spec[np.isnan(total_spec)] = 0.0

    return total_spec


def radial_stacking(gal, cube, dr=100 * u.pc, max_radius=8 * u.kpc,
                    pa_bounds=None, num_cores=1, verbose=False, how='slice',
                    return_masks=False):
    '''
    Radially stack spectra.
    '''

    max_radius = max_radius.to(dr.unit)

    radius = gal.radius(header=cube.header)

    nbins = np.int(np.floor(max_radius / dr))
    inneredge = np.linspace(0, max_radius - dr, nbins)
    outeredge = np.linspace(dr, max_radius, nbins)

    valid_mask = cube.mask.include().sum(0) > 0

    if return_masks:
        masks = []

    if pa_bounds is not None:
        # Check if they are angles
        if len(pa_bounds) != 2:
            raise IndexError("pa_bounds must contain 2 angles.")
        if not isinstance(pa_bounds, Angle):
            raise TypeError("pa_bounds must be an Angle.")

        # Return the array of PAs in the galaxy frame
        pas = gal.position_angles(header=cube.header)

        # If the start angle is greater than the end, we need to wrap about
        # the discontinuity
        if pa_bounds[0] > pa_bounds[1]:
            initial_start = pa_bounds[0].copy()
            pa_bounds = pa_bounds.wrap_at(initial_start)
            pas = pas.wrap_at(initial_start)

        pa_mask = np.logical_and(pas >= pa_bounds[0],
                                 pas < pa_bounds[1])
    else:
        pa_mask = np.ones(cube.shape[1:], dtype=bool)

    stacked_spectra = np.zeros((inneredge.size, cube.shape[0])) * cube.unit
    num_pixels = np.zeros(inneredge.size)

    for ctr, (r0, r1) in enumerate(zip(inneredge,
                                       outeredge)):

        if verbose:
            log.info("On bin {} to {}".format(r0.value, r1))

        rad_mask = np.logical_and(radius >= r0, radius < r1)

        spec_mask = np.logical_and(rad_mask, pa_mask)

        # Now account for masking in the cube
        spec_mask = np.logical_and(spec_mask, valid_mask)

        if return_masks:
            masks.append(spec_mask)

        stacked_spectra[ctr] = \
            total_profile(cube, spec_mask,  # num_cores=num_cores,
                          how=how)

        num_pixels[ctr] = spec_mask.sum()

    bin_centers = (inneredge + dr / 2.).to(dr.unit)

    if return_masks:
        return bin_centers, stacked_spectra, num_pixels, masks

    return bin_centers, stacked_spectra, num_pixels


def percentile_stacking(cube, proj, dperc=5, num_cores=1, min_val=None,
                        max_val=None, verbose=False, how='slice',
                        return_masks=False):
    '''
    Stack spectra in a cube based on the values in a given 2D image. For
    example, give the peak temperature array to stack based on percentile of
    the peak temperature distribution.

    Parameters
    ----------
    cube : `~spectral_cube.SpectralCube`
        Cube to stack from.
    proj : `~spectral_cube.Projection` or `~spectral_cube.Slice`
        A 2D image whose values to determine the percentiles to stack to.
    dperc : float, optional
        Percentile width of the bins.
    num_cores : int, optional
        Give the number of cores to run the operation on.
    '''

    # If given a min and max, mask out those values
    if min_val is None:
        min_val = np.nanmin(proj)
    if max_val is None:
        max_val = np.nanmin(proj)

    vals_mask = np.logical_and(proj >= min_val, proj <= max_val)

    unit = proj.unit
    inneredge = np.nanpercentile(proj[vals_mask],
                                 np.arange(0, 101, dperc)[:-1]) * unit
    outeredge = np.nanpercentile(proj[vals_mask],
                                 np.arange(0, 101, dperc)[1:]) * unit
    # Add something small to the 100th percentile so it is used
    outeredge[-1] += 1e-3 * unit

    stacked_spectra = np.zeros((inneredge.size, cube.shape[0])) * cube.unit
    num_pixels = np.zeros(inneredge.size)

    if return_masks:
        masks = []

    for ctr, (p0, p1) in enumerate(zip(inneredge,
                                       outeredge)):

        if verbose:
            log.info("On bin {} to {} K".format(p0, p1))

        mask = np.logical_and(proj >= p0, proj < p1)

        if return_masks:
            masks.append(mask)

        stacked_spectra[ctr] = total_profile(cube, mask, how=how)
        num_pixels[ctr] = mask.sum()

    bin_centers = inneredge + dperc / 2.

    if return_masks:
        return bin_centers, stacked_spectra, num_pixels, masks

    return bin_centers, stacked_spectra, num_pixels
