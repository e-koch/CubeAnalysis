
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from itertools import izip
from astropy import log

from .progressbar import _map_context
from .feather_cubes import get_channel_chunks


def total_profile(cube, spatial_mask=None, chunk_size=10000,
                  num_cores=1, verbose=False):
    '''
    Create the total profile over a region in a given spatial mask.
    '''

    if spatial_mask is None:
        posns = np.indices(cube[0].shape)
        posns = (posns[0].ravel(), posns[1].ravel())
    else:
        posns = np.where(spatial_mask)

    num_specs = posns[0].size
    chunksidx = get_channel_chunks(num_specs, chunk_size)

    cubelist = ([(cube.filled_data[:, jj, ii],
                 cube.mask.include(view=(slice(None), jj, ii)))
                for jj, ii in izip(posns[0][chunk], posns[1][chunk])]
                for chunk in chunksidx)

    with _map_context(num_cores, verbose=verbose,
                      num_jobs=len(chunksidx)) as map:

        stacked_spectra = \
            np.array([x for x in map(_masked_sum, cubelist)])

    # Sum each chunk together
    all_stacked = np.nansum(stacked_spectra, axis=0) * cube.unit

    return all_stacked


def _masked_sum(gen):
    '''
    Sum a list of spectra, applying their respective masks.
    '''

    for i, vals in enumerate(gen):

        spec, mask = vals

        if i == 0:
            total_stack = np.zeros_like(spec)

        total_stack[mask] += spec[mask]

    return total_stack


def radial_stacking(gal, cube, dr=100 * u.pc, max_radius=8 * u.kpc,
                    pa_bounds=None, num_cores=1, verbose=False):
    '''
    Radially stack spectra.
    '''

    max_radius = max_radius.to(dr.unit)

    radius = gal.radius(header=cube.header)

    nbins = np.int(np.floor(max_radius / dr))
    inneredge = np.linspace(0, max_radius - dr, nbins)
    outeredge = np.linspace(dr, max_radius, nbins)

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

    for ctr, (r0, r1) in enumerate(zip(inneredge,
                                       outeredge)):

        if verbose:
            log.info("On bin {} to {}".format(r0.value, r1))

        rad_mask = np.logical_and(radius >= r0, radius < r1)

        spec_mask = np.logical_and(rad_mask, pa_mask)

        stacked_spectra[ctr] = \
            total_profile(cube, spec_mask, num_cores=num_cores,
                          verbose=verbose)

    bin_centers = (inneredge + dr / 2.).to(dr.unit)

    return bin_centers, stacked_spectra


def percentile_stacking(cube, proj, dperc=5, num_cores=1, min_val=None,
                        max_val=None, verbose=False):
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

    for ctr, (p0, p1) in enumerate(zip(inneredge,
                                       outeredge)):

        if verbose:
            log.info("On bin {} to {} K".format(p0, p1))

        mask = np.logical_and(proj >= p0, proj < p1)

        stacked_spectra[ctr] = total_profile(cube, mask, num_cores=num_cores,
                                             verbose=verbose)

    bin_centers = inneredge + dperc / 2.

    return bin_centers, stacked_spectra
