
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle


def radial_profile(gal, moment, header=None, dr=100 * u.pc,
                   max_rad=10 * u.kpc, pa_bounds=None, beam=None):
    '''
    Create a radial profile of a moment 2 array, optionally with limits on
    the angles used.

    Parameters
    ----------
    gal : `~galaxies.Galaxy`
        A `Galaxy` object: https://github.com/low-sky/galaxies
    moment : `~spectral_cube.Projection` or `~astropy.units.Quantity`
        A 2D image to create the radial profile from.
    header : FITS header, optional
        The FITS header to define the galaxy coordinates from. If `moment` is
        a `Quantity`, the header must be given. Otherwise, the header information
        from the `Projection` is used.
    dr : `~astropy.units.Quantity`, optional
        Set the bin width in physical distance units.
    max_rad : `~astropy.units.Quantity`, optional
        Maximum radius to bin the data to.
    pa_bounds : list of 2 Angles, optional
        When specified, limits the angles used when calculating the profile.
        e.g. pa_bounds=Angle([0.0*u.rad, np.pi*u.rad])).
    beam : `~radio_beam.Beam`, optional
        A `Beam` object for the moment array. If `moment` is a `Projection`,
        it will be checked for an attached beam. The beam defines the number
        of independent samples within the bin annulus. This affects the
        standard deviation values in each bin.

    Returns
    -------
    radprof : `~astropy.units.Quantity`
        The radial distance of the bin centers. The units will be whatever
        unit was specified for the bin widths `dr`.
    sdprof : `~astropy.units.Quantity`
        The binned values.
    sdprof_sigma : `~astropy.units.Quantity`
        The standard deviation within each bin. This is scaled by the beam
        size, when the beam information is available.

    '''

    if moment.ndim != 2:
        raise ValueError("mom0 must be 2 dimensional.")

    if header is None:
        header = moment.header

    if beam is None:
        # See if its in the projection
        if "beam" in moment.meta:
            beam = moment.meta["beam"]
        else:
            print("No beam attached to the Projection.")

    if beam is not None:
        beam_pix = beam.sr.to(u.deg**2) / (header["CDELT2"] * u.deg)**2

    radius = gal.radius(header=header).to(u.kpc).value
    if pa_bounds is not None:
        # Check if they are angles
        if len(pa_bounds) != 2:
            raise IndexError("pa_bounds must contain 2 angles.")
        if not isinstance(pa_bounds, Angle):
            raise TypeError("pa_bounds must be an Angle.")

        # Return the array of PAs in the galaxy frame
        pas = gal.position_angles(header=header)

        # If the start angle is greater than the end, we need to wrap about
        # the discontinuity
        if pa_bounds[0] > pa_bounds[1]:
            initial_start = pa_bounds[0].copy()
            pa_bounds = pa_bounds.wrap_at(initial_start)
            pas = pas.wrap_at(initial_start)

    if max_rad is not None:
        max_rad = max_rad.to(u.kpc).value
    else:
        max_rad = radius.max() - dr

    dr = dr.to(u.kpc).value

    nbins = np.int(np.floor(max_rad / dr))

    inneredge = np.linspace(0, max_rad, nbins)
    outeredge = np.linspace(0 + dr, max_rad + dr, nbins)
    sdprof = np.zeros(nbins)
    sdprof_sigma = np.zeros(nbins)
    radprof = np.zeros(nbins)

    if pa_bounds is not None:
        pa_idx = np.logical_and(pas >= pa_bounds[0], pas < pa_bounds[1])

    for ctr, (r0, r1) in enumerate(zip(inneredge,
                                       outeredge)):

        idx = np.logical_and(radius >= r0, radius < r1)
        if pa_bounds is not None:
            idx = np.logical_and(idx, pa_idx)

        radprof[ctr] = np.nanmean(radius[idx])

        sdprof[ctr] = np.nansum(moment[idx].value) / \
            np.sum(np.isfinite(radius[idx]))
        sdprof_sigma[ctr] = \
            np.sqrt(np.nansum((moment[idx].value - sdprof[ctr])**2.) /
                    np.sum(np.isfinite(radius[idx])))
        if beam is not None:
            sdprof_sigma[ctr] /= \
                np.sqrt(np.sum(np.isfinite(radius[idx])) / beam_pix)

    # Re-apply some units
    radprof = radprof * u.kpc
    sdprof = sdprof * moment.unit
    sdprof_sigma = sdprof_sigma * moment.unit

    return radprof, sdprof, sdprof_sigma


def surfdens_radial_profile(gal, header=None, cube=None,
                            dr=100 * u.pc, mom0=None,
                            max_rad=10 * u.kpc,
                            weight_type="area",
                            mass_conversion=None,
                            restfreq=1.414 * u.GHz,
                            pa_bounds=None, beam=None):
    '''
    Create a radial profile, optionally with limits on the angles used.

    Parameters
    ----------
    weight_type : "area" or "mass"
        Return either the area weighted profile (Sum Sigma * dA / Sum dA) or
        the mass weighted profile (Sum Sigma^2 dA / Sum Sigma dA). See
        Leroy et al. (2013) for a thorough description.
    pa_bounds : list of 2 Angles, optional
        When specified, limits the angles used when calculating the profile.
        e.g. pa_bounds=Angle([0.0*u.rad, np.pi*u.rad]))
    '''

    mom0 = mom0.squeeze()

    if weight_type not in ["area", "mass"]:
        raise ValueError("weight_type must be 'area' or 'mass'.")

    if mom0.ndim != 2:
        raise ValueError("mom0 must be 2 dimensional.")

    if mom0 is None:
        if cube is None:
            raise ValueError("Must give cube when not given mom0")
        mom0 = cube.moment0()

    if header is None:
        if cube is not None:
            header = cube.header
        elif mom0 is not None:
            header = mom0.header

    if beam is None:
        if cube is not None:
            beam = cube.beam
        elif mom0 is not None:
            beam = mom0.meta["beam"]

    if beam is not None:
        beam_pix = beam.sr.to(u.deg**2) / (header["CDELT2"] * u.deg)**2

    radius = gal.radius(header=header).to(u.kpc).value
    if pa_bounds is not None:
        # Check if they are angles
        if len(pa_bounds) != 2:
            raise IndexError("pa_bounds must contain 2 angles.")
        if not isinstance(pa_bounds, Angle):
            raise TypeError("pa_bounds must be an Angle.")

        # Return the array of PAs in the galaxy frame
        pas = gal.position_angles(header=header)

        # If the start angle is greater than the end, we need to wrap about
        # the discontinuity
        if pa_bounds[0] > pa_bounds[1]:
            initial_start = pa_bounds[0].copy()
            pa_bounds = pa_bounds.wrap_at(initial_start)
            pas = pas.wrap_at(initial_start)

    if max_rad is not None:
        max_rad = max_rad.to(u.kpc).value
    else:
        max_rad = radius.max() - dr

    dr = dr.to(u.kpc).value

    nbins = np.int(np.floor(max_rad / dr))

    inneredge = np.linspace(0, max_rad, nbins)
    outeredge = np.linspace(0 + dr, max_rad + dr, nbins)
    sdprof = np.zeros(nbins)
    sdprof_sigma = np.zeros(nbins)
    radprof = np.zeros(nbins)

    if pa_bounds is not None:
        pa_idx = np.logical_and(pas >= pa_bounds[0], pas < pa_bounds[1])

    for ctr, (r0, r1) in enumerate(zip(inneredge,
                                       outeredge)):

        idx = np.logical_and(radius >= r0, radius < r1)
        if pa_bounds is not None:
            idx = np.logical_and(idx, pa_idx)

        if weight_type == "area":
            sdprof[ctr] = np.nansum(mom0[idx].value) / \
                np.sum(np.isfinite(radius[idx]))
            sdprof_sigma[ctr] = \
                np.sqrt(np.nansum((mom0[idx].value - sdprof[ctr])**2.) /
                        np.sum(np.isfinite(radius[idx])))
        else:
            # Has to be 'mass' since this is checked at the beginning
            sdprof[ctr] = np.nansum(np.power(mom0[idx].value, 2)) / \
                np.nansum(mom0[idx].value)

            # Now the std has weights of Sigma * dA, which should be
            # normalized
            weights = mom0[idx].value / np.nansum(mom0[idx].value)

            # No denominator, since the weights were normalize to unity.
            sdprof_sigma[ctr] = \
                np.sqrt(np.nansum(weights * (mom0[idx].value - sdprof[ctr])**2))

        # Rescale the sigma based on the number of independent samples
        if beam is not None:
            sdprof_sigma[ctr] /= \
                np.sqrt(np.sum(np.isfinite(radius[idx])) / beam_pix)
        radprof[ctr] = np.nanmean(radius[idx])

    # Re-apply some units
    radprof = radprof * u.kpc
    sdprof = sdprof * mom0.unit  # / u.pc ** 2
    sdprof_sigma = sdprof_sigma * mom0.unit  # / u.pc ** 2

    # Correct for the los inclinations
    sdprof *= np.cos(gal.inclination)
    sdprof_sigma *= np.cos(gal.inclination)

    # If in Jy/bm, convert to K.
    if cube is not None:
        unit = cube.unit

    if mom0 is not None:
        bases = mom0.unit.bases
        # Look for the brightness unit
        # Change to Jy/beam when astropy has better beam handling
        for bright in [u.Jy, u.K]:
            for base in bases:
                if bright.is_equivalent(base):
                    unit = base
                    break

    if unit.is_equivalent(u.Jy):
        # The beam units are sort of implied
        sdprof = sdprof.to(u.Jy * u.km / u.s)
        sdprof_sigma = sdprof_sigma.to(u.Jy * u.km / u.s)
        # Now convert to K
        if restfreq is not None:
            sdprof = sdprof * beam.jtok(restfreq)
            sdprof_sigma = sdprof_sigma * beam.jtok(restfreq)
        else:
            sdprof = sdprof * beam.jtok(mom0.header["RESTFREQ"])
            sdprof_sigma = sdprof_sigma * \
                beam.jtok(mom0.header["RESTFREQ"])

        sdprof = sdprof / u.Jy
        sdprof_sigma = sdprof_sigma / u.Jy

    if mass_conversion is not None:
        sdprof = sdprof * mass_conversion
        sdprof_sigma = sdprof_sigma * mass_conversion

    return radprof, sdprof, sdprof_sigma
