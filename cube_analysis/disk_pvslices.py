
import pvextractor as pv
from astropy.utils.console import ProgressBar
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import Angle
from regions import RectangleSkyRegion, write_ds9

from galaxies import Galaxy


def disk_pvslices(cube, gal, thetas, pv_width, max_rad, verbose=True,
                  save_name=None, quicklook=False, mom0=None,
                  save_kwargs={}, save_regions=False):
    '''
    Create a set of PV-slices across a disk at a given set of position angles.
    The position angles are measured with respect to the position angle of the
    galaxy's major axis.

    Parameters
    ----------
    cube : spectral_cube.SpectralCube
        Cube to extract slice from.
    gal : galaxies.Galaxy
        Galaxy object containing information about the system.
    thetas : astropy.units.Quantity
        Angles, with respect to the major axis of the disk, to create PV-slices
        at.
    pv_width : astropy.units.Quantity
        Width of each PV-slice in physical or angular units.
    max_rad : astropy.units.Quantity
        Maximum radius in the disk to create the profile from.
    verbose : bool, optional
        Enables a progress bar.
    save_name : str, optional
        Base file name to save the pv-slices with. Enables saving the
        pv-slices.
    quicklook : bool, optional
        Plots the path for the pv-slice on the zeroth moment map. Requires
        a spectral_cube.Projection be given for `mom0`. Requires the user
        to continue the loop at each iteration.
    mom0 : spectral_cube.Projection, optional
        A zeroth moment map used for plotting the paths with `quicklook`.
    save_kwargs : dict, optional
        Pass arguments for saving the pv-slice (e.g., `dict(overwrite=True)`)
    save_regions : bool, optional
        Enable saving the pv-slice regions as a DS9 region file.
    '''

    assert isinstance(gal, Galaxy)

    # Make sure the thetas are in degrees
    if not isinstance(thetas, u.Quantity):
        raise TypeError("thetas must be an astropy.units.Quantity")
    elif not thetas.unit.is_equivalent(u.deg):
        raise u.UnitsError("thetas must have a unit equivalent to deg.")

    thetas = thetas.to(u.deg)

    # If quicklook is enabled, you must give a moment0 projection.
    if quicklook and mom0 is None:
        raise ValueError("mom0 must be given when quicklook is enabled.")

    # Now check the pv_width
    if not isinstance(pv_width, u.Quantity):
        raise TypeError("pv_width must be an astropy.units.Quantity")

    distance = gal.distance
    pv_width_orig = pv_width.copy()

    if not max_rad.unit.is_equivalent(u.pc):
        raise u.UnitsError("max_rad must be given in physical units.")

        pv_width = phys_to_ang(pv_width, distance)

    if not max_rad.unit.is_equivalent(u.deg):
        # Try converting from a distance
        if not max_rad.unit.is_equivalent(u.pc):
            raise u.UnitsError("max_rad must be given in physical or "
                               "angular units.")

    radius = gal.radius(header=cube.header)

    pvslices = []
    paths = []
    rect_regions = []

    if verbose:
        iter = ProgressBar(thetas)
    else:
        iter = thetas

    for theta in iter:

        # Adjust path length based on inclination
        obs_rad = obs_radius(max_rad, theta, gal)

        # Now convert the physical size in the observed frame to an angular
        # size
        ang_length = 2 * phys_to_ang(obs_rad, distance)

        pv_path = pv.PathFromCenter(gal.center_position, length=ang_length,
                                    angle=theta + gal.position_angle,
                                    sample=20,
                                    width=pv_width)
        paths.append(pv_path)

        if quicklook:

            plt.imshow(mom0.value, origin='lower')
            plt.contour(radius <= max_rad, colors='r')
            center = gal.to_center_position_pixel(wcs=cube.wcs)
            plt.plot(center[0], center[1], 'bD')

            for i, posn in enumerate(pv_path.get_xy(cube.wcs)):
                if i == 0:
                    symb = "c^"
                else:
                    symb = "g^"
                plt.plot(posn[0], posn[1], symb)

            plt.draw()
            raw_input("{}".format(theta))
            plt.clf()

        # Set NaNs to zero. We're averaging over very large areas here.
        pvslice = pv.extract_pv_slice(cube, pv_path, respect_nan=False)

        if save_name is not None:
            filename = save_name + \
                "_PA_{0}_pvslice_{1}{2}_width.fits".format(int(theta.value),
                                                           pv_width_orig.value,
                                                           pv_width_orig.unit)
            pvslice.writeto(filename, **save_kwargs)
        else:
            pvslices.append(pvslice)

        if save_regions:
            # Convert the path to a rectangular region and write out as a
            # ds9 file

            rect_region = RectangleSkyRegion(center=gal.center_position,
                                             height=Angle(pv_width.to(u.deg)),
                                             width=Angle(ang_length.to(u.deg)),
                                             angle=Angle(theta + gal.position_angle))

            rect_regions.append(rect_region)

    # Save all of the slice paths as a ds9 region file
    if save_regions:
        region_name = save_name + \
            "_pvslice_{0}{1}_width.reg".format(pv_width_orig.value,
                                               pv_width_orig.unit)
        write_ds9(rect_regions, region_name)

    return pvslices


def obs_radius(radius, PA, gal):
    '''
    Radius cut-off from galaxy frame to observation frame
    '''

    ang_term = (np.cos(PA))**2 + (np.sin(PA) / np.cos(gal.inclination))**2

    return np.sqrt(radius**2 / ang_term)


def phys_to_ang(phys_size, distance):
    '''
    Convert from angular to physical scales
    '''
    return (phys_size.to(distance.unit) / distance) * u.rad
