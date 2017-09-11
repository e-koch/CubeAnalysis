
import shutil
import os
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy import log
from pandas import DataFrame
import numpy as np
from astropy.io import fits
from glob import glob
from warnings import warn
from galaxies import Galaxy

from .curve_fitting import return_smooth_model, generate_vrot_model
from .update_galaxy_params import update_galaxy_params


def run_diskfit(param_file, data_path, fits_file_wcs, overwrite=True,
                fit_model=True, gal=None):
    '''
    This is a wrapper to make running DiskFit and processing its output
    easier (at least when already working in a python environment).

    Run Diskfit provided with the parameter file and the path to the data.
    The output folder in the parameter file should be the path *from the
    given data_path*! This is due to the character limit in Fortran. For
    example, if the full path to the output folder is
    "/home/user/data/output/", the parameter file should have "output", and
    data_path will be "/home/user/data".

    DiskFit is available here: http://www.physics.rutgers.edu/~spekkens/diskfit/
    The relevant documentation and references are listed on the site.

    .. warning:: This wrapper will not work if asymmetry, radial, or warp
        components are included in the fit!

    Parameters
    ----------
    param_file : str
        DiskFit parameter file.
    data_path : str
        The path from which the output path in the parameter file is defined.
    fits_file_wcs : str
        A FITS file whose WCS information matches the moment array that has
        been fit to. The FITS files DiskFit outputs do not contain complete
        WCS information in the headers.
    '''

    if fit_model:
        if not isinstance(gal, Galaxy):
            raise ValueError("When fit_model is enabled, a Galaxy class must "
                             "be passed with `gal`.")

    # Due to the 100 character limit, copy the parameter file into the data path
    shutil.copyfile(param_file,
                    os.path.join(data_path, os.path.basename(param_file)))

    # Get the current directory so we can switch back at the end
    orig_direc = os.getcwd()

    # Now move to data_path
    os.chdir(data_path)
    # Cut to just the filename
    param_file = os.path.basename(param_file)

    # Now extract the output filename to create the folder
    with open(param_file) as f:
        content = f.readlines()

    # Output is on 8th line
    output_folder = content[7].split("'")[1].split("/")[0]
    moment_file = os.path.join(data_path, content[3].split("'")[1])

    if os.path.exists(output_folder):
        if not overwrite:
            raise OSError("Folder name already exists.")
        else:
            log.warn("The output folder exists and overwrite is selected. "
                     "Deleting existing folder.")
            shutil.rmtree(output_folder, ignore_errors=True)
    try:
        os.mkdir(output_folder)
    except OSError:
        raise OSError("Folder name already exists.")

    # Now we're ready to run it!
    os.system('echo "' + param_file + '" | ~/DiskFit')

    # When it finishes, move the parameter file into the output
    shutil.move(param_file, output_folder)

    os.chdir(output_folder)

    # Now we need to: 1) add WCS information to the model FITS, and 2) make an
    # easy to read version of the velocity curve.

    mom1 = fits.open(fits_file_wcs)
    header = mom1[0].header.copy()
    mom1.close()

    mywcs = WCS(header)

    # While thorough, the DISKFIT output isn't immediately machine-readable by
    # pandas or astropy.table. This creates a more convenient output.

    # Read in as a string
    with open('rad.out') as f:
        contents = f.readlines()

    # Find which of the parameters were left free
    disk_toggles = contents[14].split()[2:]
    disk_free_params = {disk_toggles[i][:-1]: True
                        if disk_toggles[i + 1] == "T"
                        else False for i in np.arange(0, 10, 2)}

    vel_toggles = contents[15].split()[2:]
    vel_free_params = {vel_toggles[i][:-1]: True
                       if vel_toggles[i + 1] == "T"
                       else False for i in np.arange(0, 10, 2)}


    # Read out the best fit values for the galaxy parameters and the fit results
    # Params on line 39-42, 44, 56-59
    params = {}

    if disk_free_params['PA']:
        params["PA"] = float(contents[39].split()[-3])
        params["PA_err"] = float(contents[39].split()[-1])
    else:
        params["PA"] = float(contents[19].split()[-1])
        params["PA_err"] = np.NaN

    if disk_free_params['eps']:
        params["eps"] = float(contents[40].split()[-3])
        params["eps_err"] = float(contents[40].split()[-1])
        params["inc"] = float(contents[41].split()[-3])
        params["inc_err"] = float(contents[41].split()[-1])
    else:
        params["eps"] = float(contents[20].split()[-1])
        params["eps_err"] = np.NaN
        params["inc"] = np.rad2deg(np.arccos(1 - params['eps']))
        params["inc_err"] = np.NaN

    # Both x and y are on the same line
    if disk_free_params['center']:
        params["xcent"] = float(contents[42].split()[-6])
        params["xcent_err"] = float(contents[42].split()[-4][:-1])
        params["ycent"] = float(contents[42].split()[-3])
        params["ycent_err"] = float(contents[42].split()[-1])
    else:
        params["xcent"] = float(contents[42].split()[-6])
        params["xcent_err"] = np.NaN
        params["ycent"] = float(contents[42].split()[-3])
        params["ycent_err"] = np.NaN

    # Now convert xcent and ycent to RA and Dec.
    params["RAcent"], params["Deccent"] = \
        mywcs.celestial.wcs_pix2world(params["xcent"], params["ycent"], 0)
    # Add angular uncertainties in deg.
    pix_scale = proj_plane_pixel_scales(mywcs.celestial)
    params["RAcent_err"] = pix_scale[0] * params["xcent_err"]
    params["Deccent_err"] = pix_scale[1] * params["ycent_err"]

    if vel_free_params['Vsys']:
        params["Vsys"] = float(contents[44].split()[-3])
        params["Vsys_err"] = float(contents[44].split()[-1])
    else:
        params["Vsys"] = float(contents[24].split()[-1])
        params["Vsys_err"] = np.NaN

    params["points_used"] = float(contents[56].split()[-1])
    params["iterations"] = float(contents[57].split()[-1])
    params["chi^2"] = float(contents[58].split()[-1])
    params["DOF"] = float(contents[59].split()[-1])

    # Column names are always on line 62, and data starts on 64
    colnames = contents[62].split()

    data = []

    for line in contents[64:]:
        data.append([float(val) for val in line.split()])

    data = np.array(data)
    # Sometimes Vt comes out as negative, due to PA being flipped by 180
    # degrees. The first few points may have +/- values within the error
    # so determine by checking the last 10 points.
    if (data[-10:, 2] < 0).all():
        data[:, 2] = np.abs(data[:, 2])

        # Flip PA by 180 deg.
        if params['PA'] + 180 < 360:
            params["PA"] += 180.
        else:
            params["PA"] -= 180.

    tab = Table(data=data, names=colnames)
    tab.write('rad.out.csv')

    # Can't read a dictionary directly into an astropy table?
    df = DataFrame(params, index=[0])
    df.to_csv('rad.out.params.csv')

    # Add WCS info to the output of DISKFIT

    fits_files = glob("*.fits")

    # Correct the header outputted by Diskfit

    header["COMMENT"] = "FILE GENERATED BY DISKFIT"
    keep_keys = ['INFILE', 'OUTPFILE', 'COMPS', 'SMAF', 'PIXSCALE', 'XCENOUT',
                 'YCENOUT', 'PA_OUT', 'EPS_OUT']

    for f in fits_files:
        new_header = header.copy()
        hdu = fits.open(f, mode='update')

        for key in keep_keys:
            try:
                new_header[key] = hdu[0].header[key]
            except KeyError:
                warn("Could not find keyword {} in header".format(key))

        hdu[0].header = new_header
        hdu.flush()
        hdu.close()

    if fit_model:
        # Generate a smooth model of the rotation curve
        update_galaxy_params(gal, df)

        pars, pcov = generate_vrot_model(tab)

        smooth_model = return_smooth_model(tab, header, gal)

        fit_comment = "Smooth rotation model of DISKFIT output. " \
            "Uses Eq.5 from Meidt et al. 2008. n={0:.2f}+/-{1:.2f}, " \
            "Vmax={2:.2f}+/-{3:.2f} km/s, rmax={4:.2f}+/-{5:.2f} pix".\
            format(pars[0], np.sqrt(pcov[0, 0]), pars[1], np.sqrt(pcov[1, 1]),
                   pars[2], np.sqrt(pcov[2, 2]))

        bunit = "m / s"

        # Save the smooth model
        smooth_hdu = fits.PrimaryHDU(smooth_model, header=header)
        smooth_hdu.header["BUNIT"] = bunit
        smooth_hdu.header["COMMENT"] = fit_comment
        smooth_hdu.writeto("rad.fitmod.fits", overwrite=True)

        # Also save the residuals from the smooth model
        moment_map = fits.open(moment_file)[0]
        smoothres_hdu = fits.PrimaryHDU(moment_map.data - smooth_model,
                                        header=header)
        smoothres_hdu.header["BUNIT"] = bunit
        smoothres_hdu.header["COMMENT"] = fit_comment
        smoothres_hdu.writeto("rad.fitres.fits", overwrite=True)

        # Save the fit model parameters to the parameter file.
        params['fit_n'] = pars[0]
        params['fit_n_err'] = pcov[0, 0]
        params['fit_vmax'] = pars[0]
        params['fit_vmax_err'] = pcov[0, 0]
        params['fit_n'] = pars[0]
        params['fit_n_err'] = pcov[0, 0]
        df = DataFrame(params, index=[0])
        df.to_csv('rad.out.params.csv')

    # Return the the initial directory
    os.chdir(orig_direc)


if __name__ == "__main__":

    import sys

    param_file = sys.argv[1]
    data_path = sys.argv[2]
    fits_file_wcs = sys.argv[3]

    run_diskfit(param_file, data_path, fits_file_wcs)
