
import os
from astropy import log
from datetime import datetime

from .masking import pb_masking, signal_masking, common_beam_convolve
from .moments import make_moments


def run_pipeline(cube_name, output_folder,
                 apply_pbmasking=True,
                 pb_file=None, pb_lim=0.5,
                 convolve_to_common_beam=False, combeam_kwargs={},
                 skip_existing_mask=False,
                 masking_kwargs={}, moment_kwargs={}):
    '''
    Mask and create moment arrays for a given PPV cube.
    '''

    if not os.path.exists(cube_name):
        raise IOError("Given cube name doesn't exist: {}".format(cube_name))

    # if not os.path.exists(output_folder):
    #     os.mkdir(output_folder)
    # else:
    #     inp = raw_input("This path already exists! To proceed and overwrite"
    #                     " files, enter 'y':")
    #     if inp != 'y':
    #         raise OSError("Remove the folder {} before running."
    #                       .format(output_folder))

    tstamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log.info("Starting PB masking at {}".format(tstamp))
    if pb_file is None:
        log.info("No PB file given. No PB masking applied.")
        cube_name_pbmask = cube_name
    elif apply_pbmasking is False:
        cube_name_pbmask = cube_name
    else:
        cube_name_pbmask = \
            "{0}.pbcov_gt_{1}_masked.fits".format(cube_name.rstrip(".fits"),
                                                  pb_lim)

        if os.path.exists(cube_name_pbmask):
            log.info("PB masked cube already exists")
        else:
            pb_masking(cube_name, pb_file, pb_lim, output_folder)

        # Also want to use the PB cube later on in the masking. So make
        # a masked and minimzed version of it too.

        pb_file_pbmask = \
            "{0}.pbcov_gt_{1}_masked.fits".format(pb_file.rstrip(".fits"),
                                                  pb_lim)

        if os.path.exists(pb_file_pbmask):
            log.info("PB masked cube already exists")
        else:
            pb_masking(pb_file, pb_file, pb_lim, output_folder)

    if convolve_to_common_beam:
        log.info("Convolving to a common beam size, defined by the largest"
                 " beam. This version will overwrite the PB masked cube. ")
        tstamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log.info("Convolving to a common beam at {}".format(tstamp))
        common_beam_convolve(cube_name_pbmask, output_folder, **combeam_kwargs)

    tstamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log.info("Starting signal masking at {}".format(tstamp))

    mask_name = \
        "{}_source_mask.fits".format(cube_name_pbmask.rstrip(".fits"))

    if os.path.exists(mask_name) and skip_existing_mask:
        log.info("Found existing signal masking. Skipping to making moment "
                 "images.")
    else:
        signal_masking(cube_name_pbmask, output_folder, **masking_kwargs)

    tstamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log.info("Starting moment making at {}".format(tstamp))
    make_moments(cube_name_pbmask, mask_name, output_folder,
                 **moment_kwargs)

    tstamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log.info("Finished analysis pipeline at {}".format(tstamp))
