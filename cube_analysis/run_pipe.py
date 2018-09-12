
import os
from astropy import log
from datetime import datetime

from .masking import pb_masking, signal_masking, common_beam_convolve
from .moments import make_moments


def run_pipeline(cube_name, output_folder, pb_file=None, pb_lim=0.5,
                 convolve_to_common_beam=False,
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
    else:
        pb_masking(cube_name, pb_file, pb_lim, output_folder)
        cube_name_pbmask = \
            "{0}.pbcov_gt_{1}_masked.fits".format(cube_name.rstrip(".fits"),
                                                  pb_lim)

    if convolve_to_common_beam:
        log.info("Convolving to a common beam size, defined by the largest"
                 " beam. This version will overwrite the PB masked cube. ")
        tstamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log.info("Convolving to a common beam at {}".format(tstamp))
        common_beam_convolve(cube_name_pbmask, output_folder)

    tstamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log.info("Starting signal masking at {}".format(tstamp))
    signal_masking(cube_name_pbmask, output_folder, **masking_kwargs)

    mask_name = \
        "{}_source_mask.fits".format(cube_name_pbmask.rstrip(".fits"))

    tstamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log.info("Starting moment making at {}".format(tstamp))
    make_moments(cube_name_pbmask, mask_name, output_folder,
                 **moment_kwargs)

    tstamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log.info("Finished analysis pipeline at {}".format(tstamp))
