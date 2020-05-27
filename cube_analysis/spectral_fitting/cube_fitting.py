
'''
Simplified per-spectrum fitting of a cube. Does not have many features
(on purpose).

See http://pyspeckit.readthedocs.io/en/latest/cubes.html or
https://github.com/vlas-sokolov/multicube.

'''

import numpy as np
from astropy import log
from spectral_cube import SpectralCube
from astropy.io import fits
from datetime import datetime

import sys
if sys.version_info < (3, 0):
    from itertools import izip as zip


from ..progressbar import _map_context
from ..feather_cubes import get_channel_chunks


def cube_fitter(cube_name,
                fitting_func, args=(), kwargs={},
                mask_name=None,
                spatial_mask=None,
                err_map=None,
                vcent_map=None,
                npars=None,
                nfit_stats=1,
                verbose=False, num_cores=1, chunks=10000):
    '''
    Split fitting the cube in chunks. Return the parameters, uncertainties,
    and a fit statistic in `fitting_func`.
    '''

    cube = SpectralCube.read(cube_name)

    if mask_name is not None:
        cube = cube.with_mask(fits.open(mask_name)[0].data > 0)

    if err_map is not None:
        assert cube[0].shape == err_map.shape

    if spatial_mask is not None:
        if not cube[0].shape == spatial_mask.shape:
            raise ValueError("spatial_mask must have the same shape as the "
                             "cube's spatial dimensions.")
    else:
        if err_map is None:
            spatial_mask = cube.mask.include().sum(0) > 0
        else:
            spatial_mask = np.isfinite(err_map)

    del cube

    posns = np.where(spatial_mask)

    y_chunks = get_channel_chunks(posns[0].size, chunks)
    x_chunks = get_channel_chunks(posns[1].size, chunks)

    start_time = datetime.now()

    for i, (y_chunk, x_chunk) in enumerate(zip(y_chunks, x_chunks)):

        log.info("On chunk {0} of {1} at {2}".format(i + 1, len(y_chunks),
                                                     datetime.now()))

        cube = SpectralCube.read(cube_name)
        if mask_name is not None:
            cube = cube.with_mask(fits.open(mask_name)[0].data > 0)

        gener = [(fitting_func, args, kwargs, cube[:, y, x],
                  err_map[y, x] if err_map is not None else 1.,
                  vcent_map[y, x] if vcent_map is not None else None)
                 for y, x in zip(posns[0][y_chunk],
                                 posns[1][x_chunk])]

        del cube

        with _map_context(num_cores, verbose=False,
                          num_jobs=len(y_chunk)) as map:
            out_params = map(_fitter, gener)

        if i == 0:
            if npars is None:
                if hasattr(out_params[0], "size"):
                    npars = len(out_params[0][0])
                else:
                    npars = 1

            param_cube = np.empty((npars, ) + spatial_mask.shape)
            error_cube = np.empty((npars, ) + spatial_mask.shape)
            fit_statistic = np.empty((nfit_stats, ) + spatial_mask.shape)

        for out, y, x in zip(out_params, posns[0][y_chunk],
                             posns[1][x_chunk]):
            param_cube[:, y, x] = out[0]
            error_cube[:, y, x] = out[1]
            fit_statistic[:, y, x] = out[2]

    empty_posns = np.where(~spatial_mask)
    for i in range(npars):
        param_cube[i][empty_posns] = np.NaN
        error_cube[i][empty_posns] = np.NaN
    for i in range(nfit_stats):
        fit_statistic[i][empty_posns] = np.NaN

    end_time = datetime.now()

    log.info(f"Elapsed time to fit cube {end_time - start_time}.")

    return param_cube, error_cube, fit_statistic


def _fitter(inp_args):

    fitting_func, args, kwargs, spec, err, vcent = inp_args

    return fitting_func(spec, err, vcent=vcent,
                        *args, **kwargs)
