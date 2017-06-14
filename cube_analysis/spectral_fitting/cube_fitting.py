
'''
Simplified per-spectrum fitting of a cube. Does not have many features
(on purpose).

See http://pyspeckit.readthedocs.io/en/latest/cubes.html or
https://github.com/vlas-sokolov/multicube.

'''

import numpy as np
from itertools import izip
from astropy import log

from ..progressbar import _map_context
from ..feather_cubes import get_channel_chunks


def cube_fitter(cube, fitting_func, args=(), kwargs={}, spatial_mask=None,
                verbose=False, num_cores=1, chunks=10000):
    '''
    '''

    if spatial_mask is not None:
        if not cube[0].shape == spatial_mask.shape:
            raise ValueError("spatial_mask must have the same shape as the "
                             "cube's spatial dimensions.")
    else:
        spatial_mask = cube.mask.include().sum(0) > 0

    posns = np.where(spatial_mask)

    y_chunks = get_channel_chunks(posns[0].size, chunks)
    x_chunks = get_channel_chunks(posns[1].size, chunks)

    for i, (y_chunk, x_chunk) in enumerate(izip(y_chunks, x_chunks)):

        log.info("On chunk {0} of {1}".format(i + 1, len(y_chunks)))

        gener = [(fitting_func, args, kwargs, cube[:, y, x])
                 for y, x in izip(posns[0][y_chunk], posns[1][x_chunk])]

        with _map_context(num_cores, verbose=True,
                          num_jobs=len(y_chunk)) as map:
            out_params = map(_fitter, gener)

        if i == 0:
            npars = len(out_params[0][0]) if hasattr(out_params[0], "size") \
                else 1

            param_cube = np.empty((npars, ) + spatial_mask.shape)
            error_cube = np.empty((npars, ) + spatial_mask.shape)

        for out, y, x in izip(out_params, posns[0][y_chunk],
                              posns[1][x_chunk]):
            param_cube[:, y, x] = out[0]
            error_cube[:, y, x] = out[1]

    empty_posns = np.where(~spatial_mask)
    for i in range(npars):
        param_cube[i][empty_posns] = np.NaN
        error_cube[i][empty_posns] = np.NaN

    return param_cube, error_cube


def _fitter(inp_args):

    fitting_func, args, kwargs, spec = inp_args

    return fitting_func(spec, *args, **kwargs)
