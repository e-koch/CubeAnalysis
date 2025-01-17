import contextlib
import warnings
import builtins


from tqdm.auto import tqdm
def ProgressBar(niter, **kwargs):
    return tqdm(total=niter, **kwargs)


'''
Copyright (c) 2014-2024, spectral-cube developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''

@contextlib.contextmanager
def _map_context(numcores, verbose=False, num_jobs=None, chunksize=None,):
    """
    Mapping context manager to allow parallel mapping or regular mapping
    depending on the number of cores specified.

    The builtin map is overloaded to handle python3 problems: python3 returns a
    generator, while ``multiprocessing.Pool.map`` actually runs the whole thing
    """
    if numcores is not None and numcores > 1:
        try:
            from joblib import Parallel, delayed
            from joblib.pool import has_shareable_memory
            map = lambda x,y: Parallel(n_jobs=numcores)(delayed(has_shareable_memory)(x))(y)
            parallel = True
        except ImportError:
            map = lambda x,y: list(builtins.map(x,y))
            warnings.warn("Could not import joblib.  "
                          "map will be non-parallel.",
                          ImportError
                         )
            parallel = False
    else:
        parallel = False
        map = lambda x, y: list(builtins.map(x,y))

    yield map

def choose_chunksize(nprocesses, njobs):
    '''
    Split the chunks into roughly equal portions.
    '''
    # Auto split into close to equal chunks
    if njobs % nprocesses == 0:
        chunksize = njobs / nprocesses
    else:
        # Split into smaller chunks that are still
        # roughly equal, but won't have any small
        # leftovers that would slow things down
        chunksize = njobs / (nprocesses + 1)

    return chunksize if chunksize > 0 else 1
