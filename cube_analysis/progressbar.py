
import six
import time
import signal
import multiprocessing
from functools import partial
from astropy.utils.console import (_get_stdout, isatty, isiterable,
                                   human_file_size, _CAN_RESIZE_TERMINAL,
                                   terminal_size, color_print, human_time)

import contextlib
import warnings
try:
    import builtins
except ImportError:
    # python2
    import __builtin__ as builtins

'''
Copyright (c) 2011-2016, Astropy Developers

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the name of the Astropy Team nor the names of its contributors may be
  used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''


class ProgressBar(six.Iterator):
    """
    A class to display a progress bar in the terminal.

    It is designed to be used either with the ``with`` statement::

        with ProgressBar(len(items)) as bar:
            for item in enumerate(items):
                bar.update()

    or as a generator::

        for item in ProgressBar(items):
            item.process()
    """
    def __init__(self, total_or_items, ipython_widget=False, file=None):
        """
        Parameters
        ----------
        total_or_items : int or sequence
            If an int, the number of increments in the process being
            tracked.  If a sequence, the items to iterate over.

        ipython_widget : bool, optional
            If `True`, the progress bar will display as an IPython
            notebook widget.

        file : writable file-like object, optional
            The file to write the progress bar to.  Defaults to
            `sys.stdout`.  If `file` is not a tty (as determined by
            calling its `isatty` member, if any, or special case hacks
            to detect the IPython console), the progress bar will be
            completely silent.
        """

        ipython_widget = False

        # if ipython_widget:
        #     # Import only if ipython_widget, i.e., widget in IPython
        #     # notebook
        #     if ipython_major_version < 4:
        #         from IPython.html import widgets
        #     else:
        #         from ipywidgets import widgets
        #     from IPython.display import display

        if file is None:
            file = _get_stdout()

        if not isatty(file) and not ipython_widget:
            self.update = self._silent_update
            self._silent = True
        else:
            self._silent = False

        if isiterable(total_or_items):
            self._items = iter(total_or_items)
            self._total = len(total_or_items)
        else:
            try:
                self._total = int(total_or_items)
            except TypeError:
                raise TypeError("First argument must be int or sequence")
            else:
                self._items = iter(range(self._total))

        self._file = file
        self._start_time = time.time()
        self._human_total = human_file_size(self._total)
        self._ipython_widget = ipython_widget

        self._signal_set = False
        if not ipython_widget:
            self._should_handle_resize = (
                _CAN_RESIZE_TERMINAL and self._file.isatty())
            self._handle_resize()
            if self._should_handle_resize:
                signal.signal(signal.SIGWINCH, self._handle_resize)
                self._signal_set = True

        self.update(0)

    def _handle_resize(self, signum=None, frame=None):
        terminal_width = terminal_size(self._file)[1]
        self._bar_length = terminal_width - 37

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._silent:
            if exc_type is None:
                self.update(self._total)
            self._file.write('\n')
            self._file.flush()
            if self._signal_set:
                signal.signal(signal.SIGWINCH, signal.SIG_DFL)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            rv = next(self._items)
        except StopIteration:
            self.__exit__(None, None, None)
            raise
        else:
            self.update()
            return rv

    def update(self, value=None):
        """
        Update progress bar via the console or notebook accordingly.
        """

        # Update self.value
        if value is None:
            value = self._current_value + 1
        self._current_value = value

        # Choose the appropriate environment
        if self._ipython_widget:
            self._update_ipython_widget(value)
        else:
            self._update_console(value)

    def _update_console(self, value=None):
        """
        Update the progress bar to the given value (out of the total
        given to the constructor).
        """

        if self._total == 0:
            frac = 1.0
        else:
            frac = float(value) / float(self._total)

        file = self._file
        write = file.write

        if frac > 1:
            bar_fill = int(self._bar_length)
        else:
            bar_fill = int(float(self._bar_length) * frac)
        write('\r|')
        color_print('=' * bar_fill, 'blue', file=file, end='')
        if bar_fill < self._bar_length:
            color_print('>', 'green', file=file, end='')
            write('-' * (self._bar_length - bar_fill - 1))
        write('|')

        if value >= self._total:
            t = time.time() - self._start_time
            prefix = '     '
        elif value <= 0:
            t = None
            prefix = ''
        else:
            t = ((time.time() - self._start_time) * (1.0 - frac)) / frac
            prefix = ' ETA '
        write(' {0:>4s}/{1:>4s}'.format(
            human_file_size(value),
            self._human_total))
        write(' ({0:>6s}%)'.format('{0:.2f}'.format(frac * 100.0)))
        write(prefix)
        if t is not None:
            write(human_time(t))
        self._file.flush()

    def _update_ipython_widget(self, value=None):
        """
        Update the progress bar to the given value (out of a total
        given to the constructor).

        This method is for use in the IPython notebook 2+.
        """
        pass

        # Create and display an empty progress bar widget,
        # if none exists.
        # if not hasattr(self, '_widget'):
        #     # Import only if an IPython widget, i.e., widget in iPython NB
        #     if ipython_major_version < 4:
        #         from IPython.html import widgets
        #         self._widget = widgets.FloatProgressWidget()
        #     else:
        #         from ipywidgets import widgets
        #         self._widget = widgets.FloatProgress()
        #     from IPython.display import display

        #     display(self._widget)
        #     self._widget.value = 0

        # # Calculate percent completion, and update progress bar
        # percent = (value / self._total) * 100
        # self._widget.value = percent
        # self._widget.description = \
        #     ' ({0:>6s}%)'.format('{0:.2f}'.format(percent))

    def _silent_update(self, value=None):
        pass

    @classmethod
    def map(cls, function, items, multiprocess=False, file=None, chunksize=100,
            item_len=None, nprocesses=None, **pool_kwargs):
        """
        Does a `map` operation while displaying a progress bar with
        percentage complete.

        ::

            def work(i):
                print(i)

            ProgressBar.map(work, range(50))

        Parameters
        ----------
        function : function
            Function to call for each step

        items : sequence
            Sequence where each element is a tuple of arguments to pass to
            *function*.

        multiprocess : bool, optional
            If `True`, use the `multiprocessing` module to distribute each
            task to a different processor core.

        file : writeable file-like object, optional
            The file to write the progress bar to.  Defaults to
            `sys.stdout`.  If `file` is not a tty (as determined by
            calling its `isatty` member, if any), the scrollbar will
            be completely silent.

        step : int, optional
            Update the progress bar at least every *step* steps (default: 100).
            If ``multiprocess`` is `True`, this will affect the size
            of the chunks of ``items`` that are submitted as separate tasks
            to the process pool.  A large step size may make the job
            complete faster if ``items`` is very long.
        """

        results = []

        if file is None:
            file = _get_stdout()

        if item_len is not None:
            assert isinstance(item_len, int)
            if hasattr(items, "__len__"):
                assert item_len == len(items)
        else:
            if hasattr(items, "__len__"):
                item_len = len(items)
            else:
                # Will convert to iterable. Not a good thing to do with
                # large inputs.
                items = list(items)
                item_len = len(items)

        with cls(item_len, file=file) as bar:
            if not multiprocess:
                for i, item in enumerate(items):
                    results.append(function(item))
                    if (i % chunksize) == 0:
                        bar.update(i)
            else:
                max_proc = multiprocessing.cpu_count()
                if nprocesses is None:
                    nprocesses = max_proc
                elif nprocesses > max_proc:
                    nprocesses = max_proc

                if chunksize is None:
                    chunksize = choose_chunksize(nprocesses, item_len)

                p = multiprocessing.Pool(nprocesses, **pool_kwargs)
                for i, out in enumerate(p.imap_unordered(function,
                                                         items,
                                                         chunksize=chunksize)):
                    bar.update(i)
                    results.append(out)
                p.close()
                p.join()

        return results


'''
Copyright (c) 2014, spectral-cube developers
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
def _map_context(numcores, verbose=False, num_jobs=None, chunksize=None,
                 **pool_kwargs):
    """
    Mapping context manager to allow parallel mapping or regular mapping
    depending on the number of cores specified.
    """

    if verbose:
        if numcores is not None and numcores > 1:
            parallel = True
        else:
            numcores = 1
            parallel = False
        map = lambda func, items: \
            ProgressBar.map(func, items,
                            nprocesses=numcores,
                            multiprocess=parallel,
                            item_len=num_jobs,
                            chunksize=chunksize,
                            **pool_kwargs)
    else:
        if numcores is not None and numcores > 1:
            try:
                import multiprocessing
                p = multiprocessing.Pool(processes=numcores, **pool_kwargs)
                map = partial(p.imap_unordered, chunksize=chunksize)
                parallel = True
            except ImportError:
                map = builtins.map
                warnings.warn("Could not import multiprocessing.  "
                              "map will be non-parallel.")
                parallel = False
        else:
            parallel = False
            map = builtins.map

    try:
        yield map
    finally:
        # ProgressBar.map already closes the pool
        if not verbose and parallel:
                p.close()


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

    return chunksize
