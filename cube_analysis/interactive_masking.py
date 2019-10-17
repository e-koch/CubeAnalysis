
'''
Adapted from matplotlib example:
https://matplotlib.org/3.1.0/gallery/widgets/lasso_selector_demo_sgskip.html

And the mask output is adapted from:
https://gist.github.com/tonysyu/3090704
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector

from spectral_cube import SpectralCube
from astropy.io import fits
from astropy.utils.console import ProgressBar

from .io_utils import create_huge_fits


class SelectFromImage(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    shape : tuple
        Shape of 2D image.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`. CURRENT
        DOES NOTHING.
    """

    def __init__(self, ax, shape, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.alpha_other = alpha_other

        self._shape = shape
        h, w = shape
        y, x = np.mgrid[:h, :w]

        self._mask = np.zeros(shape, dtype=bool)

        self.xys = [(xpt, ypt) for ypt, xpt in zip(y.ravel(), x.ravel())]
        self.Npts = len(self.xys)

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

        self._collections = []

    def onselect(self, verts):
        path = Path(verts)
        new_mask = path.contains_points(self.xys).reshape(self._shape)
        self._mask = np.logical_or(self._mask, new_mask)

        # Check if a contour already exists. If yes, remove it.
        if len(self.lasso.ax.collections) > 0:
            # There should only ever be one contour shown at a time.
            assert len(self.lasso.ax.collections) == 1
            self.lasso.ax.collections.pop(0)

        self.lasso.ax.contour(self.mask, colors='k', levels=[0.5],
                              linewidths=3)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()

    @property
    def mask(self):
        return self._mask


def make_interactive_image_mask(img, fig=None,
                                imshow_kwargs={'origin': 'lower'}):
    '''
    Make a boolean mask interactively given a 2D image.
    '''

    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.imshow(img, **imshow_kwargs)

    selector = SelectFromImage(ax, img.shape)

    # Record the button pressed
    event_button = []

    def accept(event):

        # Record the key so we know how to handle empty channels.
        event_button.append(event.key)

        if event.key == "enter":
            pass
        elif event.key == 'a':
            if selector.mask.any():
                print("Selected mask not empty."
                      " Overriding to give full mask.")
        elif event.key == 'z':
            if selector.mask.any():
                print("Selected mask not empty."
                      " Overriding to give empty mask.")
        else:
            raise ValueError(f"Select 'enter' to accept the mask,"
                             " 'a' for a full mask, and 'z' for an empty "
                             "mask. Received {event.key}")
        selector.disconnect()
        ax.set_title("")
        selector.lasso.ax.clear()
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("key_press_event", accept)
    ax.set_title("Press enter to accept. 'a' for full. 'z' for empty.")

    while len(event_button) == 0:
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(1.)

        if len(event_button) != 0:
            break
    fig.canvas.mpl_disconnect(cid)

    if event_button[0] == 'enter':
        return selector.mask
    elif event_button[0] == 'a':
        return np.ones_like(img, dtype=bool)
    elif event_button[0] == 'z':
        return np.ones_like(img, dtype=bool)
    else:
        raise ValueError("This should not happen...")


def test_interactive_image():
    img = np.random.uniform(0, 255, size=(100, 100))

    mask = make_interactive_image_mask(img)
    return mask


def make_interactive_cube_mask(cube_name, output_name, in_memory=False,
                               return_mask=False, overwrite=False,
                               verbose=True,
                               imshow_kwargs={'origin': 'lower'},
                               fig=None):
    '''
    Step through each channel in a cube and interactively draw a mask.

    Empty channels can be passed by pressing 'z'. Full channels
    (that is a mask everywhere with valid data), can be passed by pressing 'a'.

    '''

    if fig is not None:
        fig = plt.figure()

    if in_memory:

        cube = SpectralCube.read(cube_name)

        nchan = cube.shape[0]

        mask = np.zeros(cube.shape, dtype=bool)

        if verbose:
            iter = ProgressBar(nchan)
        else:
            iter = range(nchan)

        for chan in iter:
            interact_mask = make_interactive_image_mask(cube.unitless_filled_data[chan],
                                                        fig=fig,
                                                        imshow_kwargs=imshow_kwargs)

            # Combine current mask with the mask from the cube.
            view = (slice(chan), slice(None), slice(None))
            mask[chan] = np.logical_and(cube.mask.filled(view=view),
                                        interact_mask)

            fig.clear()

            if verbose:
                iter.update(chan + 1)

        mask_hdr = cube.header.copy()
        mask_hdr['BUNIT'] = ('', "Boolean")

        # Change BITPIX
        dtype = np.uint8
        name = dtype.name if hasattr(dtype, "name") else dtype
        mask_hdr['BITPIX'] = fits.DTYPE2BITPIX[name]

        mask_hdu = fits.PrimaryHDU(mask.astype(dtype), mask_hdr)
        mask_hdu.write(output_name, overwrite=overwrite)

        if return_mask:
            return mask

    else:
        # Assume this is a big FITS file. So we open one channel at a time.

        cube_header = fits.getheader(cube_name)

        nchan = cube_header['NAXIS3']

        mask_hdr = cube_header.copy()
        mask_hdr['BUNIT'] = ('', "Boolean")

        # Change BITPIX
        dtype = np.uint8
        name = dtype.name if hasattr(dtype, "name") else dtype
        mask_hdr['BITPIX'] = fits.DTYPE2BITPIX[name]

        create_huge_fits(output_name, mask_hdr, verbose=verbose)

        if verbose:
            iter = ProgressBar(nchan)
        else:
            iter = range(nchan)

        for chan in iter:

            cube_hdu = fits.open(cube_name, mode='denywrite', memmap=True)
            mask_hdu = fits.open(output_name, mode='update')

            interact_mask = make_interactive_image_mask(cube_hdu[0].data[chan],
                                                        fig=fig,
                                                        imshow_kwargs=imshow_kwargs)

            # Remove NaNs from the cube
            channel_mask = np.logical_and(interact_mask,
                                          np.isfinite(cube_hdu[0].data[chan]))

            mask_hdu.data[chan] = channel_mask.astype(np.uint8)

            mask_hdu.flush()
            mask_hdu.close()
            cube_hdu.close()

            fig.clear()

            if verbose:
                iter.update(chan + 1)
