
import astropy.units as u
from astropy.io import fits


def generate_header(pixel_scale, spec_scale, beamfwhm, imshape):

    header = {'CDELT1': -(pixel_scale).to(u.deg).value,
              'CDELT2': (pixel_scale).to(u.deg).value,
              'BMAJ': beamfwhm.to(u.deg).value,
              'BMIN': beamfwhm.to(u.deg).value,
              'BPA': 0.0,
              'CRPIX1': imshape[0] / 2.,
              'CRPIX2': imshape[1] / 2.,
              'CRVAL1': 0.0,
              'CRVAL2': 0.0,
              'CTYPE1': 'GLON-CAR',
              'CTYPE2': 'GLAT-CAR',
              'CUNIT1': 'deg',
              'CUNIT2': 'deg',
              'CRVAL3': 0.0,
              'CUNIT3': spec_scale.unit.to_string(),
              'CDELT3': spec_scale.value,
              'CRPIX3': 1,
              'CTYPE3': 'VRAD',
              'BUNIT': 'K',
              }

    return fits.Header(header)


def generate_hdu(data, pixel_scale, spec_scale, beamfwhm):

    imshape = data.shape[1:]

    header = generate_header(pixel_scale, spec_scale, beamfwhm, imshape)

    return fits.PrimaryHDU(data, header)
