
import astropy.units as u
from galaxies import Galaxy
from astropy.coordinates import Angle, SkyCoord


def update_galaxy_params(gal, param_table):
    '''
    Use the fit values rather than the hard-coded values in galaxies.
    '''

    assert isinstance(gal, Galaxy)

    gal.inclination = Angle(param_table["inc"][0] * u.deg)
    gal.position_angle = Angle(param_table["PA"][0] * u.deg)
    gal.vsys = param_table["Vsys"][0] * u.km / u.s

    gal.center_position = SkyCoord(param_table["RAcent"],
                                   param_table["Deccent"], unit=(u.deg, u.deg),
                                   frame='fk5')
