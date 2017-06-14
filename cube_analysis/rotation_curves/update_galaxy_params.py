
import astropy.units as u
from galaxies import Galaxy
from astropy.coordinates import Angle, SkyCoord


def update_galaxy_params(gal, param_table):
    '''
    Use the fit values rather than the hard-coded values in galaxies.
    '''

    assert isinstance(gal, Galaxy)

    gal.inclination = Angle(param_table["inc"] * u.deg)[0]
    gal.position_angle = Angle(param_table["PA"] * u.deg)[0]
    gal.vsys = (param_table["Vsys"] * u.km / u.s)[0]

    gal.center_position = SkyCoord(param_table["RAcent"],
                                   param_table["Deccent"], unit=(u.deg, u.deg),
                                   frame='fk5')
