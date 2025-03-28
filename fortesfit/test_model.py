from sys import exit
import numpy as np
from scipy import integrate, constants, interpolate
import matplotlib.pyplot as plt
from fortesfit.FortesFit_Settings import FortesFit_Cosmology as cosmo

""" FortesFit compliant readin module for the FortesFit main test model.
	This is a flat SED in nuFnu with a variable monochromatic luminosity.
"""

# No readtemplates function

# ***********************************************************************************************

def		readin(parameters, redshift, templates=None):
	""" Given a specific parameter set and a redshift, return a flat SED in nuFnu 

		The parameters are :
			MonochromLuminosity: the monochromatic luminosity in log erg/s/cm^2

	"""
	
	wave_basic = 10**(-1.0 + np.arange(1001)*(4.0/1000.0))  #  base wavelengths are in microns
	wave = wave_basic*(1.0+redshift)  # Observed wavelength 

	template_orig = np.full(1001,1e45)
	scale_factor = 10**(parameters['MonochromLuminosity'] - 45.0)
	
	lfactor = 4.0*np.pi*(cosmo.luminosity_distance(redshift).value*3.0856e24)**2.0
	observedflux = (template_orig*scale_factor/lfactor)/wave

	sed = {'observed_wavelength':wave,'observed_flux':observedflux}
	
	return sed
