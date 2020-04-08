import sys
import os
import shutil
import glob

import numpy as np

import fortesfit
# Update the python path temporarily to allow the load of the test_model.py module
fortesfitpath = os.path.dirname(fortesfit.__file__)
sys.path.insert(0,fortesfitpath)

from fortesfit import FortesFit_Settings
from fortesfit import FortesFit_Filters
from fortesfit import FortesFit_ModelManagement

""" The script sets up the Filters and Models directory for the local version of FortesFit

"""

# Use OS library tools to set up FortesFit_Filters/ and FortesFit_ModelPhotometry/ directories under $FORTESFITPATH
if os.path.isdir(FortesFit_Settings.FilterDirectory):
	print('Warning: The FortesFit_Filters directory exists already. Are you sure this is a new installation of FortesFit?')
else:
	os.mkdir(FortesFit_Settings.FilterDirectory)
if os.path.isdir(FortesFit_Settings.ModelPhotometryDirectory):
	print('Warning: The FortesFit_ModelPhotometry directory exists already. Are you sure this is a new installation of FortesFit?')
else:
	os.mkdir(FortesFit_Settings.ModelPhotometryDirectory)

# Check to see if there are any filters in the FortesFit_Filters/ directory
if len(glob.glob(FortesFit_Settings.FilterDirectory+'*')) > 0:
	print('Error: The FortesFit_Filters directory is not empty. This routine must only be run at the start of an installation.')
	sys.exit() 
else:
# Initialise a single simple filter that will be used in primary tests
# This is a flat filter from 0.9-1.1 microns
	wave = np.arange(101)*(1.1-0.9)/100.0 + 0.9
	tput = np.full(len(wave),1.0)
	filterid = FortesFit_Filters.register_filter(wave,tput,reference='Installer',description='1 micron test filter')
	# Rename this filter file to 100000.fortesfilter.xml, the only filter with this first number
	os.rename(FortesFit_Settings.FilterDirectory+'{0:6d}.fortesfilter.xml'.format(filterid),\
			  FortesFit_Settings.FilterDirectory+'100000.fortesfilter.xml')


# Check to see if there are any models in the FortesFit_ModelPhotometry/ directory
if len(glob.glob(FortesFit_Settings.ModelPhotometryDirectory+'*')) > 0:
	print('Error: The FortesFit_ModelPhotometry directory is not empty. This routine must only be run at the start of an installation.')
	sys.exit() 
else:
# Initialise a single simple model that will be used in primary tests
# This is a flat spectrum in nuFnu with monochromatic luminosity as the only parameter
	parameters = {'MonochromLuminosity':45.0} # Only a single parameter, the scale parameter
	modelid = FortesFit_ModelManagement.register_model('test_model',parameters,'MonochromLuminosity',\
											           description='Installer test flat SED model',\
													   filterids=[100000]) # Only has the test 1 micron filter
	sys.path.pop(0) # Remove the temporary change to python path. This can be removed in the final form.
	# Rename this model file to 10, the only model with this number
	if modelid != 10:
		# Catch the rare case where the modelid was already assigned to 10
		os.mkdir(FortesFit_Settings.ModelPhotometryDirectory+'Model10/')
		shutil.copy(FortesFit_Settings.ModelPhotometryDirectory+'Model{0:2d}/{1:2d}.fortesmodel.hdf5'.format(modelid,modelid),\
			  FortesFit_Settings.ModelPhotometryDirectory+'Model10/10.fortesmodel.hdf5')
		shutil.copy(FortesFit_Settings.ModelPhotometryDirectory+'Model{0:2d}/readmodule_{1:2d}.py'.format(modelid,modelid),\
			  FortesFit_Settings.ModelPhotometryDirectory+'Model10/readmodule_10.py')
		shutil.rmtree(FortesFit_Settings.ModelPhotometryDirectory+'Model{0:2d}'.format(modelid))
