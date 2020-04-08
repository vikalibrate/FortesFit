import sys
import os
import shutil
import glob
import importlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from astropy.table import Table
import h5py
from fortesfit import FortesFit_Settings
from fortesfit import FortesFit_Filters

""" Development versions of various FortesFit routines and classes
	
	This file stores test versions with altered algorithms. They are kept here for use in the future,
	but have shown worse performance than the algorithms in the main development version under the
	current test conditions. However, they may become more useful later in case they scale better
	than the main development algorithms

"""


# ***********************************************************************************************

class FitModel_test:
	""" Representation of the FORTES-FIT version of the SED model for a given model ID, 
		   redshift, set of filters
		This class is used for fitting only. It does not have any registrative functions.
		Model IDs are unique 2 digit positive integers				
	"""
	
	
	def __init__(self,ModelID,redshift,FilterIDs):
		""" Read in the FORTES-FIT version of the SED model for a given model ID, redshift, set of filters
			
			ModelID: are unique 2 digit positive integer that identifies the model	 
			         The model must be registered with FORTES-FIT, or this step will exit throwing an exception.
			redshift: Either a scalar redshift value or a two-element list-like with lower and upper redshifts in a range
			FilterIDs: A list of filter IDs that will be read from the model database
							
		"""
		
		# Store a list of filterids
		self.filterids = FilterIDs

		self.modelid = ModelID
		ModelDir = FortesFit_Settings.ModelPhotometryDirectory+'Model{0:2d}/'.format(ModelID)
		ModelFile = ModelDir+'{0:2d}.fortesmodel.hdf5'.format(ModelID)
		Model = h5py.File(ModelFile,'r')

		# Some basic information about the model
		# h5py writes string attributes in byte format, so decode when reading them
		self.description = Model.attrs['Description'].decode()
		self.scale_parameter_name = Model.attrs['ScaleParameterName'].decode()
		self.scale_parameter_value = Model.attrs['ScaleParameterValue']

		# Read the shape parameters and their pivot values
		# h5py writes string attributes in byte format, so decode when reading them
		# The order in which the shape_parameters are stored here sets the array shapes
		#  of the dependency and model photometry cubes
		self.shape_parameter_names = np.core.defchararray.decode(Model.attrs['ShapeParameterNames'])
		self.shape_parameters = {}
		for param in self.shape_parameter_names:
			self.shape_parameters.update({param:Model.attrs[param]})

		# Read the dependency cubes
		self.dependency_names = list(Model['Dependencies'])
		self.dependencies = {}
		for dependency in self.dependency_names:
			self.dependencies.update({dependency:Model['Dependencies'][dependency][()]})

		# Read the redshift pivot values from the model database
		pivot_redshifts = Model.attrs['Pivot_Redshifts']
		redshift_tolerance = 0.01 # A tolerance for redshifts to snap to the grid is delta_z/z = 1%
		self.redshift_tolerance = redshift_tolerance # This is used in the model photometry evaluation

		# Identify the range of pivot redshifts that bracket or span the input redshift(s)
		if(np.isscalar(redshift)):
			# A single redshift value
			# Determine if it lies outside the redshifts of the model
			delta_z0 = pivot_redshifts[0]/redshift - 1
			delta_zf = 1-pivot_redshifts[-1]/redshift
			if ((delta_z0 > redshift_tolerance) or (delta_zf > redshift_tolerance)):
				raise ValueError('Redshift is outside range covered by model')

			# Use the absolute normalised difference between provided redshift and the redshift grid
			# to find a bracketing pair of redshift nodes
			delta_z = np.abs((pivot_redshifts/redshift) - 1) # absolute normalised difference between redshift and grid
			sortindex = np.argsort(delta_z) # sort it
			zindex = np.sort(sortindex[0:2]) # Take the two closest nodes in order as the bracketing pair. Sort them
		elif(len(redshift) == 2):
			# Two redshift values. Save pivot redshifts that span them.	
			delta_z0 = pivot_redshifts[0]/redshift - 1
			delta_zf = 1-pivot_redshifts[-1]/redshift
			if ((delta_z0[0] > redshift_tolerance) or (delta_zf[0] > redshift_tolerance)):
				raise ValueError('Lower redshift outside range covered by model')
			if ((delta_z0[1] > redshift_tolerance) or (delta_zf[1] > redshift_tolerance)):
				raise ValueError('Upper redshift outside range covered by model')
			zindex, = np.where((pivot_redshifts >= redshift[0]) & (pivot_redshifts <= redshift[1]))
		else:
			# Incompatible redshift specification
			raise ValueError('Redshift specified incorrectly')

		self.redshifts = pivot_redshifts[zindex]

		# Set up the tuple that will be used with the SciPy regular grid interpolator
		parameter_list = [self.shape_parameters[key] for key in self.shape_parameter_names]
		parameter_list.append(pivot_redshifts[zindex])
		interpolation_tuple = tuple(parameter_list)
							
		self.model_photometry_interpolators = {}
		# Create a dictionary with keys = filterIDs and containing a list of photometry cubes for each self.redshifts
		for FilterID in FilterIDs:
			# Read in cubes that span the input redshift array, using HDF5 group/dataset == redshift/filter
			cubelist = [np.squeeze(Model['z{0:02d}/{1:6d}'.format(index,FilterID)][:]) for index in zindex]
			# Stack the list of cubes in redshift space with redshift as the last axis
			hypercube = np.stack(cubelist,axis=-1)
			# Initialise a multidimensional interpolation function using the hypercube 
			interpolation_function = interpolate.RegularGridInterpolator(interpolation_tuple, hypercube,\
									     method='linear', bounds_error=False, fill_value=-1.0*np.inf)
			# Store a dictionary of interpolation functions to be used in the evaluation of model photometry
			self.model_photometry_interpolators.update({FilterID:interpolation_function})

		Model.close()


	def evaluate(self,parameters,redshift,FilterID):
		""" Evaluate the observed model flux
			
			This is the crux of the class.
			Given a dictionary of parameter values (continuous, not just at pivots),
			a redshift and a certain FORTES-FIT filter ID, return the model flux.
			
			The redshift grid point closest to the input redshift is used for the evaluation.
							
		"""
					
		interpolants = [parameters[key] for key in self.shape_parameter_names]
		interpolants.append(redshift)
		flux = self.model_photometry_interpolators[FilterID](np.array(interpolants)) + (parameters[self.scale_parameter_name] - self.scale_parameter_value)
		if np.isfinite(flux):
			return flux
		else:
			return -1.0*np.inf 
	
# ***********************************************************************************************

