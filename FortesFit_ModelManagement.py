from sys import exit
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
 
""" A module that handles model registration and management in FORTES-FIT  """

# ***********************************************************************************************

class FullModel:
	""" Representation of the full FORTES-FIT version of the SED model for a given model ID 
		This class is used for model management only. It does not have any evaluative functions.
		Model IDs are unique 2 digit positive integers
	"""
	
	
	def __init__(self,ModelID):
		""" Read in the FORTES-FIT version of the SED model for a given model ID 
			
			The model must be registered with FORTES-FIT in this incarnation of the package.
				
		"""
		
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
		self.shape_parameter_names = np.core.defchararray.decode(Model.attrs['ShapeParameterNames'])
		self.shape_parameters = {}
		for param in self.shape_parameter_names:
			self.shape_parameters.update({param:Model.attrs[param]})
		self.shape_parameter_mesh = np.meshgrid(*[self.shape_parameters[key] for key in self.shape_parameter_names],indexing='ij')

		# Read the redshift pivot values
		self.pivot_redshifts = Model.attrs['Pivot_Redshifts']
		
		# Store a list of current filters, take from the list of datasets from the first redshift group
		self.filterids = np.array([np.int(x) for x in list(Model['z00'])])
		
		# import the readmodule from file (Python 3.4 and newer only)
		readmodulename = 'readmodule_{0:02d}'.format(self.modelid)
		readmodspec = importlib.util.spec_from_file_location(readmodulename, ModelDir+readmodulename+'.py')
		self.readmodule = importlib.util.module_from_spec(readmodspec)
		readmodspec.loader.exec_module(self.readmodule)  #  Load the readmodule
		
		Model.close()  # Close the model file, since no evaluations are needed
						

# ***********************************************************************************************

class FitModel:
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
		self.shape_parameter_names = np.core.defchararray.decode(Model.attrs['ShapeParameterNames'])
		self.shape_parameters = {}
		for param in self.shape_parameter_names:
			self.shape_parameters.update({param:Model.attrs[param]})
		self.shape_parameter_mesh = np.meshgrid(*[self.shape_parameters[key] for key in self.shape_parameter_names],indexing='ij')

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
							
		self.model_photometry = {}
		# Create a dictionary with keys = filterIDs and containing a list of photometry cubes for each self.redshifts
		for FilterID in FilterIDs:
			cubelist = [np.squeeze(Model['z{0:02d}/{1:6d}'.format(index,FilterID)][:]) for index in zindex]
			self.model_photometry.update({FilterID:cubelist})
			
		# Store a list of filterids
		self.filterids = FilterIDs
		
		# import the readmodule from file (Python 3.4 and newer only)
		readmodulename = 'readmodule_{0:02d}'.format(self.modelid)
		readmodspec = importlib.util.spec_from_file_location(readmodulename, ModelDir+readmodulename+'.py')
		self.readmodule = importlib.util.module_from_spec(readmodspec)
		readmodspec.loader.exec_module(self.readmodule)  #  Load the readmodule

		Model.close()


	def evaluate(self,parameters,redshift,FilterID):
		""" Evaluate the observed model flux
			
			This is the crux of the class.
			Given a dictionary of parameter values (continuous, not just at pivots),
			a redshift and a certain FORTES-FIT filter ID, return the model flux.
			
			The redshift grid point closest to the input redshift is used for the evaluation.
							
		"""
		
		zindex = np.argmin(np.abs(self.redshifts-redshift)) # minimum of absolute difference between input redshift and grid
			
		# Read in two cubes that span the input redshift, using HDF5 group/dataset == redshift/filter
		hypercube = self.model_photometry[FilterID][zindex]

		# If any element of the hypercube is -inf == no model photometry, skip the interpolation and set flux = -np.inf
# 		if (np.any(np.isneginf(hypercube))):
# 			flux = -1.0*np.inf
# 		else:					
		#  Use scipy.RegularGridInterpolator multi-D interpolation, linear mode, 
		#    to get the model flux for the parameter value in each cube and at the intermediate redshift
		interpolation_axes = [self.shape_parameters[key] for key in self.shape_parameter_names]
		interpolation_function = interpolate.RegularGridInterpolator(tuple(interpolation_axes), hypercube,\
										method='linear', bounds_error=False, fill_value=-1.0*np.inf)
		interpolants = [parameters[key] for key in self.shape_parameter_names]
		flux = interpolation_function(np.array(interpolants))
		return flux + (parameters[self.scale_parameter_name] - self.scale_parameter_value)
	

# 	def evaluate(self,parameters,redshift,FilterID):
# 		""" Evaluate the observed model flux
# 			
# 			This is the crux of the class.
# 			Given a dictionary of parameter values (continuous, not just at pivots),
# 			a redshift and a certain FORTES-FIT filter ID, return the model flux.
# 			
# 			The modelphotometry cubes bracketing the redshift are identified and the flux
# 			corresponding to the parameters for the filter is interpolated. The final returned
# 			flux is a linear interpolation between the values at the two pivot redshifts.
# 							
# 		"""
# 		
# 		if (len(self.redshifts) == 2):
# 			# Single bracketing pair
# 			# If the provided redshift is not within the bracketed values, return invalid photometry
# 			if ((redshift < self.redshifts[0]) or (redshift > self.redshifts[1])):
# 				return -1.0*np.inf
# 			else:
# 				zindex = np.array([0,1])
# 		else:	
# 			# More than two redshift in model.
# 			# If the provided redshift is outside the model grid, return invalid photometry
# 			if ((redshift < self.redshifts[0]) or (redshift > self.redshifts[1])):
# 				return -1.0*np.inf
# 			else:
# 				# Identify either a single redshift node that is equal to the input redshift within the redshift tolerance
# 				#   or two redshifts nodes that bracket the input redshift
# 				delta_z = np.abs((self.redshifts/redshift) - 1) # absolute normalised difference between redshift and grid
# 				sortindex = np.argsort(delta_z) # sort it
# 				zindex = np.sort(sortindex[0:2]) # Take the two closest nodes in order as the bracketing pair. Sort them
# 			
# 		# Read in two cubes that span the input redshift, using HDF5 group/dataset == redshift/filter
# 		cube1 = self.model_photometry[FilterID][zindex[1]]
# 		cube2 = self.model_photometry[FilterID][zindex[0]]
# 		# Stack the cubes along a new axis to make a hypercube with redshift as the last dimension
# 		hypercube = np.stack((cube1,cube2),axis=-1)
# 
# 		# If any element of the hypercube is -inf == no model photometry, skip the interpolation and set flux = -np.inf
# # 		if (np.any(np.isneginf(hypercube))):
# # 			flux = -1.0*np.inf
# # 		else:					
# 		#  Use scipy.RegularGridInterpolator multi-D interpolation, linear mode, 
# 		#    to get the model flux for the parameter value in each cube and at the intermediate redshift
# 		interpolation_axes = [self.shape_parameters[key] for key in self.shape_parameter_names]
# 		interpolation_axes.append(np.array([self.redshifts[zindex[0]],self.redshifts[zindex[1]]]))
# 		interpolation_function = interpolate.RegularGridInterpolator(tuple(interpolation_axes), hypercube,\
# 										method='linear', bounds_error=False, fill_value=-1.0*np.inf)
# 		interpolants = [parameters[key] for key in self.shape_parameter_names]
# 		interpolants.append(redshift)
# 		flux = interpolation_function(np.array(interpolants))
# 		return flux + (parameters[self.scale_parameter_name] - self.scale_parameter_value)
# 		


	def get_pivot_sed(self,parameters,redshift):
		""" Return the model SED at the nearest pivot point in the original parameter space of the model
			
			Given a redshift and dictionary of parameter values from a smoothly varying range,
			return the full model SED in observed wavelength and flux units at the nearest pivot point
			in the parameter space of the model
			
			This routine imports the curated version of the readfunction for the model.
			Note: the parameter names have to be decoded from binary to unicode when using the readin functions.
				  Hence the use of the decode() method below.
			
		"""
		
		pivot_paramvals = []
		# For each parameter, find the pivot value closest to the given value
		for param in self.shape_parameter_names:
			# Index of minimum difference between given value and all pivots
			index = np.argmin(np.abs(self.shape_parameters[param] - parameters[param])) 
			pivot_paramvals.append(self.shape_parameters[param][index]) # Closest pivot value stored
		
		param_dict = dict(zip(self.shape_parameter_names,pivot_paramvals))
		param_dict.update({self.scale_parameter_name:parameters[self.scale_parameter_name]})
		
		return(self.readmodule.readin(param_dict,redshift))			
				

# ***********************************************************************************************



def register_model(read_module, parameters, scale_parameter_name, \
				   description='None', \
				   redshift_array=FortesFit_Settings.PivotRedshifts,filterids=[]):
	""" Register a model for use by FORTES-FIT 
	
	A model is a family of SEDs which are identified by a set of unique parameters. Models
	can have one single scale parameter and any number of shape parameters.
	
	Combination models which could have more than one scale parameter (i.e., hybrid models)
	must be broken down into independent submodels which are registered separately. This should
	be generally valid for additive SED model libraries.
	
	read_module: a string name for a module available either in the PYTHONPATH or the working
			directory which contains a function called 'readin'. This
			accepts a value for each parameter in the form of a dictionary, 
			as well as a redshift, and returns an SED consisting of a tuple of arrays: 
			wavelength (microns) and observed flux (erg/s/cm^2/micron). 
			Extra parameters can be passed using **kwargs (already read in model arrays, for e.g.), 
			but these should not be critical for the evaluation of a model.
	
	scale_parameter_name: the name of the single unique scale parameter, string
	parameters: all model parameters, including the scale parameter normalisation value, as a dictionary
	description: a string description of the model for informational purposes, string	
	redshift_array: An array of redshifts on which to evaluate the model. Default is the FortesFit Settings redshift array
	filterids: array-like, sequence of FortesFit filter ids to register for the model. 
			   If zero-length (default), all filters in the database are added.			

	returns the new model ID
	
	The data associated with the registered model, as well as a copy of the readin module, is stored
	in a model specific directory within ModelPhotometryDirectory with the same name as the model ID.
	
	"""
	
	try:
		ReadFunction = __import__(read_module)
	except ImportError:
		print('reader module not available')
		return []
	
	# Store all filters in the form of a list of filter class objects
	FilterList = [] # Initialise the list of filters
	if (len(filterids) == 0):
		# No specific filterids provided. Use the full complement of FORTESFIT filters
		FilterFileList = glob.glob(FortesFit_Settings.FilterDirectory+'*.fortesfilter.xml')
		# Catch situation where there are no current FORTESFIT filters
		if(len(FilterFileList) == 0):
			print('No filters found. Register your first filter!')
			raise ValueError('No filter files found')
			return []
		for filterfile in FilterFileList:
			filterid = np.int(os.path.basename(filterfile).split('.')[0])
			filter = FortesFit_Filters.FortesFit_Filter(filterid)
			FilterList.append(filter)
	else:
		# At least one specific filter has been supplied
		for filterid in filterids:
			filter = FortesFit_Filters.FortesFit_Filter(filterid)
			FilterList.append(filter)

	# Read existing models and create a list of model ID numbers
	OldModelFiles = glob.glob(FortesFit_Settings.ModelPhotometryDirectory+'Model??')
	if(len(OldModelFiles) == 0):
		print('Warning: No existing models found! Check settings.')
	OldIDs = []
	for OldFile in OldModelFiles:
		OldIDs.append(np.int(os.path.basename(OldFile)[-2:]))

	# Assign a random and unique 2 digit number for the new model.
	# This approach allows for a maximum of N=90 models, which should be sufficient.
	# When the number of models approaches N, this method of assignment becomes inefficient. 
	NewIDChecked = False
	while (not NewIDChecked):
		NewID = np.random.randint(10, high=99 + 1)
		if(NewID not in OldIDs):
			NewIDChecked = True					

	# Determine the number of model parameters and the number of pivot points per parameter
	ShapeParamNames  = sorted(parameters.keys()) # Parameters in alphabetical order of their names
	ShapeParamNames.remove(scale_parameter_name) # Exclude the scale parameter
	ShapeParamPoints = []
	for param in ShapeParamNames:
		ShapeParamPoints.append(len(parameters[param]))
	print('This model has {0:<d} shape parameters, sampled at a total of {1:<d} pivot points'.\
		   format(len(ShapeParamNames),np.int(np.prod(ShapeParamPoints))))
	ch = input('If this looks reasonable, continue by entering "y" : ')
	if (ch[0] != 'y'):
		print('Exitting without registering the model.')
		exit()
	

	# Create the destination directory for this new Model
	NewModelDirectory = FortesFit_Settings.ModelPhotometryDirectory+'Model{0:2d}/'.format(NewID)
	os.mkdir(NewModelDirectory)
	
	# Copy the read_module to the model directory and rename it using the model id
	ModelReadFileName = NewModelDirectory+'readmodule_{0:2d}.py'.format(NewID)
	shutil.copy(ReadFunction.__file__ , ModelReadFileName)
	# Create a symbolic link to the readin function from the FortesFit ModelReadFunctions directory for packaging
	LinkName = FortesFit_Settings.RoutinesDirectory+'model_readfunctions/readmodule_{0:2d}.py'.format(NewID)
	os.symlink(ModelReadFileName, LinkName)							

	# Create an HDF5 file for this model
	ModelFileName = NewModelDirectory+'{0:2d}.fortesmodel.hdf5'.format(NewID)
	ModelFile = h5py.File(ModelFileName, 'w')
	ModelFile.attrs.create("Description",description,dtype=np.dtype('S{0:3d}'.format(len(description))))
	ModelFile.attrs.create("ReadinFunction",read_module,dtype=np.dtype('S{0:3d}'.format(len(read_module))))
	
	# Populate a set of higher level attributes.
	# Names of the scale parameter and its value 
	ModelFile.attrs.create("ScaleParameterName",scale_parameter_name,dtype='S20')
	ModelFile.attrs.create("ScaleParameterValue",parameters[scale_parameter_name],dtype='f4')
	# Names of all the shape parameters, the pivot values for all shape parameters. 
	ModelFile.attrs.create("ShapeParameterNames",np.array(ShapeParamNames),dtype='S20')
	for param in ShapeParamNames:
		ModelFile.attrs.create(param,parameters[param],dtype='f4')

	# Pivot redshifts
	ModelFile.attrs.create("Pivot_Redshifts",redshift_array,dtype='f4')
	
	# Initialise empty arrays that will store the photometry and temporary dictionaries	
	
	# This cube stores the photometry at pivot shape parameters and all filters.
	# Its shape is the number of all shape parameters and the number of filters (Nsp1,Nsp1,...,Nfilt)
	# Info: even though the photometry cube is stored separately for each filter in the HDF5 file, 
	#       it is more efficient to evaluate all filters together for each parameter pivot combination.
	#       Therefore, the temporary storage is a supercube of this shape.
	tempparlist = ShapeParamPoints.copy()
	tempparlist.append(len(FilterList))
	modelphot = np.empty(tuple(tempparlist),dtype='f4')
	# This dictionary is updated at each pivot point and is an argument for the readin functions
	param_subset = dict.fromkeys(ShapeParamNames)
	param_subset.update({scale_parameter_name:parameters[scale_parameter_name]}) # Include the scale parameter

	# Initialise the wavelength array that is used for filter processing (from 100 Ang to 10mm) in log microns
	ObsWave = -2.0 + np.arange(1001)*(6.0/1000.0)

	# For each pivot redshift create a group under the name z??, where ?? is the running counter of the redshift
	#    array in dd form. The group will contain one dataset for each filter. The datasets are multi-dimensional
	#    cubes with one parameter per dimension. The dataset names are filterids.
	
	for iz in range(len(redshift_array)):
		
		# Name for the HDF5 group for this redshift
		GroupName = 'z{0:02d}'.format(iz)
		zpiv = ModelFile.create_group(GroupName)
	
				
		for iparam in range(np.prod(ShapeParamPoints)):
			# Loop over all shape parameters
			# unravel indices to access the pivot points of each parameter
			paramgen = np.unravel_index(iparam,ShapeParamPoints,order='C')
			# fill the temporary parameter dictionary for the call to the read_function
			for i,key in enumerate(ShapeParamNames):
				param_subset[key] = parameters[key][paramgen[i]]
			
			# Call the readin function
			sed = ReadFunction.readin(param_subset,redshift_array[iz])
			# Interpolate the model onto the default wavelength scale
			ObsFlux = np.interp(ObsWave,np.log10(sed['observed_wavelength']),np.log10(sed['observed_flux']),\
						left=-np.inf,right=-np.inf)
			# Initialise an index list that will be used to access the modelphotometry array
			photindex = list(paramgen)
			photindex.append(0) # This is a placeholder index for the filter
			for ifilter in range(len(FilterList)):
				# Loop over all filters
				photindex[-1] = ifilter # Replace the placeholder with the filter index
				modelphotsingle = FilterList[ifilter].apply({'wavelength':10**ObsWave,'flux':10**ObsFlux})
				# Catch cases of negative or badly formed values
				if (modelphotsingle > 0.0) and (np.isfinite(modelphotsingle)):
					modelphot[tuple(photindex)] = np.log10(modelphotsingle)
				else:
					modelphot[tuple(photindex)] = -np.inf

		
		# Write out the subcubes for each filter as a separate dataset in this group. FilterID is the dataset name
		SubCubes = np.split(modelphot,len(FilterList),axis=modelphot.ndim-1) # list of cubes split into different filters
		for ifilter in range(len(FilterList)):
			DatasetName = '{0:6d}'.format(FilterList[ifilter].filterid)	
			zpiv.create_dataset(DatasetName,data=SubCubes[ifilter])
		
		ModelFile.flush() # Flush the HDF5 file to disk
		# Write out a counter to tell the user how many redshift points have been processed
		print("{0:<3d} redshift points processed".format(iz+1),end="\r")
#		if (iz > 0) and (iz%50 == 0):
#			print('{0:>3d} redshift points processed'.format(iz))
			 
	ModelFile.close()
	
	# Update the model summary table
	summarize_models()
			
	return NewID
			

# ***********************************************************************************************

def add_filter_to_model(ModelID, FilterIDs):
	""" Add a registered FORTES-FIT filter to a registered FORTES-FIT model
	
		ModelID:  FORTES-FIT local id for an existing model. 
				  It must be already registered or an exception will be thrown.	
		FilterIDs: List of FORTES-FIT local ids for an existing filter. 
				   They must be already registered or an exception will be thrown.	
		
	"""

	# Access the HDF5 file for the model
	ModelDirectory    = FortesFit_Settings.ModelPhotometryDirectory+'Model{0:2d}/'.format(ModelID)
	ModelFileName     = ModelDirectory+'{0:2d}.fortesmodel.hdf5'.format(ModelID)
	try:
		ModelFile     = h5py.File(ModelFileName, 'r+')
		Model         = FullModel(ModelID)
		# import the readmodule
#		ReadModuleName    = 'fortesfit.model_readfunctions.readmodule_{0:02d}'.format(ModelID)
#		ReadModule        = importlib.import_module(ReadModuleName)
	except IOError:
		print('Model {0:2d} has not been registered!'.format(ModelID))
		return[]	
	
	# Read in the list of FORTES filters to add to the model
	# If the filter is already included in the model, skip its processing
	FilterList = []
	for filterid in FilterIDs:
		index, = np.where(Model.filterids == filterid)
		if(len(index) > 0):
			# Filter already exists in model
			continue
		try:
			Filter = FortesFit_Filters.FortesFit_Filter(filterid)
			FilterList.append(Filter)
		except IOError:
			print('add_filter_to_model: Filter initialisation failed.') 
			ModelFile.close()
			return[]
				
	
	# After checks, only proceed if there is still >0 filters to add to this model
	if(len(FilterList) == 0):
		# No filters to add
		print('No filter to add. They may already exist in the model.')
		ModelFile.close()
		return[]	

	# Determine the number of model parameters and the number of pivot points per parameter
	ShapeParamNames  = Model.shape_parameter_names
	ShapeParamPoints = []
	for param in ShapeParamNames:
		ShapeParamPoints.append(len(Model.shape_parameters[param]))

	# Initialise empty arrays that will store the photometry and temporary dictionaries	
	
	# This cube stores the photometry at pivot shape parameters and all filters.
	# Its shape is the number of all shape parameters and the number of filters (Nsp1,Nsp1,...,Nfilt)
	# Info: even though the photometry cube is stored separately for each filter in the HDF5 file, 
	#       it is more efficient to evaluate all filters together for each parameter pivot combination.
	#       Therefore, the temporary storage is a supercube of this shape.
	tempparlist = ShapeParamPoints.copy()
	tempparlist.append(len(FilterList))
	modelphot = np.empty(tuple(tempparlist),dtype='f4')
	# This dictionary is updated at each pivot point and is an argument for the readin function
	param_subset = dict.fromkeys(ShapeParamNames)
	param_subset.update({Model.scale_parameter_name:Model.scale_parameter_value}) # Include the scale parameter

	# Initialise the wavelength array that is used for filter processing (from 100 Ang to 10mm) in log microns
	ObsWave = -2.0 + np.arange(1001)*(6.0/1000.0)

	# For each pivot redshift access the existing group under the name z??, where ?? is the running counter of the redshift
	#    array in dd form. To this group, add one dataset for each new filter. The datasets are multi-dimensional
	#    cubes with one parameter per dimension. The dataset names are filterids.
	
	for iz in range(len(Model.pivot_redshifts)):
		
		# Name for the HDF5 group for this redshift
		GroupName = 'z{0:02d}'.format(iz)
		zpiv = ModelFile[GroupName]
	
				
		for iparam in range(np.prod(ShapeParamPoints)):
			# Loop over all shape parameters
			# unravel indices to access the pivot points of each parameter
			paramgen = np.unravel_index(iparam,ShapeParamPoints,order='C')
			# fill the temporary parameter dictionary for the call to the read_function
			for i,key in enumerate(ShapeParamNames):
				param_subset[key] = Model.shape_parameters[key][paramgen[i]]
			
			# Call the readin function
			sed = Model.readmodule.readin(param_subset,Model.pivot_redshifts[iz])
			# Interpolate the model onto the default wavelength scale
			ObsFlux = np.interp(ObsWave,np.log10(sed['observed_wavelength']),np.log10(sed['observed_flux']),\
						left=-np.inf,right=-np.inf)
			# Initialise an index list that will be used to access the modelphotometry array
			photindex = list(paramgen)
			photindex.append(0) # This is a placeholder index for the filter
			for ifilter in range(len(FilterList)):
				# Loop over all filters
				photindex[-1] = ifilter # Replace the placeholder with the filter index
				modelphotsingle = FilterList[ifilter].apply({'wavelength':10**ObsWave,'flux':10**ObsFlux})
				# Catch cases of negative or badly formed values
				if (modelphotsingle > 0.0) and (np.isfinite(modelphotsingle)):
					modelphot[tuple(photindex)] = np.log10(modelphotsingle)
				else:
					modelphot[tuple(photindex)] = -np.inf

# 			for ifilter in range(len(FilterList)):
# 				# Loop over all filters
# 				modelphotsingle = FilterList[ifilter].apply({'wavelength':10**ObsWave,'flux':10**ObsFlux})
# 				# Catch cases of negative or badly formed values
# 				if (modelphotsingle > 0.0) and (np.isfinite(modelphotsingle)):
# 					modelphot[(paramgen,ifilter)] = np.log10(modelphotsingle)
# 				else:
# 					modelphot[(paramgen,ifilter)] = -np.inf

		
		# Write out the subcubes for each filter as a separate dataset in this group. FilterID is the dataset name
		SubCubes = np.split(modelphot,len(FilterList),axis=modelphot.ndim-1) # list of cubes split into different filters
		for ifilter in range(len(FilterList)):
			DatasetName = '{0:6d}'.format(FilterList[ifilter].filterid)	
			zpiv.create_dataset(DatasetName,data=SubCubes[ifilter])
		
		ModelFile.flush() # Flush the HDF5 file to disk
		print(iz)
			 
	ModelFile.close()			
	return []


# ***********************************************************************************************

def summarize_models():
	""" Create a summary in the local directory of all models available to FORTES-FIT 
	
	The summary is written out as a simple fixed-format multi-column ascii file.	
	
	"""
	
	summary_table = Table(names=('modelid','description','number of shape parameters','scale parameter'),
						  dtype=('i2',np.dtype(object),'i4',np.dtype(object)))
	
	# Read existing models and create a list of model ID numbers
	ModelDirList = glob.glob(FortesFit_Settings.ModelPhotometryDirectory+'Model??')
	
	if(len(ModelDirList) == 0):
		print('No existing models found.')
		return []
	for ModelDir in ModelDirList:
		ModelID = np.int(os.path.basename(ModelDir)[-2:])
		print('Processing model: {0:6d}'.format(ModelID))
		
		ModelFile = ModelDir+'/{0:2d}.fortesmodel.hdf5'.format(ModelID)
		Model = h5py.File(ModelFile,'r')
		ModelDesc = Model.attrs['Description']
		NParams = len(Model.attrs['ShapeParameterNames'])
		ScaleParameter = Model.attrs['ScaleParameterName']
		Model.close()
		
		summary_table.add_row([ModelID,ModelDesc,NParams,ScaleParameter])
	
	summary_table.sort('modelid')
	summary_table.write(FortesFit_Settings.ModelPhotometryDirectory+'FortesFit_models_summary.ascii',
						format='ascii.fixed_width_two_line',overwrite=True)
	
# ***********************************************************************************************
	
def delete_model(ModelID):
	""" Delete the model from the FortesFit database with the supplied modelid. 
	
	The model photometry, the reader function, and the link to the reader function are all deleted.
	The model summary is updated.
	
	ModelID: FORTES-FIT local id for an existing model. 
			 It must be already registered or an exception will be thrown.	
	
	"""
	
	# Get the model directory and the reader function link for the model
	ModelDirectory = FortesFit_Settings.ModelPhotometryDirectory+'Model{0:2d}/'.format(ModelID)
	LinkName = FortesFit_Settings.RoutinesDirectory+'model_readfunctions/readmodule_{0:2d}.py'.format(ModelID)
	
	# First delete the soft link to the reader function from the fortesfit source directory
	os.remove(LinkName)	
	# Second, remove the directory tree corresponding to the Model
	shutil.rmtree(ModelDirectory)	
		
	print('Model{0:2d} has been deleted'.format(ModelID))
	
	# Update the model summary table
	summarize_models()
