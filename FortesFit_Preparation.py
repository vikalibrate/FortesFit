import sys
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz

import matplotlib.pyplot as plt

from astropy import units as u
from astropy.table import Table

import h5py
import emcee
from fortesfit import FortesFit_Settings
from fortesfit.FortesFit_Filters import FortesFit_Filter
from fortesfit.FortesFit_ModelManagement import FitModel

""" A module with classes and functions that organise the process of setting up a fitting run  """


# ***********************************************************************************************
class PriorDistribution:
	""" This class handles a grid that specifies a prior distribution for use with FortesFit routines

	"""
	
	def __init__(self,prior_grid,parameter_range=None,Normalize=True):
		""" Initialise the FortesFit representation of a prior distribution
			
			prior_grid: Two options:
						- For a fixed parameter, a single scalar equal to the fixed value of the parameter.
						- Otherwise, a 2xN array, the first row is a grid in the parameter, 
						  the second row is the probability density function on the grid.
						  The array does not need to be sorted, and the pdf does not have to be normalised
			parameter_range: optional 2 element list-like, with [low, high] specifying the range in which
						the PDF is valid, in case it is smaller than the range of the grid.			
			Normalize: normalise the PDF to integrate to unity.
													
		"""
	
		ngrid = 1001
			
		if np.size(prior_grid) == 1:
			# Single value for fixed parameter
			self.fixed = True
			self.characteristic = prior_grid
			self.prior_grid     = np.array([prior_grid]) # Store the single value as an array

		else:
			# Check to see if this is a 2xN array
			if prior_grid.shape[0] != 2 :
				# this is the wrong shape
				raise TypeError('Input array has the wrong shape')

			# Sort the prior grid low to high in the parameter
			sortindex = np.argsort(prior_grid[0,:])

			if parameter_range is None:
				# Default, no parameter range provided
				xrange = [prior_grid[0,sortindex[0]],prior_grid[0,sortindex[-1]]]
			else:
				if parameter_range[0] >= parameter_range[1]:
					raise ValueError('User-defined parameter range must be [low, high], low < high')
				xrange = parameter_range

			# Set up a cubic spline interpolator
			interprior = interp1d(prior_grid[0,sortindex],prior_grid[1,sortindex],\
								  kind='cubic',bounds_error=False,fill_value=0.0,assume_sorted=True)

			# ngrid point sampled prior
			prior_x = xrange[0] + np.arange(ngrid)*(xrange[1]-xrange[0])/(ngrid-1)
			prior_y = interprior(prior_x) # interpolate the distribution
			if Normalize:
				norm = trapz(prior_y,prior_x)
				prior_y = prior_y/norm  #  Normalise the prior distribution to integrate to unity over its range
				
			self.fixed = False
			self.characteristic = np.sum(prior_x*prior_y)/np.sum(prior_y)
			self.prior_grid     = np.stack((prior_x,prior_y),axis=0)
		

	def	draw_prior(self,ndraws=1):
		""" Draw a given number of random values from the prior distribution.

			The algorithm uses the inverse cumulative distribution function derived from the gridded PDF
			mapped to a uniform set of samples over the range of the parameter.
			
			ndraws: number of draws to make from the prior distribution
			
			returns a numpy 1D array with the drawn values of length ndraws 		

		"""
		
		ngrid = self.prior_grid.shape[1]	
	
		# Get the CDF by integrating the well-sampled PDF at each point on the grid
		cdf = np.empty(ngrid)
		for igrid in range(ngrid):
 		   cdf[igrid] = trapz(self.prior_grid[1,0:igrid],self.prior_grid[0,0:igrid])		
		
		unifvals = np.random.random(size=ndraws)
		draws = np.interp(unifvals,cdf,self.prior_grid[0,:])

		return draws
	

# ***********************************************************************************************
class CollectData:
	""" This class summarises the information about the data of a single object to be fit with
		FortesFit.
	"""
	
	def __init__(self,filterids,fluxes,fluxerrs,fluxlimits,ID,redshift):
		""" Initialise the FortesFit data representation for a user-provided object
			
			filterids: list-like, the FortesFit ids of filters. Any order, but must match that of photometry lists.
			fluxes: list-like, the fluxes in each band corresponding to filters in argument 'filterids'. 
					Each element must be an Astropy Quantity instant. Fluxes == NaN are disregarded.
			fluxerrs: list-like, the 1sigma flux errors in each band corresponding to filters in argument 'filterids'. 
					  Each element must be an Astropy Quantity instant. Fluxerrs < 0.0 are treated as upper limits,
					  except if the corresponding flux is undefined
			fluxlimits: list-like, the associated limit (1sigma, 2 sigma, etc.) for the measurement in the band.
			
			ID: An ID string for this object. This will be used to generate the output file.
			redshift: An array of shape (2,N) or (1,). 
					  If single-element, this is the fixed redshift of the object.							
					  If 2xN, the first row is a grid of redshift, the second the probability of the redshift.
					  This is used to represent p(z) from photometric redshift distributions.
										
		"""
	
		self.id = ID  # store the ID
		
		# Process and store redshift information
		# Scope here for quality checks on redshift array information
		if np.size(redshift) == 1:
			# single redshift
			if redshift <= 0.0: raise ValueError('Redshift cannot be <= 0')
			self.redshift = PriorDistribution(redshift)
		else:
			if np.count_nonzero(redshift <= 0.0): raise ValueError('Redshifts cannot be <= 0')
			self.redshift = PriorDistribution(redshift)

		# Process and store the filters
		Filters = []
		FilterWaves = []
		for id in filterids:
			tempfilt = FortesFit_Filter(id)
			Filters.append(tempfilt)
			FilterWaves.append(tempfilt.pivot_wavelength)

		Filters = np.array(Filters)
		FilterWaves = np.array(FilterWaves)		
		
		# Flag and remove bands with no photometry
		index, = np.where(np.isfinite(fluxes)) # Disregard invalid fluxes
		n_validphot = len(index)
		fluxes      = fluxes[index]
		fluxerrs    = fluxerrs[index]
		fluxlimits  = fluxlimits[index]
		Filters     = Filters[index]
		FilterWaves = FilterWaves[index]

		# Process and store the photometry
		fitflux     = np.full(n_validphot,0.0,dtype='f4')
		fitflux_err = np.full(n_validphot,0.0,dtype='f4')
		# convert fluxes to erg/s/cm^2/micron
		for ifilt in range(n_validphot):
			# Use astropy equivalencies to convert to preferred units
			flux  = fluxes[ifilt].to(u.erg/u.s/u.cm**2/u.micron,equivalencies=u.spectral_density(FilterWaves[ifilt]*u.micron))
			eflux = fluxerrs[ifilt].to(u.erg/u.s/u.cm**2/u.micron,equivalencies=u.spectral_density(FilterWaves[ifilt]*u.micron))
			if(flux > 0.0) and (eflux > 0.0):
				# Detection
				fitflux[ifilt]     = flux.value
				fitflux_err[ifilt] = eflux.value
			if(flux > 0.0) and (eflux < 0.0):
				# Upper Limit
				fitflux[ifilt]     = flux.value
				fitflux_err[ifilt] = -1.0*fluxlimits[ifilt] # Use the correct limit for this filter

		# Filters will be stored after sorting by pivot wavelength.
		sortindex = np.argsort(FilterWaves)
		self.filters = Filters[sortindex]
		self.pivot_wavelengths = FilterWaves[sortindex]
		self.fluxes = fitflux[sortindex]
		self.flux_errors = fitflux_err[sortindex]

# ***********************************************************************************************

class CollectModel:
	""" This class summarises the information about the model and priors of a single object to be fit with
		FortesFit.
	"""
	
	def __init__(self,modellist,priordists,datacollection):
		""" Initialise the FortesFit model representation for a user-provided object
			
			modellist: list-like, the FortesFit ids of models in arbitrary order
			priordists: list-like, matching the order of modellist. 
						Each element is a dictionary with keys as the parameter names of the respective model. 
						The items of the dictionary are probability distribution representations of the prior. 
						This can be of three forms:
						 - If a single scalar value, the associated parameter is assumed to be fixed.
						 - If a 2xN shape array, the first row is the grid of the parameter, 
							the second row is the probability density of the grid values.
						 - An instance of the Scipy RVS_continuous class with a built-in PDF function.
						If a shape_parameter is missing a prior representation, it will be assumed to be uniform
						across the full range of the parameter in the model.
						Scale parameters, however, must have a prior defined.
			datacollection: An instance of the CollectData class defined in this module, which contains
						information about the filters, redshifts and photometry to be fit.
										
		"""
			
		# Determine the redshift range, or fitting redshift if a single value
		if datacollection.redshift.fixed:
			redshift_range = datacollection.redshift.characteristic
		else:
			redshift_array = datacollection.redshift.prior_grid[0,:]
			redshift_range = [redshift_array[0],redshift_array[-1]]

		# Gather the filterids
		filterids = [filter.filterid for filter in datacollection.filters]

		# Collate the models and process the prior distributions
		Models     = []
		NParams    = []
		PriorDists = []

		for imodel in range(len(modellist)):

			# Read in a model to a FortesFit FitModel instance	
			model = FitModel(modellist[imodel],redshift_range,filterids)
			priordist = priordists[imodel]
			
			# Process the prior distributions for each parameter and save
			priordict = {}

			# Loop through all the parameters of the model in order, scale parameter, then shape parameters
			# First the scale parameter
			parameter_name = model.scale_parameter_name
			if not(parameter_name in priordist.keys()):
				# No user-supplied prior, raise exception. Scale parameters need priors.
				error_message = parameter_name+' in Model{0:2d} is a scale parameter and needs a prior'.format(model.modelid)
				raise ValueError(error_message)

			prior = priordist[parameter_name]
			if np.size(prior) == 1:
				if type(prior).__name__ == 'rv_frozen':
					# A frozen Scipy rvs_continuous instance. Will raise its own exception if it isn't properly set.
					# Consider a range where the CDF goes from 1e-10 to 1 - 1e-10
					xrange = [prior.ppf(1e-10),prior.ppf(1.0-1e-10)]
					prior_x = xrange[0] + np.arange(101)*(xrange[1]-xrange[0])/100.0
					prior_y = prior.pdf(prior_x) # get the distribution using the PDF function
					priordict.update({parameter_name:PriorDistribution(np.stack((prior_x,prior_y),axis=0),Normalize=False)})
				else:					
					# single value of prior, not suitable for a scale parameter
					error_message = parameter_name+' in Model{0:2d}'.format(model.modelid)+': Scale parameter needs a non-singular prior'
					raise ValueError(error_message)
			else:
				if type(prior).__name__ == 'ndarray':
					priordict.update({parameter_name:PriorDistribution(prior)})					
				else:
					# Unclassifiable prior class. Raise exception.
					error_message = parameter_name+' in Model{0:2d}'.format(model.modelid)+': Incorrect prior format'
					raise ValueError(error_message)

			# Loop through the shape parameters
			for parameter_name in model.shape_parameter_names:

				if parameter_name in priordist.keys():
					# A prior has been provided by the user
					prior = priordists[imodel][parameter_name]				
					if np.size(prior) == 1:
						if type(prior).__name__ == 'rv_frozen':
							# A frozen Scipy rvs_continuous instance. Will raise its own exception if it isn't properly set.
							# Consider a range where the CDF goes from 1e-10 to 1 - 1e-10
							xrange = [prior.ppf(1e-10),prior.ppf(1.0-1e-10)]
							prior_x = xrange[0] + np.arange(101)*(xrange[1]-xrange[0])/100.0
							prior_y = prior.pdf(prior_x) # get the distribution using the PDF function
							priordict.update({parameter_name:PriorDistribution(np.stack((prior_x,prior_y),axis=0),Normalize=False)})
						else:					
							# single value of prior
							# If it lies within the upper and lower boundaries of the parameter range, store it
							xrange = [np.min(model.shape_parameters[parameter_name]),np.max(model.shape_parameters[parameter_name])]
							if (prior >= xrange[0]) & (prior <= xrange[1]):
								priordict.update({parameter_name:PriorDistribution(prior)})
							else:
								# the prior value is outside the model range, raise exception						 
								error_message = parameter_name+' in Model{0:2d}'.format(model.modelid)+': Singular value outside range.'
								raise ValueError(error_message)
					else:
						if type(prior).__name__ == 'ndarray':
							priordict.update({parameter_name:PriorDistribution(prior)})					
						else:
							# Unclassifiable prior class. Raise exception.
							error_message = parameter_name+' in Model{0:2d}'.format(model.modelid)+': Incorrect prior format'
							raise ValueError(error_message)
				else:
					# No prior provided. Take it to be uniform across the full range of the parameter
					xrange = [np.min(model.shape_parameters[parameter_name]),np.max(model.shape_parameters[parameter_name])]
					prior_x = xrange[0] + np.arange(101)*(xrange[1]-xrange[0])/100.0
					prior_y = np.full(101,1.0/(xrange[1]-xrange[0]))
					priordict.update({parameter_name:PriorDistribution(np.stack((prior_x,prior_y),axis=0))})
				

			n_params = len(priordict) # The number of parameters, which is handy for fitting routines
			
			Models.append(model)
			NParams.append(n_params)
			PriorDists.append(priordict)
		
		self.models = Models
		self.number_of_parameters = NParams
		self.priors = PriorDists

		# Compile a list of parameter references, including redshift. 
		# This order will be used for all further calls to likelihood, prior and sampling functions.
		# Each element is a tuple of (parameter name, model instance, prior instance, vary flag).
		# Redshift is included as the first item on the list, with a dummy model reference.
	
		parameter_reference = []
		# Redshift first
		if datacollection.redshift.fixed:
			varyflag = False
		else: varyflag = True
		parameter_reference.append(('Redshift',None,datacollection.redshift,varyflag))
		for imodel in range(len(Models)):
			for param in sorted(PriorDists[imodel].keys()):
				if PriorDists[imodel][param].fixed:
					varyflag = False
				else: varyflag = True
				parameter_reference.append((param,imodel,PriorDists[imodel][param],varyflag))
		
		self.parameter_reference = parameter_reference

# ***********************************************************************************************

def prepare_output_file(datacollection,modelcollection,OutputPath=None,description='',fittype='emcee'):
	""" This function prepares an HDF5 file to store the outputs of the SED fitting.
		
		datacollection: An FortesFit CollectData instance. See FortesFit_Preparation for details.
		modelcollection: An FortesFit CollectModel instance. See FortesFit_Preparation for details.
		OutputPath: The full path of an existing directory to store the file. Default is current working directory.
		description: An optional string to describe the fit. Defaults to the galaxy ID.
		fittype: The fitting routine for which this file will store the output. Used to initialise the storage group.

		returns the file reference of an open HDF5 file. The file should be closed after use.

	"""
	if OutputPath is None:
		# Get the current directory path if no user supplied one
		OutputPath = os.getcwd()
	
	if OutputPath[-1] != '/': OutputPath = OutputPath+'/' 
	
	GalaxyName = datacollection.id	
	FitFileName = GalaxyName+'.FortesFit_output.hdf5'  # Initialise the output file of the fit
	FitFile = h5py.File(OutputPath+FitFileName, 'w')

	FitFile.attrs.create("Object Name",GalaxyName,dtype=np.dtype('S{0:3d}'.format(len(GalaxyName))))
	if len(description) == 0:
		description = GalaxyName+ ' with FortesFit + '+fittype
	FitFile.attrs.create("Description",description,dtype=np.dtype('S{0:3d}'.format(len(description))))
			
	FitFile.attrs.create("Redshift",datacollection.redshift.prior_grid) # Store the redshift used for the fit

	# Store the photometry in a group within the file
	photometry = FitFile.create_group("Photometry")
	# The ids of the filters
	photometry.create_dataset("FilterIDs",data=[filter.filterid for filter in datacollection.filters],dtype='i8')
    # The fluxes and their errors used for the fit
	photometry.create_dataset("Fluxes",data=datacollection.fluxes)
	photometry.create_dataset("FluxErrors",data=datacollection.flux_errors)

	# Store the fitted model information in a group within the file
	# The ids of the models
	model = FitFile.create_group('Model')
	modelids = [model.modelid for model in modelcollection.models]
	model.attrs.create("ModelIDs",modelids,dtype='i4')

	for imodel in range(len(modelids)):
		subgroupid = 'Model{0:2d}'.format(modelids[imodel])
		subgroup = model.create_group(subgroupid)
		for iparam in range(len(modelcollection.parameter_reference)):
			if modelcollection.parameter_reference[iparam][1] == imodel:
				# Information about parameters and prior distribution to output file
				# Store a unique parameter id, 
				#   a combination of the unique model id and the parameter name, with a standard delimiter '_'
				param = '{0:2d}_'.format(modelids[imodel])+modelcollection.parameter_reference[iparam][0]
				prior = modelcollection.parameter_reference[iparam][2]
				subgroup.create_dataset(param,data=prior.prior_grid)
		
	# Store a unique parameter id in the order of the parameters that vary
	# This is simple a combination of the unique model id and the parameter name, with a standard delimiter '_'
	varying_parameters = []
	for iparam in range(len(modelcollection.parameter_reference)):
		# Loop through the parameters
		if modelcollection.parameter_reference[iparam][3]:
			# If the varyflag is set for this parameter, store the unique name
			imodel = modelcollection.parameter_reference[iparam][1]
			if imodel == -1:
				# The varying parameter is redshift
				varying_parameters.append(modelcollection.parameter_reference[iparam][0])
			else:
				unique_name = '{0:2d}_'.format(modelids[imodel])+modelcollection.parameter_reference[iparam][0]
				varying_parameters.append(unique_name)

	if fittype == 'emcee':
		chain = FitFile.create_group("Chain")
		note = 'Fixed parameters not included in chains.'
		chain.attrs.create("Note",note,dtype=np.dtype('S{0:3d}'.format(len(note))))
		chain.create_dataset('Varying_parameters',data=np.core.defchararray.encode(varying_parameters))

	return FitFile


# ***********************************************************************************************

def examine_priors(modelcollection):
	""" Visually examine the prior distributions for each model in turn.
		
		modelcollection: An FortesFit CollectModel instance. See FortesFit_Preparation for details.

	"""
	
	nmodels = len(modelcollection.models) # Number of models used in the fit

	for imodel in range(nmodels):

		fig = plt.figure(figsize=(8,8)) # A large plotting window, for 3x3 = 9 parameters per plot

		n_parameters = modelcollection.number_of_parameters[imodel]
		param_names  = list(modelcollection.priors[imodel].keys())
		# Loop through each parameter
		for iparam in range(n_parameters):
		
			if (iparam % 9 == 0) & (iparam != 0):
				plt.show()
				ch = input('Continue with more parameters? y or n : ')
				if ch == 'n': return
				plt.close()

			plt.subplot(3,3,(iparam % 9)+1)

			prior = modelcollection.priors[imodel][param_names[iparam]]
			if prior.fixed:
				# Single element prior >> fixed parameter
				plt.axis([0,1,0,1])
				plt.text(0.5,0.6,param_names[iparam],size='small',ha='center')
				plt.text(0.5,0.4,'Fixed at '+str(prior.characteristic),size='small',ha='center')
				plt.xticks([])
				plt.yticks([])
			else:
				# Regular prior
				plt.plot(prior.prior_grid[0,:],prior.prior_grid[1,:],'k')
				plt.plot([prior.characteristic,prior.characteristic],[0,np.max(prior.prior_grid[1,:])],'r')
				plt.yticks([])
				plt.xlabel(param_names[iparam])
	
		plt.show()
		if imodel < nmodels-1:
			ch = input('Continue with more models? y or n : ')
			if ch == 'n': return
			plt.close()

	return


	

