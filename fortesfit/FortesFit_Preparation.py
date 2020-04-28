import sys
import os
import glob

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from astropy import units as u
from astropy.table import Table

import h5py
import emcee
from fortesfit import FortesFit_Settings
from fortesfit.FortesFit_Filters import FortesFit_Filter
from fortesfit.FortesFit_ModelManagement import FitModel

""" A module with classes and functions that organise the process of setting up a fitting run  """

# ***********************************************************************************************

def process_PDF(pdf_grid,parameter_range=None,Normalize=True):
	""" Process a supplied distribution in a form useful for FortesFit applications
		
		pdf_grid: A 2xN array, the first row is a grid in the parameter, 
				  the second row is the probability density function on the grid.
				  The array does not need to be sorted, and the pdf does not have to be normalised
		parameter_range: optional 2 element list-like, with [low, high] specifying the range in which
					the PDF is valid, in case it is smaller than the range of the grid.			
		Normalize: normalise the PDF to integrate to unity.
												
	"""

	ngrid = 1001
		
	# Sort the prior grid low to high in the parameter
	sortindex = np.argsort(pdf_grid[0,:])
	pdf_orig_x = pdf_grid[0,sortindex]
	pdf_orig_y = pdf_grid[1,sortindex]

	# set all points with very low probability (< 1e-6 of the peak) to 0.0
	# SciPy RVS based distributions will already be forced to this range
	index, = np.where(pdf_orig_y < 1e-6*pdf_orig_y.max())
	pdf_orig_y[index] = 0.0

	if parameter_range is None:
		# Default, no parameter range provided
		index, = np.where(pdf_orig_y > 0.0)
		xrange = [pdf_orig_x[index].min(),pdf_orig_x[index].max()]
	else:
		if parameter_range[0] >= parameter_range[1]:
			raise ValueError('User-defined parameter range must be [low, high], low < high')
		xrange = parameter_range

	# Set up a cubic spline interpolator
	interprior = interp1d(pdf_orig_x,pdf_orig_y,\
						  kind='cubic',bounds_error=False,fill_value=0.0,assume_sorted=True)

	# ngrid point sampled prior
	pdf_x = xrange[0] + np.arange(ngrid)*(xrange[1]-xrange[0])/(ngrid-1)
	pdf_y = interprior(pdf_x) # interpolate the distribution

	# If the interpolation leads to values below values below zero, set them to zero.
	# A PDF is positive definite
	pdf_y[pdf_y < 0.0] = 0.0

	if Normalize:
		norm = trapz(pdf_y,pdf_x)
		pdf_y = pdf_y/norm  #  Normalise the prior distribution to integrate to unity over its range
		
	return np.stack([pdf_x,pdf_y])

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

			prior_new = process_PDF(prior_grid,parameter_range=parameter_range,Normalize=Normalize)
			prior_x = prior_new[0,:]
			prior_y = prior_new[1,:]
							
			self.fixed = False
			self.characteristic = np.sum(prior_x*prior_y)/np.sum(prior_y)
			self.prior_grid     = np.stack((prior_x,prior_y),axis=0)
		

	def	update_prior(self,new_prior_grid):
		""" Update the prior PDF by multiplying it with a new supplied prior PDF

			new_prior_grid: A 2xN array, the first row is a grid in the parameter, 
						    the second row is the probability density function on the grid.
						    The array does not need to be sorted, and the pdf does not have to be normalised.
		"""
			
		# Check to see if this is a 2xN array
		if new_prior_grid.shape[0] != 2 :
			# this is the wrong shape
			raise TypeError('Input array has the wrong shape')

		if not self.fixed:
		# Only update priors if the prior is not fixed

			# Sort the prior grid low to high in the parameter
			sortindex = np.argsort(new_prior_grid[0,:])
			prior_orig_x = new_prior_grid[0,sortindex]
			prior_orig_y = new_prior_grid[1,sortindex]
		 
			# set all points with very low probability (< 1e-6 of the peak) to 0.0
			# SciPy RVS based distributions will already be forced to this range
			index, = np.where(prior_orig_y < 1e-6*prior_orig_y.max())
			prior_orig_y[index] = 0.0

			# Set up a cubic spline interpolator
			interprior = interp1d(prior_orig_x,prior_orig_y,\
								  kind='cubic',bounds_error=False,fill_value=0.0,assume_sorted=True)

			# ngrid point sampled prior
			# Interpolate the new prior grid onto the original prior parameter grid
			prior_x = self.prior_grid[0,:]
			prior_y = interprior(prior_x)

			norm = trapz(prior_y,prior_x)
			prior_y = prior_y/norm  #  Normalise the prior distribution to integrate to unity over its range

			# Multiply the new prior with the original prior
			self.prior_grid[1,:] *= prior_y

			# Reanalyse the parameter range of the prior to exclude bounding regions with very low probability
			prior_x = self.prior_grid[0,:]
			prior_y = self.prior_grid[1,:]
			index, = np.where(prior_y > 0.0)
			xrange = [prior_x[index].min(),prior_x[index].max()]

			ngrid = self.prior_grid.shape[1]	

			# Set up a cubic spline interpolator
			interprior = interp1d(prior_x,prior_y,\
								  kind='cubic',bounds_error=False,fill_value=0.0,assume_sorted=True)

			# ngrid point sampled prior
			prior_x = xrange[0] + np.arange(ngrid)*(xrange[1]-xrange[0])/(ngrid-1)
			prior_y = interprior(prior_x) # interpolate the distribution

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
	
	def __init__(self,ID,redshift,filterids,fluxes,fluxerrs,Limits=None,Weights=None,MinimumError=None):
		""" Initialise the FortesFit data representation for a user-provided object
			
			ID: An ID string for this object. This will be used to generate the output file.
			redshift: An array of shape (2,N) or (1,). 
					  If single-element, this is the fixed redshift of the object.							
					  If 2xN, the first row is a grid of redshift, the second the probability of the redshift.
					  This is used to represent p(z) from photometric redshift distributions.
			filterids: list-like, the FortesFit ids of filters. Any order, but must match that of photometry lists.
			fluxes: list-like, the fluxes in each band corresponding to filters in argument 'filterids'. 
					Each element must be an Astropy Quantity instant. Fluxes == NaN are disregarded.
			fluxerrs: list-like, the 1sigma flux errors in each band corresponding to filters in argument 'filterids'. 
					  Each element must be an Astropy Quantity instant. Fluxerrs < 0.0 are treated as upper limits,
					  except if the corresponding flux is undefined
			Limits: list-like, the associated limit (1 sigma, 2 sigma, etc.) for the measurement in the band.
					 If None, all limits are assumed to be 1sigma.
			Weights: list-like, a weight to apply to the photometry, equivalent to an inverse scaling of the error.
					 Note, this is different from a normal standard weighted least-squares, where the weight is
					 proportional to the inverse variance. If None, all weights are assumed to be 1.0			
			MinimumError: A minimum percentage error. 
						  If set, flux errors are set to a minimum value of Error = (MinimumError/100)*Flux.
										
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

		# Process weights and limits
		if Weights is None:
			# No weights specified, so set equal weights for all bands
			err_scaling  = np.full(len(fluxes),1.0)
		else:
			err_scaling  = 1.0/Weights

		if not(Limits is None):
			# If limits are specified, scale the weights by the limits for non-detections
			index, = np.where(fluxerrs < 0.0)
			if len(index) > 0:
				err_scaling[index] *= Limits[index]

		# Flag and remove bands with no photometry
		index, = np.where(np.isfinite(fluxes)) # Disregard invalid fluxes
		n_validphot = len(index)
		fluxes      = fluxes[index]
		fluxerrs    = fluxerrs[index]
		err_scaling = err_scaling[index]
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
				# Set the minimum error if desired
				if not(MinimumError is None):
					if (eflux.value/flux.value) < MinimumError/100.0:
						eflux = (MinimumError/100.0)*flux
				fitflux[ifilt]     = flux.value
				fitflux_err[ifilt] = eflux.value
			if(flux > 0.0) and (eflux < 0.0):
				# Upper Limit
				fitflux[ifilt]     = flux.value
				fitflux_err[ifilt] = -1.0   # These will be scaled by the flux limit when calculating the likelihood
		
		
		# Filters will be stored after sorting by pivot wavelength.
		sortindex = np.argsort(FilterWaves)
		self.filters = Filters[sortindex]
		self.pivot_wavelengths = FilterWaves[sortindex]
		self.fluxes = fitflux[sortindex]
		self.flux_errors = fitflux_err[sortindex]
		self.error_weights = err_scaling[sortindex]
	

	# ******************************************
	
	def	plot_sed(self, **kwargs):
		
		""" Plot the SED of the photometry of the object
		"""
		
		figure = plt.figure()
		figure.show
		ax1 = plt.axes()
		ax1.set_xlim([0.1,1000])
		ax1.semilogx()

		wavelengths = self.pivot_wavelengths
		redshift = self.redshift.characteristic	
		fluxes = self.fluxes
		efluxes = self.flux_errors
		weights = self.error_weights

		# "Detections" as defined in the photometric compilation
		index, = np.where(efluxes > 0.0)
		plotflux = np.log10(fluxes[index]*wavelengths[index])
		eplotflux_hi = np.log10((fluxes[index]+efluxes[index])*wavelengths[index]) - plotflux
		eplotflux_lo = plotflux - np.log10((fluxes[index]-efluxes[index])*wavelengths[index])
		ax1.errorbar(wavelengths[index],plotflux,yerr=[eplotflux_lo,eplotflux_hi],color='black',ecolor='black',fmt='ko')
		axrange = ax1.axis()

		index1, = np.where(weights[index] != 1.0)
		if len(index1) > 0:
			ax1.plot(wavelengths[index[index1]],plotflux[index1],'ro')

		# "Limits" are defined in the photometric compilation
		# equivalent 1sigma limits will be plotted
		index, = np.where(efluxes < 0.0)
		for i in range(len(index)):
			lim = np.log10((fluxes[index[i]]/np.abs(weights[index[i]]))*wavelengths[index[i]])
			ax1.plot(wavelengths[index[i]],lim,'k+')
			ax1.arrow(wavelengths[index[i]],lim,0.0,-0.1,
					  fc='black',ec='black',head_width=0.15*wavelengths[index[i]],head_length=0.05)
		
		ax1.set_title(self.id+'    z = {0:5.3f}'.format(redshift),ha='center')
		ax1.set_xlabel(r'Observed Wavelength ($\mu$m)')
		ax1.set_ylabel(r'log $\nu$F$_{\nu}$ (erg s$^{-1}$ cm$^{-2}$)')
		
		figure.show
		
		return figure
		
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
					# Consider a range where the CDF goes from 1e-6 to 1 - 1e-6
					xrange = [prior.ppf(1e-6),prior.ppf(1.0-1e-6)]
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
					prior = priordist[parameter_name]				
					if np.size(prior) == 1:
						if type(prior).__name__ == 'rv_frozen':
							# A frozen Scipy rvs_continuous instance. Will raise its own exception if it isn't properly set.
							# Consider a range where the CDF goes from 1e-6 to 1 - 1e-6
							xrange = [prior.ppf(1e-6),prior.ppf(1.0-1e-6)]
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
				

			# Finally Process any dependency priors
			#  First initialise a grid storing the multi-dimensional likelihood from the dependencies
			#  It will have an array shape with the same dimensionality as the shape parameters
			dependency_pdf = np.full([len(model.shape_parameters[param]) for param in model.shape_parameter_names],1.0) # not normalised	
			DependencyFlag = False # A flag to identify if any dependencies are being processed

			for dependency_name in model.dependency_names:
				if dependency_name in priordist.keys():
					DependencyFlag = True # Set the dependency flag so that dependency-based priors are processed below
					# A prior has been provided by the user for this dependency
					prior = priordist[dependency_name]				
					if np.size(prior) == 1:
						if type(prior).__name__ == 'rv_frozen':
							# A frozen Scipy rvs_continuous instance. Will raise its own exception if it isn't properly set.
							# Consider a range where the CDF goes from 1e-6 to 1 - 1e-6
							xrange = [prior.ppf(1e-6),prior.ppf(1.0-1e-6)]
							prior_x = xrange[0] + np.arange(101)*(xrange[1]-xrange[0])/100.0
							prior_y = prior.pdf(prior_x) # get the distribution using the PDF function
							dep_prior = PriorDistribution(np.stack((prior_x,prior_y),axis=0),Normalize=False)
						else:					
							# single value of prior
							# Dependencies cannot be set to a single value
							error_message = dependency_name+' in Model{0:2d}'.format(model.modelid)+': The prior cannot be single-valued'
							raise ValueError(error_message)
					else:
						if type(prior).__name__ == 'ndarray':
							dep_prior = PriorDistribution(prior)
						else:
							# Unclassifiable prior class. Raise exception.
							error_message = dependency_name+' in Model{0:2d}'.format(model.modelid)+': Incorrect prior format'
							raise ValueError(error_message)
					
					# Using the dependency grid, calculate the likelihood of the shape parameters on the grid
					if model.dependencies[dependency_name].shape[-1] == 1:
						# This dependency is related to the scale parameter

						# Calculate the term to offset the dependency to the most likely prior value
						offset = dep_prior.characteristic - np.median(model.dependencies[dependency_name])

						# Update the scale parameter prior distribution to reflect the dependency prior
						# The dependency grid is calculated at the default value of the scale
						#   parameter. Therefore, the scale parameter prior due to the dependency has the
						#   same form as the dependency prior, but with a characteristic value that is the
						#   default scale parameter value + the offset mentioned above.
						new_prior = dep_prior.prior_grid.copy()
						new_prior[0,:] += offset + (model.scale_parameter_value - dep_prior.characteristic)
						priordict[model.scale_parameter_name].update_prior(new_prior)

						# Update the dependency prior
						dependency_pdf *= np.interp( \
									   np.squeeze(model.dependencies[dependency_name])+offset,\
								       dep_prior.prior_grid[0,:],dep_prior.prior_grid[1,:],\
									   left=0.0,right=0.0)
					else:
						# This dependency is independent of the scale parameter
						# Update the dependency prior only
						dependency_pdf *= np.interp( \
									   model.dependencies[dependency_name],\
								       dep_prior.prior_grid[0,:],dep_prior.prior_grid[1,:],\
									   left=0.0,right=0.0)

			# If the dependency flag is set, then obtain dependency-based updates to the priors
			if DependencyFlag:
			
				# Obtain the full integration over all shape parameters to normalise the dependency PDF
				norm = dependency_pdf.copy()
				for iax in range(len(model.shape_parameter_names),0,-1):	
					norm = trapz(norm,model.shape_parameter_value_tuple[iax-1],axis=iax-1)			
				dependency_pdf = dependency_pdf/norm

				# In turns, integrate over all shape parameters except one to yield marginalised PDFs.
				for ipar in range(len(model.shape_parameter_names)):
					prior_x = model.shape_parameters[model.shape_parameter_names[ipar]]
					temp_grid = dependency_pdf.copy()
					for iax in range(len(model.shape_parameter_names),0,-1):	
						if iax-1 != ipar:	
							temp_grid = trapz(temp_grid,model.shape_parameter_value_tuple[iax-1],axis=iax-1)			
					priordict[model.shape_parameter_names[ipar]].update_prior(np.stack((prior_x,temp_grid),axis=0))

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

def prepare_output_file(datacollection,modelcollection,fitengine,OutputPath=None,description=''):
	""" This function prepares an HDF5 file to store the outputs of the SED fitting.
		
		Positional Arguments:
		 datacollection: An FortesFit CollectData instance. See FortesFit_Preparation for details.
		 modelcollection: An FortesFit CollectModel instance. See FortesFit_Preparation for details.
		 fitengine: The fitting engine for which this file will store the output. 
				     Used to initialise the storage group, and temporary file storage directory, if needed.
		Keyword Arguments:
		 OutputPath: The full path of an existing directory to store the file. Default is current working directory.
		 description: An optional string to describe the fit. Defaults to the galaxy ID.

		Returns the file reference of an open HDF5 file. The file should be closed after use.

	"""
	if fitengine == 'multinest':
		# Create a multinest output directory if none exists
		if not os.path.isdir('multinest_output/'):
			raise FileNotFoundError('Running Multinest requires a local working directory called "multinest_output/"')

	if OutputPath is None:
		# Get the current directory path if no user supplied one
		OutputPath = os.getcwd()
	
	if OutputPath[-1] != '/': OutputPath = OutputPath+'/' 
	
	GalaxyName = datacollection.id	
	FitFileName = GalaxyName+'.FortesFit_output.'+fitengine+'.hdf5'  # Initialise the output file of the fit
	FitFile = h5py.File(OutputPath+FitFileName, 'w')

	FitFile.attrs.create("Object Name",GalaxyName,dtype=np.dtype('S{0:3d}'.format(len(GalaxyName))))
	FitFile.attrs.create("Fitting Engine",fitengine,dtype=np.dtype('S{0:3d}'.format(len(fitengine))))
	if len(description) == 0:
		description = GalaxyName+ ' with FortesFit + '+fitengine
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

		plt.close()

		modelname = modelcollection.models[imodel].description
		n_parameters = modelcollection.number_of_parameters[imodel]
		param_names  = list(modelcollection.priors[imodel].keys())
		npages = int(n_parameters / 9) + 1
		# Loop through each parameter
		for iparam in range(n_parameters):
		
			if (iparam % 9 == 0):
				# Reset the plotting window
				xstart = 0
				ystart = 0
				fig = plt.figure(figsize=(8,8)) # A large plotting window, for 3x3 = 9 parameters per plot
				fig.text(0.5,0.95,modelname,ha='center')
				if iparam != 0:
					ch = input('Continue with more parameters? y or n : ')
					if ch == 'n': return
					plt.close()

			print(param_names[iparam])
			xstart = iparam % 3
			ystart = int(iparam / 3)
			ax = fig.add_axes([0.1+xstart*0.8/3+0.01,0.9-(ystart+1)*0.8/3+0.05,0.8/3-0.01,0.6/3])			

			prior = modelcollection.priors[imodel][param_names[iparam]]
			if prior.fixed:
				# Single element prior >> fixed parameter
				ax.axis([0,1,0,1])
				plt.text(0.5,0.4,'Fixed at '+str(prior.characteristic),size='small',ha='center')
				ax.tick_params(axis='both',bottom=False,labelbottom=False,left=False,labelleft=False)		
				ax.set_title(param_names[iparam])
			else:
				# Regular prior
				ax.plot(prior.prior_grid[0,:],prior.prior_grid[1,:],'k')
				ax.plot([prior.characteristic,prior.characteristic],[0,np.max(prior.prior_grid[1,:])],'r')

				# Reasonable ticks (4 per parameter)
				xticks = ax.get_xticks()
				nskip = np.int(len(xticks)/3)
				ax.set_xticks(xticks[1::nskip])
				ax.tick_params(axis='x',labelsize='medium')
				ax.tick_params(axis='y',left=False,labelleft=False)		
				ax.set_title(param_names[iparam])
	
		plt.show()
		ch = input('Continue with more models? y or n : ')		
		if ch == 'n': return

	return


	

