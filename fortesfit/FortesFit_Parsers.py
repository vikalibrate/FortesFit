import numpy as np

from fortesfit.FortesFit_Filters import FortesFit_Filter
from fortesfit.FortesFit_ModelManagement import FullModel, FitModel
from fortesfit.FortesFit_Plots import FortesFitResult

""" A module with functions to parse FortesFit fit results

"""

#*******************************************************************
#Dev code
#*******************************************************************

def	get_bestfit_fluxes(FortesFit_OutFile, BurnIn=0, scaled_models=None, filter_scaling=None, rest=False, silent=False, old=False):
	""" 
	Get the best fit model fluxes for individual components as well as the total model for a given fit.

	Input:
	----
	FortesFit_OutFile	: (str) HDF5 file from the fit
	BurnIn (=0)			: (int) Burn in for EMCEE
	scaled_models		: (list) Model IDs for the scaled models
	filter_scaling		: (dict) Dictionary of scaling to each filter of form {filterID: scaling constant}
	rest (=False)		: (bool) Whether to return rest SED or not
	silent (=False)		: (bool) chatty (or not) function
	old (=False)		: (bool) * relic of the past *

	Return:
	----
	bestfit_fluxes 		: (array) Array of best-fit model fluxes with shape (m, n) where m is the number of filter
							n is number of model components + 1. First n - 1 columns have model fluxes for individual
							components while the last column has total model flux
	wavelength 			: (array) Observed (default) or rest-frame pivot wavelength of all filters
	redshift 			: (float) Redshift of the source
	"""

	if scaled_models is None: scaled_models = []
	if filter_scaling is None: filter_scaling = {}
				
	fitresult = FortesFitResult(FortesFit_OutFile, BurnIn=BurnIn, old=old)
	# Write out a description of the fit
	if(not silent):
		print(fitresult.fit_description)	
	ObjectName = fitresult.objectname 

	# Compile the filter wavelengths that were used in the fit
	FilterIDs = fitresult.fit_filterids
	Filters = [FortesFit_Filter(filterid) for filterid in FilterIDs]
	FilterWave = np.array([filter.pivot_wavelength for filter in Filters])	

	ModelIDs = fitresult.fit_modelids
	Nmodels = len(ModelIDs)

	# Create a list of model parameter dictionaries with best-fit/fixed parameter values. These will be changed
	#  when processing the individual model SEDs
	paramdict_plot = [{} for imodel in range(Nmodels)]
	for param in fitresult.bestfit_parameters.keys():
		for imodel in range(Nmodels):
			if param[0:2] == '{0:2d}'.format(ModelIDs[imodel]):
				paramdict_plot[imodel].update({param[3:]:fitresult.bestfit_parameters[param][0]})	

	Redshift = fitresult.redshift
	besfit_fluxes = np.zeros((len(Filters), Nmodels + 1))

	for imodel, modelid in enumerate(ModelIDs):
		if modelid in scaled_models:
			fitmodel = FitModel(modelid, Redshift, FilterIDs, filter_scaling=filter_scaling)
		else:
			fitmodel = FitModel(modelid, Redshift, FilterIDs, filter_scaling=None)

		for ifilt, filterid in enumerate(FilterIDs):
			besfit_fluxes[ifilt, imodel] = 3.63e-5*10**(fitmodel.evaluate(paramdict_plot[imodel], 
				Redshift, filterid))*FilterWave[ifilt]

	rest_wavelengths = FilterWave/(1.0 + Redshift)
	besfit_fluxes[:, -1] = np.sum(besfit_fluxes, axis=1)
	
	if rest:
		return besfit_fluxes, rest_wavelengths, Redshift

	return besfit_fluxes, FilterWave, Redshift

# def	return_sample_SEDs(FortesFit_OutFile, BurnIn=0, scaled_models=None, filter_scaling=None, wave_range = [1e-1,1e3], PDF_File='', 
# 	Nsamps=100, silent=False, old=False, legend=True):
# 	""" Plot the best-fit combined SED, model photometry. 
# 		From Nsamps SEDs drawn from the joint posterior, get the error SEDs for each component and overplot.
		
# 		FortesFit_Outfile:  HDF5 file with the outputs from FortesFit
# 		wave_range: Wavelength range to plot the SEDs, list-like two-element, ([starting, ending] wavelength in microns)
# 		BurnIn: The number of samples on the MCMC chains to exclude to allow for convergence
# 		PDF_File: A file to send the plotted SEDs. One page per fitted object.
# 		Nsamps: Number of samples to draw from joint posterior. Default = 100. Will affect the speed of the routine.
# 		silent: If True, no information messages are used. Serial plots are shown for 2 seconds.
	
# 	"""		
		
# 	# Initialise the wavelength flux array that is used for plotting the best-fit model (from 1000 Ang to 1mm) in microns
# 	ObsWave = np.log10(wave_range[0]) + np.arange(101)*(np.log10(wave_range[1]/wave_range[0])/100.0)
				
# 	fitresult = FortesFitResult(FortesFit_OutFile, BurnIn=BurnIn, old=old)
# 	# Write out a description of the fit
# 	if(not silent):
# 		print(fitresult.fit_description)	
# 	ObjectName = fitresult.objectname 

# 	# Compile the filter wavelengths that were used in the fit
# 	FilterIDs = fitresult.fit_filterids
# 	Filters = [FortesFit_Filter(filterid) for filterid in FilterIDs]
# 	FilterWave = np.array([filter.pivot_wavelength for filter in Filters])	

# 	Fluxes     = fitresult.fit_fluxes
# 	FluxErrors = fitresult.fit_fluxerrors

# 	ModelIDs = fitresult.fit_modelids
# 	Models = []
# 	for modelid in ModelIDs:
# 		Models.append(FullModel(modelid,sed_readin=True))
# 	Nmodels = len(ModelIDs)

# 	# Create a list of model parameter dictionaries with best-fit/fixed parameter values. These will be changed
# 	#  when processing the individual model SEDs
# 	paramdict_plot = [{} for imodel in range(Nmodels)]
# 	for param in fitresult.bestfit_parameters.keys():
# 		for imodel in range(Nmodels):
# 			if param[0:2] == '{0:2d}'.format(ModelIDs[imodel]):
# 				paramdict_plot[imodel].update({param[3:]:fitresult.bestfit_parameters[param][0]})

# 	# Obtain the flattened chains
# 	samples = fitresult.all_samples

# 	# Overplot the SEDs from Nsamps random samples from the burned-in chains
# 	sample_seds = np.zeros((Nsamps,len(ObsWave),Nmodels))
# 	sample_photometry = np.zeros((Nsamps,len(FilterIDs)))
# 	for isamp in range(Nsamps):
# 		# Take a random position along the chain
# 		parameter_sample  = samples[np.random.randint(samples.shape[0]),:]
# 		paramdict_varying = dict(zip(fitresult.fit_parameter_names,parameter_sample))	
# 		if fitresult.redshift == -99.0:
# 			Redshift = paramdict_varying['Redshift']
# 		else:
# 			Redshift = fitresult.redshift
# 		for imodel,modelid in enumerate(ModelIDs):
# 			model = Models[imodel]
# 			for param in paramdict_plot[imodel].keys():
# 				uparam = '{0:2d}_'.format(model.modelid)+param
# 				if uparam in paramdict_varying:
# 					paramdict_plot[imodel][param] = paramdict_varying[uparam]
# 			sed = model.get_pivot_sed(paramdict_plot[imodel],Redshift)
# 			index, = np.where(sed['observed_flux'] > 0.0) # Only interpolate over valid parts of the model SED
# 			tempflux = np.interp(ObsWave,np.log10(sed['observed_wavelength'][index]),np.log10(sed['observed_flux'][index]),\
# 								 left=-np.inf,right=-np.inf) + ObsWave	
# 			sample_seds[isamp,:,imodel] = 10**(tempflux)

# 			if modelid in scaled_models:
# 				fitmodel = FitModel(modelid,Redshift,FilterIDs,filter_scaling=filter_scaling)
# 			else:
# 				fitmodel = FitModel(modelid,Redshift,FilterIDs,filter_scaling=None)

# 			for ifilt in range(len(FilterIDs)):
# 				sample_photometry[isamp,ifilt] += \
# 					3.63e-5*10**(fitmodel.evaluate(paramdict_plot[imodel],Redshift,FilterIDs[ifilt]))*FilterWave[ifilt]

# 	BestFitFlux = np.zeros((Nmodels + 1, len(ObsWave)))  # the array for the summed best-fit SED
# 	for imodel in range(Nmodels):
# 		index, = np.where(sample_seds[0,:,imodel] != 0.0) # get range of sample data from first model SED in set
# 		scatter = np.percentile(sample_seds[:,index,imodel],[16,50,84],axis=0,interpolation='nearest')
# 		BestFitFlux[imodel, index] = scatter[2,:]
# 		BestFitFlux[-1, index] += scatter[2,:]

# 	index, = np.where((Fluxes > 0.0) & (FluxErrors > 0.0))
# 	fluxconv = FilterWave[index]
# 	plotfluxes = Fluxes[index]*fluxconv 

# 	modelFluxes = np.zeros(len(Filters),dtype='f8')
# 	for ifilt in range(len(FilterIDs)):
# 		modelFluxes[ifilt] = np.median(sample_photometry[:,ifilt])

# 	return (plotfluxes, FilterWave[index]), (modelFluxes, FilterWave), (BestFitFlux, ObsWave)

# def get_fitsummary: best-fit values, number of chains etc
# def get_posterior: take parameter name and output posterior distribution