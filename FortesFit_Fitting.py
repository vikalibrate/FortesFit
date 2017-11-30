import sys
import os

import numpy as np
from scipy.special import erf
from scipy.optimize import minimize

import emcee

from fortesfit import FortesFit_Settings
from fortesfit import FortesFit_Filters
from fortesfit import FortesFit_ModelManagement
 
""" A module with functions that are used by the model fitting routines in FortesFit 

	This version is written with EMCEE as the engine, using the first working versions of the model and filter routines.
	Redshift is taken to be fixed.

"""

# ***********************************************************************************************

def	FortesFit_FitSingle(datacollection, modelcollection, method='emcee', **kwargs):
	""" The upper level function call to fit one object with FortesFit		
		
		datacollection: An FortesFit CollectData instance. See FortesFit_Preparation for details.
		modelcollection: An FortesFit CollectModel instance. See FortesFit_Preparation for details.
		method: a keyword which identifies which fitting engine to use.
		kwargs: keyword arguments passed to the fitting engine
		
		returns the output class of the fitting engine

	"""
	
	# Obtain the indices of the varying parameters
	varying_indices = []
	for iparam in range(len(modelcollection.parameter_reference)):
		# Loop through the parameters
		if modelcollection.parameter_reference[iparam][3]:
			# If the varyflag is set for this parameter, store the index
			varying_indices.append(iparam)

	if method == 'emcee':
		return FortesFit_emcee(varying_indices, datacollection, modelcollection, **kwargs)			
	else:
		raise ValueError('Method not recognised')


# ***********************************************************************************************

def	FortesFit_emcee(varying_indices, datacollection, modelcollection, nwalkers=100, nsteps=500, tight=False):
	""" The upper level function call to fit one object with EMCEE		
		
		varying_indices: A list of indices to the parameter_reference for the varying parameters.
		datacollection: An FortesFit CollectData instance. See FortesFit_Preparation for details.
		modelcollection: An FortesFit CollectModel instance. See FortesFit_Preparation for details.
		nwalkers: number of EMCEE walkers
		nsteps: Number of MCMC steps
		tight: Use a tight ball initialisation around the characteristic values.
			   If false (the default), the prior will be sampled to initialise the walkers.

		returns the emcee sampler after running the MCMC chains

	"""
	# Initialise the walkers
	initdist = np.empty((nwalkers,len(varying_indices)))

	for ivary,varyindex in enumerate(varying_indices):
		# Loop through the varying parameters
		if tight:
			# Start with a ball of walkers of 0.1% of the initial value around the initial location
			initval = modelcollection.parameter_reference[varyindex][2].characteristic
			initdist[:,ivary] = initval*(1.0 + 0.001*(np.random.random(size=nwalkers)*2.0-1.0))
		else:
			# Draw from the prior distribution of the parameter
			prior = modelcollection.parameter_reference[varyindex][2]
			initdist[:,ivary] = prior.draw_prior(ndraws=nwalkers)

	# Move on to MCMC
	print('Starting EMCEE run')	
	# Set up the affine-invariant sampler
	sampler = emcee.EnsembleSampler(nwalkers, len(varying_indices), \
									Bayesian_probability, \
									args=(datacollection, modelcollection, varying_indices))
	for i, result in enumerate(sampler.sample(initdist, iterations=nsteps)):
		print("{0:5.1f}%".format(i*100/nsteps),end="\r")
	# sampler.run_mcmc(initdist, nsteps)
	
	return sampler
	
# ***********************************************************************************************


def	FortesFit_FitSingle_MLE(Fluxes, FluxErrors, Redshift, Filters, Models, Parameters):
	""" The upper level function call to fit one object with MLE. Used as a starting fit for MCMC.
		
		
		Fluxes: The fluxes of the object in a set of bands. Units are erg/s/cm^2/micron.
		FluxErrors: The flux errors in a set of bands with the same order as Fluxes. 
                    If the Flux is a limit, the error = -1.0*significance of limit
		Redshift: The best redshift of the object.  
		Models:  A list of FortesFit_Model class instances that will be fit to the SED
		Filters: A list of FortesFit_Filter class instances for which photometry exists. Must have the same order as Fluxes.
		Parameters: A dictionary of the parameters, which are the superset of all parameters for Models.
					The dictionary keys are the parameter names as in the Models.
					The dictionary values are an initial guess for the parameter

	"""
	
	# Compile an ordered list of parameters. This order will be preserved for all further calls
	#   to likelihood functions
	parameter_names = []
	parameter_initvals = []
	for model in Models:
		parameter_names.append(model.scale_parameter_name)
		parameter_initvals.append(Parameters[model.scale_parameter_name])
		for shapepar in model.shape_parameter_names:
			parameter_names.append(shapepar)
			parameter_initvals.append(Parameters[shapepar])
	
	# Use scipy.optimize to get a maximum likelihood estimate of the parameters
	nll = lambda *args: -1.0*Sawicki12_loglikelihood(*args)
	mleresult = minimize(nll, parameter_initvals, method='Nelder-Mead',\
						 args=(Filters,Redshift,Fluxes,FluxErrors,Models,parameter_names))

	# If the MLE fit is successful, use its values as starting values for the MCMC fit. Else use the initial values
	if (mleresult.success):
		print('MLE successful')
	else:
		print('MLE unsuccessful. Use default priors.')
	
	return mleresult

# ***********************************************************************************************
	
def	FortesFit_FitSingle_PTemcee(Fluxes, FluxErrors, Redshift, Filters, Models, Parameters, Pdz = [], \
								nwalkers=100, nsteps=500, ntemps=20):
	""" The upper level function call to fit one object with Parallel-Tempered EMCEE
		
		
		Fluxes: The fluxes of the object in a set of bands. Units are erg/s/cm^2/micron.
		FluxErrors: The flux errors in a set of bands with the same order as Fluxes. 
                    If the Flux is a limit, the error = -1.0*significance of limit
		Redshift: The best redshift of the object.  
		Models:  A list of FortesFit_Model class instances that will be fit to the SED
		Filters: A list of FortesFit_Filter class instances for which photometry exists. Must have the same order as Fluxes.
		Parameters: A dictionary of the parameters, which are the superset of all parameters for Models.
					The dictionary keys are the parameter names as in the Models.
					The dictionary values are a two element list for each parameter:
						first element: an initial guess for the parameter, which may be consistent with the prior.
						second element: an instant of a function that takes a parameter value and
					                       returns the PDF (scaled). This can be user-defined 
                                           or one may use continuous scipy.stats.<dist> PDF methods.		
		Pdz: A function instance that returns a probability (scaled) for a given redshift. 
			   This will be used (later) to allow a bootstrap analysis using the redshift uncertainty.			  
		nwalkers: number of EMCEE walkers
		nsteps: Number of MCMC steps
		ntemps: Number of temperatures

	"""
	
	# Compile an ordered list of parameters. This order will be preserved for all further calls
	#   to likelihood, prior and sampling functions
	parameter_names = []
	parameter_initvals = []
	parameter_pdfs = []
	for model in Models:
		parameter_names.append(model.scale_parameter_name)
		parameter_initvals.append(Parameters[model.scale_parameter_name][0])
		parameter_pdfs.append(Parameters[model.scale_parameter_name][1])
		for shapepar in model.shape_parameter_names:
			parameter_names.append(shapepar)
			parameter_initvals.append(Parameters[shapepar][0])
			parameter_pdfs.append(Parameters[shapepar][1])
	
	parameter_initvals = np.array(parameter_initvals)
	
	# Move on to MCMC
	print('Starting PTEMCEE run')	
	
	# Start with a ball of walkers around the initial location
	# Each parameter is scattered within with sigma of 1% of the MLE value
	initdist = [parameter_initvals + 0.001*parameter_initvals*np.random.randn(len(parameter_names)) for i in range(nwalkers)]
	
	# Set up the parallel-tempered sampler
	sampler = emcee.PTSampler(ntemps, nwalkers, len(parameter_names), \
							  Sawicki12_loglikelihood, prior_probability, \
							  loglargs=(Filters,Redshift,Fluxes,FluxErrors,Models,parameter_names,parameter_pdfs),\
								   )
	for i, result in enumerate(sampler.sample(initdist, iterations=nsteps)):
 		print("{0:5.1f}%".format(i*100/nsteps),end="\r")
	# sampler.run_mcmc(initdist, nsteps)
	
	return sampler

# ***********************************************************************************************
	
def	Bayesian_probability(varying_parameters,datacollection,modelcollection,varying_indices):
	"""  Obtains the sum of the log probabilities of likelihood and prior for the model

		 varying_parameters: a list or array of parameter values. This is the only required parameter for EMCEE 
						     or scipy.optimize.
		 datacollection: An FortesFit CollectData instance. See FortesFit_Preparation for details.
		 modelcollection: An FortesFit CollectModel instance. See FortesFit_Preparation for details.
		 varying_indices: A list of indices to the modelcollection.parameter_reference for the varying parameters.

		
		returns the log posterior probability, the sum of log likelihood and log prior probability
									  
	"""
		
	# Compile a list of dictionaries, one for each model, with the parameters update to their varying
	# values. Also get the redshift (varying or fixed) and calculate the logprior
	
	logprior = 0.0

	parameter_names = [reference[0] for reference in modelcollection.parameter_reference]
	parameter_vals  = [reference[2].characteristic for reference in modelcollection.parameter_reference]
	for ivary,varyval in enumerate(varying_parameters):
		
		# Replace the characteristic value with the varying value of the parameter, if applicable
		parameter_vals[varying_indices[ivary]] = varyval  

		# Calculate the prior probability from the respective prior instance and update logprior
		prprob = np.interp(varyval,\
						   modelcollection.parameter_reference[varying_indices[ivary]][2].prior_grid[0,:],\
						   modelcollection.parameter_reference[varying_indices[ivary]][2].prior_grid[1,:],\
						   left=0.0,right=0.0)
		if prprob > 0.0: 
			logprior += np.log(prprob)
		else:
			logprior += -np.inf	

	# Get the redshift
	redshift = 	parameter_vals[0]
	
	# Create the parameter dictionaries
	# Initialise a list to store the dictionaries
	paramdict_list = [None for imodel in range(len(modelcollection.models))]
	parstart = 1
	for imodel in range(len(modelcollection.models)):
		# Create a dictionary of the original parameter names as registered
		paramdict = dict(zip(parameter_names[parstart:parstart+modelcollection.number_of_parameters[imodel]],\
							 parameter_vals[parstart:parstart+modelcollection.number_of_parameters[imodel]]))
		paramdict_list[imodel] = paramdict
		parstart += modelcollection.number_of_parameters[imodel]		

	# Calculate the likelihood
	loglikelihood = Sawicki12_loglikelihood(redshift,\
							  datacollection.filters,datacollection.fluxes,datacollection.flux_errors,\
							  modelcollection.models,paramdict_list)
	
	# Return the total log probability (likelihood*prior)
	return loglikelihood + logprior	

# ***********************************************************************************************

def	Sawicki12_loglikelihood(redshift,filters,fluxes,flux_errors,models,parameters):
	"""	Obtains the likelihood of a given set of fluxes, as outlined in Sawicki 2012.

		redshift: redshift of the object
		filters: list of FortesFit filter instances
		fluxes: list of fluxes in erg/s/cm^2/micron in order of the filters
		flux_errors: list of errors in fluxes in erg/s/cm^2/micron in order of the filters. limits are specified as -ve*limit
		models: list of FortesFit model instances
		parameters: list of dictionaries with parameter values, in same order as the models
		
		returns the log likelihood
	
	"""
	
	
	# An array to store the model fluxes
	modelFluxes = np.zeros(len(filters),dtype='f4')
	for imodel,model in enumerate(models):
		for i,filter in enumerate(filters):
			modelFluxes[i] += 3.63e-5*10**(model.evaluate(parameters[imodel],redshift,filter.filterid))	# Include scaling from STMAG=0	
	
	loglike_det = 0.0
	index, = np.where(flux_errors > 0.0) # Detections, supplied errors are positive
	loglike_det += -0.5*np.sum(((fluxes[index] - modelFluxes[index])/flux_errors[index])**2.0) # A chisq likelihood for detections
	
	loglike_lim = 0.0
	index, = np.where(flux_errors < 0.0) # Upper limits, supplied errors are negative
	sigma = fluxes[index]/(-1.0*flux_errors[index])
	# An error function for each limited band describes the likelihood
	loglike_lim += np.sum(np.log(np.sqrt(np.pi/2.0)*sigma*(1.0 + erf((fluxes[index] - modelFluxes[index])/(np.sqrt(2.0)*sigma)))))

	# If there are any cases of FluxErrors = 0.0, they are not used to calculate likelihoods, 
	#     i.e, those fluxes are masked from fitting
	
	# Return the sum of the loglikelihoods
	return loglike_det + loglike_lim



	
