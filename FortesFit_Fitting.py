from sys import exit
import os
import glob
import numpy as np
from scipy.special import erf
from scipy.optimize import minimize
from astropy.table import Table
import h5py
import emcee
from fortesfit import FortesFit_Settings
from fortesfit import FortesFit_Filters
from fortesfit import FortesFit_ModelManagement
 
""" A module with functions that are used by the model fitting routines in FORTES-AGN  

	This version is written with EMCEE as the engine, using the first working versions of the model and filter routines.
	Redshift is taken to be fixed.
	November 13, 2016

"""

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

def	FortesFit_FitSingle_emcee(Fluxes, FluxErrors, Redshift, Filters, Models, Parameters, Pdz = [], nwalkers=100, nsteps=500):
	""" The upper level function call to fit one object with EMCEE
		
		
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
						second element: an instant of a scipy.stats.rv_ class. See SciPy documentation for details.
												
		Pdz: A function instance that returns a probability (scaled) for a given redshift. 
			   This will be used (later) to allow a bootstrap analysis using the redshift uncertainty.			  
		nwalkers: number of EMCEE walkers
		nsteps: Number of MCMC steps

	"""
	
	# Compile an ordered list of parameters. This order will be preserved for all further calls
	#   to likelihood, prior and sampling functions
	parameter_names = []
	parameter_initvals = []
	parameter_pdfs = []
	for model in Models:
		parameter_names.append(model.scale_parameter_name)
		parameter_initvals.append(Parameters[model.scale_parameter_name][0])
		parameter_pdfs.append(Parameters[model.scale_parameter_name][1].pdf)
		for shapepar in model.shape_parameter_names:
			parameter_names.append(shapepar)
			parameter_initvals.append(Parameters[shapepar][0])
			parameter_pdfs.append(Parameters[shapepar][1].pdf)
	
	parameter_initvals = np.array(parameter_initvals)
	
	# Move on to MCMC
	print('Starting EMCEE run')	
	
	# Start with a ball of walkers around the initial location
	# Each parameter is scattered within with sigma of 1% of the MLE value
	initdist = [parameter_initvals + 0.001*parameter_initvals*np.random.randn(len(parameter_names)) for i in range(nwalkers)]
	
	# Initialise the walkers using the prior distributions
	# Each parameter is drawn from the prior distribution
# 	initdist = np.empty((nwalkers,len(parameter_names)))
# 	for i in range(nwalkers):
# 		initdist[i,:] = [Parameters[param][1].rvs(size=1) for param in parameter_names]

	# Set up the affine-invariant sampler
	sampler = emcee.EnsembleSampler(nwalkers, len(parameter_names), \
									Bayesian_probability, \
									args=(Filters,Redshift,Fluxes,FluxErrors,Models,parameter_names,parameter_pdfs))
	for i, result in enumerate(sampler.sample(initdist, iterations=nsteps)):
 		print("{0:5.1f}%".format(i*100/nsteps),end="\r")
	# sampler.run_mcmc(initdist, nsteps)
	
	return sampler
	
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
	
def	Bayesian_probability(parameters,Filters,Redshift,Fluxes,FluxErrors,Models,parameter_names,parameter_pdfs):
	"""  Obtains the sum of the log probabilities of likelihood and prior for the model

		 parameters: a list or array of parameter values. This is the only required parameter for EMCEE or scipy.optimize.
		 Filters: A list of FortesFit_Filter class instances for which photometry exists. Must have the same order as Fluxes.
		 Redshift: The best redshift of the object.  
		 Fluxes: The fluxes of the object in a set of bands. Units are erg/s/cm^2/micron.
		 FluxErrors: The flux errors in a set of bands with the same order as Fluxes. 
					 If the Flux is a limit, the error = -1.0*significance of limit
		 Models:  A list of FortesFit_Model class instances that will be fit to the SED
		 parameter_names: A list of parameter names with the same order as parameters.
		 parameter_pdfs: A list of that takes a parameter value and returns the PDF (scaled). 
						 This can be user-defined or one may use continuous scipy.stats.<dist> PDF methods. 
	
	"""
	
	# Calculate the likelihood
	loglikelihood = Sawicki12_loglikelihood(parameters,Filters,Redshift,Fluxes,FluxErrors,Models,parameter_names)

	# Calculate the prior probabilities for each parameter and add the log probabilities
	logprior = 0.0
	for i in range(len(parameter_names)):
		logprior += np.log(parameter_pdfs[i](parameters[i]))
	
	# Return the total log probability (likelihood*prior)
	return loglikelihood + logprior	

# ***********************************************************************************************

def	Sawicki12_loglikelihood(parameters,Filters,Redshift,Fluxes,FluxErrors,Models,parameter_names):
	"""  Obtains the likelihood for a list of model parameters.

		 parameters: a list or array of parameter values. This is the only required parameter for EMCEE or scipy.optimize.
		 Filters: A list of FortesFit_Filter class instances for which photometry exists. Must have the same order as Fluxes.
		 Redshift: The best redshift of the object.  
		 Fluxes: The fluxes of the object in a set of bands. Units are erg/s/cm^2/micron.
		 FluxErrors: The flux errors in a set of bands with the same order as Fluxes. 
					 If the Flux is a limit, the error = -1.0*significance of limit
		 Models:  A list of FortesFit_Model class instances that will be fit to the SED
		 parameter_names: A list of parameter names with the same order as parameters.
	
	"""
	
	# Create a dictionary of the parameter names and values
	paramdict = dict(zip(parameter_names,parameters))
	
	# An array to store the model fluxes
	modelFluxes = np.zeros(len(Filters),dtype='f4')
	for i,filter in enumerate(Filters):
		for model in Models:
			modelFluxes[i] += 3.63e-5*10**(model.evaluate(paramdict,Redshift,filter.filterid))	# Include scaling from STMAG=0	
	
	index, = np.where(FluxErrors > 0.0) # Detections, supplied errors are positive
	loglike_det = -0.5*np.sum(((Fluxes[index] - modelFluxes[index])/FluxErrors[index])**2.0) # A chisq likelihood for detections
	
	index, = np.where(FluxErrors < 0.0) # Upper limits, supplied errors are negative
	sigma = Fluxes[index]/(-1.0*FluxErrors[index])
	# An error function for each limited band describes the likelihood
	loglike_lim = np.sum(np.log(np.sqrt(np.pi/2.0)*sigma*(1.0 + erf((Fluxes[index] - modelFluxes[index])/(np.sqrt(2.0)*sigma)))))

	# If there are any cases of FluxErrors = 0.0, they are not used to calculate likelihoods, 
	#     i.e, those fluxes are masked from fitting
	
	# Return the sum of the loglikelihoods
	return loglike_det + loglike_lim
	

# ***********************************************************************************************

def	prior_probability(parameters,Filters,Redshift,Fluxes,FluxErrors,Models,parameter_names,parameter_pdfs):
	"""  Obtains the sum of the log probabilities of prior for the model only. 
		 This code is only used for testing at present

		 parameters: a list or array of parameter values. This is the only required parameter for EMCEE or scipy.optimize.
		 Filters: A list of FortesFit_Filter class instances for which photometry exists. Must have the same order as Fluxes.
		 Redshift: The best redshift of the object.  
		 Fluxes: The fluxes of the object in a set of bands. Units are erg/s/cm^2/micron.
		 FluxErrors: The flux errors in a set of bands with the same order as Fluxes. 
					 If the Flux is a limit, the error = -1.0*significance of limit
		 Models:  A list of FortesFit_Model class instances that will be fit to the SED
		 parameter_names: A list of parameter names with the same order as parameters.
		 parameter_pdfs: A list of that takes a parameter value and returns the PDF (scaled). 
						 This can be user-defined or one may use continuous scipy.stats.<dist> PDF methods. 
	
	"""
	
	# Calculate the prior probabilities for each parameter and add the log probabilities
	logprior = 0.0
	for i in range(len(parameter_names)):
		logprior += np.log(parameter_pdfs[i](parameters[i]))
	
	# Return the total log prior probability
	return logprior	

	
