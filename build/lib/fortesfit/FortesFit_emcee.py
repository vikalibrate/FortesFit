import sys
import pickle

import numpy as np

import emcee

from fortesfit import FortesFit_Settings
from fortesfit import FortesFit_Filters
from fortesfit import FortesFit_ModelManagement
from fortesfit import FortesFit_Fitting

""" A module with functions that connect the common fitting functionality of FortesFit with the EMCEE Bayesian
	MCMC engine. 

"""

# ***********************************************************************************************

def	FortesFit_emcee(varying_indices, datacollection, modelcollection, nwalkers=100, nsteps=1000, tight=False, \
					write_walker_chains=False):
	""" The upper level function call to fit one object with EMCEE		
		
		varying_indices: A list of indices to the parameter_reference for the varying parameters.
		datacollection: An FortesFit CollectData instance. See FortesFit_Preparation for details.
		modelcollection: An FortesFit CollectModel instance. See FortesFit_Preparation for details.
		nwalkers: number of EMCEE walkers
		nsteps: Number of MCMC steps
		tight: Use a tight ball initialisation around the characteristic values.
			   If false (the default), the prior will be sampled to initialise the walkers.
		write_walker_chains: If set, the full chains of all walkers will be written out as a pickle file 
							called emcee_chains.pickle. This is useful for convergence checks of individual walkers.

		returns the emcee sampler after running the MCMC chains

	"""
	# Initialise the walkers
	initdist = np.empty((nwalkers,len(varying_indices)))

	for ivary,varyindex in enumerate(varying_indices):
		# Loop through the varying parameters
		if tight:
			# Start with a ball of walkers of 0.1% of the initial value around the initial location
			initval = modelcollection.parameter_reference[varyindex][2].characteristic
			initdist[:,ivary] = initval*(1.0 + 0.1*(np.random.random(size=nwalkers)*2.0-1.0))
		else:
			# Draw from the prior distribution of the parameter
			prior = modelcollection.parameter_reference[varyindex][2]
			initdist[:,ivary] = prior.draw_prior(ndraws=nwalkers)

	# Move on to MCMC
	print('Starting EMCEE run')	
	# Set up the affine-invariant sampler
	sampler = emcee.EnsembleSampler(nwalkers, len(varying_indices), \
									FortesFit_Fitting.Bayesian_probability, \
									args=(datacollection, modelcollection, varying_indices))
	for i, result in enumerate(sampler.sample(initdist, iterations=nsteps)):
		print("{0:5.1f}%".format(i*100/nsteps),end="\r")
	# sampler.run_mcmc(initdist, nsteps)
	
	if write_walker_chains:
		outfile = open('emcee_chains.pickle','wb')
		pickle.dump(sampler.chain,outfile)
		outfile.close()
	
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
		
		*** This routine is under development and should not be used at this time ***

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
