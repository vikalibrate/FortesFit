import sys
import os
import glob

import numpy as np

import pymultinest

from fortesfit import FortesFit_Settings
from fortesfit import FortesFit_Filters
from fortesfit import FortesFit_ModelManagement
from fortesfit import FortesFit_Fitting

""" A module with functions that connect the common fitting functionality of FortesFit with the Pymultinest
	Nested Sampling engine. 

"""

# ***********************************************************************************************

def	FortesFit_multinest(varying_indices, datacollection, modelcollection, 
						verbose=False, evidence_tolerance=0.5, sampling_efficiency=0.8,
						outputfiles_basename='fortesfit_'):
	""" The upper level function call to fit one object with PyMultinest		
		
		varying_indices: A list of indices to the parameter_reference for the varying parameters.
		datacollection: An FortesFit CollectData instance. See FortesFit_Preparation for details.
		modelcollection: An FortesFit CollectModel instance. See FortesFit_Preparation for details.
		keyword parameters: from PyMultinest, see PyMultinest documentation for details

		returns a pymultinest Analyzer instance, see PyMultinest documentation for details

	"""
	
	def	multinest_prior(cube, ndim, nparams):
		""" Interface function as required by Multinest/Pymultinest that takes the unit cube and
			returns the parameter cube	
		
			cube: as input, this is the unit cube with ndim parameters. As output it is the parameter cube.
			ndim: number of dimensions of parameter cube, which will be equal to number of parameters by default
			nparams: number of parameters

		"""
		# Using variables from the enclosing scope that aren't passable as arguments with Pymultinest
		nonlocal varying_indices, modelcollection
		
		# Map the cube to the parameter using the prior arrays	
		for iparam in range(nparams):
			prior_range = modelcollection.parameter_reference[varying_indices[iparam]][2].prior_grid[0,:]
			cube[iparam] = prior_range[0] + cube[iparam]*(prior_range[-1] - prior_range[0])

	def	multinest_loglikelihood(cube, ndim, nparams):
		""" Interface function as required by Multinest/Pymultinest that takes the parameter cube and
			returns the loglikelihood	
		
			cube: as input, this is the unit cube with ndim parameters. As output it is the parameter cube.
			ndim: number of dimensions of parameter cube, which will be equal to number of parameters by default
			nparams: number of parameters

		"""
		# Using variables from the enclosing scope that aren't passable as arguments with Pymultinest
		nonlocal varying_indices, datacollection, modelcollection
		
		varying_parameters = [cube[i] for i in range(ndim)]
		
		return FortesFit_Fitting.Bayesian_probability(varying_parameters,datacollection,modelcollection,varying_indices)


	outputfiles_basename = 'multinest_output/'+outputfiles_basename # Include the MultiNest working directory
	
	# Run Pymultinest with the functions defined above	
	pymultinest.run(multinest_loglikelihood, multinest_prior, len(varying_indices), 
					outputfiles_basename=outputfiles_basename,
					verbose=verbose,evidence_tolerance=evidence_tolerance)
	
	return pymultinest.Analyzer(len(varying_indices),outputfiles_basename=outputfiles_basename)

# ***********************************************************************************************
def	Multinest_cleanup(outputfiles_basename='fortesfit_'):
	""" Cleans up the local Multinest temporary working directory.
		Call when all processing from the directory is completed.
	"""

	outputfiles_basename = 'multinest_output/'+outputfiles_basename # Include the MultiNest working directory
	mnfiles = glob.glob(outputfiles_basename+'*')
	for file in mnfiles:
		os.remove(file)
	

