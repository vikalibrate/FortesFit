import sys
import os.path
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_pdf import PdfPages

from corner import corner

import h5py

from fortesfit import FortesFit_Settings
from fortesfit.FortesFit_Filters import FortesFit_Filter
from fortesfit.FortesFit_ModelManagement import FullModel, FitModel

""" A module with classes and functions to visualise and analyse FortesFit outputs  

"""
# ***********************************************************************************************

class FortesFitResult:
	""" Representation of the FORTES-FIT MCMC output for a single object """
	
	
	def __init__(self,FortesFit_OutFile, BurnIn = 100):
		""" Read in the FORTES-FIT MCMC outputs for a single object 
			
			FortesFit_OutFile: The output HDF5 file of FortesFit_FitSingle_emcee
			BurnIn: number of initial samples to exclude to allow for convergence of the chains. Default = 100 
			
		"""
				
		FitFile = h5py.File(FortesFit_OutFile, 'r')	 #  Open the HDF5 output file	

		self.objectname = FitFile.attrs['Object Name'].decode()
		self.fit_description = FitFile.attrs['Description'].decode()
			
		priors = OrderedDict()  #  A dictionary that stores priors for all parameters, including redshift
			
		# Object specific attributes
		
		# Redshift of the object
		redshift_array = FitFile.attrs['Redshift']
		if redshift_array.shape[0] == 1:
			# Redshift is fixed
			self.redshift = redshift_array[0]
		else:
			# Redshift is a varied parameter and there is no fixed value
			self.redshift = -99.0
		priors.update({'Redshift':redshift_array})   

		self.fit_filterids = FitFile['Photometry/FilterIDs'][()] # Filters of the SED used in the fit
		self.fit_fluxes = FitFile['Photometry/Fluxes'][()] # Filters of the SED used in the fit
		self.fit_fluxerrors = FitFile['Photometry/FluxErrors'][()] # Errors on the fluxes in erg/s/cm^2/micron

		# Basic attributes about the overall fits, from the main file metadata
		self.fit_modelids  = FitFile['Model'].attrs['ModelIDs']  # Models used to fit the SED

		for modelid in self.fit_modelids:
			subgroupname = 'Model/Model{0:2d}'.format(modelid)
			for uparam in FitFile[subgroupname].keys():
				dataname = subgroupname+'/'+uparam
				priors.update({uparam:FitFile[dataname][()]})
		
		self.priors = priors
		paramnames = np.array(list(priors.keys()))	
		
		self.fit_parameter_names  = np.core.defchararray.decode(FitFile['Chain/Varying_parameters'][()]) # Ordered list of parameters that were fit
		matchindex = np.zeros(len(self.fit_parameter_names),dtype='i2')
		for iparam,param in enumerate(self.fit_parameter_names):
			index, = np.where(paramnames == param)
			matchindex[iparam] = index[0]
		self.fit_parameter_indices = matchindex
		
		self.chains = FitFile['Chain/emcee_chain'][()]  #  Store the entire EMCEE output chain
		self.burn_in = BurnIn
		self.all_samples = self.chains[:,BurnIn:,:].reshape((-1,len(self.fit_parameter_names)))
				
		FitFile.close()  #  Close the HDF5 output file
		

		# Best-fit or fixed values for each parameter
		perc_pars = {}
		# First variable parameters
		for iparam,param in enumerate(self.fit_parameter_names):
			perc_pars.update({param:(np.percentile(self.all_samples[:,iparam],[50])[0],'Fit')})
		# Then fixed parameters
		for param in self.priors.keys():
			if self.priors[param].shape[0] == 1:
				# Fixed parameters have single element prior arrays
				perc_pars.update({param:(self.priors[param][0],'Fixed')})
		
		self.bestfit_parameters = perc_pars		
	
	
	def	percentiles(self, Quantiles = [16,50,84]):
		"""	Calculate percentile ranges on varying parameters for a set of input quantiles
	
			Quantiles: list or array of quantiles, Default is equivalent to -1 sigma, median, +1 sigma
		"""
		
		perc_pars = {}
		# Variable parameters only
		for iparam,param in enumerate(self.fit_parameter_names):
			perc_pars.update({param:np.percentile(self.all_samples[:,iparam],Quantiles)})
		
		return perc_pars


# ***********************************************************************************************

def	SummaryFigures(FortesFit_OutFile, BurnIn=100):
	""" Plot the chains, corner plot and model SEDs for a ForteFit output
		
		FortesFit_Outfile:  HDF5 file with the outputs from FortesFit
		BurnIn: The number of samples on the MCMC chains to exclude to allow for convergence
	
	"""		
	
	plt.close('all') # Delete all existing plots
	
	fitresult = FortesFitResult(FortesFit_OutFile, BurnIn=BurnIn)
	print(fitresult.fit_description)	
	ObjectName = fitresult.objectname 
	
	Redshift   = fitresult.redshift
	Fluxes     = fitresult.fit_fluxes
	FluxErrors = fitresult.fit_fluxerrors

	Filters = [FortesFit_Filter(filterid) for filterid in fitresult.fit_filterids]
	FilterWave = np.array([filter.pivot_wavelength for filter in Filters])	
	filter_sortindex = np.argsort(FilterWave)
	modelFluxes = np.zeros(len(Filters),dtype='f8')

	ParameterNames  = fitresult.fit_parameter_names
	
	chains = fitresult.chains
	nwalkers = chains.shape[0]
	nsamples = chains.shape[1]
	nparams  = chains.shape[2]
	samples = fitresult.all_samples
	bestparam_dict = fitresult.bestfit_parameters

	# A figure showing all the chains
	chainfig = plt.figure(figsize=(8,9))
	sampleindex = np.arange(nsamples)
	xsta = 0.1
	xend = 0.95
	dx = (xend-xsta)
	ysta = 0.05
	yend = 0.95
	dy = (yend-ysta)/nparams
	for ipar in range(nparams):
		parax = chainfig.add_axes([xsta,yend-(ipar+1)*dy,dx,0.95*dy])
		for i in range(nwalkers):
			parax.plot(sampleindex,chains[i,:,ipar],'grey',alpha=0.5)
		axrange = parax.axis()
		parax.plot([0,nsamples+1],[bestparam_dict[ParameterNames[ipar]][0],bestparam_dict[ParameterNames[ipar]][0]],'k--')
		parax.axis(axrange)
		plt.ylabel(ParameterNames[ipar],fontsize=10)
		if (ipar == 0):
			plt.title(ObjectName) 
		if (ipar == nparams-1):
			plt.xlabel(r'Samples')
		else:
			parax.tick_params(axis='x',labelbottom='off')

	# Corner plot
	param_ranges = fitresult.percentiles(Quantiles=[1,99])
	drange = [param_ranges[param][1]-param_ranges[param][0] for param in ParameterNames]
	ranges = [(param_ranges[param][0]-0.33*drange[i],param_ranges[param][1]+0.33*drange[i]) for i,param in enumerate(ParameterNames)]
	# cornerfig = ccinst.plot(figsize=0.9,extents=ranges)
	cornerfig, corneraxes = plt.subplots(nrows=nparams,ncols=nparams,figsize=(8,8))
	corner(samples,labels=ParameterNames,range=ranges,label_kwargs={'fontsize':8},fig=cornerfig)

	plt.show()
		
	return None


# ***********************************************************************************************

def	PlotModelSEDs(FortesFit_OutFile, wave_range = [1e-2,1e4], BurnIn=100, PDF_File='', Nsamps=100, silent=False):
	""" Plot the best-fit combined SED, model photometry. 
		From Nsamps SEDs drawn from the joint posterior, get the error SEDs for each component and overplot.
		
		FortesFit_Outfile:  HDF5 file with the outputs from FortesFit
		wave_range: Wavelength range to plot the SEDs, list-like two-element, ([starting, ending] wavelength in microns)
		BurnIn: The number of samples on the MCMC chains to exclude to allow for convergence
		PDF_File: A file to send the plotted SEDs. One page per fitted object.
		Nsamps: Number of samples to draw from joint posterior. Default = 100. Will affect the speed of the routine.
		silent: If True, no information messages are used. Serial plots are shown for 2 seconds.
	
	"""		
	
	# Initialise PDF output if necessary
	if len(PDF_File) > 0:
		if(not silent):
			print('Summary plots will be sent to '+PDF_File)
		output = PdfPages(PDF_File)
		
	# Initialise the wavelength and flux array that is used for plotting the best-fit model (from 100 Ang to 10mm) in microns
	ObsWave = np.log10(wave_range[0]) + np.arange(101)*(np.log10(wave_range[1]/wave_range[0])/100.0)
	ObsFlux = np.zeros(len(ObsWave))
				
	fitresult = FortesFitResult(FortesFit_OutFile, BurnIn=BurnIn)
	# Write out a description of the fit
	if(not silent):
		print(fitresult.fit_description)	
	ObjectName = fitresult.objectname 

	# Compile the filter wavelengths that were used in the fit
	FilterIDs = fitresult.fit_filterids
	Filters = [FortesFit_Filter(filterid) for filterid in FilterIDs]
	FilterWave = np.array([filter.pivot_wavelength for filter in Filters])	

	Fluxes     = fitresult.fit_fluxes
	FluxErrors = fitresult.fit_fluxerrors

	#filter_sortindex = np.argsort(FilterWave)  #  sort the filters in wavelength for cleaner plotting
	modelFluxes = np.zeros(len(Filters),dtype='f8')

	Nmodels = len(fitresult.fit_modelids)

	# Create a list of model parameter dictionaries with best-fit/fixed parameter values
	characteristics = [{} for imodel in range(Nmodels)]
	for param in fitresult.bestfit_parameters.keys():
		for imodel in range(Nmodels):
			if param[0:2] == '{0:2d}'.format(fitresult.fit_modelids[imodel]):
				characteristics[imodel].update({param.split('_')[1]:fitresult.bestfit_parameters[param][0]})

	paramdict_plot = characteristics.copy()	
	# Initialise the SED plot
	plt.close('all') # Delete all existing plots

	sedfig = plt.figure()
	ax = sedfig.add_axes([0.12,0.12,0.83,0.8])

	# Load a color map and assign colors to all models using a certain colormap
	plotnorm = Normalize(vmin=0,vmax=Nmodels-1)
	plotcmap = plt.get_cmap('brg')
	plotcols = ScalarMappable(cmap=plotcmap,norm=plotnorm)		

	# Obtain the flattened chains
	samples = fitresult.all_samples

	# Overplot the SEDs from 100 random samples from the burned-in chains
	sample_seds = np.zeros((Nsamps,len(ObsWave),Nmodels))
	for isamp in range(Nsamps):
		# Take a random position along the chain
		parameter_sample  = samples[np.random.randint(samples.shape[0]),:]
		paramdict_varying = dict(zip(fitresult.fit_parameter_names,parameter_sample))	
		if fitresult.redshift == -99.0:
			Redshift = paramdict_varying['Redshift']
		else:
			Redshift = fitresult.redshift
		for imodel,modelid in enumerate(fitresult.fit_modelids):
			model = FitModel(modelid,Redshift,FilterIDs)
			for param in characteristics[imodel].keys():
				uparam = '{0:2d}_'.format(model.modelid)+param
				if uparam in paramdict_varying:
					paramdict_plot[imodel][param] = paramdict_varying[uparam]
			sed = model.get_pivot_sed(paramdict_plot[imodel],Redshift)
			index, = np.where(sed['observed_flux'] > 0.0) # Only interpolate over valid parts of the model SED
			tempflux = np.interp(ObsWave,np.log10(sed['observed_wavelength'][index]),np.log10(sed['observed_flux'][index]),\
								 left=-np.inf,right=-np.inf) + ObsWave
	
			sample_seds[isamp,:,imodel] = 10**(tempflux)
	# Now get a lower and upper scatter at each wavelength point of the models and plot a filled polygon
	for imodel in range(Nmodels):
		index, = np.where(sample_seds[0,:,imodel] != 0.0) # get range of sample data from first model SED in set
		scatter = np.percentile(sample_seds[:,index,imodel],[16,50,84],axis=0,interpolation='nearest')
		plt.fill_between(10**(ObsWave[index]),scatter[0,:],scatter[2,:],\
						 color=plotcols.to_rgba(imodel),alpha=0.2,lw=1)
		plt.plot(10**(ObsWave[index]),scatter[1,:],color=plotcols.to_rgba(imodel),lw=2)


	# Obtain the best-fit SED components
	# Reset the array for the summed best-fit SED
	ObsFlux[:] = 0.0
	Redshift = fitresult.bestfit_parameters['Redshift'][0]
	for imodel,modelid in enumerate(fitresult.fit_modelids):
		model = FitModel(modelid,Redshift,FilterIDs)
		sed = model.get_pivot_sed(characteristics[imodel],Redshift)
		index, = np.where(sed['observed_flux'] > 0.0) # Only interpolate over valid parts of the model SED
		tempflux = np.interp(ObsWave,np.log10(sed['observed_wavelength'][index]),np.log10(sed['observed_flux'][index]),\
							 left=-np.inf,right=-np.inf) + ObsWave
		ObsFlux += 10**(tempflux)
								
	# Plot the best-fit SED as a black thick line
	plt.plot(10**(ObsWave),ObsFlux,'k',lw=2)
		
	# Plot the best-fit model photometry
	for ifilt,filter in enumerate(Filters):
		for imodel,modelid in enumerate(fitresult.fit_modelids):
			model = FitModel(modelid,Redshift,FilterIDs)
			modelFluxes[ifilt] += \
					3.63e-5*10**(model.evaluate(characteristics[imodel],Redshift,filter.filterid))*filter.pivot_wavelength
	plt.plot(FilterWave,modelFluxes,color='red',lw=0,marker='o',fillstyle='none')

	# Use the points with valid photometry to determine the plotting range
	index, = np.where(Fluxes > 0.0)
	fluxconv = FilterWave[index]
	limitfluxes = Fluxes[index]*fluxconv
	axrange = [0.8*FilterWave.min(),1.2*FilterWave.max(),\
			   0.1*limitfluxes.min(),10.0*limitfluxes.max()]
	plt.axis(axrange)
	
	# Plot the photometric points
	index, = np.where((Fluxes > 0.0) & (FluxErrors > 0.0))
	fluxconv = FilterWave[index]
	plotfluxes = Fluxes[index]*fluxconv 
	eplotfluxes = FluxErrors[index]*fluxconv 
	plt.errorbar(FilterWave[index],plotfluxes,eplotfluxes,fmt='ko',ecolor='k')
	index, = np.where((Fluxes > 0.0) & (FluxErrors < 0.0))
	fluxconv = FilterWave[index]
	plt.plot(FilterWave[index],Fluxes[index]*fluxconv,'kv',markersize=10)

	plt.loglog()
	plt.xlabel(r'log Observed Wavelength ($\mu$m)',size='x-large')
	plt.ylabel(r'log $\nu$F$_{\nu}$ (erg s$^{-1}$)',size='x-large')
	plt.title(ObjectName,size='xx-large')

	ax.tick_params(axis='both',labelsize='large')
	ax.set_xlim(wave_range[0],wave_range[1])

	if len(PDF_File) > 0:
		output.savefig(sedfig)
		output.close()
	
	return sedfig


# ***********************************************************************************************

#  Needs development
def		FortesFit_BestModelSED(FortesFit_OutFile, BurnIn=100, wave_range = [1e-2,1e4],Nsamps=100):
	""" Returns the best-fit SED components
		
		FortesFit_Outfile:  HDF5 file with the outputs from FortesFit
		ObjectGroupName: The name of the group within the output file correponding to the fitted SED.
		BurnIn: The number of samples on the MCMC chains to exclude to allow for convergence
		wave_range: Wavelength range to plot the SEDs, list-like two-element, ([starting, ending] wavelength in microns)
		Nsamps: Number of samples to draw from joint posterior. Default = 100. Will affect the speed of the routine.
		
		Returns the best-fit SED as a dictionary with keys = [ObsWave,all Models ....]
	
	"""		
	
	FitFile = h5py.File(FortesFit_OutFile, 'r')			
	
	GalGroups = list(FitFile) # Read in the group names for all objects from the file
	FitFile.close()  # Close the file. It will be opened again when setting up the FortesFitResult objects
	
	# Initialise the output SED dictionary
	outsed = {}
	
	# Initialise the wavelength and flux array that is used for plotting the best-fit model (from 100 Ang to 10mm) in microns
	ObsWave = np.log10(wave_range[0]) + np.arange(1001)*(np.log10(wave_range[1]/wave_range[0])/1000.0)
	ObsFlux = np.zeros(len(ObsWave))
	
	# Store the wavelength array to the output
	outsed.update({'ObsWave':10**(ObsWave)})

	fitresult = FortesFitResult(FortesFit_OutFile, ObjectGroupName, BurnIn=BurnIn)

	Redshift   = fitresult.redshift
	Models = [FitModel(modelid, Redshift, fitresult.fit_filterids) for modelid in fitresult.fit_modelids]
	ParameterNames  = fitresult.fit_parameter_names

	bestparam_dict = fitresult.bestfit_parameters
	
	samples = fitresult.all_samples

	# Overplot the SEDs from 100 random samples from the burned-in chains
	sample_seds = np.zeros((Nsamps,len(ObsWave),len(Models)))
	for isamp in range(Nsamps):
		parameter_sample = samples[np.random.randint(samples.shape[0]),:]
		paramdict_plot = dict(zip(ParameterNames,parameter_sample))	
		for imodel,model in enumerate(Models):
			sed = model.get_pivot_sed(paramdict_plot,Redshift)
			index, = np.where(sed['observed_flux'] > 0.0) # Only interpolate over valid parts of the model SED
			tempflux = np.interp(ObsWave,np.log10(sed['observed_wavelength'][index]),np.log10(sed['observed_flux'][index]),\
								 left=-np.inf,right=-np.inf) + ObsWave	
			sample_seds[isamp,:,imodel] = 10**(tempflux)

	
	# Take the median SED as the best one for each component. Add together to get the best total SED
	for imodel in range(len(Models)):
# 		ObsFlux[:] = 0.0
# 		index, = np.where(sample_seds[0,:,imodel] != 0.0) # get range of sample data from first model SED in set
# 		scatter = np.percentile(sample_seds[:,index,imodel],[50],axis=0,interpolation='nearest')
# 		ObsFlux[index] = scatter[0,:]
		med_sed = np.percentile(sample_seds[:,:,imodel],[50],axis=0,interpolation='nearest')
		outsed.update({'Model{0:2d}'.format(Models[imodel].modelid):med_sed[0,:]})
		ObsFlux += med_sed[0,:]

	outsed.update({'ObsFlux':ObsFlux})
	
# 
# 	# Obtain the best-fit SED components
# 	for model in Models:
# 		sed = model.get_pivot_sed(bestparam_dict,Redshift)
# 		index, = np.where(sed['observed_flux'] > 0.0) # Only interpolate over valid parts of the model SED
# 		tempflux = np.interp(ObsWave,np.log10(sed['observed_wavelength'][index]),np.log10(sed['observed_flux'][index]),\
# 							 left=-np.inf,right=-np.inf) + ObsWave					
# 		# Store the best-fit SED to the output
# 		outsed.update({'Model{0:2d}'.format(model.modelid):10**(tempflux)})

	
		
	return outsed

# ***********************************************************************************************

def		examine_model_seds(ModelID, nsamples=3, filterids=[], wave_range = [1e-2,1e4]):
	""" Plot model SEDs and model photometry at some randomly selected points on the parameter grid 
		
		ModelID:  The FortesFit ID for the model
		nsamples: The number of samples of the prior grid points to use. Default=3
		filterids: List-like, the choice of filterids for the model photometry to plot. 
					If empty (default), all filters are plotted.
		wave_range: Wavelength range to plot the SEDs, list-like two-element, ([starting, ending] wavelength in microns)
	
	"""		

	fullmodel = FullModel(ModelID)
	# Use the lowest redshift in the model grid for evaluation. zero index is a dummy
	redshift  = fullmodel.pivot_redshifts[np.int(len(fullmodel.pivot_redshifts)/2)] 
	# If no filterids are provided, instantiate the full model to get the full list of filterids
	if len(filterids) == 0:
		filterids = fullmodel.filterids
	
	# Instantiate the fitmodel used for evaluation
	model = FitModel(ModelID,redshift,filterids)
	
	# Compile the filter wavelengths that were used in the fit
	Filters = np.array([FortesFit_Filter(filterid) for filterid in filterids])
	FilterWave = np.array([filter.pivot_wavelength for filter in Filters])	
	sortindex = np.argsort(FilterWave)  #  sort the filters in wavelength for cleaner plotting
	Filters    = Filters[sortindex]
	FilterWave = FilterWave[sortindex]
	modelFluxes = np.zeros(len(Filters),dtype='f8')
	
	
	# Initialise the SED plot
	plt.close('all') # Delete all existing plots

	sedfig = plt.figure()
	ax = sedfig.add_axes([0.12,0.12,0.83,0.8])

	# Outer loop for each random set of parameters
	for isamp in range(nsamples):
		param_dict = {model.scale_parameter_name:model.scale_parameter_value}
		# Loop over shape parameters and select a random draw from each	
		for param in model.shape_parameter_names:
			randindex = np.random.randint(len(model.shape_parameters[param]))
			param_dict.update({param:model.shape_parameters[param][randindex]})

		# Obtain and plot the SED for this set of parameters
		sed = model.get_pivot_sed(param_dict,redshift)
		ax.plot(sed['observed_wavelength']/(1.0+redshift),sed['observed_wavelength']*sed['observed_flux'],'k')
			
		# Plot the model photometry for this set of parameters
		for i,filter in enumerate(Filters):
			modelFluxes[i] = \
					3.63e-5*10**(model.evaluate(param_dict,redshift,filter.filterid))*filter.pivot_wavelength
		index, = np.where(np.isfinite(modelFluxes))
		ax.plot(FilterWave[index]/(1.0+redshift),modelFluxes[index],color='red',lw=0,marker='+')
		if isamp == 0:
			yrange = [modelFluxes[index].min(),modelFluxes[index].max()]
		else:
			if (modelFluxes[index].min() < yrange[0]):
				yrange[0] = modelFluxes[index].min()
			if (modelFluxes[index].max() > yrange[1]):
				yrange[1] = modelFluxes[index].max()
		
	plt.axis([wave_range[0],wave_range[1],yrange[0],yrange[1]])

	plt.loglog()
	plt.xlabel(r'log Rest Wavelength ($\mu$m)',size='x-large')
	plt.ylabel(r'log $\nu$F$_{\nu}$ (erg s$^{-1}$)',size='x-large')
	plt.title(model.description,size='large')

	ax.tick_params(axis='both',labelsize='large')

	return sedfig
