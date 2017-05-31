from sys import exit
import os.path
import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_pdf import PdfPages
#from chainconsumer import ChainConsumer
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
	
	
	def __init__(self,FortesFit_OutFile,ObjectGroupName, BurnIn = 100):
		""" Read in the FORTES-FIT MCMC outputs for a single object 
			
			FortesFit_OutFile: The output HDF5 file of FortesFit_FitSingle_emcee
			ObjectGroupName:   The name of the group within the output file correponding to the fitted SED.
			BurnIn: number of initial samples to exclude to allow for convergence of the chains. Default = 100 
			
		"""
				
		FitFile = h5py.File(FortesFit_OutFile, 'r')	 #  Open the HDF5 output file	
			
		# Basic attributes about the overall fits, from the main file metadata
		self.fit_filterids = FitFile.attrs['FilterIDs'] # Filters of the SED used in the fit
		self.fit_modelids  = FitFile.attrs['ModelIDs']  # Models used to fit the SED
		self.fit_parameter_names  = np.core.defchararray.decode(FitFile.attrs['ParameterNames']) # Ordered list of parameters that were fit

		# Object specific attributes
		galgroup = FitFile[ObjectGroupName]
		self.redshift = galgroup.attrs['Redshift']   #   Redshift of the object
		self.fit_fluxes = galgroup.attrs['Fluxes']   #   Observed-frame fluxes of the object in erg/s/cm^2/micron
		self.fit_fluxerrors = galgroup.attrs['FluxErrors']   #   Errors on the fluxes in erg/s/cm^2/micron
		
		self.chains = galgroup['emcee_chain'][()]  #  Store the entire EMCEE output chain
		self.burn_in = BurnIn
		self.all_samples = self.chains[:,BurnIn:,:].reshape((-1,len(self.fit_parameter_names)))
				
		FitFile.close()  #  Close the HDF5 output file
		
		perc_pars = {}
		for iparam,param in enumerate(self.fit_parameter_names):
			perc_pars.update({param:np.percentile(self.all_samples[:,iparam],[50])[0]})
		self.bestfit_parameters = perc_pars		
	
	
	def	percentiles(self, Quantiles = [16,50,84]):
		"""	Calculate percentile ranges for a set of input quantiles
	
			Quantiles: list or array of quantiles, Default is equivalent to -1 sigma, median, +1 sigma
		"""
		
		perc_pars = {}
		for iparam,param in enumerate(self.fit_parameter_names):
			perc_pars.update({param:np.percentile(self.all_samples[:,iparam],Quantiles)})
		
		return perc_pars


# ***********************************************************************************************

def		FortesFit_SummaryFigures(FortesFit_OutFile, BurnIn=100, single_name=''):
	""" Plot the chains, corner plot and model SEDs for a ForteFit output
		
		FortesFit_Outfile:  HDF5 file with the outputs from FortesFit
		BurnIn: The number of samples on the MCMC chains to exclude to allow for convergence
		single_name: A string corresponding to the name of the HDF5 group which contains the chains for a single object.
				    This is useful if the SEDs for only one object is required
	
	"""		
	
	FitFile = h5py.File(FortesFit_OutFile, 'r')		
	print(FitFile.attrs['Description'].decode())
	
	GalGroups = list(FitFile) # Read in the group names for all objects from the file
	FitFile.close()  # Close the file. It will be opened again when setting up the FortesFitResult objects
	
	for iobj, objectname in enumerate(GalGroups):
		
		plt.close('all') # Delete all existing plots
		
		if (len(single_name) > 0):
			if(objectname != single_name):
				continue
				
		fitresult = FortesFitResult(FortesFit_OutFile, objectname, BurnIn=BurnIn)

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
			parax.plot([0,nsamples+1],[bestparam_dict[ParameterNames[ipar]],bestparam_dict[ParameterNames[ipar]]],'k--')
			parax.axis(axrange)
			plt.ylabel(ParameterNames[ipar],fontsize=10)
			if (ipar == 0):
				plt.title(objectname)
			if (ipar == nparams-1):
				plt.xlabel(r'Samples')
			else:
				parax.tick_params(axis='x',labelbottom='off')

		param_ranges = fitresult.percentiles(Quantiles=[1,99])
		drange = [param_ranges[param][1]-param_ranges[param][0] for param in ParameterNames]
		ranges = [(param_ranges[param][0]-0.33*drange[i],param_ranges[param][1]+0.33*drange[i]) for i,param in enumerate(ParameterNames)]
		# cornerfig = ccinst.plot(figsize=0.9,extents=ranges)
		cornerfig, corneraxes = plt.subplots(nrows=nparams,ncols=nparams,figsize=(8,8))
		corner(samples,labels=ParameterNames,range=ranges,label_kwargs={'fontsize':8},fig=cornerfig)

		plt.show()

		ch = input('Continue? y/n  : ')
		if (ch[0] == 'n'):
			print('Exitting')
			return False
		
	return True


# ***********************************************************************************************

def		FortesFit_PlotModelSEDs_Raw(FortesFit_OutFile, BurnIn=100, PDF_File='',single_name='', silent=False):
	""" Plot the best-fit combined SED, model photometry, and 100 random model SEDs for a given FortesFit output file
		
		FortesFit_Outfile:  HDF5 file with the outputs from FortesFit
		BurnIn: The number of samples on the MCMC chains to exclude to allow for convergence
		PDF_File: A file to send the plotted SEDs. One page per fitted object.
		single_name: A string corresponding to the name of the HDF5 group which contains the chains for a single object.
				    This is useful if the SEDs for only one object is required
		silent: If True, no information messages are used. Serial plots are shown for 2 seconds.
	
	"""		
	
	# Initialise PDF output if necessary
	if len(PDF_File) > 0:
		if(not silent):
			print('Summary plots will be sent to '+PDF_File)
		output = PdfPages(PDF_File)
		
	FitFile = h5py.File(FortesFit_OutFile, 'r')			
	# Write out a description of the fit
	if(not silent):
		print(FitFile.attrs['Description'].decode())
	
	GalGroups = list(FitFile) # Read in the group names for all objects from the file
	FitFile.close()  # Close the file. It will be opened again when setting up the FortesFitResult objects
	
	# Initialise the wavelength and flux array that is used for plotting the best-fit model (from 100 Ang to 10mm) in microns
	ObsWave = -2.0 + np.arange(1001)*(6.0/1000.0)
	ObsFlux = np.zeros(len(ObsWave))
	
	for iobj, objectname in enumerate(GalGroups):
		
		if (len(single_name) > 0):
			if(objectname != single_name):
				continue
				
		fitresult = FortesFitResult(FortesFit_OutFile, objectname, BurnIn=BurnIn)

		Redshift   = fitresult.redshift
		Fluxes     = fitresult.fit_fluxes
		FluxErrors = fitresult.fit_fluxerrors

		Filters = [FortesFit_Filter(filterid) for filterid in fitresult.fit_filterids]
		FilterWave = np.array([filter.pivot_wavelength for filter in Filters])	
		filter_sortindex = np.argsort(FilterWave)
		modelFluxes = np.zeros(len(Filters),dtype='f8')
	
		Models = [FitModel(modelid, Redshift, fitresult.fit_filterids) for modelid in fitresult.fit_modelids]
		ParameterNames  = fitresult.fit_parameter_names
	
		bestparam_dict = fitresult.bestfit_parameters
		
		# Initialise the SED plot
		plt.close('all') # Delete all existing plots
		sedfig = plt.figure()

		# Overplot the SEDs from 100 random samples from the burned-in chains
		samples = fitresult.all_samples
		for isamp in range(100):
			parameter_sample = samples[np.random.randint(samples.shape[0]),:]
			paramdict_plot = dict(zip(ParameterNames,parameter_sample))	
			for model in Models:
				sed = model.get_pivot_sed(paramdict_plot,Redshift)
				index, = np.where(sed['observed_flux'] > 0.0) # Only interpolate over valid parts of the model SED
				tempflux = np.interp(ObsWave,np.log10(sed['observed_wavelength'][index]),np.log10(sed['observed_flux'][index]),\
									 left=-np.inf,right=-np.inf) + ObsWave
		
				plt.plot(10**(ObsWave),10**(tempflux),'grey',alpha=0.1,lw=1)

		# Obtain the best-fit SED components
		# Reset the array for the summed best-fit SED
		ObsFlux[:] = 0.0
		for model in Models:
			sed = model.get_pivot_sed(bestparam_dict,Redshift)
			index, = np.where(sed['observed_flux'] > 0.0) # Only interpolate over valid parts of the model SED
			tempflux = np.interp(ObsWave,np.log10(sed['observed_wavelength'][index]),np.log10(sed['observed_flux'][index]),\
								 left=-np.inf,right=-np.inf) + ObsWave
			ObsFlux += 10**(tempflux)
									
		# Plot the best-fit SED as a red thick line
		plt.plot(10**(ObsWave),ObsFlux,'r',lw=2)
			
		# Plot the best-fit model photometry
		for i,filter in enumerate(Filters):
			for model in Models:
				modelFluxes[i] += \
						3.63e-5*10**(model.evaluate(bestparam_dict,Redshift,filter.filterid))*filter.pivot_wavelength
		plt.plot(FilterWave,modelFluxes,color='red',lw=0,marker='o',fillstyle='none')

		index, = np.where((Fluxes > 0.0) & (FluxErrors > 0.0))
		fluxconv = FilterWave[index]
		plotfluxes = Fluxes[index]*fluxconv 
		eplotfluxes = FluxErrors[index]*fluxconv 
		plt.errorbar(FilterWave[index],plotfluxes,eplotfluxes,fmt='ko',ecolor='k')
		# Use the points with valid photometry to determine the plotting range
		axrange = [0.8*FilterWave.min(),1.2*FilterWave.max(),\
				   0.1*plotfluxes.min(),10.0*plotfluxes.max()]
		plt.axis(axrange)
		index, = np.where((Fluxes > 0.0) & (FluxErrors < 0.0))
		fluxconv = FilterWave[index]
		plt.plot(FilterWave[index],Fluxes[index]*fluxconv,'kv',markersize=10)

		plt.loglog()
		plt.xlabel(r'log Observed Wavelength ($\mu$m)')
		plt.ylabel(r'log $\nu$F$_{\nu}$ (erg s$^{-1}$)')
		plt.title(objectname)
	
		plt.show()

		if len(PDF_File) > 0:
			plt.pause(2)
			output.savefig(sedfig)
		else:
			if(not silent):
				ch = input('Continue? y/n  : ')
				if (ch[0] == 'n'):
					print('Exitting')
					return []
					break
			else:
				plt.pause(2)
		
	if len(PDF_File) > 0:
		output.close()
	
	return sedfig


# ***********************************************************************************************

def		FortesFit_PlotModelSEDs(FortesFit_OutFile, single_name='', wave_range = [1e-2,1e4], \
								BurnIn=100, PDF_File='', Nsamps=100, silent=False):
	""" Plot the best-fit combined SED, model photometry. 
		From Nsamps SEDs drawn from the joint posterior, get the error SEDs for each component and overplot.
		
		FortesFit_Outfile:  HDF5 file with the outputs from FortesFit
		single_name: A string corresponding to the name of the HDF5 group which contains the chains for a single object.
				    This is useful if the SEDs for only one object is required
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
		
	FitFile = h5py.File(FortesFit_OutFile, 'r')			
	# Write out a description of the fit
	if(not silent):
		print(FitFile.attrs['Description'].decode())
	
	GalGroups = list(FitFile) # Read in the group names for all objects from the file
	FitFile.close()  # Close the file. It will be opened again when setting up the FortesFitResult objects
	
	# Initialise the wavelength and flux array that is used for plotting the best-fit model (from 100 Ang to 10mm) in microns
	ObsWave = np.log10(wave_range[0]) + np.arange(101)*(np.log10(wave_range[1]/wave_range[0])/100.0)
	ObsFlux = np.zeros(len(ObsWave))
	
	for iobj, objectname in enumerate(GalGroups):
		
		# Skip all except the choice if single_name is set
		if (len(single_name) > 0):
			if(objectname != single_name):
				continue
				
		fitresult = FortesFitResult(FortesFit_OutFile, objectname, BurnIn=BurnIn)

		Redshift   = fitresult.redshift
		Fluxes     = fitresult.fit_fluxes
		FluxErrors = fitresult.fit_fluxerrors

		# Compile the filter wavelengths that were used in the fit
		Filters = [FortesFit_Filter(filterid) for filterid in fitresult.fit_filterids]
		FilterWave = np.array([filter.pivot_wavelength for filter in Filters])	
		filter_sortindex = np.argsort(FilterWave)  #  sort the filters in wavelength for cleaner plotting
		modelFluxes = np.zeros(len(Filters),dtype='f8')
	
		Models = [FitModel(modelid, Redshift, fitresult.fit_filterids) for modelid in fitresult.fit_modelids]
		ParameterNames  = fitresult.fit_parameter_names
	
		bestparam_dict = fitresult.bestfit_parameters
		
		# Initialise the SED plot
		plt.close('all') # Delete all existing plots

		sedfig = plt.figure()
		ax = sedfig.add_axes([0.12,0.12,0.83,0.8])

		# Load a color map and assign colors to all models using a certain colormap
		plotnorm = Normalize(vmin=0,vmax=len(Models)-1)
		plotcmap = plt.get_cmap('brg')
		plotcols = ScalarMappable(cmap=plotcmap,norm=plotnorm)		

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
		# Now get a lower and upper scatter at each wavelength point of the models and plot a filled polygon
		for imodel in range(len(Models)):
			index, = np.where(sample_seds[0,:,imodel] != 0.0) # get range of sample data from first model SED in set
			scatter = np.percentile(sample_seds[:,index,imodel],[16,50,84],axis=0,interpolation='nearest')
			plt.fill_between(10**(ObsWave[index]),scatter[0,:],scatter[2,:],\
							 color=plotcols.to_rgba(imodel),alpha=0.2,lw=1)
			plt.plot(10**(ObsWave[index]),scatter[1,:],color=plotcols.to_rgba(imodel),lw=2)


		# Obtain the best-fit SED components
		# Reset the array for the summed best-fit SED
		ObsFlux[:] = 0.0
		for model in Models:
			sed = model.get_pivot_sed(bestparam_dict,Redshift)
			index, = np.where(sed['observed_flux'] > 0.0) # Only interpolate over valid parts of the model SED
			tempflux = np.interp(ObsWave,np.log10(sed['observed_wavelength'][index]),np.log10(sed['observed_flux'][index]),\
								 left=-np.inf,right=-np.inf) + ObsWave
			ObsFlux += 10**(tempflux)
									
		# Plot the best-fit SED as a red thick line
		plt.plot(10**(ObsWave),ObsFlux,'k',lw=2)
			
		# Plot the best-fit model photometry
		for i,filter in enumerate(Filters):
			for model in Models:
				modelFluxes[i] += \
						3.63e-5*10**(model.evaluate(bestparam_dict,Redshift,filter.filterid))*filter.pivot_wavelength
		plt.plot(FilterWave,modelFluxes,color='red',lw=0,marker='o',fillstyle='none')

		index, = np.where((Fluxes > 0.0) & (FluxErrors > 0.0))
		fluxconv = FilterWave[index]
		plotfluxes = Fluxes[index]*fluxconv 
		eplotfluxes = FluxErrors[index]*fluxconv 
		plt.errorbar(FilterWave[index],plotfluxes,eplotfluxes,fmt='ko',ecolor='k')
		# Use the points with valid photometry to determine the plotting range
		axrange = [0.8*FilterWave.min(),1.2*FilterWave.max(),\
				   0.1*plotfluxes.min(),10.0*plotfluxes.max()]
		plt.axis(axrange)
		index, = np.where((Fluxes > 0.0) & (FluxErrors < 0.0))
		fluxconv = FilterWave[index]
		plt.plot(FilterWave[index],Fluxes[index]*fluxconv,'kv',markersize=10)

		plt.loglog()
		plt.xlabel(r'log Observed Wavelength ($\mu$m)',size='x-large')
		plt.ylabel(r'log $\nu$F$_{\nu}$ (erg s$^{-1}$)',size='x-large')
		plt.title(objectname,size='xx-large')
	
		ax.tick_params(axis='both',labelsize='large')
		ax.set_xlim(wave_range[0],wave_range[1])

		if len(PDF_File) > 0:
			plt.pause(2)
			output.savefig(sedfig)
		else:
			if(not silent):
				ch = input('Continue? y/n  : ')
				if (ch[0] == 'n'):
					print('Exitting')
					return []
					break
			else:
				plt.pause(2)
		
	if len(PDF_File) > 0:
		output.close()
	
	return sedfig

# ***********************************************************************************************


def		FortesFit_BestModelSED(FortesFit_OutFile, ObjectGroupName, BurnIn=100):
	""" Returns the best-fit SED components
		
		FortesFit_Outfile:  HDF5 file with the outputs from FortesFit
		ObjectGroupName: The name of the group within the output file correponding to the fitted SED.
		BurnIn: The number of samples on the MCMC chains to exclude to allow for convergence
		
		Returns the best-fit SED as a dictionary with keys = [ObsWave,all Models ....]
	
	"""		
	
	FitFile = h5py.File(FortesFit_OutFile, 'r')			
	
	GalGroups = list(FitFile) # Read in the group names for all objects from the file
	FitFile.close()  # Close the file. It will be opened again when setting up the FortesFitResult objects
	
	# Initialise the output SED dictionary
	outsed = {}
	
	# Initialise the wavelength and flux array that is used for plotting the best-fit model (from 100 Ang to 10mm) in microns
	ObsWave = -2.0 + np.arange(1001)*(6.0/1000.0)
	ObsFlux = np.zeros(len(ObsWave))
	
	# Store the wavelength array to the output
	outsed.update({'ObsWave':10**(ObsWave)})

	fitresult = FortesFitResult(FortesFit_OutFile, ObjectGroupName, BurnIn=BurnIn)

	Redshift   = fitresult.redshift
	Models = [FitModel(modelid, Redshift, fitresult.fit_filterids) for modelid in fitresult.fit_modelids]
	bestparam_dict = fitresult.bestfit_parameters
	
	# Obtain the best-fit SED components
	for model in Models:
		sed = model.get_pivot_sed(bestparam_dict,Redshift)
		index, = np.where(sed['observed_flux'] > 0.0) # Only interpolate over valid parts of the model SED
		tempflux = np.interp(ObsWave,np.log10(sed['observed_wavelength'][index]),np.log10(sed['observed_flux'][index]),\
							 left=-np.inf,right=-np.inf) + ObsWave					
		# Store the best-fit SED to the output
		outsed.update({'Model{0:2d}'.format(model.modelid):10**(tempflux)})
		
	return outsed


# ***********************************************************************************************


def		FortesFit_SingleParameterDistribution(ParameterName, FortesFit_OutFile, BurnIn=100, single_name=''):
	""" Plot the marginalised distributions of a single parameter
		
		
		ParameterName:  The parameter for which the distribution should be plotted
		FortesFit_Outfile:  HDF5 file with the outputs from FortesFit
		BurnIn: The number of samples on the MCMC chains to exclude to allow for convergence
		single_name: A string corresponding to the name of the HDF5 group which contains the chains for a single object.
				    This is useful if the SEDs for only one object is required
	
	"""		
		
	FitFile = h5py.File(FortesFit_OutFile, 'r')		
	print(FitFile.attrs['Description'].decode())
	
	GalGroups = list(FitFile) # Read in the group names for all objects from the file
	FitFile.close()  # Close the file. It will be opened again when setting up the FortesFitResult objects
	
	for iobj, objectname in enumerate(GalGroups):
		
		plt.close('all') # Delete all existing plots
		
		if (len(single_name) > 0):
			if(objectname != single_name):
				continue
				
		fitresult = FortesFitResult(FortesFit_OutFile, objectname, BurnIn=BurnIn)

		Redshift   = fitresult.redshift
		ParameterNames  = fitresult.fit_parameter_names
		parindex, = np.where(ParameterNames == ParameterName) # Identify the parameter to plot	

		samples = fitresult.all_samples
		plotvals = np.squeeze(samples[:,parindex])
			
		distfig = plt.figure()
		ax = plt.axes()
		n,bins,patches = ax.hist(plotvals,bins=100,histtype='stepfilled')
		plt.xlabel(ParameterName)
		plt.title(objectname)

		plt.show()

		ch = input('Continue? y/n  : ')
		if (ch[0] == 'n'):
			print('Exitting')
			break
		
	
	return []


# ***********************************************************************************************

def		FortesFit_ExaminePrior(parameter_names, prior_dictionary, nsamples=1000):
	""" Plot the prior distributions and mark the initialisation value
		
		parameter_names:   A list of parameter names 
		prior_dictionary:  A dictionary with parameter names as keys. The values are 2-element lists
						    with the initisation value and and instance of 
						    a function that provides random independent samples from the prior (such as SciPy RVS)
		nsamples: The number of samples to make to generate the prior distributions. 1000 as default.
	
	"""		

	plt.close('all')
	plt.ion()
	
	NumberParameters = len(parameter_names)
	
	priorfig = plt.figure(figsize=(9,9))
	xsta = 0.01
	xend = 0.99
	dx = (xend-xsta)/2
	ysta = 0.15
	yend = 0.99
	dy = (yend-ysta)/(NumberParameters/2)
	
	for ipar in range(NumberParameters):
		
		ixx = ipar % 2
		iyy = np.int(ipar/2)
		
		parax = priorfig.add_axes([xsta+ixx*dx+0.05*dx,yend-(iyy+1)*dy+0.2*dy,0.9*dx,0.8*dy])
		prior_samples = prior_dictionary[parameter_names[ipar]][1].rvs(size=nsamples)				
		histprops = parax.hist(prior_samples,bins=100,\
							   color='green',histtype='stepfilled',normed=True,alpha=0.5)

		# Control the ticks and labels
		parax.set_xlabel(parameter_names[ipar],fontsize=12)
		parax.tick_params(axis='y',left='off',right='off',labelleft='off')
		parax.tick_params(axis='x',labelsize=10)

		# Determine the plotting ranges, including a small buffer around the edges of the distribution
		dxx = np.max(prior_samples) - np.min(prior_samples)
		newxrange = [np.min(prior_samples) - 0.1*dxx, np.max(prior_samples) + 0.1*dxx]
		parax.set_xlim(left=newxrange[0], right=newxrange[1])		
		
		# Plot a line for the initialisation value
		initval = prior_dictionary[parameter_names[ipar]][0]
		yrange = parax.get_ylim()
		parax.plot([initval,initval],yrange,'r-',lw=3)	

	print('Do these priors look sensible? Please confirm before we move to time-intensive fits.')
	ch = input('Continue? y/n  : ')
	if (ch[0] == 'n'):
		raise(ValueError('Priors are malformed.'))
	else:
		plt.ioff()
		return ch

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
