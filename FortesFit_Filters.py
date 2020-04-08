from sys import exit
import os
import glob
import numpy as np
from scipy import integrate
from astropy.table import Table
from astropy.io import votable as votable_routines
from fortesfit import FortesFit_Settings
 
""" A module that handles filter registration and management in FORTES-AGN  """

# ***********************************************************************************************

class FortesFit_Filter:
	""" Representation of the FORTES-AGN version of the filter throughput for a given filter ID 
		
		Filter IDs are unique 6 digit positive integers """
	
	def __init__(self,FilterID):
		""" Read in the FORTES-AGN version of the filter throughput for a given filter ID """
		
		self.filterid = FilterID
		FilterFile = FortesFit_Settings.FilterDirectory+'{0:6d}.fortesfilter.xml'.format(FilterID)
		try:
			filter_table = votable_routines.parse(FilterFile)
		except IOError:
			print('Filter {0:6d} has not been registered!'.format(FilterID))
		
		self.wavelength   = filter_table.resources[0].tables[0].array['Wavelength'].data
		self.throughput   = filter_table.resources[0].tables[0].array['Throughput'].data
		for info in filter_table.resources[0].infos:
			if(info.name == 'format'):
				self.format = info.value
			if(info.name == 'description'):
				self.description = info.value

		# Evaluate the pivot wavelength (which can be used to convert between fnu and flambda)
		# See Bessell and Murphy (2012) for details.
		if (self.format == 'energy'):
			FilterFlux = integrate.trapz(self.throughput*np.full(len(self.wavelength),1.0),self.wavelength)
			FilterNorm = integrate.trapz(self.throughput*(1.0/(self.wavelength**2.0)),self.wavelength)
		else:
			FilterFlux = integrate.trapz(self.throughput*self.wavelength*np.full(len(self.wavelength),1.0),self.wavelength)
			FilterNorm = integrate.trapz(self.throughput*self.wavelength*(1.0/(self.wavelength**2.0)),self.wavelength)
		self.pivot_wavelength =  np.sqrt(FilterFlux/FilterNorm)
	
	def recast(self, wavelength):
		""" Interpolate the filter throughput onto the grid given by input wavelength

		Uses 1D linear interpolation
		wavelength in microns, listlike
		
		"""
		WaveLength = np.array(wavelength)
		return np.interp(WaveLength, self.wavelength, self.throughput, left=0.0, right=0.0)

	def apply(self, sed):
		""" Apply the filter to the supplied SED

		Uses 1D linear interpolation
		sed is a dictionary with keys (wavelength, flux) in units of (microns, /micron), individually listlike
		sed may contain a third key called zeropoint in the same units as flux, listlike. This is integrated
		over to normalise the filter-equivalent flux. If not provided, the zeropoint is assumed to be flat
		in wavelength with a value = 3.63e-5 erg/s/cm^2/micron = 0 STMAG
		
		"""
		WaveLength = np.array(sed['wavelength'])
		FluxLam = np.array(sed['flux'])
		if ('zeropoint' in sed):
			ZeroPoint = np.array(sed['zeropoint'])
		else:
			ZeroPoint = np.full(len(WaveLength),3.63e-5)
		
		ApplyFilter = np.interp(WaveLength, self.wavelength, self.throughput, left=0.0, right=0.0)
		index, = np.where(ApplyFilter > 0.0) # Range of wavelengths over which the filter is non-zero
		
		if len(index) == 0:
			return 0.0
		else:
			intslice = slice(index.min(),index.max())
			
			if (self.format == 'energy'):
				FilterFlux = integrate.trapz(ApplyFilter[intslice]*FluxLam[intslice],WaveLength[intslice])
				FilterNorm = integrate.trapz(ApplyFilter[intslice]*ZeroPoint[intslice],WaveLength[intslice])
			else:
				FilterFlux = integrate.trapz(ApplyFilter[intslice]*WaveLength[intslice]*FluxLam[intslice],WaveLength[intslice])
				FilterNorm = integrate.trapz(ApplyFilter[intslice]*WaveLength[intslice]*ZeroPoint[intslice],WaveLength[intslice])
		
			return FilterFlux/FilterNorm


# ***********************************************************************************************


def register_filter(wavelength, throughput, format='photon', reference='User', description='None'):
	""" Register a filter for use by FORTES-AGN 
	
	wavelength in microns, listlike
	throughput in any unit, listlike
	
	The routine sorts the wavelength array to ensure that inverted or misformed inputs are dealt with
		at the point of filter registry
	
	returns the new filter ID
	
	"""
					
	# Read existing filters and create a list of filter ID numbers (which are the same as the filternames)
	OldFilterFiles = glob.glob(FortesFit_Settings.FilterDirectory+'*.fortesfilter.xml')
	if(len(OldFilterFiles) == 1):
		print('You are registering your first filter. Exciting!')
	OldIDs = []
	for OldFile in OldFilterFiles:
		OldIDs.append(np.int(os.path.basename(OldFile).split('.')[0]))
	OldIDs = np.array(OldIDs,dtype=int)		

	# Assign a random and unique 6 digit number for the new filter.
	# This approach allows for a maximum of N=900000 filters, which should be sufficient.
	# When the number of filters approaches N, this method of assignment becomes in efficient. 
	NewIDChecked = False
	while (not NewIDChecked):
		NewID = np.random.randint(100000, high=999999 + 1)
		index, = np.where(OldIDs == NewID)
		if(len(index) == 0):
			NewIDChecked = True					

	# Convert inputs to Numpy Arrays
	WaveLength = np.array(wavelength)
	ThroughPut = np.array(throughput)
	
	# Sort the inputs by wavelength low to high
	sortindex = np.argsort(WaveLength)
	WaveLength = WaveLength[sortindex]
	ThroughPut = ThroughPut[sortindex]

	# Clean the throughput. If < 1e-4*max, set to 0.0
	MaxThroughPut = ThroughPut.max()
	CleanedThroughPut = np.where(ThroughPut > 1.0e-4*MaxThroughPut, ThroughPut, np.full(len(ThroughPut),0.0))
	
	# Create the output table
	filter_table = Table([WaveLength, CleanedThroughPut],\
						  names = ['Wavelength', 'Throughput'])
	maintable = votable_routines.tree.VOTableFile.from_table(filter_table,'FORTESAGN')
	maintable.resources[0].infos.append(votable_routines.tree.Info(name='format',value=format))
	maintable.resources[0].infos.append(votable_routines.tree.Info(name='description',value=description))
	maintable.resources[0].links.append(votable_routines.tree.Link(href=reference))
	maintable.resources[0].tables[0].fields[0].unit = '10-6m'
	maintable.resources[0].tables[0].fields[1].unit = ''
	
	# Write the filter function to a FITS file
	OutFile = FortesFit_Settings.FilterDirectory+'{0:6d}.fortesfilter.xml'.format(NewID)
	maintable.to_xml(OutFile)
	
	summarize_filters()
	
	return NewID


# ***********************************************************************************************

def print_filter_info(filterlist):
	""" Print a summary of filter information to stdout
		
		filterlist: Either a list-like set of filterids, or the name of a Fortes filterset in the main filters directory	
	"""

	if type(filterlist).__name__ == 'str':
		# A string, therefore assume this is a Fortes filterset file
		filtersetfile = FortesFit_Settings.FilterDirectory+filterlist
		if os.path.isfile(filtersetfile):
			# The Fortes filterset file exists in the filters directory, read the first column, which are the filter ids.
			filterids = np.loadtxt(filtersetfile,usecols=(0,),dtype='i4',comments='#')
		else:
			print('Fortes filterset file does not exist')
			return
	else:
		# Otherwise a list of filter ids
		filterids = filterlist	
	
	# Get a list of filter instances and information
	Filters = []
	FilterNames = []
	FilterWave = []
	for id in filterids:
		tempfilt = FortesFit_Filter(id)
		Filters.append(tempfilt)
		FilterNames.append(tempfilt.description)
		FilterWave.append(tempfilt.pivot_wavelength)

	print('FilterID            Name       Pivot Wavelength (microns)')
	print('---------------------------------------------------------')
	for ifilt in range(len(filterids)):
		print('{0:6d}  {1:25s}   {2:<10.1e}'.format(filterids[ifilt],FilterNames[ifilt],FilterWave[ifilt]))
	print(' ')	
	
	return
	
# ***********************************************************************************************

def get_filterset(filtersetfile):
	""" Obtain a list of filterids given a Fortes filterset name
		
		filtersetfile: The name of a Fortes filterset file in the main filters directory
		
		If file exists, returns a list of FortesFit filter IDs in the order of the filterset, as a numpy array.
	"""

	filtersetfile = FortesFit_Settings.FilterDirectory+filtersetfile
	if os.path.isfile(filtersetfile):
		# The Fortes filterset file exists in the filters directory, read the first column, which are the filter ids.
		filterids = np.loadtxt(filtersetfile,usecols=(0,),dtype='i4',comments='#')
	else:
		print('Fortes filterset file does not exist')
		return None
		
	return filterids

# ***********************************************************************************************


def summarize_filters():
	""" Create a summary in the local directory of all filters available to FortesFit 
	
	The summary is written out as a simple fixed-format multi-column ascii file.
	
	"""
	
	summary_table = Table(names=('filterid','format','pivot wavelength','description'),
						  dtype=('i8',np.dtype(object),'f8',np.dtype(object)))
	summary_table['pivot wavelength'].format = '15.4e'
	
	# Read existing filters and create a list of filter ID numbers (which are the same as the filternames)
	FilterFileList = glob.glob(FortesFit_Settings.FilterDirectory+'*.fortesfilter.xml')
	if(len(FilterFileList) == 0):
		print('No existing filters found.')
		return []
	for FilterFile in FilterFileList:
		FilterID = np.int(os.path.basename(FilterFile).split('.')[0])
		Filter = FortesFit_Filter(FilterID)
		FilterFormat = Filter.format
		FilterDesc = Filter.description
		PivotWavelength = Filter.pivot_wavelength		
		summary_table.add_row([FilterID,FilterFormat,PivotWavelength,FilterDesc])
	
	summary_table.sort('pivot wavelength')
	summary_table.write(FortesFit_Settings.FilterDirectory+'FortesFit_filters_summary.ascii',
						format='ascii.fixed_width_two_line',overwrite=True)
	
	
