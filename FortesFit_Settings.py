import os
import numpy as np
from astropy.cosmology import WMAP9 as FortesFit_Cosmology

#  Paths
FortesFit_Path  = os.getenv('FORTESFITPATH','None')
if FortesFit_Path[-1] != '/':
	FortesFit_Path += '/'

FilterDirectory = FortesFit_Path+'FortesFit_Filters/'
ModelPhotometryDirectory = FortesFit_Path+'FortesFit_ModelPhotometry/'

# Redshift range and sampling settings
FortesFit_MinimumRedshift = 0.001
FortesFit_MaximumRedshift = 10.0
FortesFit_DeltaRedshiftSharp = 0.01  #  The redshift sampling in a geometric series of 1+z
NumRedshifts   = np.int( (np.log((1.0+FortesFit_MaximumRedshift)/(1.0+FortesFit_MinimumRedshift)) / np.log(1.0+FortesFit_DeltaRedshiftSharp)) + 1.0)
PivotRedshifts = ( (1.0+FortesFit_MinimumRedshift)*((1.0+FortesFit_DeltaRedshiftSharp)**np.arange(NumRedshifts)) ) - 1.0
if NumRedshifts >= 1000:
	raise ValueError('Number of pivot redshifts must be less than 1000')
