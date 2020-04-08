import sys
import time

import numpy as np

import astropy.units as u

from fortesfit import FortesFit_Preparation
from fortesfit import FortesFit_Fitting

""" End-to-end test of FortesFit with out-of-the-box EMCEE fitting.	
	Very simple model fitting one point of data with a fixed slope model.
"""


flux = [1.0]*u.mJy # flux of the test photometry in erg/s/cm^2/micron, corresponding to 1 mJy
err  = [0.1]*u.mJy # test with a 10% error
limit = [3.0] # test with a 3sigma error
redshift = 1.0 # test with a redshift of 1

datacollection = FortesFit_Preparation.CollectData('installer_test',redshift,[100000],flux,err,Limits=limit)

prior = [{'MonochromLuminosity':np.array([40.0+np.arange(101)*(10.0/100.0),np.full(101,1.0)])}]

modelcollection = FortesFit_Preparation.CollectModel([10],prior,datacollection)

fitfile = FortesFit_Preparation.prepare_output_file(datacollection,modelcollection,'emcee',\
													description='Installer test output')	
FortesFit_Fitting.FortesFit_FitSingle(datacollection, modelcollection, fitfile)

fitfile.close()

