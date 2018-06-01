# FortesFit

Overview: FortesFit is a Python-based multi-component SED fitting package. At its core is a set of classes that allow easy management of parameterised SED model libraries. These models serve as the building blocks of an additive model that can be fit to an observed SED, which allow constraints on the model parameters. FortesFit is built to allow the inclusion of different fitting engines: the default currently is the Affine-Invariant Ensemble sampler from EMCEE (http://dfm.io/emcee/current/).

Within astronomy, Spectral Energy Distributions (SEDs) are discretely sampled functions of energy that describe the luminous output of an astronomical object. The independent variable is the photon energy of radiation, more often represented by frequency or wavelength. The dependent variable is the energy flux from the object, usually represented in energy units such as erg/s or Watts. Observed SEDs are usually cast in luminance units (e.g., erg/s/cm^2 or W/m^2), and often in specific flux units (e.g., erg/s/cm^2/micron or W/M^2/Hz), the energy received from the source on the surface of a hypothetical Earth that is free of atmospheric or local absorption effects.

SEDs are the fundamental spectral (i.e., energy-dependent) descriptions of astronomical bodies, since electromagnetic radiation is often the only messenger of information available for objects in deep space. An SED may be understood as the emission from various components that add (or absorb) linearly along the line of sight to an astronomical body. Therefore, the main aim of SED analysis is to use an observed SED to identify and constrain the contributions from these different components.

In FortesFit, we distinguish between SEDs and spectrograms (or spectra). The latter are generally continguous and offer a higher spectral resolution compared to SEDs. For our purposes, we define an SED as a collection of photometry taken through individual filter bands (physical or synthetic). A filter band describes the response of the astronomical observation (composed of the atmosphere, telescope, instrument, and detector) to the light from the source. Therefore, it encapsulates the actual spectral information inherent in the corresponding photometry from the SED. In FortesFit, the independent variable is not specified by the monochromatic energy of the radiation, but the filter bands used to observe the astronomical source. 

Installation: The code can be installed using PIP ($ pip install fortesfit) or cloned directly from GitHub. PIP installation is recommended. After installation, set an environment variable called FORTESFITPATH to a location where filters and model photometry will be stored. Then run the modules FortesFit_Init, which sets up a test filter and model, followed by FortesFit_installation_test, which does an end-to-end fit using the testing setup. 

Once this is successful, the user writes a Python program that incorporates FortesFit classes and functions to set up the data, models, priors, engine, and output for a fit. The FortesFit_Preparation module includes handy functions that help organise and prepare the photometry, models, prior descriptions and output files before running a fit. The FortesFit_Fitting module contains the wrappers to the fitting engines. At present, an MCMC engine from the EMCEE package is the default approach. If MultiNest and PyMultinest are installed on the user's system, this may also be used for fitting. Both simpler MLE or least-squares fitting engines are planned. The structure of FortesFit separates the handling and processing of data and models from the fitting engines themselves, giving the user the functionality to design and test various classes of models in an agile fashion.  


The fitting process usually involves:

  - Registration of the filters used for the fits, using the routines in FortesFit_Filters. These are stored to disk and can be used for future fitting projects.
  
 - Registration of the models used for the fits, using the routines in FortesFit_ModelManagement. These are stored to disk and can be used for future fitting projects. Filters can be freely added to existing models after registration.
 
 - Setting up the observed SED(s) for fitting routines in FortesFit_Preparation. An example script is included with best practices. The units functionality of Astropy is used by FortesFit to homogenise photometry.
  
 - Initialisation of the priors used in the fit. FortesFit is unique in astronomical SED fitting codes for its versatile use of priors. These may be specified in the form of a grid or using the distributions available from the stats subpackage in SciPy. Parameters without user-defined priors are assigned a default uninformative prior. However, parameters that determine the scale of the SED components (for e.g., the luminosity or stellar mass of a template) must have their priors defined.
 
 - Setting up the combination of models and priors for the fit, using routines in FortesFit_Preparation. This is also the point where the mechanics of the fit (output files, fitting algorithm, etc.) are chosen.
 
 - Running the fit using functions under FortesFit_Fitting.
 
 - Examining the fit results using routines under FortesFit_Plots
 
FortesFit has been used successfully to fit:
  - The infra-red SEDs of nearby AGN from 1-1000 um with stellar, AGN and SF-heated dust components (Rosario+ 2017)
  - The mid-far infrared SEDs of a few local AGN with clumpy torus SED models (5-10 parameters) and SF-heated dust.
  - UV to sub-mm SEDs of high-redshift AGN in the KASHz survey.
  - Your project here
