# FortesFit

Overview: FortesFit is a Python-based multi-component SED fitting package. Its core design concept is a set of classes that allow easy management of parameterised SED model libraries. These models serve as the building blocks of an observed SED, which is used to constrain the model parameters. FortesFit is built on an MCMC engine that is used for the default fitting, though other fitting methods can also be used.

Spectral Energy Distributions (SEDs) are discretely sampled functions of energy that describe the luminous output of an astronomical object. The independent variable is the photon energy of radiation, more often represented by frequency or wavelength. The dependent variable is the energy flux from the object, usually represented in energy units such as erg/s or Watts. Observed SEDs are usually cast in luminance units (e.g., erg/s/cm^2 or W/m^2), and often in specific flux units (e.g., erg/s/cm^2/micron or W/M^2/Hz), the energy received from the source on the surface of a hypothetical Earth that is free of atmospheric or local absorption effects.

SEDs are the fundamental spectral (i.e., energy-dependent) descriptions of astronomical bodies, since electromagnetic radiation is often the only messenger of information available for objects in deep space. An SED may be understood as the emission from various components that add linearly along the line of sight to an astronomical body. Therefore, the main aim of SED analysis is to use an observed SED to identify and constrain the contributions from these different components.

In FortesFit, we distinguish between SEDs and spectrograms (or spectra). The latter are generally close to continguous and offer a higher spectral resolution compared to SEDs. For our purposes, we define an SED as a collection of photometry taken through individual filter bands (physical or synthetic). The independent variable is therefore not taken to be energy of the radiation itself, but the actually filter bands used to observe the astronomical source. A filter band describes the response of the astronomical observation (composed of the atmosphere, telescope, instrument, and detector) to the light from the source. Therefore, it encapsulates the actual spectral information inherent in the corresponding piece of photometry from the SED.

A basis fitting procedure involves:

  - Setting up the observed SED(s) for fitting. This must be done by the user, but an example script is included with best practices.
  - Registration of the filters used for the fits, using the routines in FortesFit_Filters. These are stored to disk and can be used for future fitting projects.
 - Registration of the models used for the fits, using the routines in FortesFit_ModelManagement. These are stored to disk and can be used for future fitting projects. Filters can be added to existing models.
 - Initialisation of the priors used in the fit. FortesFit is unique in astronomical SED fitting codes for its versatile use of priors built on the SciPy statistical subpackage. Therefore, priors must be provided on parameters for each fit. However, these may be chosen to be uniformative.
 
While the setup of FortesFit is intensive, a single setup can be used to fit large numbers of objects, especially if the prior initialisation part can be made programmatic (i.e., written in Python code). Defaults are in place to underline best practice, so a user only needs to understand a default (i.e., uninformative prior defaults) if they need to change them. 

FortesFit has been used successfully to fit:
  - The infra-red SEDs of nearby AGN from 1-1000 um with stellar, AGN and SF-heated dust components (Rosario+ 2017a)
  - The mid-far infrared SEDs of a few local AGN with clumpy torus SED models (5-10 parameters) and SF-heated dust.
  - .....
