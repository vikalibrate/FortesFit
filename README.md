# FortesFit

FortesFit is a Python-based multi-component SED (Spectral Energy Distribution) fitting package. It is a set of classes that enable easy management of parameterised SED model libraries. These models serve as building blocks of an additive model that can be fit to an observed SED to constrain the model parameters.

## Background
In astronomy, Spectral Energy Distributions (SEDs) are discretely sampled functions of energy that describe the luminous output of an astronomical object. The independent variable is the photon energy of radiation, often represented by frequency or wavelength. The dependent variable is the energy flux from the object, usually represented in energy units such as erg/s or Watts. 

SEDs are the fundamental spectral (i.e., energy-dependent) descriptions of astronomical bodies, since electromagnetic radiation is often the only messenger of information available for objects in deep space. An SED may be understood as the emission from various components that add (or absorb) linearly along the line of sight to an astronomical body. Therefore, the main aim of SED analysis is to use an observed SED to identify and constrain the contributions from these different components.

FortesFit distinguishes between SEDs and spectrograms (or spectra). Unlike spectra, which are contiguous and have higher resolution, SEDs are composed of photometric data taken through different filter bands (physical or synthetic). In this context, A filter band describes the response of the astronomical observation (composed of the atmosphere, telescope, instrument, and detector) to the light from the source. Therefore, it encapsulates the actual spectral information inherent in the corresponding photometry from the SED.

## Installation and initial setup
The recommended way to install FortesFit is [through pip](https://pypi.org/project/fortesfit/). The latest version can be installed using
```bash
pip install fortesfit
```
You will need to do the following initial setup before you can use FortesFit for your science.
1. Set an environment variable ```FORTESFITPATH```. This is the directory where all the filters and models will be stored. On Unix systems, you can do this using:  
   ```bash
   export FORTESFITPATH="/path/to/your/directory"
   ```  
> Note: You will have to do this each time you open a new terminal session. Alternatively, you can [automatically set the environment variable upon activation of a specific conda environemt](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux), or add the path to your ```.bashrc``` or equivalent.
2. Create test filters and models:
   ```bash
   python -c "from fortesfit import FortesFit_Init"
   ```
3. Run a test script which does an end-to-end fit using the testing setup:
   ```bash
   python -c "from fortesfit import FortesFit_installation_test"
   ```

Once the setup is successful, the user writes a Python program that incorporates FortesFit classes and functions to set up the data, models, priors, engine, and output for a fit. A typical workflow is given [below](#Basic-fitting-workflow).

## Using FortesFit
### Features
  - **Full control on filters and models**. This makes it easy to add any filters or your own models! You are also free to define the model parameters as you please. However, this can be daunting at first, please refer to [examples repository](https://github.com/DevangLiya/fortesfit_resources) for pre-packaged models.
  - **Model dependencies** allow users to define quantities derived from the original model parameters. These dependencies can then be used to set priors or even tie together fluxes from different filters.
  - **Choice of fitting engine** through ```FortesFit_Preparation``` module. At present, FortesFit offers an MCMC engine from the [EMCEE package](https://emcee.readthedocs.io/en/stable/), and nested sampling engine MultiNest through [PyMultinest](https://johannesbuchner.github.io/PyMultiNest/). (Note: MultiNest needs to be installed seperately by the user.)
  - **Full control on priors** by specifying your own prior in form of probability density function.

### Basic fitting workflow
A basic FortesFit workflow may look like following
  1. Register filters using the routines in ```FortesFit_Filters```. These will be stored on disk for future projects.
  2. Register models using the routines in ```FortesFit_ModelManagement```. These will also be stored on disk for future use. Filters can be freely added to existing models after registration.
  3. Set up observed SED(s) using routines in ```FortesFit_Preparation```. The photometry is homogenised at this stage using astropy's units functionality.
  4. Initialise priors. FortesFit is unique in astronomical SED fitting codes for its versatile use of priors. These may be specified in the form of a grid or using [SciPy's built-in distributions](https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions). Parameters without user-defined priors are assigned a default uniform prior. However, parameters that determine the scale of the SED components (for e.g., the luminosity or stellar mass of a template) must have their priors defined.
  5. Set up the fit by combining models and priors using routines in ```FortesFit_Preparation```. This is also the point where the mechanics of the fit (output files, fitting algorithm, etc.) are chosen.
  6. Run the fit using routines in ```FortesFit_Fitting```.
  7. Examin the fit results using routines in ```FortesFit_Plots``` and ```FortesFit_Parsers```.

### Tutorials, examples and additional resources
A number of resources related to FortesFit can be found in [this repository](https://github.com/DevangLiya/fortesfit_resources). Some of the things included in the repository are tutorials (in form of jupyter notebooks), demonstrations of advanced features, pre-made models and filters.
 
## Who has used FortesFit?
  - The infra-red SEDs of nearby AGN from 1-1000 um with stellar, AGN and SF-heated dust components (Rosario+ 2017)
  - The mid-far infrared SEDs of a few local AGN with clumpy torus SED models (5-10 parameters) and SF-heated dust.
  - UV to sub-mm SEDs of high-redshift AGN in the KASHz survey.
  - Your project here!
