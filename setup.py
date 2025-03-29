from setuptools import setup

setup(name='fortesfit',
      version='2.0.0',
      install_requires=['numpy','scipy','matplotlib',
                        'astropy','emcee','h5py','corner','tqdm'],
      description='Flexible SED fitting for astrophysics',
      url='http://github.com/vikalibrate/FortesFit',
      author='David Rosario, Devang Liya',
      author_email='david.rosario@newcastle.ac.uk, d.h.liya2@newcastle.ac.uk',
      license='MIT',
      packages=['fortesfit'],
)
