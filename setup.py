from setuptools import setup

setup(name='fortesfit',
      version='1.0.6',
      install_requires=['numpy','scipy','matplotlib',
                        'astropy','emcee','h5py','corner'],
      description='Flexible SED fitting for astrophysics',
      url='http://github.com/vikalibrate/FortesFit',
      author='David Rosario',
      author_email='david.rosario@durham.ac.uk',
      license='MIT',
      packages=['fortesfit'],
)
