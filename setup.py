from setuptools import setup

setup(name='astroalign',
      version='1.0a1',
      description='Astrometric Alignment of Images',
      author='Martin Beroiz',
      author_email='martinberoiz@gmail.com',
      url='https://github.com/toros-astro/astroalign',
      packages=['astroalign', ],
      install_requires=["numpy>=1.6.2",
                        "scipy>=0.15",
                        "astropy>=1.0",
                        "scikit-image>=0.11"
                        ],
      test_suite='tests',
      )
