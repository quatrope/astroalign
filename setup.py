from setuptools import setup

setup(name='astroalign',
      version='1.0.0.dev1',
      description='Astrometric Alignment of Images',
      author='Martin Beroiz',
      author_email='martinberoiz@gmail.com',
      url='https://github.com/toros-astro/astroalign',
      py_modules=['astroalign', ],
      install_requires=["numpy>=1.6.2",
                        "scipy>=0.15",
                        "astropy>=1.0",
                        "scikit-image",
                        "sep",
                        ],
      test_suite='tests',
      )
