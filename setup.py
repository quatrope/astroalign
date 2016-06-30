from setuptools import setup
import os

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

setup(name='astroalign',
      version='1.0a4',
      description='Astrometric Alignment of Images',
      long_description=README,
      author='Martin Beroiz',
      author_email='martinberoiz@gmail.com',
      url='https://github.com/toros-astro/astroalign',
      py_modules=['astroalign', ],
      install_requires=["numpy>=1.6.2",
                        "scipy>=0.15",
                        "astropy>=1.0",
                        ],
      test_suite='tests',
      )
