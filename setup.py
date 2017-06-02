from setuptools import setup

# Get the version from astroalign file itself (not imported)
with open('astroalign.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            _, _, aa_version = line.replace("'", '').split()
            break

setup(name='astroalign',
      version=aa_version,
      description='Astrometric Alignment of Images',
      author='Martin Beroiz',
      author_email='martinberoiz@gmail.com',
      url='https://github.com/toros-astro/astroalign',
      py_modules=['astroalign', ],
      install_requires=["numpy>=1.6.2",
                        "scipy>=0.15",
                        "scikit-image",
                        "astropy>=1.0",
                        "sep",
                        ],
      test_suite='tests',
      )
