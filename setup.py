from setuptools import setup

# Get the version from astroalign file itself (not imported)
with open('astroalign.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            _, _, aa_version = line.replace("'", '').split()
            break

from os import path
root_dir = path.abspath(path.dirname(__file__))
with open(path.join(root_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='astroalign',
      version=aa_version,
      description='Astrometric Alignment of Images',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Martin Beroiz',
      author_email='martinberoiz@gmail.com',
      url='https://github.com/toros-astro/astroalign',
      py_modules=['astroalign', ],
      install_requires=["numpy>=1.6.2",
                        "scipy>=0.15",
                        "scikit-image",
                        "sep",
                        ],
      test_suite='tests',
      )
