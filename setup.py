from setuptools import setup

setup(name='astroalign',
      version='1.0a0',
      description='Astrometric Alignment of Images',
      author='Martin Beroiz',
      author_email='martinberoiz@gmail.com',
      url='http://martinberoiz.org',
      packages=['astroalign',],
      install_requires=["numpy", "scipy", "astropy", "scikit-image"],
      test_suite='tests',
     )
