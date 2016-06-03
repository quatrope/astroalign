#ASTROALIGN

[![Build Status](https://travis-ci.org/toros-astro/astroalign.svg?branch=master)](https://travis-ci.org/toros-astro/astroalign)
[![Coverage](https://codecov.io/github/toros-astro/astroalign/coverage.svg?branch=master)](https://codecov.io/github/toros-astro/astroalign)
[![Documentation Status](https://readthedocs.org/projects/astroalign/badge/?version=latest)](http://astroalign.readthedocs.org/en/latest/?badge=latest)

**ASTROALIGN** is a simple package that will try to align two stellar astronomical images, especially when there is no WCS information available.

It does so by finding similar 3-point asterisms (triangles) in both images and deducing the
affine transformation between them.

General align routines try to match interesting points, using corner detection routines to make the point correspondence.

These generally fail for stellar astronomical images, since stars have very little stable structure and so, in general, indistinguishable from each other.

Asterism matching is more robust and closer to the human way of matching images.

Astro align can match images of very different field of view, point-spread functions, seeing and atmospheric conditions.

It may not work, or work with special care, on images of extended objects with few point-like sources or in very crowded fields.

***

##Installation

Using setuptools:

    pip install astroalign

or from this distribution with

    python setup.py install

***

Usage example

    >>> import astroalign as aa
    >>> aligned_image = aa.align_image(reference_image, image2transform)

In this example image2transform will be interpolated by a transformation to coincide pixel to pixel with reference_img and stored in aligned_image.

Sometimes, CCD defects can confuse the alignment. In cases where it's necessary you can mask out bad regions using a mask (True on bad) and pass a [numpy masked array](http://docs.scipy.org/doc/numpy/reference/maskedarray.html) that will ignore the masked pixels. 

The returned aligned_image will be the same type as image2transform and its mask, if any, will be transformed in the same way as the image.

Pixels outside the boundaries of the transformation are filled with the image median value and masked True if mask is provided.

More information available on docstrings

***

*This package is inspired by the [astrometry.net](http://astrometry.net) program*

TOROS Dev Team

<martinberoiz@gmail.com>

