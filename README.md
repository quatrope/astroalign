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

    $ pip install astroalign

or from this distribution with

    $ python setup.py install

***

Usage example

    >>> import astroalign as aa
    >>> aligned_image = aa.align_image(source_image, target_image)

In this example `source_image` will be interpolated by a transformation to coincide pixel to pixel with `target_image` and stored in `aligned_image`.

If we are only interested in knowing the transformation and the correspondence of control points in both images, use `get_transform` will return the transformation in a scikit-image SimilarityTransform object and a list of stars in source with the corresponding stars in target.

    >>> transf, (s_list, t_list) = aa.get_transform(source, target)

`source` and `target` can each either be the numpy array of the image, or an iterable of (x, y) pairs of star positions on the image.

The returned `transf` object is a scikit-image [`SimilarityTranform`](http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.SimilarityTransform) object that contains the transformation matrix along with the scale, rotation and translation parameters.

`s_list` and `t_list` are numpy arrays of (x, y) star positions on `source` and `target` respectively. `transf` applied to `s_list` will approximately render `t_list`.

***

*This package is inspired by the [astrometry.net](http://astrometry.net) program*

TOROS Dev Team

<martinberoiz@gmail.com>

