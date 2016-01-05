#ASTROALIGN

**ASTROALIGN** is a simple package that will try to align two stellar astronomical images.

It does so by finding similar 3-point asterisms (triangles) in both images and deducing the
affine transformation between them.

General align routines try to match interesting points, using corner detection routines to make the point correspondence.

These generally fail for stellar astronomical images, since stars have very little stable structure and so, in general, indistinguishable from each other.

Asterism matching is more robust and closer to the human way of matching images.

Astro align can match images of very different field of view, point-spread functions, seeing and atmospheric conditions.

***

*This package is inspired by the [astrometry.net](http://astrometry.net) program*

Author: Martin Beroiz

<martinberoiz@gmail.com>

[![Build Status](https://travis-ci.org/toros-astro/astroalign.svg?branch=master)](https://travis-ci.org/toros-astro/astroalign)
