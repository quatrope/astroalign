.. _index:

Astroalign documentation
========================

**ASTROALIGN** is a simple package that will try to align two stellar astronomical images, especially when there is no WCS information available.

It does so by finding similar 3-point asterisms (triangles) in both images and estimating the
affine transformation between them.

General registration routines try to match feature points, using corner detection routines to make the point correspondence.
These generally fail for stellar astronomical images, since stars have very little stable structure and so, in general, indistinguishable from each other.
Asterism matching is more robust, and closer to the human way of matching stellar images.

Astroalign can match images of very different field of view, point-spread functions, seeing and atmospheric conditions.

It may not work, or work with special care, on images of extended objects with few point-like sources or in very crowded fields.


Guide:
^^^^^^

.. toctree::
   :maxdepth: 2
   
   installation
   tutorial
   methods
   mask


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
