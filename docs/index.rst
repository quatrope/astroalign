.. _index:

.. image:: aa_bw.jpg
    :width: 200px
    :align: left

Astroalign documentation
========================

**ASTROALIGN** is a simple package that will try to register (align) two stellar astronomical images,
especially when there is no WCS information available.

It does so by finding similar 3-point asterisms (triangles) in both images and estimating the
affine transformation between them.

Generic registration routines try to match point features, using corner detection routines to make the point correspondence.
These generally fail for stellar astronomical images, since stars have very little stable structure and so, in general, indistinguishable from each other.
Asterism matching is more robust, and closer to the human way of matching stellar images.

Astroalign can match images of very different fields of view, point-spread functions, seeing and atmospheric conditions.

You can find a Jupyter notebook example with the main features at http://toros-astro.github.io/astroalign.

.. note::

    It may not work, or work with special care, on images of extended objects with few point-like sources or in very crowded fields.

.. note::
    If your images contain a large number of hot pixels, this may result in an incorrect registration.
    Please refer to the tutorial for how to solve this problem using `CCDProc's cosmic-ray remover <https://ccdproc.readthedocs.io/en/latest/api/ccdproc.cosmicray_lacosmic.html>`_.


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
