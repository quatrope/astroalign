.. _index:

.. image:: images/logo-seal-dark-light.png
    :width: 200px
    :align: left

Astroalign documentation
========================

**ASTROALIGN** is a python module that will try to register (align) two stellar astronomical images,
especially when there is no WCS information available.

It does so by finding similar 3-point asterisms (triangles) in both images and estimating the
affine transformation between them.

Generic registration routines try to match point features, using corner detection routines to make the point correspondence.
These generally fail for stellar astronomical images, since stars have very little stable structure and so, in general, indistinguishable from each other.
Asterism matching is more robust, and closer to the human way of matching stellar images.

Because of the geometric nature of its core algorithm, it is not greatly effected by point-spread function differences, seeing or atmospheric conditions.

You can find a Jupyter notebook example with the main features at http://quatrope.github.io/astroalign.

.. note::

    It may not work, or work with special care, on images of extended objects with few point-like sources or in very crowded fields.

.. note::
    If your images contain a large number of hot pixels, this may result in an incorrect registration.
    Please refer to the tutorial for how to solve this problem using `CCDProc's cosmic-ray remover <https://ccdproc.readthedocs.io/en/latest/api/ccdproc.cosmicray_lacosmic.html>`_.

.. note::
    This library is compatible with `bottleneck <https://github.com/pydata/bottleneck>`_ and may offer performance improvements in some cases.
    `Astroalign` will pick this optional dependency up and use its performance improved ``mean`` and ``median`` functions.


Citation
--------

If you use astroalign in a scientific publication, we would appreciate
citations to the following paper:

    Astroalign: A Python module for astronomical image registration.
    Beroiz, M., Cabral, J. B., & Sanchez, B.
    Astronomy and Computing, Volume 32, July 2020, 100384.


Bibtex entry::

    @article{BEROIZ2020100384,
    title = "Astroalign: A Python module for astronomical image registration",
    journal = "Astronomy and Computing",
    volume = "32",
    pages = "100384",
    year = "2020",
    issn = "2213-1337",
    doi = "https://doi.org/10.1016/j.ascom.2020.100384",
    url = "http://www.sciencedirect.com/science/article/pii/S221313372030038X",
    author = "M. Beroiz and J.B. Cabral and B. Sanchez",
    keywords = "Astronomy, Image registration, Python package",
    abstract = "We present an algorithm implemented in the Astroalign Python module for image registration     in astronomy. Our module does not rely on WCS information and instead matches three-point asterisms (    triangles) on the images to find the most accurate linear transformation between them. It is especially     useful in the context of aligning images prior to stacking or performing difference image analysis.     Astroalign can match images of different point-spread functions, seeing, and atmospheric conditions."
    }

**Full Publication:** https://www.sciencedirect.com/science/article/pii/S221313372030038X


Or cite the project itself from `ASCL <http://ascl.net/1906.001>`_:

    Beroiz, M. I. (2019). Astroalign: Asterism-matching alignment of
    astronomical images. Astrophysics Source Code Library.

Bibtex::

    @article{beroiz2019astroalign,
        title={Astroalign: Asterism-matching alignment of astronomical images},
        author={Beroiz, Martin I},
        journal={Astrophysics Source Code Library},
        year={2019}
    }


Guide:
^^^^^^

.. toctree::
   :maxdepth: 2

   installation
   tutorial
   mask
   examples
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
