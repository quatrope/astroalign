.. _examples:

Examples
========

Very few stars on the field
---------------------------

.. note::
    The minimum number of stars necessary to find a transformation is 3

If your field has few stars on the field, of about 3 to 6, you may want to
restrict astroalign to only pick that number of stars, to prevent catching
noisy structures as sources.

Use ``max_control_points`` keyword argument to do so::

    >>> import astroalign as aa
    >>> registered_image, footprint = aa.register(source, target, max_control_points=3)

This keyword will also work in ``find_transform``.

Faint stars
-----------

If your stars are faint, they may not be bright enough to pass the
:math:`5 \sigma` threshold. If you need to lower the detection :math:`\sigma`
used in the source detection process, adjust the ``detection_sigma`` keyword argument::

    >>> import astroalign as aa
    >>> registered_image, footprint = aa.register(source, target, detection_sigma=2)

This keyword will also work in ``find_transform``.

Avoiding hot pixels and other CCD artifacts
-------------------------------------------

If your CCD is dominated by persistent defects like hot or dead pixels, they may be taken
as legitimate sources and output the identity transformation.

We suggest cleaning the image first using `CCDProc's cosmicray_lacosmic <https://ccdproc.readthedocs.io/en/latest/api/ccdproc.cosmicray_lacosmic.html>`_ to clean the image before trying registration::

    >>> from ccdproc import cosmicray_lacosmic as lacosmic
    >>> clean_source, mask = lacosmic(myimage)
    >>> registered_image, footprint = aa.register(clean_source, clean_target, min_area=9)

Another quick fix can be increasing the expected connected pixels in order to
be considered a source. Increment ``min_area`` from default value of 5::

    >>> import astroalign as aa
    >>> registered_image, footprint = aa.register(source, target, min_area=9)
