Tutorial
========

A simple usage example
----------------------

.. note::
    Check this `Jupyter notebook <http://toros-astro.github.io/astroalign/>`_ for a more complete example.

Suppose we have two images of about the same portion of the sky, and we would like to transform one of them to fit on top of the other one.
Suppose we do not have WCS information, but we are confident that we could do it by eye, by matching some obvious asterisms on the two images.

In this particular use case, astroalign can be of great help to automatize the process.

After we load our images into numpy arrays, we simple choose one to be the source image to be transformed and the other to be the target.

The usage for this simple most common case would be as follows::

    >>> import astroalign as aa
    >>> registered_image, footprint = aa.register(source, target)

``registered_image`` is now a transformed (numpy array) image of ``source`` that will match pixel to pixel to ``target``.

``footprint`` is a boolean numpy array, `True` for masked pixels with no information.

.. warning::
    Flux may not be conserved after the transformation.


Finding the transformation
--------------------------

In some cases it may be necessary to inspect first the transformation parameters before applying it,
or we may be interested only in a star to star correspondence between the images.
For those cases, we can use ``find_transform``.

``find_transform`` will return a `scikit-image <http://scikit-image.org>`_ `SimilarityTransform <http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.SimilarityTransform>`_ object that encapsulates the matrix transformation,
and the transformation parameters.
It will also return a tuple with two lists of star positions of ``source`` and its corresponding ordered star postions on
the ``target`` image.::

    >>> transf, (source_list, target_list) = aa.find_transform(source, target)

source and target here can be either numpy arrays of the image pixels, or any iterable (x, y) pair,
corresponding to a star position.

The transformation parameters can be found in ``transf.rotation``, ``transf.traslation``, ``transf.scale``
and the transformation matrix in ``transf.params``.

If the transformation is satisfactory, we can apply it to the image with ``apply_transform``.
Continuing our example::

    >>> if transf.rotation > MIN_ROT:
    ...     registered_image = aa.apply_transform(transf, source, target)

If you know the star-to-star correspondence
-------------------------------------------

.. note::
    `estimate_transform <http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.estimate_transform>`_
    from `scikit-image` is imported into astroalign as a convenience.

If for any reason you know which star corresponds to which other, you can call ``estimate_transform``.

Let us suppose we know the correspondence:

- (127.03, 85.98) in source --> (175.13, 111.36) in target
- (23.11, 31.87) in source --> (0.58, 119.04) in target
- (98.84, 142.99) in source --> (181.55, 206.49) in target
- (150.93, 85.02) in source --> (205.60, 91.89) in target
- (137.99, 12.88) in source --> (134.61, 7.94) in target

Then we can estimate the transform::

    >>> src = np.array([(127.03, 85.98), (23.11, 31.87), (98.84, 142.99),
    ...                 (150.93, 85.02), (137.99, 12.88)])
    >>> dst = np.array([(175.13, 111.36), (0.58, 119.04), (181.55, 206.49),
    ...                 (205.60, 91.89), (134.61, 7.94)])
    >>> tform = aa.estimate_transform('affine', src, dst)

And apply it to an image with ``apply_transform`` or to a set of points with ``matrix_transform``.

Applying a transformation to a set of points
--------------------------------------------

.. note::
    `matrix_transform <http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.matrix_transform>`_
    from `scikit-image` is imported into astroalign as a convenience.

To apply a known transform to a set of points, we use `matrix_transform`.
Following the example in the previous section::

    >>> dst_calc = aa.matrix_transform(src, tform.params)

``dst_calc`` should be a 5 by 2 array similar to the ``dst`` array.


Dealing with hot pixels
-----------------------

Hot pixels always appear on the same position of the CCD.
If your image is dominated by hot pixels, the source detection algorithm may pick those up
and output the identity tranformation.

To avoid this, you can use `CCDProc's cosmicray_lacosmic <https://ccdproc.readthedocs.io/en/latest/api/ccdproc.cosmicray_lacosmic.html>`_ to clean the image before trying registration::

    from ccdproc import cosmicray_lacosmic as lacosmic
    clean_image = lacosmic(myimage)

----------------------------------------

See :ref:`api` for the API specification.
