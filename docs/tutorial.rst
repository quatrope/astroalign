Tutorial
========

A simple usage example
----------------------

Suppose we have two images of about the same portion of the sky, and we would like to transform one of them to fit on top of the other.
Suppose we do not have WCS information, but we are confident that we could do it by eye, by matching some obvious asterisms on the two images.

In this particular use case, astroalign can be of great help to automatize the process.

After we load our images into numpy arrays, we simple choose one to be the source image to be transformed, and the other to be the target.

The usage for this simple most common case would be as follows::

    >>> import astroalign as aa
    >>> registered_image, footprint = aa.register(source, target)

``registered_image`` is now a transformed (numpy array) image of ``source`` that will match pixel to pixel to ``target``.

``footprint`` is a boolean numpy array, ``True`` for masked pixels with no information.

.. note::
    * If instead of images, you have lists of bright, reference star positions on each image,
      see :ref:`ftransf`.

    * ``astroalign.register`` will also accept as input, data objects with ``data`` and ``mask`` properties,
      like ``NDData``, ``CCDData`` and ``Numpy`` masked arrays.
      See :ref:`dataobjs`.

    * Check this `Jupyter notebook <http://quatrope.github.io/astroalign/>`_ for a more complete example.

.. warning::
    Flux may not be conserved after the transformation.

.. note::
    If your image requires special care see :ref:`examples`.


Images with RGB channels
------------------------

Astroalign can work with color images provided the channel index be the last axis in the array.
Adding the channel dimension in the last axis of the array is the default behavior for
`pillow <https://pillow.readthedocs.io>`_
and `scikit-image <https://scikit-image.org/docs/dev/user_guide/numpy_images.html>`__.
The transformation is found on the ``mean`` average of all the channels.
PNG images with RGBA channels work similarly.

Example:

.. code-block:: python

    from PIL import Image
    import astroalign as aa
    source = Image.open("source.jpg")
    target = Image.open("target.jpg")
    registered, footprint = aa.register(source, target)
    # Convert back to pillow image if necessary:
    registered = Image.fromarray(registered.astype("unit8"))

*Pillow may require array to be unsigned 8-bit integer format.*


Mask Fill Value
---------------

If you need to mask the aligned image with a special value over the region where transformation had no pixel information,
you can use the ``footprint`` mask to do so::

    >>> registered_image, footprint = aa.register(source, target)
    >>> registered_image[footprint] = -99999.99

Or you can pass the value to the ``fill_value`` argument::

    >>> registered_image, footprint = aa.register(source, target, fill_value=-99999.99)

Both will yield the same result.

.. _ftransf:

Finding the transformation
--------------------------

In some cases it may be necessary to inspect first the transformation parameters before applying it,
or we may be interested only in a star to star correspondence between the images.
For those cases, we can use ``find_transform``::

    >>> transf, (source_list, target_list) = aa.find_transform(source, target)

The inputs ``source`` and ``target`` can be either numpy arrays of the image pixels,
**or any iterable of (x, y) pairs**, corresponding to star positions.

Having an iterable of (x, y) pairs is especially useful in situations where source detection requires special care.
In situations like that, source detection can be done separately and the resulting catalogs fed to ``find_transform``.

``find_transform`` returns a `scikit-image <http://scikit-image.org>`__ `SimilarityTransform <http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.SimilarityTransform>`_ object that encapsulates the matrix transformation,
and the transformation parameters.
It will also return a tuple with two lists of star positions of ``source`` and its corresponding ordered star postions on
the ``target`` image.

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

To apply a known transform to a set of points, we use ``matrix_transform``.
Following the example in the previous section::

    >>> dst_calc = aa.matrix_transform(src, tform.params)

``dst_calc`` should be a 5 by 2 array similar to the ``dst`` array.


.. _dataobjs:

Objects with ``data`` and ``mask`` property
-------------------------------------------

If your image is stored in objects with ``data`` and ``mask`` properties,
such as `ccdproc <https://ccdproc.readthedocs.io>`_'s
`CCDData <http://docs.astropy.org/en/stable/api/astropy.nddata.CCDData.html>`_
or `astropy <https://www.astropy.org>`_'s
`NDData <https://docs.astropy.org/en/stable/api/astropy.nddata.NDData.html>`_
or a `NumPy <http://www.numpy.org>`_
`masked array <https://www.numpy.org/devdocs/reference/maskedarray.generic.html>`_
you can use them as input for ``register``, ``find_transform`` and ``apply_transform``.

In general in these cases it is convenient to transform their masks
along with the data and to add the ``footprint`` onto the mask.

Astroalign provides this functionality with the ``propagate_mask`` argument to ``register`` and ``apply_transform``.

For example::

    >>> from astropy.nddata import NDData
    >>> nd = NDData([[0, 1], [2, 3]], [[True, False], [False, False]])

and we want to apply a clockwise 90 degree rotation::

    >>> import numpy as np
    >>> from skimage.transform import SimilarityTransform
    >>> transf = SimilarityTransform(rotation=np.pi/2., translation=(1, 0))

Then we can call astroalign as usual, but with the `propagate_mask` set to True::

    >>> aligned_image, footprint = aa.apply_transform(transf, nd, nd, propagate_mask=True)

This will transform ``nd.data`` and ``nd.mask`` simultaneously and add the
``footprint`` mask from the transformation onto ``nd.mask``::

    >>> aligned_image
    array([[2., 0.],
       [3., 1.]])
    >>> footprint
    array([[False,  True],
       [False, False]])

Creating a new object of the same input type is now easier::

    >>> new_nd = NDData(aligned_image, mask=footprint)

The same will apply for ``CCDData`` objects and ``NumPy`` masked arrays.

----------------------------------------

See :ref:`api` for the API specification.
