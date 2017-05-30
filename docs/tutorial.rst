A simple usage example
======================

Suppose we have two images of about the same portion of the sky, and we would like to transform one of them to fit on top of the other one.
Suppose we do not have WCS information, but we are confident that we could do it by eye, by matching some obvious asterisms on the two images.

In this particular use case, astroalign can be of great help to automatize the process.

After we load our images into numpy array, we simple choose one to be the source image and the other to be the target.

The usage for this simple most common case would be as follows::


    >>> import astroalign as aa
    >>> aligned_image = aa.align_image(source, target)

``aligned_image`` is now a transformed (numpy array) image of ``source`` that will match pixel to pixel to ``target``.

In some cases it may be necessary to inspect first the transformation parameters before applying it,
or we may be interested only in a star to star correspondance between the images.
For those cases, we can use get_transform.

``get_transform`` will return a `scikit-image <http://scikit-image.org>`_ `SimilarityTransform <http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.SimilarityTransform>`_ object that encapsulates the matrix transformation,
and the transformation parameters. 
It will also return a tuple with two lists of star positions of ``source`` and its corresponding ordered star postions on 
the ``target`` image.::


    >>> transf, (source_list, target_list) = aa.get_transform(source, target)

source and target here can be either numpy arrays of the image pixels, or any iterable (x, y) pair, 
corresponding to a star position.

The transformation parameters can be found in ``transf.rotation``, ``transf.traslation``, ``transf.scale`` 
and the transformation matrix in ``transf.params``.

If the transformation is satisfactory we can apply it to the image with ``apply_transform``::

    >>> aligned_image = aa.apply_transform(transf, source, target)

In most cases, this will be all the functionality you will need from this tool.
As a convenience, ``estimate_transform`` and ``matrix_transform`` from ``scikit-image`` are imported in astroalign as well.

See :ref:`methods` for more information.
