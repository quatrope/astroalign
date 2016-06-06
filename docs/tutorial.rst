A simple usage example
======================

Suppose we have two images of about the same portion of the sky, and we would like to transform one of them to fit on top of the other one.
Suppose we do not have WCS information, but we are confident that we could do it by eye, by matching some obvious asterisms on the two images.

In this particular use case, astroalign can be of great help to automatize the process.

After we upload the data of our images into numpy array, we simple choose one to be the reference and the other to be the one to be transformed.

The usage for this simple most common case would be as follows::


    >>> import astroalign as aa
    >>> aligned_image = aa.align_image(image_ref, image_trans)

aligned_image is now a transformed image of image_trans that will match pixel to pixel to image_ref.

In most cases, this will be all the functionality you will need from this tool.
However Astroalign provides other tools that can help you apply the transformation to other objects like source pixel coordinates.

See :ref:`methods` for more information.
