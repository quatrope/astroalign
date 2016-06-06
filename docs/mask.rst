Working with masks
==================

Sometimes, CCD defects can confuse the alignment algorithm because of misplaced star centroids, or fake point like sources in the image.

In those cases, you may want to mask those artifacts so they are not counted as control points.

The way to do so is to wrap your image in a `numpy masked array <http://docs.scipy.org/doc/numpy/reference/maskedarray.html>`_::

    marray = np.ma.array(myarray, mask=badpixelmask)

and mask bad pixels with True, following the numpy masked array convention.

Either the refeerence image or the input image can be masked this way.

You can call astro_align now the usual way::

    >>> import astroalign as aa
    >>> aligned_image = aa.align_image(image_ref, myarray)

The type of the returned aligned_image wil be the same type as the input image, regardless of the type of image_ref.

That is, if input image is a masked array, the output will also be a masked array, 
with the masked transformed in the same way as the input image and filled with True
for pixels outside the boundary.
