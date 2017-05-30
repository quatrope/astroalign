.. _mask:

Working with masks
==================

Sometimes, CCD defects can confuse the alignment algorithm because of misplaced star centroids, or fake point-like sources on the image.
In those cases, you may want to mask those artifacts so they are not counted as control points.

The way to do so is to wrap your image in a `numpy masked array <http://docs.scipy.org/doc/numpy/reference/maskedarray.html>`_::

    >>> myarray = np.ma.array(myarray, mask=badpixelmask)

and mask bad pixels with True, following the numpy masked array convention.

You can now call astroalign methods in the usual way::

    >>> import astroalign as aa
    >>> aligned_image = aa.align_image(myarray, target)

The type of the returned ``aligned_image`` wil be the same type as the input image, regardless of the type of ``target``.

That is, if the source image is a masked array, the output will also be a masked array, 
with the masked transformed in the same way as the source image and filled with True
for pixels outside the boundary.
