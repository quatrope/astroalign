Working with masks
==================

Sometimes, CCD defects can confuse the alignment algorithm because of misplaced star centroids, or fake point like sources in the image.

In those cases, you may want to mask those artifacts so they are not counted as control points.

The way to do so is to wrap your image in a `numpy masked array <http://docs.scipy.org/doc/numpy/reference/maskedarray.html>`_

Like so::

    marray = np.ma.array(myarray, mask=badpixelmask)

