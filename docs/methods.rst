.. _methods:

Other Methods
=============


``get_transform`` (`source`, `target`)

Estimate a SimilarityTransform object T that maps pixel x, y indices
from the source image s = (x, y) into the target (destination) image t = (x, y)
t = T * s.
T contains parameters of the tranformation T.rotation, T.translation, T.scale

Parameters
    ``source``
      Either a numpy array of the source image to be transformed
      or an interable of (x, y) coordinates of the target control points.
    ``target``
      Either a numpy array of the target (destination) image
      or an interable of (x, y) coordinates of the target control points.

Returns
    ``(source_control_points, target_control_points)``
      A tuple of matched star positions in source and target.


-----------------------------------------------------------------------------


``apply_transform`` (`transform`, `source`, `target`)

Applies the transformation ``transform`` to source.
The output image will have the same shape as target.

Parameters
    ``source``, ``target``:
      2D numpy arrays (not necessarily the same shape)
    ``transform``:
      A ``SimilarityTransform`` object.

Returns
    ``aligned_image``
      Numpy 2D array (image) the same type as ``source``.
      Masks will be transformed the same way as ``source``.


-----------------------------------------------------------------------------


``align_image`` (`source`, `target`)

Return the registered image of ``source`` to coincide with ``target``.

Parameters
    ``source``, ``target``:
      2D numpy arrays (not necessarily the same shape)
    ``transform``:
      A ``SimilarityTransform`` object.

Returns
    ``aligned_image``
      Numpy 2D array (image) the same type as ``source``.
      Masks will be transformed the same way as ``source``.

-----------------------------------------------------------------------------


``estimate_transform`` (`ttype`, `src`, `dst`, `**kwargs`)


Given a list of (x, y) positions on source corresponding to those of dest, 
estimate the ``ttype`` that transform ``src`` into ``dst``.

This is a convenient import from scikit-image.
Refer to `scikit-image 
<http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.estimate_transform>`_ 
documentation for more information.

-----------------------------------------------------------------------------

``matrix_transform`` (`coords`, `matrix`)


Apply 2D matrix transform to coords.

This is a convenient import from scikit-image.
Refer to `scikit-image 
<http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.matrix_transform>`_ 
documentation for more information.

-----------------------------------------------------------------------------

``MAX_CONTROL_POINTS`` 


The maximum control points (stars) to use to build the invariants.

Default: 50

-----------------------------------------------------------------------------

``PIXEL_TOL`` 


The pixel distance tolerance to assume two invariant points are the same.

Default: 2

-----------------------------------------------------------------------------

``MIN_MATCHES_FRACTION``


The minimum fraction of triangle matches to accept a transformation.

If the minimum fraction yields more than 10 triangles, 10 is used instead.

Default: 0.8

-----------------------------------------------------------------------------

``NUM_NEAREST_NEIGHBORS``


The number of nearest neighbors of a given star (including itself) to construct 
the triangle invariants.

Default: 5
