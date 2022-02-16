# MIT License

# Copyright (c) 2016 Martin Beroiz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
ASTROALIGN is a simple package that will try to align two stellar astronomical
images, especially when there is no WCS information available.

It does so by finding similar 3-point asterisms (triangles) in both images and
deducing the affine transformation between them.

General registration routines try to match feature points, using corner
detection routines to make the point correspondence.
These generally fail for stellar astronomical images, since stars have very
little stable structure and so, in general, indistinguishable from each other.

Asterism matching is more robust, and closer to the human way of matching
stellar images.

Astroalign can match images of very different field of view, point-spread
functions, seeing and atmospheric conditions.

(c) Martin Beroiz
"""


__version__ = "2.4.1"

__all__ = [
    "MIN_MATCHES_FRACTION",
    "MaxIterError",
    "NUM_NEAREST_NEIGHBORS",
    "PIXEL_TOL",
    "apply_transform",
    "estimate_transform",
    "find_transform",
    "matrix_transform",
    "register",
]

try:
    import bottleneck as bn
except ImportError:
    HAS_BOTTLENECK = False
else:
    HAS_BOTTLENECK = True

import numpy as _np
from skimage.transform import estimate_transform
from skimage.transform import matrix_transform

PIXEL_TOL = 2
"""The pixel distance tolerance to assume two invariant points are the same.

Default: 2"""

MIN_MATCHES_FRACTION = 0.8
"""The minimum fraction of triangle matches to accept a transformation.

If the minimum fraction yields more than 10 triangles, 10 is used instead.

Default: 0.8
"""

NUM_NEAREST_NEIGHBORS = 5
"""
The number of nearest neighbors of a given star (including itself) to construct
the triangle invariants.

Default: 5
"""

_default_median = bn.nanmedian if HAS_BOTTLENECK else _np.nanmedian  # pragma: no cover
"""
Default median function when/if optional bottleneck is available
"""

_default_average = bn.nanmean if HAS_BOTTLENECK else _np.nanmean  # pragma: no cover
"""
Default mean function when/if optional bottleneck is available
"""

_default_sum = bn.nansum if HAS_BOTTLENECK else _np.nansum  # pragma: no cover
"""
Default sum function when/if optional bottleneck is available
"""

_default_std = bn.nanstd if HAS_BOTTLENECK else _np.nanstd  # pragma: no cover
"""
Default std deviation function when/if optional bottleneck is available
"""


def _invariantfeatures(x1, x2, x3):
    "Given 3 points x1, x2, x3, return the invariant features for the set."
    sides = _np.sort(
        [
            _np.linalg.norm(x1 - x2),
            _np.linalg.norm(x2 - x3),
            _np.linalg.norm(x1 - x3),
        ]
    )
    return [sides[2] / sides[1], sides[1] / sides[0]]


def _arrangetriplet(sources, vertex_indices):
    """Return vertex_indices ordered in an (a, b, c) form where:
      a is the vertex defined by L1 & L2
      b is the vertex defined by L2 & L3
      c is the vertex defined by L3 & L1
    and L1 < L2 < L3 are the sides of the triangle
    defined by vertex_indices."""
    ind1, ind2, ind3 = vertex_indices
    x1, x2, x3 = sources[vertex_indices]

    side_ind = _np.array([(ind1, ind2), (ind2, ind3), (ind3, ind1)])
    side_lengths = list(map(_np.linalg.norm, (x1 - x2, x2 - x3, x3 - x1)))
    l1_ind, l2_ind, l3_ind = _np.argsort(side_lengths)

    # the most common vertex in the list of vertices for two sides is the
    # point at which they meet.
    from collections import Counter

    count = Counter(side_ind[[l1_ind, l2_ind]].flatten())
    a = count.most_common(1)[0][0]
    count = Counter(side_ind[[l2_ind, l3_ind]].flatten())
    b = count.most_common(1)[0][0]
    count = Counter(side_ind[[l3_ind, l1_ind]].flatten())
    c = count.most_common(1)[0][0]

    return _np.array([a, b, c])


def _generate_invariants(sources):
    """Return an array of (unique) invariants derived from the array `sources`.
    Return an array of the indices of `sources` that correspond to each
    invariant, arranged as described in _arrangetriplet."""
    from scipy.spatial import KDTree
    from itertools import combinations
    from functools import partial

    arrange = partial(_arrangetriplet, sources=sources)

    inv = []
    triang_vrtx = []
    coordtree = KDTree(sources)
    # The number of nearest neighbors to request (to work with few sources)
    knn = min(len(sources), NUM_NEAREST_NEIGHBORS)
    for asrc in sources:
        __, indx = coordtree.query(asrc, knn)

        # Generate all possible triangles with the 5 indx provided, and store
        # them with the order (a, b, c) defined in _arrangetriplet
        all_asterism_triang = [
            arrange(vertex_indices=list(cmb)) for cmb in combinations(indx, 3)
        ]
        triang_vrtx.extend(all_asterism_triang)

        inv.extend(
            [
                _invariantfeatures(*sources[triplet])
                for triplet in all_asterism_triang
            ]
        )

    # Remove here all possible duplicate triangles
    uniq_ind = [
        pos for (pos, elem) in enumerate(inv) if elem not in inv[pos + 1 :]
    ]
    inv_uniq = _np.array(inv)[uniq_ind]
    triang_vrtx_uniq = _np.array(triang_vrtx)[uniq_ind]

    return inv_uniq, triang_vrtx_uniq


class _MatchTransform:
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def fit(self, data):
        """
        Return the best 2D similarity transform from the points given in data.

        data: N sets of similar corresponding triangles.
            3 indices for a triangle in ref
            and the 3 indices for the corresponding triangle in target;
            arranged in a (N, 3, 2) array.
        """
        d1, d2, d3 = data.shape
        s, d = data.reshape(d1 * d2, d3).T
        approx_t = estimate_transform(
            "similarity", self.source[s], self.target[d]
        )
        return approx_t

    def get_error(self, data, approx_t):
        d1, d2, d3 = data.shape
        s, d = data.reshape(d1 * d2, d3).T
        resid = approx_t.residuals(self.source[s], self.target[d]).reshape(
            d1, d2
        )
        error = resid.max(axis=1)
        return error


def _data(image):
    if hasattr(image, "data") and isinstance(image.data, _np.ndarray):
        return image.data
    else:
        return _np.asarray(image)


def _bw(image):
    "Return a 2D numpy array for an array of arbitrary channels"
    if image.ndim == 2:
        return image
    return _default_average(image, axis=-1)


def _shape(image):
    "Return a 2D shape for the image, ignoring channel info"
    if image.ndim == 2:
        return image.shape
    h, w, ch = image.shape
    return h, w


def find_transform(
    source, target, max_control_points=50, detection_sigma=5, min_area=5
):
    """Estimate the transform between ``source`` and ``target``.

    Return a SimilarityTransform object ``T`` that maps pixel x, y indices from
    the source image s = (x, y) into the target (destination) image t = (x, y).
    T contains parameters of the tranformation: ``T.rotation``,
    ``T.translation``, ``T.scale``, ``T.params``.

    Args:
        source (array-like): Either a NumPy, CCData or NDData array of the
            source image to be transformed or an interable of (x, y)
            coordinates of the target control points.
        target (array-like): Either a NumPy, CCData or NDData array of the
            target (destination) image or an interable of (x, y) coordinates of
            the target control points.
        max_control_points: The maximum number of control point-sources to find
            the transformation.
        detection_sigma: Factor of background std-dev above which is considered
            a detection. This value is ignored if input are not images.
        min_area: Minimum number of connected pixels to be considered a source.
            This value is ignored if input are not images.

    Returns:
        The transformation object and a tuple of corresponding star positions
        in source and target.::

            T, (source_pos_array, target_pos_array)

    Raises:
        TypeError: If input type of ``source`` or ``target`` is not supported.
        ValueError: If it cannot find more than 3 stars on any input.
    """
    from scipy.spatial import KDTree

    try:
        if len(_data(source)[0]) == 2:
            # Assume it's a list of (x, y) pairs
            source_controlp = _np.array(source)[:max_control_points]
        else:
            # Assume it's a 2D image
            source_controlp = _find_sources(
                _bw(_data(source)),
                detection_sigma=detection_sigma,
                min_area=min_area,
            )[:max_control_points]
    except Exception:
        raise TypeError("Input type for source not supported.")

    try:
        if len(_data(target)[0]) == 2:
            # Assume it's a list of (x, y) pairs
            target_controlp = _np.array(target)[:max_control_points]
        else:
            # Assume it's a 2D image
            target_controlp = _find_sources(
                _bw(_data(target)),
                detection_sigma=detection_sigma,
                min_area=min_area,
            )[:max_control_points]
    except Exception:
        raise TypeError("Input type for target not supported.")

    # Check for low number of reference points
    if len(source_controlp) < 3:
        raise ValueError(
            "Reference stars in source image are less than the "
            "minimum value (3)."
        )
    if len(target_controlp) < 3:
        raise ValueError(
            "Reference stars in target image are less than the "
            "minimum value (3)."
        )

    source_invariants, source_asterisms = _generate_invariants(source_controlp)
    source_invariant_tree = KDTree(source_invariants)

    target_invariants, target_asterisms = _generate_invariants(target_controlp)
    target_invariant_tree = KDTree(target_invariants)

    # r = 0.1 is the maximum search distance, 0.1 is an empirical value that
    # returns about the same number of matches than inputs
    # matches_list is a list of lists such that for each element
    # source_invariant_tree.data[i], matches_list[i] is a list of the indices
    # of its neighbors in target_invariant_tree.data
    matches_list = source_invariant_tree.query_ball_tree(
        target_invariant_tree, r=0.1
    )

    # matches unravels the previous list of matches into pairs of source and
    # target control point matches.
    # matches is a (N, 3, 2) array. N sets of similar corresponding triangles.
    # 3 indices for a triangle in ref
    # and the 3 indices for the corresponding triangle in target;
    matches = []
    # t1 is an asterism in source, t2 in target
    for t1, t2_list in zip(source_asterisms, matches_list):
        for t2 in target_asterisms[t2_list]:
            matches.append(list(zip(t1, t2)))
    matches = _np.array(matches)

    inv_model = _MatchTransform(source_controlp, target_controlp)
    n_invariants = len(matches)
    # Set the minimum matches to be between 1 and 10 asterisms
    min_matches = max(1, min(10, int(n_invariants * MIN_MATCHES_FRACTION)))
    if (len(source_controlp) == 3 or len(target_controlp) == 3) and len(
        matches
    ) == 1:
        best_t = inv_model.fit(matches)
        inlier_ind = _np.arange(len(matches))  # All of the indices
    else:
        best_t, inlier_ind = _ransac(
            matches, inv_model, PIXEL_TOL, min_matches
        )
    triangle_inliers = matches[inlier_ind]
    d1, d2, d3 = triangle_inliers.shape
    inl_arr = triangle_inliers.reshape(d1 * d2, d3)
    inl_unique = set(tuple(pair) for pair in inl_arr)
    # In the next, multiple assignements to the same source point s are removed
    # We keep the pair (s, t) with the lowest reprojection error.
    inl_dict = {}
    for s_i, t_i in inl_unique:
        # calculate error
        s_vertex = source_controlp[s_i]
        t_vertex = target_controlp[t_i]
        t_vertex_pred = matrix_transform(s_vertex, best_t.params)
        error = _np.linalg.norm(t_vertex_pred - t_vertex)

        # if s_i not in dict, or if its error is smaller than previous error
        if s_i not in inl_dict or (error < inl_dict[s_i][1]):
            inl_dict[s_i] = (t_i, error)
    inl_arr_unique = _np.array(
        [[s_i, t_i] for s_i, (t_i, e) in inl_dict.items()]
    )
    s, d = inl_arr_unique.T

    return best_t, (source_controlp[s], target_controlp[d])


def apply_transform(
    transform, source, target, fill_value=None, propagate_mask=False
):
    """Applies the transformation ``transform`` to ``source``.

    The output image will have the same shape as ``target``.

    Args:
        transform: A scikit-image ``SimilarityTransform`` object.
        source (numpy array): A 2D NumPy, CCData or NDData array of the source
            image to be transformed.
        target (numpy array): A 2D NumPy, CCData or NDData array of the target
            image. Only used to set the output image shape.
        fill_value (float): A value to fill in the areas of aligned_image
            where footprint == True.
        propagate_mask (bool): Wether to propagate the mask in source.mask
            onto footprint.

    Return:
        A tuple (aligned_image, footprint).
        aligned_image is a numpy 2D array of the transformed source
        footprint is a mask 2D array with True on the regions
        with no pixel information.
    """
    from skimage.transform import warp

    source_data = _data(source)
    target_shape = _data(target).shape

    aligned_image = warp(
        source_data,
        inverse_map=transform.inverse,
        output_shape=target_shape,
        order=3,
        mode="constant",
        cval=_default_median(source_data),
        clip=True,
        preserve_range=True,
    )

    footprint = warp(
        _np.zeros(_shape(source_data), dtype="float32"),
        inverse_map=transform.inverse,
        output_shape=target_shape,
        cval=1.0,
    )
    footprint = footprint > 0.4

    if hasattr(source, "mask") and propagate_mask:
        source_mask = _np.array(source.mask)
        if source_mask.shape == source_data.shape:
            source_mask_rot = warp(
                source_mask.astype("float32"),
                inverse_map=transform.inverse,
                output_shape=target_shape,
                cval=1.0,
            )
            source_mask_rot = source_mask_rot > 0.4
            footprint = footprint | source_mask_rot
    if fill_value is not None:
        aligned_image[footprint] = fill_value

    return aligned_image, footprint


def register(
    source,
    target,
    fill_value=None,
    propagate_mask=False,
    max_control_points=50,
    detection_sigma=5,
    min_area=5,
):
    """Transform ``source`` to coincide pixel to pixel with ``target``.

    Args:
        source (numpy array): A 2D NumPy, CCData or NDData array of the source
            image to be transformed.
        target (numpy array): A 2D NumPy, CCData or NDData array of the target
            image. Used to set the output image shape as well.
        fill_value (float): A value to fill in the areas of aligned_image
            where footprint == True.
        propagate_mask (bool): Wether to propagate the mask in source.mask
            onto footprint.
        max_control_points: The maximum number of control point-sources to find
            the transformation.
        detection_sigma: Factor of background std-dev above which is considered
            a detection.
        min_area: Minimum number of connected pixels to be considered a source.

    Return:
        A tuple (aligned_image, footprint).
        aligned_image is a numpy 2D array of the transformed source
        footprint is a mask 2D array with True on the regions
        with no pixel information.


    """
    t, __ = find_transform(
        source=source,
        target=target,
        max_control_points=max_control_points,
        detection_sigma=detection_sigma,
        min_area=min_area,
    )
    aligned_image, footprint = apply_transform(
        t, source, target, fill_value, propagate_mask
    )
    return aligned_image, footprint


def _find_sources(img, detection_sigma=5, min_area=5):
    "Return sources (x, y) sorted by brightness."

    import sep

    if isinstance(img, _np.ma.MaskedArray):
        image = img.filled(fill_value=_default_median(img)).astype("float32")
    else:
        image = img.astype("float32")
    bkg = sep.Background(image)
    thresh = detection_sigma * bkg.globalrms
    sources = sep.extract(image - bkg.back(), thresh, minarea=min_area)
    sources.sort(order="flux")
    return _np.array([[asrc["x"], asrc["y"]] for asrc in sources[::-1]])


# Copyright (c) 2004-2007, Andrew D. Straw. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.

#     * Neither the name of the Andrew D. Straw nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# a PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# Modified by Martin Beroiz


class MaxIterError(RuntimeError):
    pass


def _ransac(data, model, thresh, min_matches):
    """fit model parameters to data using the RANSAC algorithm

    This implementation written from pseudocode found at
    http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

    Given:
        data: a set of data points
        model: a model that can be fitted to data points
        thresh: a threshold value to determine when a data point fits a model
        min_matches: the min number of matches required to assert that a model
            fits well to data
    Return:
        bestfit: model parameters which best fit the data (or nil if no good
                  model is found)"""
    good_fit = None
    n_data = data.shape[0]
    all_idxs = _np.arange(n_data)
    _np.random.shuffle(all_idxs)

    for iter_i in range(n_data):
        # Partition indices into two random subsets
        maybe_idxs = all_idxs[iter_i : iter_i + 1]
        test_idxs = list(all_idxs[:iter_i])
        test_idxs.extend(list(all_idxs[iter_i + 1 :]))
        test_idxs = _np.array(test_idxs)
        maybeinliers = data[maybe_idxs, :]
        test_points = data[test_idxs, :]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error(test_points, maybemodel)
        # select indices of rows with accepted points
        also_idxs = test_idxs[test_err < thresh]
        alsoinliers = data[also_idxs, :]
        if len(alsoinliers) >= min_matches:
            good_data = _np.concatenate((maybeinliers, alsoinliers))
            good_fit = model.fit(good_data)
            break

    if good_fit is None:
        raise MaxIterError(
            "List of matching triangles exhausted before an acceptable "
            "transformation was found"
        )

    better_fit = good_fit
    for i in range(3):
        test_err = model.get_error(data, better_fit)
        better_inlier_idxs = _np.arange(n_data)[test_err < thresh]
        better_data = data[better_inlier_idxs]
        better_fit = model.fit(better_data)
    best_fit = better_fit
    best_inlier_idxs = better_inlier_idxs
    return best_fit, best_inlier_idxs
