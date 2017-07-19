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

import numpy as _np

__version__ = '1.0.1'

from skimage.transform import estimate_transform
from skimage.transform import matrix_transform # noqa


MAX_CONTROL_POINTS = 50
"""The maximum control points (stars) to use to build the invariants.

Default: 50"""

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


def _invariantfeatures(x1, x2, x3):
    "Given 3 points x1, x2, x3, return the invariant features for the set."
    sides = _np.sort([_np.linalg.norm(x1 - x2), _np.linalg.norm(x2 - x3),
                     _np.linalg.norm(x1 - x3)])
    return [sides[2] / sides[1], sides[1] / sides[0]]


def _arrangetriplet(sources, vertex_indices):
    """Return vertex_indices ordered in an (a, b, c) form where:
  a is the vertex defined by L1 & L2
  b is the vertex defined by L2 & L3
  c is the vertex defined by L3 & L1
and L1 < L2 < L3 are the sides of the triangle defined by vertex_indices."""
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
Return an array of the indices of `sources` that correspond to each invariant,
arranged as described in _arrangetriplet.
"""
    from scipy.spatial import KDTree
    from itertools import combinations
    from functools import partial
    arrange = partial(_arrangetriplet, sources=sources)

    inv = []
    triang_vrtx = []
    coordtree = KDTree(sources)
    for asrc in sources:
        __, indx = coordtree.query(asrc, NUM_NEAREST_NEIGHBORS)

        # Generate all possible triangles with the 5 indx provided, and store
        # them with the order (a, b, c) defined in _arrangetriplet
        all_asterism_triang = [arrange(vertex_indices=list(cmb))
                               for cmb in combinations(indx, 3)]
        triang_vrtx.extend(all_asterism_triang)

        inv.extend([_invariantfeatures(*sources[triplet])
                    for triplet in all_asterism_triang])

    # Remove here all possible duplicate triangles
    uniq_ind = [pos for (pos, elem) in enumerate(inv)
                if elem not in inv[pos + 1:]]
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
        approx_t = estimate_transform('similarity',
                                      self.source[s], self.target[d])
        return approx_t

    def get_error(self, data, approx_t):
        d1, d2, d3 = data.shape
        s, d = data.reshape(d1 * d2, d3).T
        resid = approx_t.residuals(self.source[s], self.target[d])\
            .reshape(d1, d2)
        error = resid.max(axis=1)
        return error


def find_transform(source, target):
    """Estimate the transform between ``source`` and ``target``.

    Return a SimilarityTransform object ``T`` that maps pixel x, y indices from
    the source image s = (x, y) into the target (destination) image t = (x, y).
    T contains parameters of the tranformation: ``T.rotation``,
    ``T.translation``, ``T.scale``, ``T.params``.

    Args:
        source (array-like): Either a numpy array of the source image to be
            transformed or an interable of (x, y) coordinates of the target
            control points.
        target (array-like): Either a numpy array of the target (destination)
            image or an interable of (x, y) coordinates of the target
            control points.

    Returns:
        The transformation object and a tuple of corresponding star positions
        in source and target.::

            T, (source_pos_array, target_pos_array)

    Raises:
        TypeError: If input type of ``source`` or ``target`` is not supported.
        Exception: If it cannot find more than 3 stars on any input.
    """
    from scipy.spatial import KDTree

    try:
        if len(source[0]) == 2:
            # Assume it's a list of (x, y) pairs
            source_controlp = _np.array(source)[:MAX_CONTROL_POINTS]
        else:
            # Assume it's a 2D image
            source_controlp = _find_sources(source)[:MAX_CONTROL_POINTS]
    except:
        raise TypeError('Input type for source not supported.')

    try:
        if len(target[0]) == 2:
            # Assume it's a list of (x, y) pairs
            target_controlp = _np.array(target)[:MAX_CONTROL_POINTS]
        else:
            # Assume it's a 2D image
            target_controlp = _find_sources(target)[:MAX_CONTROL_POINTS]
    except:
        raise TypeError('Input type for target not supported.')

    # Check for low number of reference points
    if len(source_controlp) < 3:
        raise Exception("Reference stars in source image are less than the "
                        "minimum value (3).")
    if len(target_controlp) < 3:
        raise Exception("Reference stars in target image are less than the "
                        "minimum value (3).")

    source_invariants, source_asterisms = _generate_invariants(source_controlp)
    source_invariant_tree = KDTree(source_invariants)

    target_invariants, target_asterisms = _generate_invariants(target_controlp)
    target_invariant_tree = KDTree(target_invariants)

    # r = 0.03 is the maximum search distance, 0.03 is an empirical value that
    # returns about the same number of matches than inputs
    # matches_list is a list of lists such that for each element
    # source_invariant_tree.data[i], matches_list[i] is a list of the indices
    # of its neighbors in target_invariant_tree.data
    matches_list = \
        source_invariant_tree.query_ball_tree(target_invariant_tree, r=0.03)

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
    max_iter = n_invariants
    min_matches = min(10, int(n_invariants * MIN_MATCHES_FRACTION))
    best_t, inlier_ind = _ransac(matches, inv_model, 1, max_iter, PIXEL_TOL,
                                 min_matches)
    triangle_inliers = matches[inlier_ind]
    d1, d2, d3 = triangle_inliers.shape
    inl_arr = triangle_inliers.reshape(d1 * d2, d3)
    inl_unique = set(tuple(pair) for pair in inl_arr)
    inl_arr_unique = _np.array(list(list(apair) for apair in inl_unique))
    s, d = inl_arr_unique.T

    return best_t, (source_controlp[s], target_controlp[d])


def apply_transform(transform, source, target):
    """Applies the transformation ``transform`` to ``source``.

    The output image will have the same shape as ``target``.

    Args:
        transform: A scikit-image ``SimilarityTransform`` object.
        source (numpy array): A 2D numpy array of the source image to be
            transformed.
        target (numpy array): A 2D numpy array of the target image. Only used
            to set the output image shape.

    Return:
        A numpy 2D array of the transformed source. If source is a masked array
        the returned image will also be a masked array with outside pixels set
        to True.
    """

    from skimage.transform import warp
    aligned_image = warp(source, inverse_map=transform.inverse,
                         output_shape=target.shape, order=3, mode='constant',
                         cval=_np.median(source), clip=False,
                         preserve_range=False)

    if isinstance(source, _np.ma.MaskedArray):
        # it could be that source's mask is just set to False
        if isinstance(source.mask, _np.ndarray):
            aligned_image_mask = warp(source.mask.astype('float32'),
                                      inverse_map=transform.inverse,
                                      output_shape=target.shape,
                                      cval=1.0)
            aligned_image_mask = aligned_image_mask > 0.4
            aligned_image = _np.ma.array(aligned_image,
                                         mask=aligned_image_mask)
        else:
            # If source is masked array with mask set to false, we
            # return the same
            aligned_image = _np.ma.array(aligned_image)
    return aligned_image


def register(source, target):
    """Transform ``source`` to coincide pixel to pixel with ``target``.

    Args:
        source (numpy array): A 2D numpy array of the source image to be
            transformed.
        target (numpy array): A 2D numpy array of the target image. Only used
            to set the output image shape.

    Return:
        A numpy 2D array of the transformed source. If source is a masked array
        the returned image will also be a masked array with outside pixels set
        to True.
    """
    t, __ = find_transform(source=source, target=target)
    aligned_image = apply_transform(t, source, target)
    return aligned_image


def align_image(ref_image, img2transf, n_ref_src=50, n_img_src=70, px_tol=2.):
    "Deprecated: Alias for ``register`` for backwards compatibility."
    return register(img2transf, ref_image)


def find_affine_transform(test_srcs, ref_srcs, max_pix_tol=2.,
                          min_matches_fraction=0.8, invariant_map=None):
    "Deprecated: Alias for ``find_transform`` for backwards compatibility."
    transf, _ = find_transform(ref_srcs, test_srcs)
    return transf.params


def _find_sources(img):
    "Return sources (x, y) sorted by brightness."

    import sep
    if isinstance(img, _np.ma.MaskedArray):
        image = img.filled(fill_value=_np.median(img)).astype('float32')
    else:
        image = img.astype('float32')
    bkg = sep.Background(image)
    thresh = 3. * bkg.globalrms
    sources = sep.extract(image - bkg.back(), thresh)
    sources.sort(order='flux')
    return _np.array([[asrc['x'], asrc['y']] for asrc in sources[::-1]])


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

def _ransac(data, model, min_data_points, max_iter, thresh, min_matches):
    """fit model parameters to data using the RANSAC algorithm

This implementation written from pseudocode found at
http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

Given:
    data: a set of data points
    model: a model that can be fitted to data points
    min_data_points: the minimum number of data values required to fit the
        model
    max_iter: the maximum number of iterations allowed in the algorithm
    thresh: a threshold value to determine when a data point fits a model
    min_matches: the min number of matches required to assert that a model
        fits well to data
Return:
    bestfit: model parameters which best fit the data (or nil if no good model
              is found)
"""
    iterations = 0
    bestfit = None
    best_inlier_idxs = None
    n_data = data.shape[0]
    n = min_data_points
    all_idxs = _np.arange(n_data)

    while iterations < max_iter:
        # Partition indices into two random subsets
        _np.random.shuffle(all_idxs)
        maybe_idxs, test_idxs = all_idxs[:n], all_idxs[n:]
        maybeinliers = data[maybe_idxs, :]
        test_points = data[test_idxs, :]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error(test_points, maybemodel)
        # select indices of rows with accepted points
        also_idxs = test_idxs[test_err < thresh]
        alsoinliers = data[also_idxs, :]
        if len(alsoinliers) > min_matches:
            betterdata = _np.concatenate((maybeinliers, alsoinliers))
            bestfit = model.fit(betterdata)
            best_inlier_idxs = _np.concatenate((maybe_idxs, also_idxs))
            break
        iterations += 1
    if bestfit is None:
        raise ValueError("Did not meet fit acceptance criteria")

    return bestfit, best_inlier_idxs
