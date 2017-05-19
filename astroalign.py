"""
ASTROALIGN is a simple package that will try to align two stellar astronomical
images.

It does so by finding similar 3-point asterisms (triangles) in both images and
deducing the affine transformation between them.

General align routines try to match interesting points, using corner detection
routines to make the point correspondence.

These generally fail for stellar astronomical images, since stars have very
little stable structure and so, in general, indistinguishable from each other.

Asterism matching is more robust and closer to the human way of matching
images.

Astro align can match images of very different field of view, point-spread
functions, seeing and atmospheric conditions.

(c) Martin Beroiz
"""

import numpy as _np

__version__ = '1.0.0.dev0'


MAX_CONTROL_POINTS = 50
PIXEL_TOL = 2
MIN_MATCHES_FRACTION = 0.8
NUM_NEAREST_NEIGHBORS = 5


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
    side_lengths = map(_np.linalg.norm, (x1 - x2, x2 - x3, x3 - x1))
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
    def __init__(self, ref_srcs, target_srcs):
        self.ref = ref_srcs
        self.target = target_srcs

    def fit(self, data):
        """
    Return the best 2D similarity transform from the points given in data.

    data: N sets of 3 indices for a triangle in ref
        and the 3 indices of the corresponding triangle in target;
        arranged in a (N, 3, 2) array.
        """
        from skimage.transform import estimate_transform
        d1, d2, d3 = data.shape
        s, d = data.reshape(d1 * d2, d3).T
        approx_t = estimate_transform('similarity',
                                      self.ref[s], self.target[d])
        approxm = approx_t.params[:2, :]
        return approxm

    def get_error(self, data, approxm):
        error = []
        for atrianglematch in data:
            max_err = 0.
            for ind_r, ind_t in atrianglematch:
                x = self.ref[ind_r]
                y = self.target[ind_t]
                y_fit = approxm.dot(_np.append(x, 1))
                max_err = max(max_err, _np.linalg.norm(y - y_fit))
            error.append(max_err)
        error = _np.array(error)
        return error


def get_transform(source, target):
    """Return the 2 by 3 affine transformation M that maps pixel x, y indices
from the source image s = (x, y) into the target (destination) image t = (x, y)
t = M * s.
Return parameters (rotation_angle, translation_x, translation_y, scale_factor)
from the transformation.
Return an iterable of matched sources.

source:
  Either a numpy array of the source image to be transformed
  or an interable of (x, y) coordinates of the source control points.
target:
  Either a numpy array of the target (destination) image
  or an interable of (x, y) coordinates of the target control points.
"""
    from scipy.spatial import KDTree

    try:
        import sep  # noqa
    except ImportError:
        source_finder = _find_sources
    else:
        source_finder = _find_sources_with_sep

    try:
        if len(source[0]) == 2:
            # Assume it's a list of (x, y) pairs
            source_controlp = _np.array(source)[:MAX_CONTROL_POINTS]
        else:
            # Assume it's a 2D image
            source_controlp = source_finder(source)[:MAX_CONTROL_POINTS]
    except:
        raise TypeError('Input type for source not supported.')

    try:
        if len(target[0]) == 2:
            # Assume it's a list of (x, y) pairs
            target_controlp = _np.array(target)[:MAX_CONTROL_POINTS]
        else:
            # Assume it's a 2D image
            target_controlp = source_finder(target)[:MAX_CONTROL_POINTS]
    except:
        raise TypeError('Input type for target not supported.')

    # Check for low number of reference points
    if len(source_controlp) < 3:
        raise Exception("Reference stars in source image are less than the "
                        "minimum value of points (3).")
    if len(target_controlp) < 3:
        raise Exception("Reference stars in target image are less than the "
                        "minimum value of points (3).")

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

    matches = []
    # t1 is an asterism in source, t2 in target
    for t1, t2_list in zip(source_asterisms, matches_list):
        for t2 in target_asterisms[t2_list]:
            matches.append(zip(t2, t1))
    matches = _np.array(matches)

    inv_model = _MatchTransform(target_controlp, source_controlp)
    n_invariants = len(matches)
    max_iter = n_invariants
    min_matches = min(10, int(n_invariants * MIN_MATCHES_FRACTION))
    best_m, inliers = _ransac(matches, inv_model, 1, max_iter, PIXEL_TOL,
                              min_matches)

    return best_m


def align_image(source, target):
    """Return a transformed (aligned) image of source that coincides
    pixel to pixel with target.

    align_image accepts a numpy array or a numpy masked array, and returns a
    realigned image interpolated to coincide with target.
    Sometimes bad CCD sections can confuse the alignment.
    Bad pixels can be masked (True on bad) in a masked array to facilitate the
    process.
    The returned image will be the same type as source.
    Masks will be transformed the same way as source.

    Return aligned_image"""

    from scipy.ndimage.interpolation import affine_transform

    try:
        import sep  # noqa
    except ImportError:
        source_finder = _find_sources
    else:
        source_finder = _find_sources_with_sep

    ref_srcs = source_finder(target)[:MAX_CONTROL_POINTS]
    img_sources = source_finder(source)[:MAX_CONTROL_POINTS]

    m = get_transform(source=ref_srcs, target=img_sources)

    # SciPy Affine transformation transform a (row,col) pixel according to pT+s
    # where p is in the _output_ image, T is the rotation and s the translation
    # offset, so some mathematics is required to put it into a suitable form
    # In particular, affine_transform() requires the inverse transformation
    # that registration returns but for (row, col) instead of (x,y)
    def inverse_transform(m):
        m_rot_inv = _np.linalg.inv(m[:2, :2])
        m_offset_inv = -m_rot_inv.dot(m[:2, 2])
        m_inv = _np.zeros(m.shape)
        m_inv[:2, :2] = m_rot_inv
        m_inv[:2, 2] = m_offset_inv
        if m.shape == (3, 3):
            m_inv[2, 2] = 1
        return m_inv

    m_inv = inverse_transform(m)
    # p will transform from (x,y) to (row, col)=(y,x)
    p = _np.array([[0, 1], [1, 0]])
    mrcinv_rot = p.dot(m_inv[:2, :2]).dot(p)
    mrcinv_offset = p.dot(m_inv[:2, 2])

    aligned_image = affine_transform(source, mrcinv_rot,
                                     offset=mrcinv_offset,
                                     output_shape=target.shape,
                                     cval=_np.median(source)
                                     )
    if isinstance(source, _np.ma.MaskedArray):
        # it could be that source's mask is just set to False
        if type(source.mask) is _np.ndarray:
            aligned_image_mask = \
                affine_transform(source.mask.astype('float32'),
                                 mrcinv_rot,
                                 offset=mrcinv_offset,
                                 output_shape=target.shape,
                                 cval=1.0
                                 )
            aligned_image_mask = aligned_image_mask > 0.4
            aligned_image = _np.ma.array(aligned_image,
                                         mask=aligned_image_mask)
        else:
            # If source is masked array with mask set to false, we
            # return the same
            aligned_image = _np.ma.array(aligned_image)
    return aligned_image


def _find_sources(image):
    """Return sources (x, y) sorted by brightness.
    """
    from scipy import ndimage
    from astropy.stats import mad_std

    img1 = image.copy().astype('float32')
    m, s = _np.median(image), mad_std(image)
    src_mask = image > m + 3.0 * s
    # set the background to the min value of the sources
    img1[~src_mask] = img1[src_mask].min()
    # this rescales (min,max) to (0,1)
    img1 = (img1.min() - img1) / (img1.min() - img1.max())
    img1[~src_mask] = 0.

    def obj_params_with_offset(img, labels, aslice, label_idx):
        y_offset = aslice[0].start
        x_offset = aslice[1].start
        thumb = img[aslice]
        lb = labels[aslice]
        yc, xc = ndimage.center_of_mass(thumb, labels=lb, index=label_idx)
        br = thumb[lb == label_idx].sum()  # the intensity of the source
        return [br, xc + x_offset, yc + y_offset]

    srcs_labels, num_srcs = ndimage.label(img1)

    # Eliminate here all 1 pixel sources
    all_objects = [[ind + 1, aslice] for ind, aslice
                   in enumerate(ndimage.find_objects(srcs_labels))
                   if srcs_labels[aslice].shape != (1, 1)]
    lum = _np.array([obj_params_with_offset(img1, srcs_labels, aslice, lab_idx)
                    for lab_idx, aslice in all_objects])

    lum = lum[lum[:, 0].argsort()[::-1]]  # sort by brightness descending order

    return lum[:, 1:]


def _find_sources_with_sep(img):
    """Return sources (x, y) sorted by brightness. Use SEP package.
    """
    import sep
    if isinstance(img, _np.ma.MaskedArray):
        image = img.filled(fill_value=_np.median(img)).astype('float32')
    else:
        image = img.astype('float32')

    bkg = sep.Background(image)
    thresh = 3. * bkg.globalrms
    try:
        sources = sep.extract(image - bkg.back(), thresh)
    except Exception as e:
        buff_message = 'internal pixel buffer full'
        if e.message[0:26] == buff_message:
            sep.set_extract_pixstack(600000)
        try:
            sources = sep.extract(image - bkg.back(), thresh)
        except Exception as e:
            if e.message[0:26] == buff_message:
                sep.set_extract_pixstack(900000)
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

{{{
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
iterations = 0
bestfit = nil
besterr = something really large
while iterations < max_iter {
    maybeinliers = n randomly selected values from data
    maybemodel = model parameters fitted to maybeinliers
    alsoinliers = empty set
    for every point in data not in maybeinliers {
        if point fits maybemodel with an error smaller than thresh
             add point to alsoinliers
    }
    if the number of elements in alsoinliers is > min_matches {
        % this implies that we may have found a good model
        % now test how good it is
        bettermodel = model parameters fitted to all points in maybeinliers and
        alsoinliers
        thiserr = a measure of how well model fits these points
        if thiserr < besterr {
            bestfit = bettermodel
            besterr = thiserr
        }
    }
    increment iterations
}
return bestfit
}}}
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
