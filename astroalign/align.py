import numpy as np
from scipy.spatial import KDTree
from itertools import combinations
import ransac


class InvariantTriangleMapping():

    def invariantfeat(self, sources, ind1, ind2, ind3):
        x1, x2, x3 = sources[[ind1, ind2, ind3]]
        sides = np.sort([np.linalg.norm(x1 - x2), np.linalg.norm(x2 - x3),
                        np.linalg.norm(x1 - x3)])
        return [sides[2] / sides[1], sides[1] / sides[0]]

    def generate_invariants(self, sources, nearest_neighbors=5):
        # Helping function
        def arrangetriplet(sources, vertex_indices):
            side1 = np.array([vertex_indices[0], vertex_indices[1]])
            side2 = np.array([vertex_indices[1], vertex_indices[2]])
            side3 = np.array([vertex_indices[0], vertex_indices[2]])

            sidelengths = [np.linalg.norm(sources[p1_ind] - sources[p2_ind])
                           for p1_ind, p2_ind in [side1, side2, side3]]
            lengths_arg = np.argsort(sidelengths)

            # Sides sorted from shortest to longest
            sides = np.array([side1, side2, side3])[lengths_arg]

            # now I order the points inside the side this way:
            # [(x2,x0),(x0,x1),(x1,x2)]
            for i in range(-1, 2):
                if sides[i][0] in sides[i + 1]:
                    # swap the points
                    sides[i] = sides[i][[1, 0]]

            return sides[:, 1]

        inv = []
        triang_vrtx = []
        coordtree = KDTree(sources)
        for asrc in sources:
            __, indx = coordtree.query(asrc, 5)
            all_asterism_triang = [list(cmb) for cmb in combinations(indx, 3)]
            inv.extend([self.invariantfeat(sources, *triplet)
                        for triplet in all_asterism_triang])
            triang_vrtx.extend(all_asterism_triang)

        # Remove here many duplicate triangles for close-tight neighbors
        inv_uniq = np.array([elem for (pos, elem) in enumerate(inv)
                            if elem not in inv[pos + 1:]])
        triang_vrtx_uniq = [triang_vrtx[pos] for (pos, elem) in enumerate(inv)
                            if elem not in inv[pos + 1:]]

        # This will order the vertices in the triangle in a determined way to
        # make a point to point correspondance with other triangles
        # (basically going around the triangle from smallest to largest side)
        triang_vrtx_uniq = np.array([arrangetriplet(sources, triplet)
                                     for triplet in triang_vrtx_uniq])

        return inv_uniq, triang_vrtx_uniq

    class MatchTransform:
        def __init__(self, ref_srcs, target_srcs):
            self.ref = ref_srcs
            self.target = target_srcs

        def fit(self, data):
            # numpy arrays require an explicit 'in' method
            def in_np_array(elem, arr):
                return np.any([np.all(elem == el) for el in arr])

            # Collect all matches, forget triangle info
            d1, d2, d3 = data.shape
            point_matches = data.reshape(d1 * d2, d3)
            m = []
            b = []
            for match_ind, amatch in enumerate(point_matches):
                # add here the matches that don't repeat
                if not in_np_array(amatch, point_matches[match_ind + 1:]):
                    ind_r, ind_t = amatch
                    x_r, y_r = self.ref[ind_r]
                    x_t, y_t = self.target[ind_t]
                    m.extend([[x_r, y_r, 1, 0], [y_r, -x_r, 0, 1]])
                    b.extend([x_t, y_t])
            m = np.array(m)
            b = np.array(b)
            sol, resid, rank, sv = np.linalg.lstsq(m, b.T)
            # lc,s is l (scaling) times cos,sin(alpha); alpha is the rot angle
            # ltx,y is l (scaling) times the translation in the x,y direction
            lc = sol.item(0)
            ls = sol.item(1)
            ltx = sol.item(2)
            lty = sol.item(3)
            approxm = np.array([[lc, ls, ltx], [-ls, lc, lty]])
            return approxm

        def get_error(self, data, approxm):
            error = []
            for amatch in data:
                max_err = 0.
                for ind_r, ind_t in amatch:
                    x = self.ref[ind_r]
                    y = self.target[ind_t]
                    y_fit = approxm.dot(np.append(x, 1))
                    max_err = max(max_err, np.linalg.norm(y - y_fit))
                error.append(max_err)
            return np.array(error)


def find_affine_transform(test_srcs, ref_srcs, max_pix_tol=2.,
                          min_matches_fraction=0.8, invariant_map=None):
    """
    TODO: Docstring
    """
    if len(test_srcs) < 3:
        raise Exception(
            "Test sources are less than the minimum value of points (3).")

    if invariant_map is None:
        inv_map = InvariantTriangleMapping()

    if len(ref_srcs) < 3:
        raise Exception(
            "Ref sources are less than the minimum value of points (3).")
    # generate_invariants should return a list of the invariant tuples for each
    # asterism and a corresponding list of the indices that make up the astrsm
    ref_invariants, ref_asterisms = \
        inv_map.generate_invariants(ref_srcs, nearest_neighbors=7)
    ref_invariant_tree = KDTree(ref_invariants)

    test_invariants, test_asterisms = \
        inv_map.generate_invariants(test_srcs, nearest_neighbors=5)
    test_invariant_tree = KDTree(test_invariants)

    # 0.03 is just an empirical number that returns about the same number of
    # matches than inputs
    matches_list = \
        test_invariant_tree.query_ball_tree(ref_invariant_tree, 0.03)

    matches = []
    # t1 is an asterism in test, t2 in ref
    for t1, t2_list in zip(test_asterisms, matches_list):
        for t2 in np.array(ref_asterisms)[t2_list]:
            matches.append(zip(t2, t1))
    matches = np.array(matches)

    inv_model = inv_map.MatchTransform(ref_srcs, test_srcs)
    n_invariants = len(matches)
    max_iter = n_invariants
    min_matches = min(10, int(n_invariants * min_matches_fraction))
    best_m = ransac.ransac(matches, inv_model, 1, max_iter, max_pix_tol,
                           min_matches)
    return best_m


def align_image(image, image_ref):
    """Return an aligned image that coincides pixel to pixel with image_ref.

    align_image accepts a numpy array (masked or not) and returns a realigned
    image interpolated to coincide with ref_image.
    Sometimes bled stars or bad pixels can confuse the alignment. If the image
    is a masked array, it will return an aligned masked array with the mask
    transformed as well.

    Return aligned_image"""

    from scipy import ndimage

    try:
        import sep  # noqa
    except ImportError:
        source_finder = find_sources
    else:
        source_finder = find_sources_with_sep

    test_srcs = source_finder(image)[:50]
    ref_sources = source_finder(image_ref)[:70]

    m = find_affine_transform(test_srcs, ref_srcs=ref_sources)

    # SciPy Affine transformation transform a (row,col) pixel according to pT+s
    # where p is in the _output_ image, T is the rotation and s the translation
    # offset, so some mathematics is required to put it into a suitable form
    # In particular, affine_transform() requires the inverse transformation
    # that registration returns but for (row, col) instead of (x,y)
    def inverse_transform(m):
        m_rot_inv = np.linalg.inv(m[:2, :2])
        m_offset_inv = -m_rot_inv.dot(m[:2, 2])
        m_inv = np.zeros(m.shape)
        m_inv[:2, :2] = m_rot_inv
        m_inv[:2, 2] = m_offset_inv
        if m.shape == (3, 3):
            m_inv[2, 2] = 1
        return m_inv

    m_inv = inverse_transform(m)
    # p will transform from (x,y) to (row, col)=(y,x)
    p = np.array([[0, 1], [1, 0]])
    mrcinv_rot = p.dot(m_inv[:2, :2]).dot(p)
    mrcinv_offset = p.dot(m_inv[:2, 2])

    aligned_image = \
        ndimage.interpolation.affine_transform(image_ref, mrcinv_rot,
                                               offset=mrcinv_offset,
                                               output_shape=image.shape)
    if isinstance(image_ref, np.ma.MaskedArray):
        aligned_image_mask = \
            ndimage.interpolation.affine_transform(image_ref.mask, mrcinv_rot,
                                                   offset=mrcinv_offset,
                                                   output_shape=image.shape)
        aligned_image = np.ma.array(aligned_image, mask=aligned_image_mask)
    return aligned_image


def find_sources(image):
    """Return sources (x, y) sorted by brightness.
    """
    from scipy import ndimage
    from skimage import exposure
    from astropy.stats import mad_std

    img1 = image.copy()
    m, s = np.median(image), mad_std(image)
    src_mask = image > m + 3.0 * s
    # set the background to the min value of the sources
    img1[~src_mask] = img1[src_mask].min()
    # this rescales (min,max) to (0,1)
    img1 = exposure.rescale_intensity(img1)
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

    if num_srcs < 10:
        print("WARNING: Only %d sources found." % (num_srcs))

    # Eliminate here all 1 pixel sources
    all_objects = [[ind + 1, aslice] for ind, aslice
                   in enumerate(ndimage.find_objects(srcs_labels))
                   if srcs_labels[aslice].shape != (1, 1)]
    lum = np.array([obj_params_with_offset(img1, srcs_labels, aslice, lab_idx)
                    for lab_idx, aslice in all_objects])

    lum = lum[lum[:, 0].argsort()[::-1]]  # sort by brightness descending order

    return lum[:, 1:]


def find_sources_with_sep(img):
    """Return sources (x, y) sorted by brightness. Use SEP package.
    """
    import sep
    image = img.astype('float32')
    bkg = sep.Background(image)
    thresh = 3. * bkg.globalrms
    sources = sep.extract(image - bkg.back(), thresh)
    sources.sort(order='flux')
    return np.array([[asrc['x'], asrc['y']] for asrc in sources[::-1]])
