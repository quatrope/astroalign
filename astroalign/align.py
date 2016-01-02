import numpy as np
from scipy.spatial import KDTree
from itertools import combinations
import os
from astropy.io import fits
import ransac
import shlex, subprocess

class InvariantTriangleMapping():
    
    def invariantFeat(self, sources, ind1, ind2, ind3):
        x1,x2,x3 = sources[[ind1,ind2,ind3]]
        sides = np.sort([np.linalg.norm(x1-x2),np.linalg.norm(x2-x3),np.linalg.norm(x1-x3)])
        return [sides[2]/sides[1],sides[1]/sides[0]]
    
    def generateInvariants(self, sources, nearest_neighbors = 5):
        #Helping function
        def arrangeTriplet(sources, vertex_indices):
            side1 = np.array([vertex_indices[0], vertex_indices[1]])
            side2 = np.array([vertex_indices[1], vertex_indices[2]])
            side3 = np.array([vertex_indices[0], vertex_indices[2]])

            sideLengths = [np.linalg.norm(sources[p1_ind]-sources[p2_ind]) for p1_ind, p2_ind in [side1, side2, side3]]
            lengths_arg = np.argsort(sideLengths)
    
            #Sides sorted from shortest to longest
            sides = np.array([side1, side2, side3])[lengths_arg]
        
            #now I order the points inside the side this way: [(x2,x0),(x0,x1),(x1,x2)]
            for i in range(-1,2):
                if sides[i][0] in sides[i+1]: 
                    #swap the points
                    sides[i] = sides[i][[1,0]]
            
            return sides[:,1]
        
        inv = []
        triang_vrtx = []
        coordTree = KDTree(sources)
        for asrc in sources:
            __, indx = coordTree.query(asrc, 5)
            all_asterism_triang = [list(acomb) for acomb in combinations(indx, 3)]
            inv.extend([self.invariantFeat(sources, *triplet) for triplet in all_asterism_triang])
            triang_vrtx.extend(all_asterism_triang)

        #Remove here many duplicate triangles for close-tight neighbors
        inv_uniq = np.array([elem for (pos, elem) in enumerate(inv) if elem not in inv[pos + 1:]])
        triang_vrtx_uniq = [triang_vrtx[pos] for (pos, elem) in enumerate(inv) if elem not in inv[pos + 1:]]
   
        #This will order the vertices in the triangle in a determined way to make a point to point correspondance with other triangles
        #(basically going around the triangle from smallest side to largest side)
        triang_vrtx_uniq = np.array([arrangeTriplet(sources, triplet) for triplet in triang_vrtx_uniq])

        return inv_uniq, triang_vrtx_uniq

    class matchTransform:
        def __init__(self, ref_srcs, target_srcs):
            self.ref = ref_srcs
            self.target = target_srcs
        
        def fit(self, data):
            #numpy arrays require an explicit 'in' method
            def in_np_array(elem, arr):
                return np.any([np.all(elem == el) for el in arr])
        
            #Collect all matches, forget triangle info
            d1, d2, d3 = data.shape
            point_matches = data.reshape(d1*d2,d3)
            A = []; b = [];
            for match_ind, amatch in enumerate(point_matches):
                #add here the matches that don't repeat
                if not in_np_array(amatch, point_matches[match_ind + 1:]):
                    ind_r, ind_t = amatch
                    x_r, y_r = self.ref[ind_r]
                    x_t, y_t = self.target[ind_t]
                    A.extend([[x_r, y_r, 1, 0],[y_r, -x_r, 0, 1]])
                    b.extend([x_t, y_t])
            A = np.array(A)
            b = np.array(b)
            sol, resid, rank, sv = np.linalg.lstsq(A,b.T)
            #lc,s is l (scaling) times cos,sin(alpha); alpha is the rotation angle
            #ltx,y is l (scaling) times the translation in the x,y direction
            lc = sol.item(0)
            ls = sol.item(1)
            ltx = sol.item(2)
            lty = sol.item(3)
            approxM = np.array([[lc, ls, ltx],[-ls, lc, lty]])
            return approxM
    
        def get_error(self, data, approxM):
            error = []
            for amatch in data:
                max_err = 0.
                for ind_r, ind_t in amatch:
                    x = self.ref[ind_r]
                    y = self.target[ind_t]
                    y_fit = approxM.dot(np.append(x,1))
                    max_err = max(max_err, np.linalg.norm(y - y_fit))
                error.append(max_err)
            return np.array(error)


def findAffineTransform(test_srcs, ref_srcs, max_pix_tol = 2., min_matches_fraction = 0.8, invariantMap=None):
    """
    TODO: Docstring
    """
    if len(test_srcs) < 3:
        raise Exception("Test sources has less than the minimum value of points (3).")
    
    if invariantMap is None:
        invMap = InvariantTriangleMapping()
    
    if len(ref_srcs) < 3:
        raise Exception("Test sources has less than the minimum value of points (3).")
    #generateInvariants should return a list of the invariant tuples for each asterism and 
    # a corresponding list of the indices that make up the asterism 
    ref_invariants, ref_asterisms = invMap.generateInvariants(ref_srcs, nearest_neighbors = 7)
    ref_invariant_tree = KDTree(ref_invariants)

    test_invariants, test_asterisms = invMap.generateInvariants(test_srcs, nearest_neighbors = 5)
    test_invariant_tree = KDTree(test_invariants)

    #0.03 is just an empirical number that returns about the same number of matches than inputs
    matches_list = test_invariant_tree.query_ball_tree(ref_invariant_tree, 0.03)

    matches = []
    #t1 is an asterism in test, t2 in ref
    for t1, t2_list in zip(test_asterisms, matches_list):
        for t2 in np.array(ref_asterisms)[t2_list]:
            matches.append(zip(t2, t1))
    matches = np.array(matches)
    
    invModel = invMap.matchTransform(ref_srcs, test_srcs)
    nInvariants = len(matches)
    max_iter = nInvariants
    min_matches = min(10, int(nInvariants * min_matches_fraction))
    bestM = ransac.ransac(matches, invModel, 1, max_iter, max_pix_tol, min_matches)
    return bestM

 
def alignImage(image, image_ref):
    """Return an aligned image that coincides pixel to pixel with image_ref.

    alignImage accepts a numpy array (masked or not) and returns a realigned image 
    interpolated to coincide with ref_image.
    Sometimes bled stars or bad pixels can confuse the alignment. If the image is a 
    masked array, it will return an aligned masked array with the mask transformed 
    as well.

    Return aligned_image"""
    
    from scipy import ndimage

    try:
        import sep
    except ImportError:
        sourceFinder = findSources
    else:
        sourceFinder = findSourcesWithSEP

    test_srcs = sourceFinder(image)[:50]
    ref_sources = sourceFinder(image_ref)[:70]

    M = findAffineTransform(test_srcs, ref_srcs = ref_sources)

    #SciPy Affine transformation transform a (row,col) pixel according to pT+s where p is in the _output_ image,
    #T is the rotation and s the translation offset, so some mathematics is required to put it into a suitable form
    #In particular, affine_transform() requires the inverse transformation that registration returns but for (row, col) instead of (x,y)
    def inverseTransform(M):
        M_rot_inv = np.linalg.inv(M[:2,:2])
        M_offset_inv = -M_rot_inv.dot(M[:2,2])
        Minv = np.zeros(M.shape)
        Minv[:2,:2] = M_rot_inv
        Minv[:2,2] = M_offset_inv
        if M.shape == (3,3): Minv[2,2] = 1
        return Minv

    Minv = inverseTransform(M)
    #P will transform from (x,y) to (row, col)=(y,x)
    P = np.array([[0,1],[1,0]])
    Mrcinv_rot = P.dot(Minv[:2,:2]).dot(P)
    Mrcinv_offset = P.dot(Minv[:2,2])

    aligned_image = ndimage.interpolation.affine_transform(image_ref, Mrcinv_rot, offset=Mrcinv_offset, output_shape=image.shape)
    if isinstance(image_ref, np.ma.MaskedArray): 
        aligned_image_mask = ndimage.interpolation.affine_transform(image_ref.mask, Mrcinv_rot, offset=Mrcinv_offset, output_shape=image_in.shape)
        aligned_image = np.ma.array(aligned_image, mask=aligned_image_mask)
    return aligned_image


def bkgNoiseSigma(dataImg, noiseLvl = 3.0):
    """Return background mean and std. dev. of sky background.

    Calculate the background (sky) mean value and a measure of its standard deviation.
    Background is considered anything below 'noiseLvl' sigmas.
    goodPixMask is a mask containing the good pixels that should be considered.
    Return mean, std_dev
    """
    m = dataImg.mean()
    s = dataImg.std()

    prevSgm = 2*s #This will make the first while condition true
    tol = 1E-2
    while abs(prevSgm - s)/s > tol:
        prevSgm = s
        bkgMask = np.logical_and(dataImg < m + noiseLvl*s, dataImg > m - noiseLvl*s)
        #The 1.*m hack is to force m to be a float, instead of possibly a masked ndarray
        m, s = 1.*dataImg[bkgMask].mean(), dataImg[bkgMask].std()

    return m, s


def findSources(image):
    """Return sources (x, y) sorted by brightness.
    """
    from scipy import ndimage
    from skimage import exposure
    from astropy.stats import mad_std

    img1 = image.copy()
    m, s = np.median(image), mad_std(image)
    src_mask = image > m + 3.0*s
    #set the background to the min value of the sources
    img1[~src_mask] = img1[src_mask].min()
    #this rescales (min,max) to (0,1)
    img1 = exposure.rescale_intensity(img1)
    img1[~src_mask] = 0.

    def obj_params_with_offset(img, labels, aslice, label_idx):
        y_offset = aslice[0].start
        x_offset = aslice[1].start
        thumb = img[aslice]
        lb = labels[aslice]
        yc, xc = ndimage.center_of_mass(thumb, labels=lb, index=label_idx)
        br = thumb[lb == label_idx].sum() #the intensity of the source
        return [br, xc + x_offset, yc + y_offset]

    srcs_labels, num_srcs = ndimage.label(img1)

    if num_srcs < 10:
        print("WARNING: Only %d sources found." % (num_srcs))

    #Eliminate here all 1 pixel sources
    all_objects = [[ind + 1, aslice] for ind, aslice in enumerate(ndimage.find_objects(srcs_labels))
                                                if srcs_labels[aslice].shape != (1,1)]
    lum = np.array([obj_params_with_offset(img1, srcs_labels, aslice, lab_idx)
                for lab_idx, aslice in all_objects])

    lum = lum[lum[:,0].argsort()[::-1]]  #sort by brightness highest to smallest

    return lum[:,1:]


def findSourcesWithSEP(img):
    """Return sources (x, y) sorted by brightness. Use SEP package.
    """
    import sep
    image = img.astype('float32')
    bkg = sep.Background(image)
    thresh = 3.*bkg.globalrms
    sources = sep.extract(image - bkg.back(), thresh)
    sources.sort(order='flux')
    return np.array([[asrc['x'],asrc['y']] for asrc in sources[::-1]])




