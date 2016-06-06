.. _methods:

Other Methods
=============


find_affine_transform

(test_srcs, ref_srcs, max_pix_tol=2.0, min_matches_fraction=0.8, invariant_map=None)


Return the 2 by 3 affine transformation M that maps pixel coordinates (indices) from the reference image r = (x, y) into the test image t = (x, y). t = M * r

astroalign.find_sources(image)
Return sources (x, y) sorted by brightness.

astroalign.find_sources_with_sep
