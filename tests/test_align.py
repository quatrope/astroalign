import unittest
import numpy as np
import astroalign as aa


def gauss(shape=(11, 11), center=None, sx=2, sy=2):
    "Returns a Gaussian of given shape, normalized to 1."
    h, w = shape
    if center is None:
        center = ((h - 1) / 2., (w - 1) / 2.)
    x0, y0 = center
    x, y = np.meshgrid(range(w), range(h))
    krnl = np.exp(-0.5 * ((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2))
    krnl /= krnl.sum()
    return krnl


class TestAlign(unittest.TestCase):
    def setUp(self):
        from scipy import signal
        self.h = 512  # image height
        self.w = 512  # image width
        kh = 10  # kernel height
        kw = 10  # kernel width
        noise_level = 500  # counts
        num_stars = 1500
        psf = gauss(shape=(21, 21), sx=1.5, sy=1.5)

        # Transformation parameters
        self.x_offset = 10
        self.y_offset = -20
        self.rot_angle = 50. * np.pi / 180.

        big_r = (0.5 * np.sqrt(self.h ** 2 + self.w ** 2)
                 + max(abs(self.x_offset), abs(self.y_offset)))

        self.image_ref = np.random.poisson(noise_level,
                                           size=(self.h + kh, self.w + kw)
                                           ).astype('float64')
        self.image = np.random.poisson(noise_level,
                                       size=(self.h + kh, self.w + kw)
                                       ).astype('float64')

        # x and y of stars in the ref frame (int's)
        self.star_refx = np.random.randint(low=int(-big_r) + self.w / 2,
                                           high=int(big_r) + self.w / 2,
                                           size=(num_stars,))
        self.star_refy = np.random.randint(low=int(-big_r) + self.h / 2,
                                           high=int(big_r) + self.h / 2,
                                           size=(num_stars,))
        # Fluxes of stars
        a, m = 0.8, 3. * self.image_ref.std()  # This are Pareto dist coeff's
        self.star_f = (1. + np.random.pareto(a, num_stars)) * m
        # self.star_f = 1.*np.random.exponential(1600., size=(num_stars,))

        # inframe will contain the stars in the reference image
        inframe = []
        ymax, xmax = self.image_ref.shape
        for x, y, f in zip(self.star_refx, self.star_refy, self.star_f):
            if x > 0 and x < xmax and y > 0 and y < ymax:
                inframe.append((int(x), int(y), f))
        self.ref_cols, self.ref_rows, self.ref_flux = \
            np.array(inframe).astype(int).T

        self.image_ref[self.ref_rows, self.ref_cols] += self.ref_flux
        self.image_ref = signal.convolve2d(self.image_ref, psf, mode='same')
        self.image_ref = self.image_ref[kh // 2: -kh // 2, kw // 2: -kw // 2]
        # Adjust here the positions of rows and cols after cropping image
        self.ref_cols -= kw // 2
        self.ref_rows -= kh // 2

        newx, newy = [], []
        for x, y in zip(self.star_refx, self.star_refy):
            x -= self.w / 2
            y -= self.h / 2
            xp = (x * np.cos(self.rot_angle) - y * np.sin(self.rot_angle)
                  + self.x_offset)
            yp = (x * np.sin(self.rot_angle) + y * np.cos(self.rot_angle)
                  + self.y_offset)
            xp += self.w / 2
            yp += self.h / 2
            newx.append(xp)
            newy.append(yp)
        # x and y of stars in the new frame (float's)
        self.star_newx = np.array(newx)
        self.star_newy = np.array(newy)

        inframe = []
        ymax, xmax = self.image.shape
        for x, y, f in zip(self.star_newx, self.star_newy, self.star_f):
            if (x > 0 and x < xmax and y > 0 and y < xmax):
                inframe.append((int(x), int(y), f))
        self.new_cols, self.new_rows, self.new_flux = \
            np.array(inframe).astype(int).T

        self.image[self.new_rows, self.new_cols] += self.new_flux
        self.image = signal.convolve2d(self.image, psf, mode='same')
        self.image = self.image[kh // 2: -kh // 2, kw // 2: -kw // 2]
        # Adjust here the positions of rows and cols after cropping image
        self.new_cols -= kw // 2
        self.new_rows -= kh // 2

        self.star_ref_pos = np.array(list(zip(self.ref_cols, self.ref_rows)))
        self.star_new_pos = np.array(list(zip(self.new_cols, self.new_rows)))

    def test_find_transform_givensources(self):
        from skimage.transform import estimate_transform, matrix_transform
        source = np.array([[1.4, 2.2], [5.3, 1.0], [3.7, 1.5],
                           [10.1, 9.6], [1.3, 10.2], [7.1, 2.0]])
        nsrc = source.shape[0]
        scale = 1.5  # scaling parameter
        alpha = np.pi / 8.  # rotation angle
        mm = scale * np.array([[np.cos(alpha), -np.sin(alpha)],
                               [np.sin(alpha), np.cos(alpha)]])
        tx, ty = 2.0, 1.0  # translation parameters
        transl = np.array([nsrc * [tx], nsrc * [ty]])
        dest = (mm.dot(source.T) + transl).T
        t_true = estimate_transform('similarity', source, dest)

        # disorder dest points so they don't match the order of source
        np.random.shuffle(dest)

        t, (src_pts, dst_pts) = aa.find_transform(source, dest)
        self.assertLess(t_true.scale - t.scale, 1E-10)
        self.assertLess(t_true.rotation - t.rotation, 1E-10)
        self.assertLess(np.linalg.norm(t_true.translation - t.translation),
                        1E-10)
        self.assertEqual(src_pts.shape[0], dst_pts.shape[0])
        self.assertEqual(src_pts.shape[1], 2)
        self.assertEqual(dst_pts.shape[1], 2)
        dst_pts_test = matrix_transform(src_pts, t.params)
        self.assertLess(np.linalg.norm(dst_pts_test - dst_pts), 1E-10)

    def test_register(self):
        def compare_image(the_image):
            """Return the fraction of sources found in the reference image"""
            # pixel comparison is not good, doesn't work. Compare catalogs.
            if isinstance(the_image, np.ma.MaskedArray):
                full_algn = the_image.filled(fill_value=np.median(the_image))\
                    .astype('float32')
            else:
                full_algn = the_image.astype('float32')
            # full_algn[the_image == 0] = np.median(the_image)
            import sep
            bkg = sep.Background(full_algn)
            thresh = 3.0 * bkg.globalrms
            allobjs = sep.extract(full_algn - bkg.back(), thresh)
            allxy = np.array([[obj['x'], obj['y']] for obj in allobjs])

            from scipy.spatial import KDTree
            ref_coordtree = KDTree(self.star_ref_pos)

            # Compare here srcs list with self.star_ref_pos
            num_sources = 0
            for asrc in allxy:
                found_source = ref_coordtree.query_ball_point(asrc, 3)
                if found_source:
                    num_sources += 1
            fraction_found = float(num_sources) / float(len(allxy))
            return fraction_found

        registered_img, footp = aa.register(source=self.image,
                                            target=self.image_ref)
        self.assertIs(type(registered_img), np.ndarray)
        self.assertIs(type(footp), np.ndarray)
        self.assertIs(footp.dtype, np.dtype('bool'))
        fraction = compare_image(registered_img)
        self.assertGreater(fraction, 0.85)

    def test_register_nddata(self):
        from astropy.nddata import NDData
        from skimage.transform import SimilarityTransform
        transf = SimilarityTransform(rotation=np.pi/2., translation=(1, 0))
        nparr = np.array([[0., 1.], [2., 3.]])
        mask = [[True, False], [False, False]]

        nd = NDData(nparr, mask=mask)
        registered_img, footp = aa.apply_transform(
            transf, nd, nd, propagate_mask=True)
        err = np.linalg.norm(registered_img - np.array([[2., 0.], [3., 1.]]))
        self.assertLess(err, 1E-6)
        err_mask = (footp == np.array([[False, True], [False, False]]))
        self.assertTrue(all(err_mask.flatten()))

        # Test now if there is no assigned mask during creation
        nd = NDData(nparr)
        registered_img, footp = aa.apply_transform(
            transf, nd, nd, propagate_mask=True)
        err = np.linalg.norm(registered_img - np.array([[2., 0.], [3., 1.]]))
        self.assertLess(err, 1E-6)
        err_mask = (footp == np.array([[False, False], [False, False]]))
        self.assertTrue(all(err_mask.flatten()))

    def test_register_ccddata(self):
        from ccdproc import CCDData
        from skimage.transform import SimilarityTransform
        transf = SimilarityTransform(rotation=np.pi/2., translation=(1, 0))
        nparr = np.array([[0., 1.], [2., 3.]])
        mask = [[True, False], [False, False]]

        cd = CCDData(nparr, mask=mask, unit='adu')
        registered_img, footp = aa.apply_transform(
            transf, cd, cd, propagate_mask=True)
        err = np.linalg.norm(registered_img - np.array([[2., 0.], [3., 1.]]))
        self.assertLess(err, 1E-6)
        err_mask = (footp == np.array([[False, True], [False, False]]))
        self.assertTrue(all(err_mask.flatten()))

        cd = CCDData(nparr, unit='adu')
        registered_img, footp = aa.apply_transform(
            transf, cd, cd, propagate_mask=True)
        err = np.linalg.norm(registered_img - np.array([[2., 0.], [3., 1.]]))
        self.assertLess(err, 1E-6)
        err_mask = (footp == np.array([[False, False], [False, False]]))
        self.assertTrue(all(err_mask.flatten()))

    def test_register_npma(self):
        from skimage.transform import SimilarityTransform
        transf = SimilarityTransform(rotation=np.pi/2., translation=(1, 0))
        nparr = np.array([[0., 1.], [2., 3.]])
        mask = [[True, False], [False, False]]

        ma = np.ma.array(nparr, mask=mask)
        registered_img, footp = aa.apply_transform(
            transf, ma, ma, propagate_mask=True)
        err = np.linalg.norm(registered_img - np.array([[2., 0.], [3., 1.]]))
        self.assertLess(err, 1E-6)
        err_mask = (footp == np.array([[False, True], [False, False]]))
        self.assertTrue(all(err_mask.flatten()))

        ma = np.ma.array(nparr)
        registered_img, footp = aa.apply_transform(
            transf, ma, ma, propagate_mask=True)
        err = np.linalg.norm(registered_img - np.array([[2., 0.], [3., 1.]]))
        self.assertLess(err, 1E-6)
        err_mask = (footp == np.array([[False, False], [False, False]]))
        self.assertTrue(all(err_mask.flatten()))


    def test_fill_value(self):
        registered_img, footp = aa.register(source=self.image,
                                            target=self.image_ref,
                                            fill_value=-9999.99,
                                            )
        self.assertTrue(all(registered_img[footp] == -9999.99))
        self.assertTrue(all(registered_img[~footp] != -9999.99))

    def test_find_sources(self):
        srcs = aa._find_sources(self.image_ref)

        from scipy.spatial import KDTree
        ref_coordtree = KDTree(self.star_ref_pos)

        # Compare here srcs list with self.star_ref_pos
        num_sources = 0
        for asrc in srcs:
            found_source = ref_coordtree.query_ball_point(asrc, 3)
            if found_source:
                num_sources += 1
        fraction_found = float(num_sources) / float(len(srcs))
        self.assertGreater(fraction_found, 0.85)

    def test_dtypes(self):
        # aa.register(self.image.astype('float16'), self.image_ref)
        aa.register(self.image.astype('float32'), self.image_ref)
        aa.register(self.image.astype('float64'), self.image_ref)
        aa.register(self.image.astype('int32'), self.image_ref)
        aa.register(self.image.astype('int64'), self.image_ref)

    def test_consistent_invert(self):
        t, __ = aa.find_transform(self.image, self.image_ref)
        tinv, __ = aa.find_transform(self.image_ref, self.image)
        rpoint = np.random.rand(3) * self.h
        rpoint[2] = 1.0
        rtransf = tinv.params.dot(t.params.dot(rpoint))
        err = np.linalg.norm(rpoint - rtransf) / np.linalg.norm(rpoint)
        self.assertLess(err, 1E-2)

if __name__ == "__main__":
    unittest.main()
