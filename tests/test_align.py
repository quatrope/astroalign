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
        a, m = 0.8, 3. * self.image_ref.std()
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
        scale = 1.5
        alpha = np.pi / 8.
        mm = scale * np.array([[np.cos(alpha), -np.sin(alpha)],
                               [np.sin(alpha), np.cos(alpha)]])
        tx = 2.0
        ty = 1.0
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

        registered_img = aa.register(source=self.image,
                                     target=self.image_ref)

        # Test that image returned is not masked
        self.assertIs(type(registered_img), np.ndarray)
        fraction = compare_image(registered_img)
        self.assertGreater(fraction, 0.85)

        # Test masked arrays
        # Make some masks...
        mask = np.zeros(self.image.shape, dtype='bool')
        mask[self.h // 10:self.h // 10 + 10, :] = True
        mask_ref = np.zeros(self.image_ref.shape, dtype='bool')
        mask_ref[:, self.w // 10:self.w // 10 + 10] = True
        image_masked = np.ma.array(self.image, mask=mask)
        image_ref_masked = np.ma.array(self.image_ref, mask=mask_ref)

        def testalignment(source, target):
            registered_img = aa.register(source=source, target=target)
            self.assertIs(type(registered_img), type(source))
            fraction = compare_image(registered_img)
            self.assertGreater(fraction, 0.85)

        # Test it works with masked image:
        testalignment(image_masked, self.image_ref)

        # Test it works with masked ref:
        testalignment(self.image, image_ref_masked)

        # Test it works with both masked image and masked ref:
        testalignment(image_masked, image_ref_masked)

        # Test it works when given a masked array with no mask set
        testalignment(np.ma.array(self.image), self.image_ref)

        # Test it works when given a reference masked array with no mask set
        testalignment(self.image, np.ma.array(self.image_ref))

        # Test if it works when both images are masked, but with no mask set
        testalignment(np.ma.array(self.image), np.ma.array(self.image_ref))

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


if __name__ == "__main__":
    unittest.main()
