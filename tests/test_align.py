import unittest
import numpy as np
import astroalign
# import scipy


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

        self.star_ref_pos = np.array(zip(self.ref_cols, self.ref_rows))
        self.star_new_pos = np.array(zip(self.new_cols, self.new_rows))

    def test_find_affine_transform(self):

        star_ref_pos_b = self.star_ref_pos[np.argsort(self.ref_flux)]
        star_new_pos_b = self.star_new_pos[np.argsort(self.new_flux)]

        m = astroalign.find_affine_transform(star_new_pos_b[50::-1],
                                             star_ref_pos_b[70::-1])
        alpha = self.rot_angle
        xoff_corrected = ((1 - np.cos(alpha)) * self.w / 2 + np.sin(alpha) *
                          self.h / 2 + self.x_offset)
        yoff_corrected = (-np.sin(alpha) * self.w / 2 + (1 - np.cos(alpha)) *
                          self.h / 2 + self.y_offset)
        mtrue = np.array([[np.cos(alpha), -np.sin(alpha), xoff_corrected],
                          [np.sin(alpha), np.cos(alpha), yoff_corrected]])
        # Pass the test if less than 2% relative error in result
        rel_error = np.linalg.norm(m - mtrue, 1) / np.linalg.norm(mtrue, 1)
        self.assertLess(rel_error, 2E-2)

    def test_align_image(self):
        def compare_image(the_image):
            """Return the fraction of sources found in the original image"""
            # pixel comparison is not good, doesn't work. Compare catalogs.
            if isinstance(the_image, np.ma.MaskedArray):
                full_algn = the_image.filled(fill_value=np.median(the_image))\
                    .astype('float32')
            else:
                full_algn = the_image.astype('float32')
            full_algn[the_image == 0] = np.median(the_image)
            import sep
            bkg = sep.Background(full_algn)
            thresh = 3.0 * bkg.globalrms
            allobjs = sep.extract(full_algn - bkg.back(), thresh)
            allxy = np.array([[obj['x'], obj['y']] for obj in allobjs])

            from scipy.spatial import KDTree
            ref_coordtree = KDTree(self.star_new_pos)

            # Compare here srcs list with self.star_ref_pos
            num_sources = 0
            for asrc in allxy:
                found_source = ref_coordtree.query_ball_point(asrc, 3)
                if found_source:
                    num_sources += 1
            fraction_found = float(num_sources) / float(len(allxy))
            return fraction_found

        image_aligned = astroalign.align_image(self.image, self.image_ref)
        # Test that image returned is not masked
        self.assertIs(type(image_aligned), np.ndarray)
        fraction = compare_image(image_aligned)
        self.assertGreater(fraction, 0.85)

        # Test masked arrays
        # Make some masks...
        mask = np.zeros(self.image.shape, dtype='bool')
        mask[self.h / 10:self.h / 10 + 10, :] = True
        mask_ref = np.zeros(self.image_ref.shape, dtype='bool')
        mask_ref[:, self.w / 10:self.w / 10 + 10] = True
        image_masked = np.ma.array(self.image, mask=mask)
        image_ref_masked = np.ma.array(self.image_ref, mask=mask_ref)

        # Test it works with masked image:
        image_aligned = astroalign.align_image(image_masked, self.image_ref)
        self.assertIs(type(image_aligned), np.ndarray)
        fraction = compare_image(image_aligned)
        self.assertGreater(fraction, 0.85)

        # Test it works with masked ref:
        image_aligned = astroalign.align_image(self.image, image_ref_masked)
        self.assertIs(type(image_aligned), np.ma.MaskedArray)
        fraction = compare_image(image_aligned)
        self.assertGreater(fraction, 0.85)

        # Test it works with both masked image and masked ref:
        image_aligned = astroalign.align_image(image_masked, image_ref_masked)
        self.assertIs(type(image_aligned), np.ma.MaskedArray)
        fraction = compare_image(image_aligned)
        self.assertGreater(fraction, 0.85)

        # Test it works when given a masked array with no mask set
        image_aligned = astroalign.align_image(np.ma.array(self.image),
                                               self.image_ref)
        self.assertIs(type(image_aligned), np.ndarray)
        fraction = compare_image(image_aligned)
        self.assertGreater(fraction, 0.85)

        # Test it works when given a reference masked array with no mask set
        image_aligned = astroalign.align_image(self.image,
                                               np.ma.array(self.image_ref))
        self.assertIs(type(image_aligned), np.ma.MaskedArray)
        fraction = compare_image(image_aligned)
        self.assertGreater(fraction, 0.85)

    def test_find_sources(self):
        srcs = astroalign.find_sources(self.image_ref)

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


#class TestRANSAC(unittest.TestCase):
#    def test_ransac(self):
#
#        class LinearLeastSquaresModel:
#            """linear system solved using linear least squares
#
#            This class serves as an example that fulfills the model interface
#            needed by the ransac() function.
#
#            """
#            def __init__(self, input_columns, output_columns, debug=False):
#                self.input_columns = input_columns
#                self.output_columns = output_columns
#                self.debug = debug
#
#            def fit(self, data):
#                a = np.vstack([data[:, i] for i in self.input_columns]).T
#                b = np.vstack([data[:, i] for i in self.output_columns]).T
#                x, resids, rank, s = scipy.linalg.lstsq(a, b)
#                return x
#
#            def get_error(self, data, model):
#                a = np.vstack([data[:, i] for i in self.input_columns]).T
#                b = np.vstack([data[:, i] for i in self.output_columns]).T
#                b_fit = scipy.dot(a, model)
#                # sum squared error per row
#                err_per_point = np.sum((b - b_fit) ** 2, axis=1)
#                return err_per_point
#
#        # generate perfect input data
#        n_samples = 500
#        n_inputs = 1
#        n_outputs = 1
#        a_exact = 20 * np.random.random((n_samples, n_inputs))
#        # the model
#        perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))
#        b_exact = scipy.dot(a_exact, perfect_fit)
#        assert b_exact.shape == (n_samples, n_outputs)
#
#        # add a little gaussian noise (linear least squares alone should handle
#        # this well)
#        a_noisy = a_exact + np.random.normal(size=a_exact.shape)
#        b_noisy = b_exact + np.random.normal(size=b_exact.shape)
#
#        # add some outliers
#        n_outliers = 100
#        all_idxs = np.arange(a_noisy.shape[0])
#        np.random.shuffle(all_idxs)
#        outlier_idxs = all_idxs[:n_outliers]
#        non_outlier_idxs = all_idxs[n_outliers:]
#        a_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
#        b_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers,
#                                                            n_outputs))
#
#        # setup model
#        all_data = np.hstack((a_noisy, b_noisy))
#        input_columns = range(n_inputs)  # the first columns of the array
#        # the last columns of the array
#        output_columns = [n_inputs + i for i in range(n_outputs)]
#        debug = False
#        model = LinearLeastSquaresModel(input_columns, output_columns,
#                                        debug=debug)
#
#        linear_fit, resids, rank, s = \
#            scipy.linalg.lstsq(all_data[:, input_columns],
#                               all_data[:, output_columns])
#
#        # run RANSAC algorithm
#        ransac_fit, ransac_data = astroalign.ransac(all_data, model, 50, 1000, 7e3,
#                                               300, debug=debug,
#                                               return_all=True)
#        # if 1:
#        #    import pylab
#        #
#        #    sort_idxs = np.argsort(a_exact[:, 0])
#        #    a_col0_sorted = a_exact[sort_idxs]  # maintain as rank-2 array
#        #
#        #    if 1:
#        #        pylab.plot(a_noisy[:, 0], b_noisy[:, 0], 'k.', label='data')
#        #        pylab.plot(a_noisy[ransac_data['inliers'], 0],
#        #                   b_noisy[ransac_data['inliers'], 0], 'bx',
#        #                   label='RANSAC data')
#        #    else:
#        #        pylab.plot(a_noisy[non_outlier_idxs, 0],
#        #                   b_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
#        #        pylab.plot(a_noisy[outlier_idxs, 0],
#        #                   b_noisy[outlier_idxs, 0], 'r.', label='outlier data')
#        #    pylab.plot(a_col0_sorted[:, 0],
#        #               np.dot(a_col0_sorted, ransac_fit)[:, 0],
#        #               label='RANSAC fit')
#        #    pylab.plot(a_col0_sorted[:, 0],
#        #               np.dot(a_col0_sorted, perfect_fit)[:, 0],
#        #               label='exact system')
#        #    pylab.plot(a_col0_sorted[:, 0],
#        #               np.dot(a_col0_sorted, linear_fit)[:, 0],
#        #               label='linear fit')
#        #    pylab.legend()
#        #    pylab.show()
#
#    def tearDown(self):
#        self.image = None
#        self.image_ref = None
#

if __name__ == "__main__":
    unittest.main()
