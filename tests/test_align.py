# MIT License

# Copyright (c) 2016-2019 Martin Beroiz

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

import unittest
import numpy as np
import astroalign as aa
from astropy.nddata import NDData
from ccdproc import CCDData
from skimage.transform import SimilarityTransform
from skimage.transform import estimate_transform, matrix_transform
import tempfile
from PIL import Image


def gauss(shape=(11, 11), center=None, sx=2, sy=2):
    "Returns a Gaussian of given shape, normalized to 1."
    h, w = shape
    if center is None:
        center = ((h - 1) / 2.0, (w - 1) / 2.0)
    x0, y0 = center
    x, y = np.meshgrid(range(w), range(h))
    krnl = np.exp(-0.5 * ((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2))
    krnl /= krnl.sum()
    return krnl


def simulate_image_pair(
    shape=(512, 512),
    kshape=(10, 10),
    noise_level=500,
    gshape=(21, 21),
    gsigma=1.5,
    translation=(10, -20),
    rot_angle_deg=50.0,
    num_stars=1500,
    star_refx=None,
    star_refy=None,
    star_flux=None,
):
    from scipy import signal

    h, w = shape  # image height and width
    kh, kw = kshape  # kernel height and width
    psf = gauss(shape=gshape, sx=gsigma, sy=gsigma)
    # Transformation parameters
    x_offset, y_offset = translation

    rot_angle = rot_angle_deg * np.pi / 180.0

    big_r = 0.5 * np.sqrt(h ** 2 + w ** 2) + max(abs(x_offset), abs(y_offset))

    image_ref = np.random.poisson(noise_level, size=(h + kh, w + kw)).astype(
        "float64"
    )
    image = np.random.poisson(noise_level, size=(h + kh, w + kw)).astype(
        "float64"
    )

    # x and y of stars in the ref frame (int's)
    if star_refx is None:
        star_refx = np.random.randint(
            low=int(-big_r) + w / 2, high=int(big_r) + w / 2, size=(num_stars,)
        )
    if star_refy is None:
        star_refy = np.random.randint(
            low=int(-big_r) + h / 2, high=int(big_r) + h / 2, size=(num_stars,)
        )
    # Fluxes of stars
    if star_flux is None:
        a, m = 0.8, 3.0 * image_ref.std()  # This are Pareto dist coeff's
        star_flux = (1.0 + np.random.pareto(a, num_stars)) * m

    # inframe will contain the stars in the reference image
    inframe = []
    ymax, xmax = image_ref.shape
    for x, y, f in zip(star_refx, star_refy, star_flux):
        if x > 0 and x < xmax and y > 0 and y < ymax:
            inframe.append((int(x), int(y), f))
    ref_cols, ref_rows, ref_flux = np.array(inframe).astype(int).T

    image_ref[ref_rows, ref_cols] += ref_flux
    image_ref = signal.convolve2d(image_ref, psf, mode="same")
    image_ref = image_ref[kh // 2 : -kh // 2, kw // 2 : -kw // 2]
    # Adjust here the positions of rows and cols after cropping image
    ref_cols -= kw // 2
    ref_rows -= kh // 2

    newx, newy = [], []
    for x, y in zip(star_refx, star_refy):
        x -= w / 2
        y -= h / 2
        xp = x * np.cos(rot_angle) - y * np.sin(rot_angle) + x_offset
        yp = x * np.sin(rot_angle) + y * np.cos(rot_angle) + y_offset
        xp += w / 2
        yp += h / 2
        newx.append(xp)
        newy.append(yp)
    # x and y of stars in the new frame (float's)
    star_newx = np.array(newx)
    star_newy = np.array(newy)

    inframe = []
    ymax, xmax = image.shape
    for x, y, f in zip(star_newx, star_newy, star_flux):
        if x > 0 and x < xmax and y > 0 and y < xmax:
            inframe.append((int(x), int(y), f))
    new_cols, new_rows, new_flux = np.array(inframe).astype(int).T

    image[new_rows, new_cols] += new_flux
    image = signal.convolve2d(image, psf, mode="same")
    image = image[kh // 2 : -kh // 2, kw // 2 : -kw // 2]
    # Adjust here the positions of rows and cols after cropping image
    new_cols -= kw // 2
    new_rows -= kh // 2

    star_ref_pos = np.array(list(zip(ref_cols, ref_rows)))
    star_new_pos = np.array(list(zip(new_cols, new_rows)))

    return image, image_ref, star_ref_pos, star_new_pos


def simulate_image_single(
    shape=(512, 512),
    kshape=(10, 10),
    noise_level=500,
    gshape=(21, 21),
    gsigma=1.5,
    num_stars=1500,
    star_refx=None,
    star_refy=None,
    star_flux=None,
):
    from scipy import signal

    h, w = shape  # image height and width
    kh, kw = kshape  # kernel height and width
    psf = gauss(shape=gshape, sx=gsigma, sy=gsigma)

    big_r = 0.5 * np.sqrt(h ** 2 + w ** 2)

    # Sky background
    image = np.random.poisson(noise_level, size=(h + kh, w + kw)).astype(
        "float64"
    )

    # x and y of stars in the ref frame (int's)
    if star_refx is None:
        star_refx = np.random.randint(
            low=int(-big_r) + w / 2, high=int(big_r) + w / 2, size=(num_stars,)
        )
    if star_refy is None:
        star_refy = np.random.randint(
            low=int(-big_r) + h / 2, high=int(big_r) + h / 2, size=(num_stars,)
        )
    # Fluxes of stars
    if star_flux is None:
        a, m = 0.8, 3.0 * image.std()  # This are Pareto dist coeff's
        star_flux = (1.0 + np.random.pareto(a, num_stars)) * m

    # inframe will contain the stars in the reference image
    inframe = []
    ymax, xmax = image.shape
    for x, y, f in zip(star_refx, star_refy, star_flux):
        if x > 0 and x < xmax and y > 0 and y < ymax:
            inframe.append((int(x), int(y), f))
    cols, rows, flux = np.array(inframe).astype(int).T

    image[rows, cols] += flux
    image = signal.convolve2d(image, psf, mode="same")
    image = image[kh // 2 : -kh // 2, kw // 2 : -kw // 2]
    # Adjust here the positions of rows and cols after cropping image
    cols -= kw // 2
    rows -= kh // 2

    star_pos = np.array(list(zip(cols, rows)))

    return image, star_pos


class TestAlign(unittest.TestCase):
    def setUp(self):
        self.h = 512  # image height
        self.w = 512  # image width
        self.x_offset = 10
        self.y_offset = -20
        self.rot_angle = 50.0 * np.pi / 180.0
        (
            self.image,
            self.image_ref,
            self.star_ref_pos,
            self.star_new_pos,
        ) = simulate_image_pair(
            shape=(self.h, self.w),
            translation=(self.x_offset, self.y_offset),
            rot_angle_deg=50.0,
        )
        self.image_mask = np.zeros((self.h, self.w), dtype="bool")
        self.image_ref_mask = np.zeros((self.h, self.w), dtype="bool")
        self.image_mask[10:30, 70:90] = True
        self.image_ref_mask[10:30, 20:50] = True

    def test_find_transform_givensources(self):

        source = np.array(
            [
                [1.4, 2.2],
                [5.3, 1.0],
                [3.7, 1.5],
                [10.1, 9.6],
                [1.3, 10.2],
                [7.1, 2.0],
            ]
        )
        nsrc = source.shape[0]
        scale = 1.5  # scaling parameter
        alpha = np.pi / 8.0  # rotation angle
        mm = scale * np.array(
            [[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]]
        )
        tx, ty = 2.0, 1.0  # translation parameters
        transl = np.array([nsrc * [tx], nsrc * [ty]])
        dest = (mm.dot(source.T) + transl).T
        t_true = estimate_transform("similarity", source, dest)

        # disorder dest points so they don't match the order of source
        np.random.shuffle(dest)

        t, (src_pts, dst_pts) = aa.find_transform(source, dest)
        self.assertLess(t_true.scale - t.scale, 1e-10)
        self.assertLess(t_true.rotation - t.rotation, 1e-10)
        self.assertLess(
            np.linalg.norm(t_true.translation - t.translation), 1e-10
        )
        self.assertEqual(src_pts.shape[0], dst_pts.shape[0])
        self.assertEqual(src_pts.shape[1], 2)
        self.assertEqual(dst_pts.shape[1], 2)
        dst_pts_test = matrix_transform(src_pts, t.params)
        self.assertLess(np.linalg.norm(dst_pts_test - dst_pts), 1e-10)

    def compare_image(self, the_image):
        """Return the fraction of sources found in the reference image"""
        # pixel comparison is not good, doesn't work. Compare catalogs.
        full_algn = the_image.astype("float32")
        import sep

        bkg = sep.Background(full_algn)
        thresh = 5.0 * bkg.globalrms
        allobjs = sep.extract(full_algn - bkg.back(), thresh)
        allxy = np.array([[obj["x"], obj["y"]] for obj in allobjs])

        from scipy.spatial import KDTree

        ref_coordtree = KDTree(self.star_ref_pos)

        # Compare here srcs list with self.star_ref_pos
        num_sources = 0
        for asrc in allxy:
            found_source = ref_coordtree.query_ball_point(asrc, 3)
            if found_source:
                num_sources += 1
        fraction_found = num_sources / len(allxy)
        return fraction_found

    def test_register(self):
        registered_img, footp = aa.register(
            source=self.image, target=self.image_ref
        )
        self.assertIsInstance(registered_img, np.ndarray)
        self.assertIsInstance(footp, np.ndarray)
        self.assertIs(footp.dtype, np.dtype("bool"))
        fraction = self.compare_image(registered_img)
        self.assertGreater(fraction, 0.85)

    def test_register_nddata(self):
        nd_image = NDData(self.image, mask=self.image_mask)
        nd_image_ref = NDData(self.image_ref, mask=self.image_ref_mask)
        registered_img, footp = aa.register(
            source=nd_image, target=nd_image_ref
        )
        self.assertIsInstance(registered_img, np.ndarray)
        self.assertIsInstance(footp, np.ndarray)
        self.assertIs(footp.dtype, np.dtype("bool"))
        fraction = self.compare_image(registered_img)
        self.assertGreater(fraction, 0.85)

    def test_register_ccddata(self):
        ccd_image = CCDData(
            self.image,
            mask=self.image_mask,
            meta={"object": "fake galaxy", "filter": "R"},
            unit="adu",
        )
        ccd_image_ref = CCDData(
            self.image_ref,
            mask=self.image_ref_mask,
            meta={"object": "fake galaxy", "filter": "R"},
            unit="adu",
        )
        registered_img, footp = aa.register(
            source=ccd_image, target=ccd_image_ref
        )
        self.assertIsInstance(registered_img, np.ndarray)
        self.assertIsInstance(footp, np.ndarray)
        self.assertIs(footp.dtype, np.dtype("bool"))
        fraction = self.compare_image(registered_img)
        self.assertGreater(fraction, 0.85)

    def test_register_npma(self):
        ma_image = np.ma.array(self.image, mask=self.image_mask)
        ma_image_ref = np.ma.array(self.image_ref, mask=self.image_ref_mask)
        registered_img, footp = aa.register(
            source=ma_image, target=ma_image_ref
        )
        self.assertIsInstance(registered_img, np.ndarray)
        self.assertIsInstance(footp, np.ndarray)
        self.assertIs(footp.dtype, np.dtype("bool"))
        fraction = self.compare_image(registered_img)
        self.assertGreater(fraction, 0.85)

    def test_apply_transform_nddata(self):
        transf = SimilarityTransform(rotation=np.pi / 2.0, translation=(1, 0))

        nd = NDData(
            [[0.0, 1.0], [2.0, 3.0]], mask=[[True, False], [False, False]]
        )
        registered_img, footp = aa.apply_transform(
            transf, nd, nd, propagate_mask=True
        )
        err = np.linalg.norm(
            registered_img - np.array([[2.0, 0.0], [3.0, 1.0]])
        )
        self.assertLess(err, 1e-6)
        err_mask = footp == np.array([[False, True], [False, False]])
        self.assertTrue(all(err_mask.flatten()))

        # Test now if there is no assigned mask during creation
        nd = NDData([[0.0, 1.0], [2.0, 3.0]])
        registered_img, footp = aa.apply_transform(
            transf, nd, nd, propagate_mask=True
        )
        err = np.linalg.norm(
            registered_img - np.array([[2.0, 0.0], [3.0, 1.0]])
        )
        self.assertLess(err, 1e-6)
        err_mask = footp == np.array([[False, False], [False, False]])
        self.assertTrue(all(err_mask.flatten()))

    def test_apply_transform_ccddata(self):

        transf = SimilarityTransform(rotation=np.pi / 2.0, translation=(1, 0))

        cd = CCDData(
            [[0.0, 1.0], [2.0, 3.0]],
            mask=[[True, False], [False, False]],
            unit="adu",
        )
        registered_img, footp = aa.apply_transform(
            transf, cd, cd, propagate_mask=True
        )
        err = np.linalg.norm(
            registered_img - np.array([[2.0, 0.0], [3.0, 1.0]])
        )
        self.assertLess(err, 1e-6)
        err_mask = footp == np.array([[False, True], [False, False]])
        self.assertTrue(all(err_mask.flatten()))

        cd = CCDData([[0.0, 1.0], [2.0, 3.0]], unit="adu")
        registered_img, footp = aa.apply_transform(
            transf, cd, cd, propagate_mask=True
        )
        err = np.linalg.norm(
            registered_img - np.array([[2.0, 0.0], [3.0, 1.0]])
        )
        self.assertLess(err, 1e-6)
        err_mask = footp == np.array([[False, False], [False, False]])
        self.assertTrue(all(err_mask.flatten()))

    def test_apply_transform_npma(self):
        from skimage.transform import SimilarityTransform

        transf = SimilarityTransform(rotation=np.pi / 2.0, translation=(1, 0))
        nparr = np.array([[0.0, 1.0], [2.0, 3.0]])
        mask = [[True, False], [False, False]]

        ma = np.ma.array(nparr, mask=mask)
        registered_img, footp = aa.apply_transform(
            transf, ma, ma, propagate_mask=True
        )
        err = np.linalg.norm(
            registered_img - np.array([[2.0, 0.0], [3.0, 1.0]])
        )
        self.assertLess(err, 1e-6)
        err_mask = footp == np.array([[False, True], [False, False]])
        self.assertTrue(all(err_mask.flatten()))

        ma = np.ma.array(nparr)
        registered_img, footp = aa.apply_transform(
            transf, ma, ma, propagate_mask=True
        )
        err = np.linalg.norm(
            registered_img - np.array([[2.0, 0.0], [3.0, 1.0]])
        )
        self.assertLess(err, 1e-6)
        err_mask = footp == np.array([[False, False], [False, False]])
        self.assertTrue(all(err_mask.flatten()))

    def test_fill_value(self):
        registered_img, footp = aa.register(
            source=self.image, target=self.image_ref, fill_value=-9999.99
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
        aa.register(self.image.astype("float32"), self.image_ref)
        aa.register(self.image.astype("float64"), self.image_ref)
        aa.register(self.image.astype("int32"), self.image_ref)
        aa.register(self.image.astype("int64"), self.image_ref)

    def test_consistent_invert(self):
        t, __ = aa.find_transform(self.image, self.image_ref)
        tinv, __ = aa.find_transform(self.image_ref, self.image)
        rpoint = np.random.rand(3) * self.h
        rpoint[2] = 1.0
        rtransf = tinv.params.dot(t.params.dot(rpoint))
        err = np.linalg.norm(rpoint - rtransf) / np.linalg.norm(rpoint)
        self.assertLess(err, 1e-2)

    def test_unrepeated_sources(self):
        source = np.array(
            [[0.0, 2.0], [1.0, 3.0], [2.1, 1.75], [3.5, 1.0], [4.0, 2.0]]
        )
        R = np.array(
            [
                [np.cos(30.0 * np.pi / 180), np.sin(30.0 * np.pi / 180)],
                [-np.sin(30.0 * np.pi / 180), np.cos(30.0 * np.pi / 180)],
            ]
        )
        tr = np.array([-0.5, 2.5])
        target = R.dot(source.T).T + tr
        best_t, (s_list, t_list) = aa.find_transform(source, target)
        self.assertEqual(len(s_list), len(t_list))
        self.assertLessEqual(len(s_list), len(source))
        # Assert no repeated sources used
        source_set = set((x, y) for x, y in s_list)
        self.assertEqual(len(s_list), len(source_set))
        # Assert no repeated targets used
        target_set = set((x, y) for x, y in t_list)
        self.assertEqual(len(t_list), len(target_set))
        # Assert s_list is a subset of source
        self.assertTrue(source_set <= set((x, y) for x, y in source))
        # Assert t_list is a subset of target
        self.assertTrue(target_set <= set((x, y) for x, y in target))

    def test_consistent_result(self):
        t1, __ = aa.find_transform(source=self.image, target=self.image_ref)
        for i in range(5):
            t2, __ = aa.find_transform(
                source=self.image, target=self.image_ref
            )
            self.assertLess(np.linalg.norm(t1.params - t2.params), 1e-10)


class TestFewSources(unittest.TestCase):
    def setUp(self):
        self.h = 512  # image height
        self.w = 512  # image width
        self.x_offset = 10
        self.y_offset = -20
        self.rot_angle = 50.0 * np.pi / 180.0

    def check_if_findtransform_ok(self, numstars):
        """Helper function to test find_transform with common test code
        for 3, 4, 5, and 6 stars"""

        if numstars > 6:
            raise NotImplementedError

        # x and y of stars in the ref frame (int's)
        self.star_refx = np.array([100, 120, 400, 400, 200, 200])[:numstars]
        self.star_refy = np.array([150, 200, 200, 320, 210, 350])[:numstars]
        self.num_stars = numstars
        # Fluxes of stars
        self.star_f = np.array(numstars * [700.0])

        (
            self.image,
            self.image_ref,
            self.star_ref_pos,
            self.star_new_pos,
        ) = simulate_image_pair(
            shape=(self.h, self.w),
            translation=(self.x_offset, self.y_offset),
            rot_angle_deg=50.0,
            num_stars=self.num_stars,
            star_refx=self.star_refx,
            star_refy=self.star_refy,
            star_flux=self.star_f,
        )

        source = self.star_ref_pos
        dest = self.star_new_pos.copy()
        t_true = estimate_transform("similarity", source, dest)

        # disorder dest points so they don't match the order of source
        np.random.shuffle(dest)

        t, (src_pts, dst_pts) = aa.find_transform(source, dest)
        self.assertLess(t_true.scale - t.scale, 1e-10)
        self.assertLess(t_true.rotation - t.rotation, 1e-10)
        self.assertLess(
            np.linalg.norm(t_true.translation - t.translation), 1.0
        )
        self.assertEqual(src_pts.shape[0], dst_pts.shape[0])
        self.assertLessEqual(src_pts.shape[0], source.shape[0])
        self.assertEqual(src_pts.shape[1], 2)
        self.assertEqual(dst_pts.shape[1], 2)
        dst_pts_test = matrix_transform(src_pts, t.params)
        self.assertLess(np.linalg.norm(dst_pts_test - dst_pts), 1.0)

    def test_find_transform_twosources(self):
        with self.assertRaises(Exception):
            self.check_if_findtransform_ok(2)

    def test_find_transform_threesources(self):
        self.check_if_findtransform_ok(3)

    def test_find_transform_foursources(self):
        self.check_if_findtransform_ok(4)

    def test_find_transform_fivesources(self):
        self.check_if_findtransform_ok(5)

    def test_find_transform_sixsources(self):
        self.check_if_findtransform_ok(6)

    def check_if_register_ok(self, numstars):
        """Helper function to test register with common test code
        for 3, 4, 5, and 6 stars"""

        if numstars > 6:
            raise NotImplementedError

        # x and y of stars in the ref frame (int's)
        self.star_refx = np.array([100, 120, 400, 400, 200, 200])[:numstars]
        self.star_refy = np.array([150, 200, 200, 320, 210, 350])[:numstars]
        self.num_stars = numstars
        # Fluxes of stars
        self.star_f = np.array(numstars * [700.0])

        (
            self.image,
            self.image_ref,
            self.star_ref_pos,
            self.star_new_pos,
        ) = simulate_image_pair(
            shape=(self.h, self.w),
            translation=(self.x_offset, self.y_offset),
            rot_angle_deg=50.0,
            noise_level=50,
            num_stars=self.num_stars,
            star_refx=self.star_refx,
            star_refy=self.star_refy,
            star_flux=self.star_f,
        )

        aligned, footprint = aa.register(self.image_ref, self.image)

        source = self.star_ref_pos
        dest = self.star_new_pos.copy()
        t_true = estimate_transform("similarity", source, dest)
        aligned_true, fp = aa.apply_transform(
            t_true, self.image_ref, self.image
        )

        err = np.linalg.norm((aligned_true - aligned)[fp], 1) / np.linalg.norm(
            (aligned_true)[fp], 1
        )
        self.assertLess(err, 1e-1)

    def test_register_twosources(self):
        with self.assertRaises(Exception):
            self.check_if_register_ok(2)

    def test_register_threesources(self):
        self.check_if_register_ok(3)

    def test_register_foursources(self):
        self.check_if_register_ok(4)

    def test_register_fivesources(self):
        self.check_if_register_ok(5)

    def test_register_sixsources(self):
        self.check_if_register_ok(6)


class TestColorImages(unittest.TestCase):
    def setUp(self):
        def convert_to_uint8(sky_arr):
            sky_max, sky_min = sky_arr.max(), sky_arr.min()
            sky_arr = (sky_arr - sky_min) * 512 / (sky_max - sky_min)
            sky_arr = np.clip(sky_arr, 0, 512)
            return sky_arr.astype("uint8")

        self.h = 512  # image height
        self.w = 512  # image width
        self.x_offset = 10
        self.y_offset = -20
        self.rot_angle = 50.0 * np.pi / 180.0
        (
            image_new,
            image_ref,
            self.star_ref_pos,
            self.star_new_pos,
        ) = simulate_image_pair(
            shape=(self.h, self.w),
            translation=(self.x_offset, self.y_offset),
            rot_angle_deg=50.0,
            noise_level=10.0,
            num_stars=150,
            star_flux=np.array([1000.0] * 150),
        )
        self.image_rgb_new = np.array(
            [image_new.copy(), image_new.copy(), image_new.copy()]
        )
        self.image_rgb_ref = np.array(
            [image_ref.copy(), image_ref.copy(), image_ref.copy()]
        )
        self.image_rgb_new = np.moveaxis(self.image_rgb_new, 0, -1)
        self.image_rgb_ref = np.moveaxis(self.image_rgb_ref, 0, -1)

        self.image_rgba_new = np.array(
            [
                image_new.copy(),
                image_new.copy(),
                image_new.copy(),
                255.0 * np.ones(image_new.shape),
            ]
        )
        self.image_rgba_ref = np.array(
            [
                image_ref.copy(),
                image_ref.copy(),
                image_ref.copy(),
                255.0 * np.ones(image_new.shape),
            ]
        )
        self.image_rgba_new = np.moveaxis(self.image_rgba_new, 0, -1)
        self.image_rgba_ref = np.moveaxis(self.image_rgba_ref, 0, -1)

        self.jpgref_fp = tempfile.TemporaryFile()
        sky_ref = convert_to_uint8(self.image_rgb_ref)
        Image.fromarray(sky_ref).save(self.jpgref_fp, "jpeg")

        self.jpgnew_fp = tempfile.TemporaryFile()
        sky_new = convert_to_uint8(self.image_rgb_new)
        Image.fromarray(sky_new).save(self.jpgnew_fp, "jpeg")

        self.pngref_fp = tempfile.TemporaryFile()
        sky_ref = convert_to_uint8(self.image_rgba_ref)
        Image.fromarray(sky_ref).save(self.pngref_fp, "png")

        self.pngnew_fp = tempfile.TemporaryFile()
        sky_new = convert_to_uint8(self.image_rgba_new)
        Image.fromarray(sky_new).save(self.pngnew_fp, "png")

    def tearDown(self):
        self.jpgref_fp.close()
        self.jpgnew_fp.close()
        self.pngref_fp.close()
        self.pngnew_fp.close()

    def compare_image(self, the_image):
        """Return the fraction of sources found in the reference image"""
        # pixel comparison is not good, doesn't work. Compare catalogs.
        full_algn = np.mean(the_image, axis=-1, dtype="float32")
        import sep

        bkg = sep.Background(full_algn)
        thresh = 5.0 * bkg.globalrms
        allobjs = sep.extract(full_algn - bkg.back(), thresh)
        allxy = np.array([[obj["x"], obj["y"]] for obj in allobjs])

        from scipy.spatial import KDTree

        ref_coordtree = KDTree(self.star_ref_pos)

        # Compare here srcs list with self.star_ref_pos
        num_sources = 0
        for asrc in allxy:
            found_source = ref_coordtree.query_ball_point(asrc, 3)
            if found_source:
                num_sources += 1
        fraction_found = num_sources / len(allxy)
        return fraction_found

    def test_register_rgb_channels(self):
        "Test register works with RGB images"
        registered, footp = aa.register(
            source=self.image_rgb_new, target=self.image_rgb_ref
        )
        self.assertEqual(registered.ndim, self.image_rgb_new.ndim)
        fraction = self.compare_image(registered)
        self.assertGreater(fraction, 0.70)
        self.assertTrue(footp.ndim == 2)
        self.assertTrue(footp.shape == (self.h, self.w))

    def test_register_rgba_channels(self):
        "Test register works with RGB images"
        registered, footp = aa.register(
            source=self.image_rgba_new, target=self.image_rgba_ref
        )
        self.assertEqual(registered.ndim, self.image_rgba_new.ndim)
        fraction = self.compare_image(registered)
        self.assertGreater(fraction, 0.60)
        self.assertTrue(footp.ndim == 2)
        self.assertTrue(footp.shape == (self.h, self.w))

    def test_register_jpg_image(self):
        source = Image.open(self.jpgnew_fp)
        target = Image.open(self.jpgref_fp)
        registered, footp = aa.register(source, target)
        self.assertEqual(registered.ndim, self.image_rgb_new.ndim)
        fraction = self.compare_image(registered)
        self.assertGreater(fraction, 0.70)
        self.assertTrue(footp.ndim == 2)
        self.assertTrue(footp.shape == (self.h, self.w))

    def test_register_png_image(self):
        source = Image.open(self.pngnew_fp)
        target = Image.open(self.pngref_fp)
        registered, footp = aa.register(source, target)
        self.assertEqual(registered.ndim, self.image_rgba_new.ndim)
        fraction = self.compare_image(registered)
        self.assertGreater(fraction, 0.70)
        self.assertTrue(footp.ndim == 2)
        self.assertTrue(footp.shape == (self.h, self.w))


if __name__ == "__main__":
    unittest.main()
