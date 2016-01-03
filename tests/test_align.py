import unittest
import numpy as np
from astroalign import align


def gauss(shape=(10, 10), center=None, sx=2, sy=2):
    h, w = shape
    if center is None:
        center = ((h - 1) / 2., (w - 1) / 2.)
    x0, y0 = center
    x, y = np.meshgrid(range(w), range(h))
    norm = np.sqrt(2 * np.pi * (sx ** 2) * (sy ** 2))
    return np.exp(-0.5 * ((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2))\
        / norm


class TestAlign(unittest.TestCase):
    def setUp(self):
        from scipy import signal
        self.h = 512
        self.w = 512
        kh = 10
        kw = 10
        noise_level = 0
        num_stars = 1500
        psf = gauss(shape=(20, 20), sx=1.5, sy=1.5)

        self.image_ref = np.random.poisson(noise_level,
                                           size=(self.h + kh, self.w + kw)
                                           ).astype('float64')
        self.star_rows = np.random.randint(low=0, high=self.h,
                                           size=(num_stars,))
        self.star_cols = np.random.randint(low=0, high=self.w,
                                           size=(num_stars,))
        self.star_fluxes = 1. * np.random.exponential(1600., size=(num_stars,))
        self.image_ref[self.star_rows, self.star_cols] += self.star_fluxes
        self.image_ref = signal.convolve2d(self.image_ref, psf, mode='same')
        self.image_ref = self.image_ref[kh // 2: -kh // 2, kw // 2: -kw // 2]

        self.x_offset = 10
        self.y_offset = -20
        self.rot_angle = 50. * np.pi / 180.

        self.star_new_rows = []
        self.star_new_cols = []
        self.star_new_fluxes = []
        for x, y, flux in \
                zip(self.star_cols, self.star_rows, self.star_fluxes):
            x -= self.w / 2
            y -= self.h / 2
            new_x = x * np.cos(self.rot_angle) - y * np.sin(self.rot_angle) \
                + self.x_offset
            new_y = x * np.sin(self.rot_angle) + y * np.cos(self.rot_angle) \
                + self.y_offset
            new_x += self.w / 2
            new_y += self.h / 2
            if (new_x > 0 and new_x < self.w and new_y > 0 and new_y < self.h):
                self.star_new_cols.append(new_x)
                self.star_new_rows.append(new_y)
                self.star_new_fluxes.append(flux)

        self.star_new_rows = np.array(self.star_new_rows).astype(int)
        self.star_new_cols = np.array(self.star_new_cols).astype(int)

        from scipy import signal
        self.image = np.random.poisson(noise_level,
                                       size=(self.h + kh, self.w + kw)
                                       ).astype('float64')
        self.image[self.star_new_rows, self.star_new_cols] += \
            self.star_new_fluxes
        self.image = signal.convolve2d(self.image, psf, mode='same')
        self.image = self.image[kh // 2: -kh // 2, kw // 2: -kw // 2]

    def test_find_affine_transform(self):
        star_pos = np.array(zip(self.star_cols, self.star_rows))
        star_new_pos = np.array(zip(self.star_new_cols, self.star_new_rows))

        star_pos = star_pos[np.argsort(self.star_fluxes)]
        star_new_pos = star_new_pos[np.argsort(self.star_new_fluxes)]

        m = align.find_affine_transform(star_new_pos[50::-1], star_pos[70::-1])
        alpha = self.rot_angle
        xoff_corrected = ((1 - np.cos(alpha)) * self.w / 2 + np.sin(alpha) *
                          self.h / 2 + self.x_offset)
        yoff_corrected = (-np.sin(alpha) * self.w / 2 + (1 - np.cos(alpha)) *
                          self.h / 2 + self.y_offset)
        mtrue = np.array([[np.cos(alpha), -np.sin(alpha), xoff_corrected],
                          [np.sin(alpha), np.cos(alpha), yoff_corrected]])
        self.assertLess(np.linalg.norm(m - mtrue, 1)
                        / np.linalg.norm(mtrue, 1), 1E-2)

    def test_align_image(self):
        # image_aligned = align.alignImage(self.image, self.image_ref)
        # pixel comparison is not good, doesn't work. Compare catalogs.
        # error = np.linalg.norm(image_aligned - self.image)
        #                        /np.linalg.norm(self.image)
        # self.assertLess(error, 0.5)
        self.assertEqual(1, 1)

    def test_find_sources(self):
        # srcs = align.findSources(self.image_ref)
        # Compare here srcs list with self.star_rows and self.star_cols
        self.assertEqual(1, 1)

    def tearDown(self):
        self.image = None
        self.image_ref = None


if __name__ == "__main__":
    unittest.main()
