import unittest
import numpy as np
from astroalign import align

def gauss(shape = (10,10), center=None, sx=2, sy=2):
    h, w = shape
    if center is None: center = ((h-1)/2., (w-1)/2.)
    x0,y0 = center
    x,y = np.meshgrid(range(w),range(h))
    norm = np.sqrt(2*np.pi*(sx**2)*(sy**2))
    return np.exp(-0.5*((x-x0)**2/sx**2 + (y-y0)**2/sy**2))/norm


class TestAlign(unittest.TestCase):
    def setUp(self):
        from scipy import signal
        h = 512
        w = 512
        kh = 10
        kw = 10
        noise_level = 0
        num_stars = 1500
        psf = gauss(shape=(20,20), sx=1.5, sy=1.5)

        self.image_ref = np.random.poisson(noise_level, size=(h+kh,w+kw)).astype('float64')
        self.star_rows = np.random.randint(low=0, high=h, size=(num_stars,))
        self.star_cols = np.random.randint(low=0, high=w, size=(num_stars,))
        self.star_fluxes = 1.0*np.random.exponential(160., size=(num_stars,))
        self.image_ref[self.star_rows, self.star_cols] += self.star_fluxes
        self.image_ref = signal.convolve2d(self.image_ref, psf, mode='same')[kh//2:-kh//2,kw//2:-kw//2]

        self.x_offset = 10
        self.y_offset = -20
        self.rot_angle = 50.*np.pi/180.

        self.star_new_rows = []
        self.star_new_cols = []
        self.star_new_fluxes = []
        for x, y, flux in zip(self.star_cols, self.star_rows, self.star_fluxes):
            x -= w / 2
            y -= h / 2
            new_x = x * np.cos(self.rot_angle) - y * np.sin(self.rot_angle) + self.x_offset
            new_y = x * np.sin(self.rot_angle) + y * np.cos(self.rot_angle) + self.y_offset
            new_x += w/2
            new_y += h/2
            if (new_x > 0 and new_x < w and new_y > 0 and new_y < h):
                self.star_new_rows.append(new_x)
                self.star_new_cols.append(new_y)
                self.star_new_fluxes.append(flux)

        self.star_new_rows = np.array(self.star_new_rows).astype(int)
        self.star_new_cols = np.array(self.star_new_cols).astype(int)

        from scipy import signal
        self.image = np.random.poisson(noise_level*1.5, size=(h+kh,w+kw)).astype('float64')
        self.image[self.star_new_rows, self.star_new_cols] += self.star_new_fluxes
        self.image = signal.convolve2d(self.image, psf, mode='same')[kh//2:-kh//2,kw//2:-kw//2]

    def test_align(self):
        np.save("image", self.image)
        np.save("image_ref", self.image_ref)
        image_aligned = align.alignImage(self.image, self.image_ref)
        error = np.linalg.norm(image_aligned - self.image_ref)
        self.assertLess(error, 1E-6)

    def tearDown(self):
        self.image = None
        self.image_ref = None


if __name__ == "__main__":
    unittest.main()
