# MIT License

# Copyright (c) 2016-2019 Martin Beroiz, Juan B. Cabral, Bruno Sanchez

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


# =============================================================================
# IMPORTS
# =============================================================================
import sys
import datetime as dt
import argparse

import numpy as np

import astroalign as aa

import pandas as pd

import sep

from skimage.transform import SimilarityTransform

from scipy import stats

sys.path.insert(0, '..')  # noqa
from tests.test_align import simulate_image_single  # noqa


# =============================================================================
# CONSTANTS
# =============================================================================

SIZE = 256

STARS = 300

NOISE = 100

REPEATS = 35

DEFAULT_SIZE = 6.4, 4.8


# =============================================================================
# FUNCTIONS
# =============================================================================

def get_image(size, stars, noise, seed):
    """Retrieves a single image"""
    if seed is not None:
        np.random.seed(seed)
    shape = (size, size)
    image = simulate_image_single(
        shape=shape, num_stars=stars, noise_level=noise)[:2]
    return image


def benchmark(size=SIZE, stars=STARS, noise=NOISE, repeats=REPEATS, seed=None):
    # get image
    image = get_image(size, stars, noise, seed)
    imagedata = np.ascontiguousarray(image[0])

    # detect sources (we know where they are, actually)
    bkg = sep.Background(imagedata)
    thresh = 3. * bkg.globalrms
    sources = sep.extract(imagedata - bkg.back(), thresh)
    sources.sort(order='flux')

    # perform photometry
    flux, fluxerr, flag = sep.sum_circle(
        imagedata-bkg.back(), sources['x'],
        sources['y'], 3.0, err=bkg.globalrms, gain=1.0)

    dframes = []
    # transform it
    for i_trsf in range(repeats):
        dx, dy = np.random.randint(
            low=-1 * size / 32., high=size / 32., size=2)
        theta = (np.random.random()-0.5)*0.125*np.pi
        s = 0.85+np.random.random()*0.3
        trf = SimilarityTransform(
            translation=(dx, dy), rotation=theta, scale=s)

        target = np.zeros(shape=np.array(imagedata.shape) * 2)
        newimage = aa.apply_transform(trf, imagedata - bkg.back(), target)

        # perform photometry on new places
        src_coords = np.array([sources['x'], sources['y']]).T
        new_coords = trf(src_coords).T
        nflux, nfluxerr, nflag = sep.sum_circle(
            newimage[0], new_coords[0], new_coords[1], 3.0 * s,
            err=bkg.globalrms, gain=1.0)

        # compare fluxes
        good_flux = nflag == 0
        new_to_orig = nflux[good_flux]/flux[good_flux]

        # put everything in a pd dataframe
        df = pd.DataFrame()

        df["idx"] = np.array([i_trsf] * sum(good_flux))
        df["seed"] = np.array([seed] * sum(good_flux))
        df["repeats"] = np.array([repeats] * sum(good_flux))

        df['orig_x'] = sources['x'][good_flux]
        df['orig_y'] = sources['y'][good_flux]
        df['orig_flux'] = flux[good_flux]
        df['orig_fluxerr'] = fluxerr[good_flux]
        df['orig_flag'] = flag[good_flux]

        df['new_x'] = new_coords[0][good_flux]
        df['new_y'] = new_coords[1][good_flux]
        df['new_flux'] = nflux[good_flux]
        df['new_fluxerr'] = nfluxerr[good_flux]
        df['new_flag'] = nflag[good_flux]

        df['flux_ratio'] = new_to_orig

        df['trf_theta'] = theta
        df['trf_dx'] = dx
        df['trf_dy'] = dy
        df['trf_scale'] = s

        slp, intpt, r_val, p_val, std_err = stats.linregress(
            flux[good_flux], nflux[good_flux])
        df['stats_slope'] = slp
        df['stats_intpt'] = intpt
        df['flux_per_area_ratio'] = df['flux_ratio'] / (df['trf_scale'] ** 2)

        dframes.append(df)

    final_df = pd.concat(dframes)

    return final_df


def describe(results):
    repetitions = results.repeats.values[0]
    resume = results[["flux_per_area_ratio"]].describe()
    return repetitions, resume


def plot(results, ax):

    bins = np.arange(0.95, 1.05, 0.001)
    ax.hist(
        results.flux_per_area_ratio, normed=True,
        histtype='step', bins=bins, label='Data')

    ax.plot(
        bins + (bins[1] - bins[0]) / 2.,
        stats.norm.pdf(
            bins,
            loc=np.mean(results.flux_per_area_ratio),
            scale=np.std(results.flux_per_area_ratio)),
        label='Gaussian')

    ax.legend(loc='best')

    ax.set_title("Flux ratio per unit area")
    ax.set_xlabel('Flux ratio per unit area')
    ax.set_ylabel('Normalized N')

    return ax


# =============================================================================
# CLI MAIN
# =============================================================================

class CLI:

    def __init__(self):
        self._parser = argparse.ArgumentParser(
                description="Astroalign flux benchmark tool")
        self._parser.set_defaults(
            callback=lambda ns: self.parser.print_usage())

        self._parser.add_argument(
            '--version', action='version', version='%(prog)s 2019.10')

        subparsers = self._parser.add_subparsers()

        # =====================================================================
        # benchmark subparser
        # =====================================================================

        benchmark = subparsers.add_parser(
            "benchmark",
            help="Execute and collect the flux benchmark data of astroalign")
        benchmark.set_defaults(callback=self.benchmark_command)

        benchmark.add_argument(
            "--size", dest="size", type=int, default=SIZE,
            help=("The size in pixels of the image. This parameter creates "
                  f"square figure (defaults={SIZE})."))

        benchmark.add_argument(
            "--stars", dest="stars", type=int, default=STARS,
            help=("The total numbers of stars in the image "
                  f"(defaults={STARS})."))

        benchmark.add_argument(
            "--noise", dest="noise", type=int, default=NOISE,
            help=f"lambda parameter for poisson noise (default={NOISE})")

        benchmark.add_argument(
            "--number", dest="repeats", type=int, default=REPEATS,
            help=f"How many flux tests must be executed (default={REPEATS})")

        benchmark.add_argument(
            "--seed", dest="seed", type=int, default=None,
            help=("Random seed used to initialize the pseudo-random number "
                  "generator. if seed is None, then random-state will try to "
                  "read data from /dev/urandom (or the Windows analogue) if "
                  "available or seed from the clock otherwise "
                  "(default=None)."))

        benchmark.add_argument(
            "--out", "-o", dest="out", required=True,
            type=argparse.FileType('w'),
            help="Output file path. The data was stored in CSV format")

        # =====================================================================
        # describe subparser
        # =====================================================================

        describe = subparsers.add_parser(
            "describe",
            help="Show a resume and (optionally) of the benchmark results")
        describe.set_defaults(callback=self.describe_command)

        describe.add_argument(
            "--file", "-f", dest="file", required=True,
            type=argparse.FileType('r'),
            help="File path of the flux benchmark data in CSV format")

        # =====================================================================
        # plot subparser
        # =====================================================================

        plot = subparsers.add_parser(
            "plot", help="Show the histogram of a given results")
        plot.set_defaults(callback=self.plot_command)

        plot.add_argument(
            "--file", "-f", dest="file", required=True,
            type=argparse.FileType('r'),
            help="File path of the flux benchmark data in CSV format")

        plot.add_argument(
            "--size", dest="size", nargs=2, type=float,
            help=("The size of the entire figure in inches in the format "
                  f"'width height' (default={DEFAULT_SIZE})."))

        plot.add_argument(
            "--out", "-o", dest="out",
            help=("A file to store the generated plot. "
                  "By default the default matplotlib backend shows the plot"))

    def parse_and_run(self, *args, **kwargs):
        ns = self._parser.parse_args(*args, **kwargs)
        return ns.callback(ns)

    def plot_command(self, ns):
        import matplotlib.pyplot as plt

        results = pd.read_csv(ns.file)

        size = ns.size if ns.size else DEFAULT_SIZE

        fig, ax = plt.subplots()
        fig.set_size_inches(*size)

        plot(results, ax)

        fig.suptitle("")
        plt.tight_layout()
        if ns.out is None:
            print(f"Showing plot for data stored in '{ns.file.name}'...")
            fig.canvas.set_window_title(f"{self.parser.prog} - {ns.file.name}")
            plt.show()
        else:
            print(
                f"Storing plot for data in '{ns.file.name}' -> '{ns.out}'...")
            plt.savefig(ns.out)
            print("DONE!")

    def describe_command(self, ns):
        results = pd.read_csv(ns.file)

        repetitions, resume = describe(results)

        print(f"Data size: {len(results)}")
        print(f"\twith {repetitions} repetitions \n")
        print(">>>>> Resume <<<<<")
        print(resume)
        print("")

    def benchmark_command(self, ns):
        if ns.repeats <= 0:
            self._parser.error(f"'repeats' must be > 0. Found {ns.repeats}")

        now = dt.datetime.now

        print(
            f"[{now()}] Starting flux benchmark "
            f"for astroalign {aa.__version__}...")
        print("")
        results = benchmark(
            size=ns.size, stars=ns.stars, noise=ns.noise,
            repeats=ns.repeats, seed=ns.seed)

        repetitions, resume = describe(results)

        print(f"[{now()}] Data size: {len(results)}")
        print(f"\twith {repetitions} repetitions \n")

        print(">>>>> Resume <<<<<")
        print(resume)
        print("")

        results.to_csv(ns.out, index=False)

    @property
    def parser(self):
        return self._parser


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = CLI()
    parser.parse_and_run()
