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
import os
import timeit
import datetime as dt
import argparse
from collections import OrderedDict

import numpy as np

import astroalign as aa

import pandas as pd

import joblib

from sklearn.model_selection import ParameterGrid

import tqdm

test_path = os.path.abspath(os.path.dirname(aa.__file__))
sys.path.insert(0, test_path)

from tests.test_align import simulate_image_pair  # noqa


# =============================================================================
# CONSTANTS
# =============================================================================

SIZES = (256, 512, 768, 1024)

STARS = (300, 500, 1000, 10_000)

NOISES = (100, 500, 1000, 5000)

COMB_NUMBER = 10

STATEMENT = "aa.register(source, target)"

REPEATS = 50

COLSROWS = {
    "vertical": {"ncols": 1, "nrows": 3},
    "horizontal": {"ncols": 3, "nrows": 1}
}

DEFAULT_SIZES = {
    "vertical": (4, 12),
    "horizontal": (12, 4)
}


# =============================================================================
# FUNCTIONS
# =============================================================================

def get_images(size, stars, noise, seed):
    """Retrieves a pair source and target image"""
    if seed is not None:
        np.random.seed(seed)
    shape = (size, size)
    source, target = simulate_image_pair(
        shape=shape, num_stars=stars, noise_level=noise)[:2]
    return source, target


def get_parameters(sizes, stars, noises, comb_number, repeats, seed):
    """Create a list of dictionaries with all the combinations of the given
    parameters.

    """
    grid = ParameterGrid({
        "size": sizes, "stars": stars, "noise": noises})
    grid = list(grid) * comb_number

    # set the random state for run in parallel
    random = np.random.RandomState(seed)
    images_seeds = random.randint(1_000_000, size=len(grid))

    for idx, g in enumerate(grid):
        g["idx"] = idx
        g["seed"] = seed
        g["images_seed"] = images_seeds[idx]
        g["repeats"] = repeats

    return grid


def _test(idx, size, stars, noise, seed, images_seed, repeats):

    # create the two images
    source, target = get_images(
        size=size, stars=stars, noise=noise, seed=images_seed)

    # create the timer
    test_globals = {"aa": aa, "source": source, "target": target}
    timer = timeit.Timer(stmt=STATEMENT, globals=test_globals)

    # find the number of loops
    loops = timer.autorange()[0]

    # create a copy of the params to be returned ad result
    result = OrderedDict({
        "idx": idx, "size": size, "noise": noise, "stars": stars, "seed": seed,
        "images_seed": images_seed, "repeats": repeats, "loops": loops})

    # execute the timeit
    times = timer.repeat(repeats, loops)

    # store the times into the result
    result["time"] = np.min(np.array(times) / loops)
    for tidx, time in enumerate(times):
        result[f"time_{tidx}"] = time

    return result


def benchmark(sizes=SIZES, stars=STARS, noises=NOISES,
              comb_number=10, seed=None, repeats=REPEATS, n_jobs=-1):

    grid = get_parameters(
        sizes=sizes, stars=stars, noises=noises,
        comb_number=comb_number, seed=seed, repeats=repeats)

    with joblib.Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(
            joblib.delayed(_test)(**params) for params in tqdm.tqdm(grid))

    df = pd.DataFrame(results)
    return df


def describe(results):
    repetitions = results.repeats.values[0]
    resume = results[["time", "loops"]].describe()
    return repetitions, resume


def plot(results, ax_size, ax_stars, ax_noise):
    results = results[["size", "stars", "noise", "time"]]

    def plots(df, by, ax):
        df.boxplot(by=by, column=["time"], grid=False, ax=ax)
        ax.set_title(f"Time by {by.title()}")
        ax.set_ylabel("Seconds")
        ax.set_xlabel(by.title())

    plots(results, "noise", ax_noise)
    plots(results, "stars", ax_stars)
    plots(results, "size", ax_size)

    return (ax_size, ax_stars, ax_noise)


# =============================================================================
# CLI MAIN
# =============================================================================

class CLI:

    def __init__(self):
        self._parser = argparse.ArgumentParser(
                description="Astroalign time benchmark tool based on timeit")
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
            help="Execute and collect the benchmark data of astroalign")
        benchmark.set_defaults(callback=self.benchmark_command)

        benchmark.add_argument(
            "--sizes", dest="sizes", type=int, default=SIZES, nargs="+",
            help=("The size in pixels of the image. This parameter creates "
                  f"square figure (defaults={SIZES})."))

        benchmark.add_argument(
            "--stars", dest="stars", type=int, default=STARS, nargs="+",
            help=("The total numbers of stars in the image "
                  f"(defaults={STARS})."))

        benchmark.add_argument(
            "--noises", dest="noises", type=int, default=NOISES, nargs="+",
            help=f"lambda parameter for poisson noise (default={NOISES})")

        benchmark.add_argument(
            "--number", dest="comb_number", type=int, default=10,
            help=("How many random images pairs must be created for one "
                  "combination of sizes, stars and noise (default=10)."))

        benchmark.add_argument(
            "--seed", dest="seed", type=int, default=None,
            help=("Random seed used to initialize the pseudo-random number "
                  "generator. if seed is None, then random-state will try to "
                  "read data from /dev/urandom (or the Windows analogue) if "
                  "available or seed from the clock otherwise "
                  "(default=None)."))

        benchmark.add_argument(
            "--repeats", dest="repeats", type=int, default=REPEATS,
            help=("How many measurements must be taken for every image pair. "
                  "The final 'time' is the lower bound of all the times. "
                  "Docs: https://docs.python.org/3.7/library/timeit.html"))

        benchmark.add_argument(
            "--jobs", dest="n_jobs", type=int, default=-1,
            help=("The number of CPU to run the benchmars. "
                  "-1 uses all the available CPUS (default=-1)"))

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
            help="File path of the time benchmark data in CSV format")

        # =====================================================================
        # plot subparser
        # =====================================================================

        plot = subparsers.add_parser(
            "plot", help="Show three boxplots of a given results")
        plot.set_defaults(callback=self.plot_command)

        plot.add_argument(
            "--file", "-f", dest="file", required=True,
            type=argparse.FileType('r'),
            help="File path of the time benchmark data in CSV format")

        plot.add_argument(
            "--orientation", dest="orientation",
            choices=list(COLSROWS.keys()), default="horizontal",
            help=("If the plots will be a single row (horizontal) "
                  f"or vertical for a single column (default='horizontal')"))

        plot.add_argument(
            "--size", dest="size", nargs=2, type=float,
            help=("The size of the entire figure in inches in the format "
                  "'width height' for horizontal orientation the size by "
                  f"default is {DEFAULT_SIZES['horizontal']} and for "
                  f"vertical {DEFAULT_SIZES['vertical']}."))

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

        orientation = COLSROWS[ns.orientation]
        size = ns.size if ns.size else DEFAULT_SIZES[ns.orientation]

        fig, axes = plt.subplots(**orientation)
        fig.set_size_inches(*size)

        plot(results, *axes)

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

        print(f"Executed: {len(results)} cases")

        print(f"\twith {repetitions} repetitions \n")
        print(">>>>> Resume <<<<<")
        print(resume)
        print("")

    def benchmark_command(self, ns):
        if ns.repeats <= 0:
            self._parser.error(f"'repeats' must be > 0. Found {ns.repeats}")

        now = dt.datetime.now

        print(
            f"[{now()}] Starting benchmark for astroalign {aa.__version__}...")
        print("")
        results = benchmark(
            sizes=ns.sizes, stars=ns.stars, noises=ns.noises,
            comb_number=ns.comb_number, seed=ns.seed,
            repeats=ns.repeats, n_jobs=ns.n_jobs)

        repetitions, resume = describe(results)

        print(f"[{now()}] Executed: {len(results)} cases")

        print(f"\twith {repetitions} repetitions \n")
        print(">>>>> Resume <<<<<")
        print(resume)
        print("")

        results.to_csv(ns.out, index=False)
        print(f"[{now()}] Data stored in '{ns.out.name}'")

    @property
    def parser(self):
        return self._parser


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = CLI()
    parser.parse_and_run()
