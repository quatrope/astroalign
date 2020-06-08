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

from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd

import joblib

import tqdm


test_path = os.path.abspath(os.path.dirname(aa.__file__))
sys.path.insert(0, test_path)

from tests.test_align import simulate_image_pair # noqa


# =============================================================================
# CONSTANTS
# =============================================================================

SIZES = (256, 512, 768, 1024)

STARS = 10000

NOISE = 1000

STEP = 10

STATEMENT = "aa.register(source, target)"

REPEATS = 50

COMB_NUMBER = 10

DEFAULT_SIZE = (8, 8)


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


def get_parameters(min_size, max_size, step_size, stars,
                   noise, seed, comb_number, repeats):
    """Create a list of dictionaries with all the combinations of the given
    parameters.

    """

    sample_size = int((max_size - min_size) / step_size)
    sizes = np.linspace(min_size, max_size, sample_size, dtype=int)

    grid = ParameterGrid({
        "size": sizes, "stars": [stars],
        "noise": [noise], "repeats": [repeats]})
    grid = list(grid) * comb_number

    # set the random state for run in parallel
    random = np.random.RandomState(seed)
    images_seeds = random.randint(1_000_000, size=len(grid))

    for idx, g in enumerate(grid):
        g["idx"] = idx
        g["seed"] = seed
        g["min_size"] = min_size
        g["max_size"] = max_size
        g["step_size"] = step_size
        g["images_seed"] = images_seeds[idx]
    return grid


def _test(idx, min_size, max_size, step_size, size,
          stars, noise, seed, repeats, images_seed):

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
        "idx": idx, "min_size": min_size, "max_size": max_size,
        "step_size": step_size, "size": size, "noise": noise,
        "stars": stars, "seed": seed, "images_seed": images_seed,
        "repeats": repeats, "loops": loops})

    # execute the timeit
    times = timer.repeat(repeats, loops)

    # store the times into the result
    result["time"] = np.min(np.array(times) / loops)
    for tidx, time in enumerate(times):
        result[f"time_{tidx}"] = time

    return result


def benchmark(min_size=min(SIZES), max_size=max(SIZES), step_size=STEP,
              stars=STARS, noise=NOISE, seed=None, repeats=REPEATS,
              n_jobs=-1, comb_number=COMB_NUMBER):

    grid = get_parameters(
        min_size=min_size, max_size=max_size, step_size=step_size,
        repeats=repeats, stars=stars, noise=noise, seed=seed,
        comb_number=comb_number)

    with joblib.Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(
            joblib.delayed(_test)(**params) for params in tqdm.tqdm(grid))

    df = pd.DataFrame(results)
    return df


def describe(results):
    repetitions = results.repeats.values[0]
    resume = results[["time", "size", "loops"]].describe()
    return repetitions, resume


def plot(results, ax):
    df = results[["size", "time"]]

    df.plot.scatter(x='size', y='time', c='LightBlue', ax=ax, marker=".")

    # linear regression
    x = df["size"].values.reshape((-1, 1))
    y = df["time"].values
    linear = LinearRegression().fit(x, y)
    y_pred = linear.predict(x)

    mqe = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    ax.plot(x, y_pred, color='DarkBlue', linewidth=2)

    ax.set_title(
        "Linear regression between size and time "
        f"\n$mse={mqe:.3f}$ - $R^2={r2:.3f}$")
    ax.set_xlabel("Size")
    ax.set_ylabel("Seconds")

    return ax


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
            help="Execute and collect the regression benchmark of astroalign")
        benchmark.set_defaults(callback=self.benchmark_command)

        benchmark.add_argument(
            "--max", dest="max_size", type=int, default=max(SIZES),
            help=("The size in pixels of the bigger square image. "
                  f"(defaults={max(SIZES)})."))

        benchmark.add_argument(
            "--min", dest="min_size", type=int, default=min(SIZES),
            help=("The size in pixels of the smallest square image. "
                  f"(defaults={max(SIZES)})."))

        benchmark.add_argument(
            "--step", dest="step_size", type=int, default=STEP,
            help=f"The size between every image (defaults={STEP}).")

        benchmark.add_argument(
            "--stars", dest="stars", type=int, default=STARS,
            help=("The total numbers of stars in the image "
                  f"(defaults={STARS})."))

        benchmark.add_argument(
            "--noise", dest="noise", type=int, default=NOISE,
            help=f"lambda parameter for poisson noise (default={NOISE})")

        benchmark.add_argument(
            "--number", dest="comb_number", type=int, default=10,
            help=("How many random images pairs must be created for one "
                  f"size (default={COMB_NUMBER})."))

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

        print(f"Executed: {len(results)} cases")

        print(f"\twith {repetitions} repetitions \n")
        print(">>>>> Resume <<<<<")
        print(resume)
        print("")

    def benchmark_command(self, ns):
        if ns.step_size <= 0:
            self._parser.error(f"'step' must be > 0. Found {ns.step_size}")

        now = dt.datetime.now

        print(
            f"[{now()}] Starting benchmark for astroalign {aa.__version__}...")
        print("")
        results = benchmark(
            max_size=ns.max_size, min_size=ns.min_size, step_size=ns.step_size,
            stars=ns.stars, noise=ns.noise, seed=ns.seed,
            repeats=ns.repeats, n_jobs=ns.n_jobs, comb_number=ns.comb_number)

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
