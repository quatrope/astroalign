# MIT License

# Copyright (c) 2016-2019 Martin Beroiz, Juan B. Cabral

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

from tests.test_align import simulate_image_pair

# =============================================================================
# CONSTANTS
# =============================================================================

SIZES = (256, 512, 768, 1024)

STARS = (300, 500, 1000, 10_000)

NOISES = (100, 500, 1000, 5000)

COMB_NUMBER = 10

STATEMENT = "aa.register(source, target)"

REPEATS = 50


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


def get_parameters(sizes, stars, noises, comb_number, seed, repeats):

    grid = ParameterGrid({
        "size": sizes, "stars": stars, "noise": noises})

    # set the random state for run in paralell
    if seed is not None:
        np.random.seed(42)

    grid = list(grid) * comb_number
    for idx, g in enumerate(grid):
        g["idx"] = idx
        g["seed"] = np.random.randint(1_000_000)
        g["repeats"] = repeats

    return grid


def _test(idx, size, stars, noise, seed, repeats):

    # create the two images
    source, target = get_images(size=size, stars=stars, noise=noise, seed=seed)

    # create the timer
    test_globals = {"aa": aa, "source": source, "target": target}
    timer = timeit.Timer(stmt=STATEMENT, globals=test_globals)

    # find the number of loops
    loops = timer.autorange()[0]

    # create a copy of the params to be returned ad result
    result = OrderedDict({
        "idx": idx, "size": size, "noise": noise,
        "seed": seed, "repeats": repeats, "loops": loops})

    # execute the timeit
    times = timer.repeat(repeats, loops)

    # store the times into the result
    result["time"] = np.min(np.array(times) / loops)
    for tidx, time in enumerate(times):
        result[f"time_{tidx}"] = time

    return result


def do_benchmark(sizes=SIZES, stars=STARS, noises=NOISES,
                 comb_number=10, seed=None, repeats=REPEATS, n_jobs=-1):

    grid = get_parameters(
        sizes=sizes, stars=stars, noises=noises,
        comb_number=comb_number, seed=seed, repeats=repeats)

    with joblib.Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(
            joblib.delayed(_test)(**params) for params in tqdm.tqdm(grid))

    df = pd.DataFrame(results)
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Astroalign benchmark tool based on timeit")

    parser.add_argument(
        '--version', action='version', version='%(prog)s 2019.10')

    parser.add_argument(
        "--sizes", dest="sizes", type=int, default=SIZES, nargs="+",
        help="")
    parser.add_argument(
        "--stars", dest="stars", type=int, default=STARS, nargs="+",
        help="")

    parser.add_argument(
        "--noises", dest="noises", type=int, default=SIZES, nargs="+",
        help="")

    parser.add_argument(
        "--cnumber", dest="comb_number", type=int, default=10,
        help="")

    parser.add_argument(
        "--seed", dest="seed", type=int, default=None,
        help="")

    parser.add_argument(
        "--repeats", dest="repeats", type=int, default=REPEATS,
        help="")

    parser.add_argument(
        "--jobs", dest="n_jobs", type=int, default=-1,
        help=("The number of CPU to run the tests. "
              "-1 uses all the available CPUS (default=-1)"))

    parser.add_argument(
        "--out", "-o", nargs='?', dest="out", required=True,
        type=argparse.FileType('w'))

    ns = parser.parse_args()

    now = dt.datetime.now

    print(f"[{now()}] Starting benchmark for astroalign {aa.__version__}...\n")
    results = do_benchmark(
        sizes=ns.sizes, stars=ns.stars, noises=ns.noises,
        comb_number=ns.comb_number, seed=ns.seed,
        repeats=ns.repeats, n_jobs=ns.n_jobs)

    print(f"[{now()}] Executed: {len(results)} cases")
    print(f"\twith {ns.repeats} repetitions \n")
    print(">>>>> Resume <<<<<")
    print(results[["time", "loops"]].describe())
    print("")

    results.to_csv(ns.out, index=False)
    print(f"[{now()}] Data stored in '{ns.out.name}'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    main()
