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

import timeit
import datetime as dt
import argparse
from collections import OrderedDict

import numpy as np

import astroalign as aa

import pandas as pd

import sep 

import joblib

from sklearn.model_selection import ParameterGrid
from skimage.transform import SimilarityTransform
from scipy import stats 
import tqdm

from tests.test_align import simulate_image_single


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

def get_image(size, stars, noise, seed):
    """Retrieves a single image"""
    if seed is not None:
        np.random.seed(seed)
    shape = (size, size)
    image = simulate_image_single(
        shape=shape, num_stars=stars, noise_level=noise)[:2]
    return image


def get_parameters(sizes, stars, noises, comb_number, repeats, seed):
    """Create a list of dictionaries with all the combinations of the given
    parameters.

    """
    grid = ParameterGrid({
        "size": sizes, "stars": stars, "noise": noises})
    grid = list(grid) * comb_number

    # set the random state for run in paralell
    random = np.random.RandomState(seed)
    images_seeds = random.randint(1_000_000, size=len(grid))

    for idx, g in enumerate(grid):
        g["idx"] = idx
        g["seed"] = seed
        g["images_seed"] = images_seeds[idx]
        g["repeats"] = repeats

    return grid


def _test(size, stars, noise, seed, n_tests):
    # get image
    image = get_image(size, stars, noise, seed)
    imagedata = np.ascontiguousarray(image[0])
    # detect sources (we know where they are, actually)
    bkg = sep.Background(imagedata)
    thresh = 3. * bkg.globalrms
    sources = sep.extract(imagedata - bkg.back(), thresh)
    sources.sort(order='flux')
    
    # perform photometry
    flux, fluxerr, flag = sep.sum_circle(imagedata-bkg.back(), 
                                         sources['x'], sources['y'], 3.0, 
                                         err=bkg.globalrms, gain=1.0)

    dframes = []
    # transform it
    for i_trsf in range(n_tests):
        dx, dy = np.random.randint(low=-1*size//32, high=size//32, size=2)
        theta = (np.random.random()-0.5)*0.125*np.pi
        s = 0.85+np.random.random()*0.3
        trf = SimilarityTransform(translation=(dx, dy), rotation=theta, scale=s)
        
        target = np.zeros(shape=np.array(imagedata.shape)*2)
        newimage = aa.apply_transform(trf, imagedata-bkg.back(), target)

        # perform photometry on new places
        src_coords = np.array([sources['x'], sources['y']]).T
        new_coords = trf(src_coords).T
        nflux, nfluxerr, nflag = sep.sum_circle(newimage[0], new_coords[0], 
                                                new_coords[1], 3.0*s, 
                                                err=bkg.globalrms, 
                                                gain=1.0)
        
        # compare fluxes
        good_flux = nflag==0
        new_to_orig = nflux[good_flux]/flux[good_flux]

        # put everything in a pd dataframe
        df = pd.DataFrame()
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
        
        slp, intpt, r_val, p_val, std_err = stats.linregress(flux[good_flux],
                                                             nflux[good_flux])
        df['stats_slope'] = slp
        df['stats_intpt'] = intpt
        dframes.append(df)
    
    final_df = pd.concat(dframes)

    return final_df



