import numpy
import scipy  # use numpy if scipy unavailable
import scipy.linalg  # use numpy if scipy unavailable

# Copyright (c) 2004-2007, Andrew D. Straw. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.

#     * Neither the name of the Andrew D. Straw nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# a PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__version__ = '0.1'


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """fit model parameters to data using the RANSAC algorithm

This implementation written from pseudocode found at
http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

{{{
Given:
    data - a set of observed data points
    model - a model that can be fitted to data points
    n - the minimum number of data values required to fit the model
    k - the maximum number of iterations allowed in the algorithm
    t - a threshold value for determining when a data point fits a model
    d - the number of close data values required to assert that a model fits
        well to data
Return:
    bestfit - model parameters which best fit the data (or nil if no good model
              is found)
iterations = 0
bestfit = nil
besterr = something really large
while iterations < k {
    maybeinliers = n randomly selected values from data
    maybemodel = model parameters fitted to maybeinliers
    alsoinliers = empty set
    for every point in data not in maybeinliers {
        if point fits maybemodel with an error smaller than t
             add point to alsoinliers
    }
    if the number of elements in alsoinliers is > d {
        % this implies that we may have found a good model
        % now test how good it is
        bettermodel = model parameters fitted to all points in maybeinliers and
        alsoinliers
        thiserr = a measure of how well model fits these points
        if thiserr < besterr {
            bestfit = bettermodel
            besterr = thiserr
        }
    }
    increment iterations
}
return bestfit
}}}
"""
    iterations = 0
    bestfit = None
    # besterr = numpy.inf
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybeinliers = data[maybe_idxs, :]
        test_points = data[test_idxs, :]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error(test_points, maybemodel)
        # select indices of rows with accepted points
        also_idxs = test_idxs[test_err < t]
        alsoinliers = data[also_idxs, :]
        if len(alsoinliers) > d:
            betterdata = numpy.concatenate((maybeinliers, alsoinliers))
            bestfit = model.fit(betterdata)
            # better_errs = model.get_error(betterdata, bettermodel)
            # thiserr = numpy.mean(better_errs)
            # if thiserr < besterr:
            # bestfit = bettermodel
            # besterr = thiserr
            best_inlier_idxs = numpy.concatenate((maybe_idxs, also_idxs))
            break
        iterations += 1
    if bestfit is None:
        raise ValueError("did not meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = numpy.arange(n_data)
    numpy.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


class LinearLeastSquaresModel:
    """linear system solved using linear least squares

    This class serves as an example that fulfills the model interface
    needed by the ransac() function.

    """
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        a = numpy.vstack([data[:, i] for i in self.input_columns]).T
        b = numpy.vstack([data[:, i] for i in self.output_columns]).T
        x, resids, rank, s = scipy.linalg.lstsq(a, b)
        return x

    def get_error(self, data, model):
        a = numpy.vstack([data[:, i] for i in self.input_columns]).T
        b = numpy.vstack([data[:, i] for i in self.output_columns]).T
        b_fit = scipy.dot(a, model)
        # sum squared error per row
        err_per_point = numpy.sum((b - b_fit) ** 2, axis=1)
        return err_per_point


def test():
    # generate perfect input data

    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    a_exact = 20 * numpy.random.random((n_samples, n_inputs))
    # the model
    perfect_fit = 60 * numpy.random.normal(size=(n_inputs, n_outputs))
    b_exact = scipy.dot(a_exact, perfect_fit)
    assert b_exact.shape == (n_samples, n_outputs)

    # add a little gaussian noise (linear least squares alone should handle
    # this well)
    a_noisy = a_exact + numpy.random.normal(size=a_exact.shape)
    b_noisy = b_exact + numpy.random.normal(size=b_exact.shape)

    # add some outliers
    n_outliers = 100
    all_idxs = numpy.arange(a_noisy.shape[0])
    numpy.random.shuffle(all_idxs)
    outlier_idxs = all_idxs[:n_outliers]
    non_outlier_idxs = all_idxs[n_outliers:]
    a_noisy[outlier_idxs] = 20 * numpy.random.random((n_outliers, n_inputs))
    b_noisy[outlier_idxs] = 50 * numpy.random.normal(size=(n_outliers,
                                                           n_outputs))

    # setup model
    all_data = numpy.hstack((a_noisy, b_noisy))
    input_columns = range(n_inputs)  # the first columns of the array
    # the last columns of the array
    output_columns = [n_inputs + i for i in range(n_outputs)]
    debug = False
    model = LinearLeastSquaresModel(input_columns, output_columns, debug=debug)

    linear_fit, resids, rank, s = \
        scipy.linalg.lstsq(all_data[:, input_columns],
                           all_data[:, output_columns])

    # run RANSAC algorithm
    ransac_fit, ransac_data = ransac(all_data, model,
                                     50, 1000, 7e3, 300,  # misc. parameters
                                     debug=debug, return_all=True)
    if 1:
        import pylab

        sort_idxs = numpy.argsort(a_exact[:, 0])
        a_col0_sorted = a_exact[sort_idxs]  # maintain as rank-2 array

        if 1:
            pylab.plot(a_noisy[:, 0], b_noisy[:, 0], 'k.', label='data')
            pylab.plot(a_noisy[ransac_data['inliers'], 0],
                       b_noisy[ransac_data['inliers'], 0], 'bx',
                       label='RANSAC data')
        else:
            pylab.plot(a_noisy[non_outlier_idxs, 0],
                       b_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(a_noisy[outlier_idxs, 0],
                       b_noisy[outlier_idxs, 0], 'r.', label='outlier data')
        pylab.plot(a_col0_sorted[:, 0],
                   numpy.dot(a_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        pylab.plot(a_col0_sorted[:, 0],
                   numpy.dot(a_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        pylab.plot(a_col0_sorted[:, 0],
                   numpy.dot(a_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()
        pylab.show()

if __name__ == '__main__':
    test()
