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


# =============================================================================
# IMPORTS
# =============================================================================

import os

from ez_setup import use_setuptools

use_setuptools()

from setuptools import setup  # noqa


# =============================================================================
# PATH TO THIS MODULE
# =============================================================================

PATH = os.path.abspath(os.path.dirname(__file__))


# =============================================================================
# Get the version from astroalign file itself (not imported)
# =============================================================================

ASTROALIGN_PY_PATH = os.path.join(PATH, "astroalign.py")

with open(ASTROALIGN_PY_PATH, "r") as f:
    for line in f:
        if line.startswith("__version__"):
            _, _, AA_VERSION = line.replace('"', "").split()
            break


# =============================================================================
# RETRIEVE TE README
# =============================================================================

README_MD_PATH = os.path.join(PATH, "README.md")

with open(README_MD_PATH, "r") as f:
    LONG_DESCRIPTION = f.read()


# =============================================================================
# THE REQUIREMENTS!
# =============================================================================

REQUIREMENTS = ["numpy>=1.11", "scipy>=0.15", "scikit-image", "sep"]


# =============================================================================
# THE SETUP ITSELF!
# =============================================================================


def run():
    setup(
        name="astroalign",
        version=AA_VERSION,
        description="Astrometric Alignment of Images",
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author="Martin Beroiz",
        author_email="martinberoiz@gmail.com",
        url="https://astroalign.readthedocs.io/",
        py_modules=["astroalign", "ez_setup"],
        install_requires=REQUIREMENTS,
        test_suite="tests",
    )


if __name__ == "__main__":
    run()
