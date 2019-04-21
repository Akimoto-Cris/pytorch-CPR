from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import math
import cv2
import matplotlib
import numpy as np
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

setup(
    name="post",
    ext_modules=cythonize(
        (Extension("post",
                   sources=["evaluation/post.pyx"],
                   language="c++",
                   include_dirs=[np.get_include()]
                   ))
    ),

)