from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = [Extension('utils',sources=["utils.pyx"], include_dirs=[numpy.get_include()])]

setup(
  name = 'utils',
    ext_modules = cythonize(ext)
)
