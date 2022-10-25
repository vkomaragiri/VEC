#author: Vasundhara Komaragiri

from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy

# define an extension that will be cythonized and compiled
setup(name='TPM',
      ext_modules = cythonize([
      Extension("Variable", ["Variable.pyx"], include_dirs=[numpy.get_include()], language="c"),# extra_compile_args=["-std=c++14"]), 
      Extension("Util", ["Util.pyx"], include_dirs=[numpy.get_include()], language="c"),# extra_compile_args=["-std=c++14"]), 
      Extension("Function", ["Function.pyx"], include_dirs=[numpy.get_include()], language="c"),# extra_compile_args=["-std=c++14"]),
      Extension("MN", ["MN.pyx"], include_dirs=[numpy.get_include()], language="c"),#, extra_compile_args=["-std=c++14"]),
      Extension("BTP", ["BTP.pyx"], include_dirs=[numpy.get_include()], language="c"),#++", extra_compile_args=["-std=c++14"]),
      ])
)
