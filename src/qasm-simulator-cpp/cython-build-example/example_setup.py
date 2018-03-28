import sys
import os
from distutils.core import setup, Extension
from distutils.spawn import find_executable
from Cython.Build import cythonize
from numpy.__config__ import get_info as np_config

# Build options
include = [os.path.abspath('../src'),
           os.path.abspath('../src/backends'),
           os.path.abspath('../src/engines'),
           os.path.abspath('../src/utilities'),
           os.path.abspath('../src/third-party'),
           os.path.abspath('../src/third-party/headers')]
warnings = ['-pedantic', '-Wall', '-Wextra', '-Wfloat-equal', '-Wundef',
            '-Wcast-align', '-Wwrite-strings', '-Wmissing-declarations',
            '-Wshadow', '-Woverloaded-virtual']
opt = ['-ffast-math', '-O3', '-march=native']
if sys.platform != 'win32':
    extra_compile_args = ['-g', '-std=c++11'] + opt + warnings
else:
    extra_compile_args = ['/W1', '/Ox']

extra_link_args = []
libraries = []
library_dirs = []
include_dirs = include

# Numpy BLAS
blas_info = np_config('blas_mkl_info')
if blas_info == {}:
    blas_info = np_config('blas_opt_info')
extra_compile_args += blas_info.get('extra_compile_args', [])
extra_link_args += blas_info.get('extra_link_args', [])
libraries += blas_info.get('libraries', [])
library_dirs += blas_info.get('library_dirs', [])
include_dirs += blas_info.get('include_dirs', [])

# MacOS Specific build instructions
if sys.platform == 'darwin':
    # Set minimum os version to support C++11 headers
    min_macos_version = '-mmacosx-version-min=10.9'
    extra_compile_args.append(min_macos_version)
    extra_link_args.append(min_macos_version)

    # Check for OpenMP compatible GCC compiler
    for gcc in ['g++-7', 'g++-6', 'g++-5']:
        path = find_executable(gcc)
        if path is not None:
            # Use most recent GCC compiler
            os.environ['CC'] = path
            os.environ['CXX'] = path
            extra_compile_args.append('-fopenmp')
            extra_link_args.append('-fopenmp')
            break
elif sys.platform == 'win32':
    extra_compile_args.append('/openmp')
else:
    # Linux
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')

# Remove -Wstrict-prototypes from cflags
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
if "CFLAGS" in cfg_vars:
    cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")

# Simulator extension
qasm_simulator = Extension('qasm_simulator',
                           sources=['qasm_simulator.pyx'],
                           extra_link_args=extra_link_args,
                           extra_compile_args=extra_compile_args,
                           libraries=libraries,
                           library_dirs=library_dirs,
                           include_dirs=include_dirs,
                           language='c++')

setup(
    name='qasm_simulator',
    packages=['qasm_simulator'],
    ext_modules=cythonize(qasm_simulator)
)
