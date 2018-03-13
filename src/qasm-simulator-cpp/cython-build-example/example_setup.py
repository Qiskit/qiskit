import sys
import os
from distutils.core import setup, Extension
from distutils.spawn import find_executable
from Cython.Build import cythonize

# Build options
include = ['-I../src',
           '-I../src/backends',
           '-I../src/engines',
           '-I../src/utilities',
           '-I../src/third-party',
           '-I../src/third-party/headers']
warnings = ['-pedantic', '-Wall', '-Wextra', '-Wfloat-equal', '-Wundef',
            '-Wcast-align', '-Wwrite-strings', '-Wmissing-declarations',
            '-Wshadow', '-Woverloaded-virtual']
opt = ['-ffast-math', '-O3', '-march=native']
compile_args = ['-g', '-std=c++11'] + opt + include + warnings
link_args = []

# MacOS Specific build instructions
if sys.platform == 'darwin':

    # Set minimum os version to support C++11 headers
    min_macos_version = '-mmacosx-version-min=10.9'
    compile_args.append(min_macos_version)
    link_args.append(min_macos_version)

    # Link to Apple Accelerate framework for BLAS
    link_args += ['-Wl,-framework', '-Wl,Accelerate']

    # Check for OpenMP compatible GCC compiler
    for gcc in ['g++-7', 'g++-6', 'g++-5']:
        path = find_executable(gcc)
        if path is not None:
            # Use most recent GCC compiler
            os.environ['CC'] = path
            os.environ['CXX'] = path
            compile_args.append('-fopenmp')
            link_args.append('-fopenmp')
            break
else:
    compile_args.append('-fopenmp')
    link_args += ['-fopenmp', '-llapack', '-lblas']
    # TODO Check BLAS linkage for Windows and Linux
    # TODO Link to Anaconda MKL if available

# Simulator extension
qasm_simulator = Extension('qasm_simulator',
                           sources=['qasm_simulator.pyx'],
                           extra_link_args=link_args,
                           extra_compile_args=compile_args,
                           language='c++')

setup(
    name='qasm_simulator',
    packages=['qasm_simulator'],
    ext_modules=cythonize(qasm_simulator)
)
