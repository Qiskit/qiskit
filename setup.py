# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Software for developing quantum computing programs"""

import os
import sys
import shutil
import distutils.sysconfig
import subprocess
import unittest
from setuptools import setup, Extension, Command
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from numpy.__config__ import get_info as np_config


NAME = "qiskit"
URL = "https://github.com/QISKit/qiskit-sdk-py"
AUTHOR = "QISKit Development Team"
AUTHOR_EMAIL = "qiskit@us.ibm.com"
LICENSE = "Apache 2.0"
MAJOR = 0
MINOR = 5
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
KEYWORDS = "qiskit sdk quantum"
PLATFORMS = ["Linux", "OSX", "Unix", "Windows"]
DOCLINES = __doc__.split('\n')
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
EXTRA_KWARGS = {}


def git_short_hash():
    """Get the git short hash.
    """
    try:
        git_str = "+" + os.popen('git log -1 --format="%h"').read().strip()
    except ValueError:
        git_str = ""
    else:
        if git_str == '+':  # fixes setuptools PEP issues with versioning
            git_str = ''
    return git_str


FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'+str(MICRO)+git_short_hash()

CLASSIFIERS = [
    "Environment :: Console",
    "License :: OSI Approved :: Apache Software License",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: MacOS",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Topic :: Scientific/Engineering"]

REQUIRES = [
    "IBMQuantumExperience (>=1.8.26)",
    "matplotlib (>=2.0,<=2.1)",
    "networkx (>=1.11,<1.12)",
    "numpy (>=1.13,<=1.14)",
    "ply (==3.10)",
    "scipy (>=0.19,<=1.0)",
    "Sphinx (>=1.6)",
    "sympy (>=1.0)"]

INSTALL_REQUIRES = REQUIRES

PACKAGES = ["qiskit",
            "qiskit.backends",
            "qiskit.dagcircuit",
            "qiskit.extensions",
            "qiskit.extensions.standard",
            "qiskit.extensions.qasm_simulator_cpp",
            "qiskit.extensions.quantum_initializer",
            "qiskit.mapper",
            "qiskit.qasm",
            "qiskit.qasm._node",
            "qiskit.unroll",
            "qiskit.tools",
            "qiskit.tools.apps",
            "qiskit.tools.qcvv",
            "qiskit.tools.qi",
            "qiskit.cython"]

PACKAGE_DATA = {}
HEADERS = []
EXT_MODULES = []
# Add Cython files
# Build options
include = [os.path.abspath('qiskit/cython/src'),
           os.path.abspath('qiskit/cython/src/backends'),
           os.path.abspath('qiskit/cython/src/engines'),
           os.path.abspath('qiskit/cython/src/utilities'),
           os.path.abspath('qiskit/cython/src/third-party'),
           os.path.abspath('qiskit/cython/src/third-party/headers')]
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

    # Add OMP flags if compiler explicitly set
    # and compiler not Apple's, i.e. in /usr/bin
    if ('CC' in os.environ.keys() and 'CXX' in os.environ.keys()):
        which_compiler = shutil.which(os.environ['CC'])
        if not which_compiler.startswith('/usr/bin'):
            extra_compile_args.append('-fopenmp')
            extra_link_args.append('-fopenmp')
elif sys.platform == 'win32':
    extra_compile_args.append('/openmp')
else:
    # Linux
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')

# Remove -Wstrict-prototypes from cflags
cfg_vars = distutils.sysconfig.get_config_vars()
if "CFLAGS" in cfg_vars:
    cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")

# Simulator extension
qasm_simulator = Extension('qiskit.cython.qasm_simulator',
                           sources=['qiskit/cython/qasm_simulator.pyx'],
                           extra_link_args=extra_link_args,
                           extra_compile_args=extra_compile_args,
                           libraries=libraries,
                           library_dirs=library_dirs,
                           include_dirs=include_dirs,
                           language='c++')

EXT_MODULES.append(qasm_simulator)

# Add command for running unittests from setup.py
class TestCommand(Command):
    """Run unittests from setup."""
    description = 'Run unittests from setup'
    user_options = [
        # The format is (long option, short option, description).
        ('abc', None, 'abc')]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run command."""
        command = 'python3 -m unittest discover -s test -v'
        subprocess.run(command, shell=True, stderr=subprocess.STDOUT)


# Add command for running pylint from setup.py
class PylintCommand(Command):
    """Run Pylint on all QISKit Python source files."""
    description = 'Run Pylint on QISKIT Python source files'
    user_options = [
      # The format is (long option, short option, description).
      ('pylint-rcfile=', None, 'path to Pylint config file')]

    def initialize_options(self):
        """Set default values for options."""
        # Each user option must be listed here with their default value.
        self.pylint_rcfile = ''

    def finalize_options(self):
        """Post-process options."""
        if self.pylint_rcfile:
            assert os.path.exists(self.pylint_rcfile), (
                'Pylint config file %s does not exist.' % self.pylint_rcfile)

    def run(self):
        """Run command."""
        command = ['pylint']
        if self.pylint_rcfile:
            command.append('--rcfile=%s' % self.pylint_rcfile)
        command.append(os.getcwd()+"/qiskit")
        subprocess.run(command, stderr=subprocess.STDOUT)


# Add command for running PEP8 tests from setup.py
class PEP8Command(Command):
    """Run pep8 from setup."""
    description = 'Run pep8 from setup'
    user_options = [
        # The format is (long option, short option, description).
        ('abc', None, 'abc')]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run command."""
        command = 'pycodestyle --max-line-length=100 qiskit test'
        subprocess.run(command, shell=True, stderr=subprocess.STDOUT)


# Add command for running profile from setup.py
class ProfileCommand(Command):
    """Run profile from setup."""
    description = 'Run profile from setup'
    user_options = [
        # The format is (long option, short option, description).
        ('abc', None, 'abc')]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run command."""
        command = 'python3 -m unittest discover -p "profile*.py" -v'
        subprocess.run(command, shell=True, stderr=subprocess.STDOUT)


# Add command for running coverage from setup.py
class CoverageCommand(Command):
    """Runcoverage from setup."""
    description = 'Run coverage from setup'
    user_options = [
        # The format is (long option, short option, description).
        ('abc', None, 'abc')]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run command."""
        command = 'coverage erase; coverage run --source qiskit -m unittest ' \
                'discover -s test -q; coverage html'
        subprocess.run(command, shell=True, stderr=subprocess.STDOUT)


# Setup commands go here
setup(
    name=NAME,
    version=FULLVERSION,
    packages=PACKAGES,
    include_package_data=True,
    include_dirs=include_dirs,
    headers=HEADERS,
    ext_modules=cythonize(EXT_MODULES),
    cmdclass={'build_ext': build_ext, 'test': TestCommand, 
              'lint': PylintCommand, 'pep8': PEP8Command,
              'profile': ProfileCommand,
              'coverage': CoverageCommand},
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    keywords=KEYWORDS,
    url=URL,
    classifiers=CLASSIFIERS,
    platforms=PLATFORMS,
    requires=REQUIRES,
    python_requires='>3.5.0',
    package_data=PACKAGE_DATA,
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
    **EXTRA_KWARGS
)
