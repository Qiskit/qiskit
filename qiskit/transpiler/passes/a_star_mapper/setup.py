#!/usr/bin/env python
"""The A-star Mapper
"""

DOCLINES = __doc__.split('\n')

CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: BSD
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Operating System :: MacOS
Operating System :: POSIX
Operating System :: Unix
Operating System :: Microsoft :: Windows
"""

# import statements
import os
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

# all information about QuTiP goes here
MAJOR = 0
MINOR = 1
MICRO = 0
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
REQUIRES = ['cython (>=0.28)']
INSTALL_REQUIRES = ['cython>=0.28']
PACKAGES = ['a_star_mapper']
PACKAGE_DATA = {}

INCLUDE_DIRS = []
HEADERS = []
NAME = "a_star_mapper"
AUTHOR = ("")
AUTHOR_EMAIL = ("")
LICENSE = "BSD"
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
KEYWORDS = "cython"
CLASSIFIERS = [_f for _f in CLASSIFIERS.split('\n') if _f]
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]
EXT_MODULES = []
_compiler_flags = []
_link_args = []

# Add Cython extensions here
cy_exts = ['a_star_mapper']

# If on Win
if sys.platform == 'win32':
    _compiler_flags += ['/w', '/Ox']
# Everything else
else:
    _compiler_flags += ['-w', '-O3', '-march=native', '-funroll-loops']
    if sys.platform == 'darwin':
        _compiler_flags.append('-mmacosx-version-min=10.9')
        _link_args.append('-mmacosx-version-min=10.9')


# Add Cython files
for ext in cy_exts:
    _mod = Extension(ext,
            sources = [ext+'.pyx'],
            include_dirs = INCLUDE_DIRS,
            extra_compile_args=_compiler_flags,
            extra_link_args=_link_args,
            language='c++')
    EXT_MODULES.append(_mod)

# Remove -Wstrict-prototypes from cflags
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
if "CFLAGS" in cfg_vars:
    cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")


# Setup commands go here
setup(
    name = NAME,
    version = VERSION,
    packages = PACKAGES,
    include_package_data=True,
    include_dirs = INCLUDE_DIRS,
    headers = HEADERS,
    ext_modules = cythonize(EXT_MODULES),
    cmdclass = {'build_ext': build_ext},
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    license = LICENSE,
    description = DESCRIPTION,
    long_description = LONG_DESCRIPTION,
    keywords = KEYWORDS,
    classifiers = CLASSIFIERS,
    platforms = PLATFORMS,
    requires = REQUIRES,
    package_data = PACKAGE_DATA,
    zip_safe = False,
    install_requires=INSTALL_REQUIRES
)
