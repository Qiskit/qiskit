#!/usr/bin/env python
"""The A-star Mapper
"""

# import statements
# import os
import sys
import distutils.sysconfig
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext


DOCLINES = __doc__.split("\n")

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

MAJOR = 0
MINOR = 1
MICRO = 0
VERSION = "%d.%d.%d" % (MAJOR, MINOR, MICRO)
REQUIRES = ["cython (>=0.28)"]
INSTALL_REQUIRES = ["cython>=0.28"]
PACKAGES = ["a_star_mapper"]
PACKAGE_DATA = {}

INCLUDE_DIRS = []
HEADERS = []
NAME = "a_star_mapper"
AUTHOR = ""
AUTHOR_EMAIL = ""
LICENSE = "BSD"
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
KEYWORDS = "cython"
CLASSIFIERS = [_f for _f in CLASSIFIERS.split("\n") if _f]
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]
EXT_MODULES = []
COMPILER_FLAGS = []
LINK_ARGS = []

# Add Cython extensions here
CY_EXTS = ["a_star_mapper"]

# If on Win
if sys.platform == "win32":
    COMPILER_FLAGS += ["/w", "/Ox"]
# Everything else
else:
    COMPILER_FLAGS += ["-w", "-O3", "-march=native", "-funroll-loops"]
    if sys.platform == "darwin":
        COMPILER_FLAGS.append("-mmacosx-version-min=10.9")
        LINK_ARGS.append("-mmacosx-version-min=10.9")


# Add Cython files
for ext in CY_EXTS:
    _mod = Extension(
        ext,
        sources=[ext + ".pyx"],
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=COMPILER_FLAGS,
        extra_link_args=LINK_ARGS,
        language="c++",
    )
    EXT_MODULES.append(_mod)

# Remove -Wstrict-prototypes from cflags

CFG_VARS = distutils.sysconfig.get_config_vars()
if "CFLAGS" in CFG_VARS:
    CFG_VARS["CFLAGS"] = CFG_VARS["CFLAGS"].replace("-Wstrict-prototypes", "")


# Setup commands go here
setup(
    name=NAME,
    version=VERSION,
    packages=PACKAGES,
    include_package_data=True,
    include_dirs=INCLUDE_DIRS,
    headers=HEADERS,
    ext_modules=cythonize(EXT_MODULES),
    cmdclass={"build_ext": build_ext},
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    keywords=KEYWORDS,
    classifiers=CLASSIFIERS,
    platforms=PLATFORMS,
    requires=REQUIRES,
    package_data=PACKAGE_DATA,
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
)
