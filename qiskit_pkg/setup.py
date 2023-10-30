# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# This file is the setup.py file for the qiskit package. Because python
# packaging doesn't offer a mechanism to have qiskit supersede qiskit-terra
# and cleanly upgrade from one to the other, there needs to be a separate
# package shim to ensure no matter how people installed qiskit < 0.45.0 the
# upgrade works.

import os

from setuptools import setup

README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = readme_file.read()

requirements = ["qiskit-terra==1.0.0"]

setup(
    name="qiskit",
    version="1.0.0",
    description="Software for developing quantum computing programs",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://qiskit.org/",
    author="Qiskit Development Team",
    author_email="hello@qiskit.org",
    license="Apache 2.0",
    py_modules=[],
    packages=[],
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum",
    install_requires=requirements,
    project_urls={
        "Bug Tracker": "https://github.com/Qiskit/qiskit/issues",
        "Documentation": "https://qiskit.org/documentation/",
        "Source Code": "https://github.com/Qiskit/qiskit",
    },
    include_package_data=True,
    python_requires=">=3.8",
    extras_require={
        "qasm3-import": ["qiskit-terra[qasm3-import]"],
        "visualization": ["qiskit-terra[visualization]"],
        "crosstalk-pass": ["qiskit-terra[crosstalk-pass]"],
        "csp-layout-pass": ["qiskit-terra[csp-layout-pass]"],
        "all": ["qiskit-terra[all]"],
    },
)
