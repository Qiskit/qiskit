# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import os

from skbuild import setup
from setuptools import find_packages
import unittest

requirements = [
    "jsonschema>=2.6,<2.7",
    "IBMQuantumExperience>=1.9.8",
    "matplotlib>=2.1",
    "networkx>=2.0",
    "numpy>=1.13",
    "ply>=3.10",
    "scipy>=0.19",
    "sympy>=1.0",
    "pillow>=4.2.1",
    "scikit-build>=0.8"
]


def load_version():
    qiskit_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(qiskit_dir, "qiskit" + os.path.sep + "VERSION.txt"), "r") as version_file:
        return version_file.read().strip()

setup(
    name="qiskit",
    version=load_version(),
    description="Software for developing quantum computing programs",
    long_description="""Qiskit is a software development kit for writing
        quantum computing experiments, programs, and applications. Works with
        Python 3.5 and 3.6""",
    url="https://github.com/Qiskit/qiskit-terra",
    author="Qiskit Development Team",
    author_email="qiskit@us.ibm.com",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum",
    packages=find_packages(exclude=['test*']),
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.5",
    test_suite="test"
)
