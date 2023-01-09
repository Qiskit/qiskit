# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"The Qiskit Terra setup file."

import os
import re
import sys
from setuptools import setup, find_packages, Extension
from setuptools_rust import Binding, RustExtension


with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH) as readme_file:
    README = re.sub(
        "<!--- long-description-skip-begin -->.*<!--- long-description-skip-end -->",
        "",
        readme_file.read(),
        flags=re.S | re.M,
    )


visualization_extras = [
    "matplotlib>=3.3",
    "ipywidgets>=7.3.0",
    "pydot",
    "pillow>=4.2.1",
    "pylatexenc>=1.4",
    "seaborn>=0.9.0",
    "pygments>=2.4",
]
z3_requirements = [
    "z3-solver>=4.7",
]
bip_requirements = ["cplex", "docplex"]
csp_requirements = ["python-constraint>=1.4"]
toqm_requirements = ["qiskit-toqm>=0.1.0"]

setup(
    name="qiskit-terra",
    version="0.23.0",
    description="Software for developing quantum computing programs",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Qiskit/qiskit-terra",
    author="Qiskit Development Team",
    author_email="hello@qiskit.org",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum",
    packages=find_packages(exclude=["test*"]),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    python_requires=">=3.7",
    extras_require={
        "visualization": visualization_extras,
        "bip-mapper": bip_requirements,
        "crosstalk-pass": z3_requirements,
        "csp-layout-pass": csp_requirements,
        "toqm": toqm_requirements,
        # Note: 'all' only includes extras that are stable and work on the majority of Python
        # versions and OSes supported by Terra. You have to ask for anything else explicitly.
        "all": visualization_extras + z3_requirements + csp_requirements,
    },
    project_urls={
        "Bug Tracker": "https://github.com/Qiskit/qiskit-terra/issues",
        "Documentation": "https://qiskit.org/documentation/",
        "Source Code": "https://github.com/Qiskit/qiskit-terra",
    },
    rust_extensions=[RustExtension("qiskit._accelerate", "Cargo.toml", binding=Binding.PyO3)],
    zip_safe=False,
    entry_points={
        "qiskit.unitary_synthesis": [
            "default = qiskit.transpiler.passes.synthesis.unitary_synthesis:DefaultUnitarySynthesis",
            "aqc = qiskit.transpiler.synthesis.aqc.aqc_plugin:AQCSynthesisPlugin",
            "sk = qiskit.transpiler.passes.synthesis.solovay_kitaev_synthesis:SolovayKitaevSynthesis",
        ],
        "qiskit.synthesis": [
            "clifford.default = qiskit.transpiler.passes.synthesis.high_level_synthesis:DefaultSynthesisClifford",
            "linear_function.default = qiskit.transpiler.passes.synthesis.high_level_synthesis:DefaultSynthesisLinearFunction",
        ],
        "qiskit.transpiler.routing": [
            "basic = qiskit.transpiler.preset_passmanagers.builtin_plugins:BasicSwapPassManager",
            "stochastic = qiskit.transpiler.preset_passmanagers.builtin_plugins:StochasticSwapPassManager",
            "lookahead = qiskit.transpiler.preset_passmanagers.builtin_plugins:LookaheadSwapPassManager",
            "sabre = qiskit.transpiler.preset_passmanagers.builtin_plugins:SabreSwapPassManager",
            "none = qiskit.transpiler.preset_passmanagers.builtin_plugins:NoneRoutingPassManager",
        ],
    },
)
