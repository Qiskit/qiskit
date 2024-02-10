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
from setuptools import setup, find_packages
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

# If RUST_DEBUG is set, force compiling in debug mode. Else, use the default behavior of whether
# it's an editable installation.
rust_debug = True if os.getenv("RUST_DEBUG") == "1" else None

# If modifying these optional extras, make sure to sync with `requirements-optional.txt` and
# `qiskit.utils.optionals` as well.
qasm3_import_extras = [
    "qiskit-qasm3-import>=0.1.0",
]
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
csp_requirements = ["python-constraint>=1.4"]


setup(
    name="qiskit-terra",
    version="0.46.0",
    description="Software for developing quantum computing programs",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://www.ibm.com/quantum/qiskit",
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
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum",
    packages=find_packages(exclude=["test*"]),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    python_requires=">=3.8",
    extras_require={
        "qasm3-import": qasm3_import_extras,
        "visualization": visualization_extras,
        "crosstalk-pass": z3_requirements,
        "csp-layout-pass": csp_requirements,
        "all": visualization_extras + z3_requirements + csp_requirements + qasm3_import_extras,
    },
    project_urls={
        "Documentation": "https://docs.quantum.ibm.com",
        "API Reference": "https://docs.quantum.ibm.com/api/qiskit",
        "Repository": "https://github.com/Qiskit/qiskit",
        "Issues": "https://github.com/Qiskit/qiskit/issues",
        "Changelog": "https://docs.quantum.ibm.com/api/qiskit/release-notes",
    },
    rust_extensions=[
        RustExtension(
            "qiskit._accelerate",
            "crates/accelerate/Cargo.toml",
            binding=Binding.PyO3,
            debug=rust_debug,
        ),
        RustExtension(
            "qiskit._qasm2",
            "crates/qasm2/Cargo.toml",
            binding=Binding.PyO3,
            debug=rust_debug,
        ),
    ],
    options={"bdist_wheel": {"py_limited_api": "cp38"}},
    zip_safe=False,
    entry_points={
        "qiskit.unitary_synthesis": [
            "default = qiskit.transpiler.passes.synthesis.unitary_synthesis:DefaultUnitarySynthesis",
            "aqc = qiskit.transpiler.passes.synthesis.aqc_plugin:AQCSynthesisPlugin",
            "sk = qiskit.transpiler.passes.synthesis.solovay_kitaev_synthesis:SolovayKitaevSynthesis",
        ],
        "qiskit.synthesis": [
            "clifford.default = qiskit.transpiler.passes.synthesis.high_level_synthesis:DefaultSynthesisClifford",
            "clifford.ag = qiskit.transpiler.passes.synthesis.high_level_synthesis:AGSynthesisClifford",
            "clifford.bm = qiskit.transpiler.passes.synthesis.high_level_synthesis:BMSynthesisClifford",
            "clifford.greedy = qiskit.transpiler.passes.synthesis.high_level_synthesis:GreedySynthesisClifford",
            "clifford.layers = qiskit.transpiler.passes.synthesis.high_level_synthesis:LayerSynthesisClifford",
            "clifford.lnn = qiskit.transpiler.passes.synthesis.high_level_synthesis:LayerLnnSynthesisClifford",
            "linear_function.default = qiskit.transpiler.passes.synthesis.high_level_synthesis:DefaultSynthesisLinearFunction",
            "linear_function.kms = qiskit.transpiler.passes.synthesis.high_level_synthesis:KMSSynthesisLinearFunction",
            "linear_function.pmh = qiskit.transpiler.passes.synthesis.high_level_synthesis:PMHSynthesisLinearFunction",
            "permutation.default = qiskit.transpiler.passes.synthesis.high_level_synthesis:BasicSynthesisPermutation",
            "permutation.kms = qiskit.transpiler.passes.synthesis.high_level_synthesis:KMSSynthesisPermutation",
            "permutation.basic = qiskit.transpiler.passes.synthesis.high_level_synthesis:BasicSynthesisPermutation",
            "permutation.acg = qiskit.transpiler.passes.synthesis.high_level_synthesis:ACGSynthesisPermutation",
        ],
        "qiskit.transpiler.init": [
            "default = qiskit.transpiler.preset_passmanagers.builtin_plugins:DefaultInitPassManager",
        ],
        "qiskit.transpiler.translation": [
            "translator = qiskit.transpiler.preset_passmanagers.builtin_plugins:BasisTranslatorPassManager",
            "unroller = qiskit.transpiler.preset_passmanagers.builtin_plugins:UnrollerPassManager",
            "synthesis = qiskit.transpiler.preset_passmanagers.builtin_plugins:UnitarySynthesisPassManager",
        ],
        "qiskit.transpiler.routing": [
            "basic = qiskit.transpiler.preset_passmanagers.builtin_plugins:BasicSwapPassManager",
            "stochastic = qiskit.transpiler.preset_passmanagers.builtin_plugins:StochasticSwapPassManager",
            "lookahead = qiskit.transpiler.preset_passmanagers.builtin_plugins:LookaheadSwapPassManager",
            "sabre = qiskit.transpiler.preset_passmanagers.builtin_plugins:SabreSwapPassManager",
            "none = qiskit.transpiler.preset_passmanagers.builtin_plugins:NoneRoutingPassManager",
        ],
        "qiskit.transpiler.optimization": [
            "default = qiskit.transpiler.preset_passmanagers.builtin_plugins:OptimizationPassManager",
        ],
        "qiskit.transpiler.layout": [
            "default = qiskit.transpiler.preset_passmanagers.builtin_plugins:DefaultLayoutPassManager",
            "trivial = qiskit.transpiler.preset_passmanagers.builtin_plugins:TrivialLayoutPassManager",
            "dense = qiskit.transpiler.preset_passmanagers.builtin_plugins:DenseLayoutPassManager",
            "noise_adaptive = qiskit.transpiler.preset_passmanagers.builtin_plugins:NoiseAdaptiveLayoutPassManager",
            "sabre = qiskit.transpiler.preset_passmanagers.builtin_plugins:SabreLayoutPassManager",
        ],
        "qiskit.transpiler.scheduling": [
            "alap = qiskit.transpiler.preset_passmanagers.builtin_plugins:AlapSchedulingPassManager",
            "asap = qiskit.transpiler.preset_passmanagers.builtin_plugins:AsapSchedulingPassManager",
            "default = qiskit.transpiler.preset_passmanagers.builtin_plugins:DefaultSchedulingPassManager",
        ],
    },
)
