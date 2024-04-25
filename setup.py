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

"The Qiskit setup file."

import os
from setuptools import setup
from setuptools_rust import Binding, RustExtension

# Most of this configuration is managed by `pyproject.toml`.  This only includes the extra bits to
# configure `setuptools-rust`, because we do a little dynamic trick with the debug setting, and we
# also want an explicit `setup.py` file to exist so we can manually call
#
#   python setup.py build_rust --inplace --release
#
# to make optimised Rust components even for editable releases, which would otherwise be quite
# unergonomic to do otherwise.


# If RUST_DEBUG is set, force compiling in debug mode. Else, use the default behavior of whether
# it's an editable installation.
rust_debug = True if os.getenv("RUST_DEBUG") == "1" else None

setup(
    rust_extensions=[
        RustExtension(
            "qiskit._accelerate",
            "crates/pyext/Cargo.toml",
            binding=Binding.PyO3,
            debug=rust_debug,
        )
    ],
    options={"bdist_wheel": {"py_limited_api": "cp38"}},
)
