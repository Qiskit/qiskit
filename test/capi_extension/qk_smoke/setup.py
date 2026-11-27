# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import sys

from setuptools import Extension, setup

import qiskit.capi

extra_compile_args = []
if sys.platform != "win32":
    extra_compile_args.extend(["-std=c11", "-Wall", "-Wextra", "-Wno-unused-parameter", "-Werror"])

setup(
    ext_modules=[
        Extension(
            name="qk_smoke._core",
            sources=["src/qk_smoke/_coremodule.c"],
            include_dirs=[qiskit.capi.get_include()],
            extra_compile_args=extra_compile_args,
        )
    ]
)
