#!/usr/bin/env python3

# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Utility script to invoke cargo test within the current Python environment.

Notably, this sets up the environment variables necessary for Qiskit to be
found by PyO3 / our Rust test executable.
"""

import os
import subprocess
import site
import sys
import sysconfig

# This allows the Python interpreter baked into our test executable to find the
# Qiskit installed in the active environment.
os.environ["PYTHONPATH"] = os.pathsep.join([os.getcwd()] + site.getsitepackages())

# Uncomment to debug PyO3's build / link against Python.
# os.environ["PYO3_PRINT_CONFIG"] = "1"

# On Linux, the test executable's RPATH doesn't contain libpython, so we add it
# to the dlopen search path here.
os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
    filter(None, [sysconfig.get_config_var("LIBDIR"), os.getenv("LD_LIBRARY_PATH")])
)

# The '--no-default-features' flag is used here to disable PyO3's
# 'extension-module' when running the tests (which would otherwise cause link
# errors).
subprocess.run(["cargo", "test", "--no-default-features"] + sys.argv[1:], check=True)
