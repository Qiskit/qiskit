# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=undefined-all-variable

"""Module containing transpiler synthesize."""

import importlib

__all__ = ["graysynth", "cnot_synth"]


def __getattr__(name):
    if name in __all__:
        return getattr(importlib.import_module(".graysynth", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
