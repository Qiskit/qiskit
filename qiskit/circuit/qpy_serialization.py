# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Alias for Qiskit QPY import."""


def __getattr__(name):
    import warnings
    from qiskit import qpy

    # Skip warning on special Python dunders, which Python occasionally queries on its own accord.
    if f"__{name[2:-2]}__" != name:
        warnings.warn(
            f"Module '{__name__}' is deprecated since Qiskit Terra 0.23,"
            " and will be removed in a future release. Please import from 'qiskit.qpy' instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
    return getattr(qpy, name)
