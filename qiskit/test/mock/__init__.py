# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Utilities for mocking the IBMQ provider, including job responses and backends.

The module includes dummy provider, backends, and jobs.
The purpose of these classes is to fake backends for testing purposes:
testing local timeouts, arbitrary responses or behavior, etc.

The mock devices are mainly for testing the compiler.
"""

import warnings

import qiskit.providers.fake_provider


def __getattr__(name):
    if name.startswith("_"):
        # Some Python components (including tests) do funny things with dunders.
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    warnings.warn(
        f"The module '{__name__}' is deprecated since "
        "Qiskit Terra 0.21.0, and will be removed 3 months or more later. "
        "Instead, you should import the desired object directly 'qiskit.providers.fake_provider'.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return getattr(qiskit.providers.fake_provider, name)
