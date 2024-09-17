# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Deprecation functions for Qiskit Pulse
"""

import warnings
from functools import wraps
from qiskit.utils.deprecation import deprecate_func


def deprecate_pulse_func(func):
    """Deprecation message for functions and classes"""
    return deprecate_func(
        since="1.3",
        removal_timeline="in Qiskit 2.0",
        additional_msg="This is part of the entire Qiskit Pulse package deprecation. "
        "The package will be moved to the Qiskit Dynamics repository: "
        "https://github.com/qiskit-community/qiskit-dynamics/",
    )(func)


def ignore_pulse_deprecation_warnings(func):
    """Filter Pulse deprecation warnings in a decorated function"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=DeprecationWarning, message=".*Qiskit Pulse package"
            )
            return func(*args, **kwargs)

    return wrapper
