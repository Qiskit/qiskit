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
Deprecation functions for Qiskit Pulse. To be removed in Qiskit 2.0.
"""

from qiskit.utils.deprecation import deprecate_func, deprecate_arg


def deprecate_pulse_func(func):
    """Deprecation message for functions and classes"""
    return deprecate_func(
        since="1.3",
        removal_timeline="in Qiskit 2.0",
        additional_msg="The entire Qiskit Pulse package is being deprecated "
        "and will be moved to the Qiskit Dynamics repository: "
        "https://github.com/qiskit-community/qiskit-dynamics",
    )(func)


def deprecate_pulse_dependency(*args, **kwargs):
    """Deprecation message for functions and classes which use or depend on Pulse"""

    decorator = deprecate_func(
        since="1.3",
        removal_timeline="in Qiskit 2.0",
        additional_msg="The entire Qiskit Pulse package is being deprecated "
        "and this is a dependency on the package.",
        **kwargs,
    )

    # Taken when `deprecate_pulse_dependency` is used with no arguments and with empty parentheses,
    # in which case the decorated function is passed
    if args:
        return decorator(args[0])
    return decorator


def deprecate_pulse_arg(arg_name, **kwargs):
    """Deprecation message for arguments related to Pulse"""
    return deprecate_arg(
        name=arg_name,
        since="1.3",
        removal_timeline="in Qiskit 2.0",
        additional_msg="The entire Qiskit Pulse package is being deprecated "
        "and this argument uses a dependency on the package.",
        **kwargs,
    )
