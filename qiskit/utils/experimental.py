# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
.. currentmodule:: qiskit.utils.experimental_warning

Qiskit's experimental APIs have to raise a user-visible warning. If your code raises this warning,
it might break at any minor release.
"""


class QiskitWarning(UserWarning):
    """General Qiskit Warning"""

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return str(self.message)


class ExperimentalQiskitAPI(QiskitWarning):
    """Warning for experimental APIs. If you see this warning, the API you are using has no stability
    guarantees and it might change or get removed at any minor Qiskit release"""

    def __init__(self, api_name):
        self.api_name = api_name
        super().__init__(
            f"Calling {self.api_name} is experimental and it might be changed or removed at "
            "any point"
        )


# TODO: add the decorators a la qiskit.utils.deprecation.*
