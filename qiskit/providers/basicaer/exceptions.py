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

"""
Exception for errors raised by Basic Aer.
"""

from qiskit.exceptions import QiskitError
from qiskit.utils.deprecation import deprecate_func


class BasicAerError(QiskitError):
    """Base class for errors raised by Basic Aer."""

    @deprecate_func(
        since="0.46.0",
        removal_timeline="in Qiskit 1.0.0",
        additional_msg="The qiskit.providers.basicaer module has been superseded "
        "by qiskit.providers.basic_provider. "
        "Use the new qiskit.providers.basic_provider.BasicProviderError class instead.",
    )
    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)
