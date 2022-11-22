# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Exception for errors raised by Opflow module."""

from qiskit.exceptions import QiskitError
from qiskit.utils.deprecation import deprecate_function


class OpflowError(QiskitError):
    """Deprecation: For Opflow specific errors."""

    @deprecate_function(
        "The OpflowError opflow class is deprecated as of Qiskit Terra 0.23.0 "
        "and will be removed no sooner than 3 months after the release date. "
    )
    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
