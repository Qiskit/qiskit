# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Exceptions for errors raised by the validation."""


from marshmallow import ValidationError
from qiskit.exceptions import QiskitError


class ModelValidationError(ValidationError, QiskitError):
    """Raised when a sequence subscript is out of range."""
    def __init__(self, message, field_names=None, fields=None, data=None,
                 **kwargs):
        super().__init__(message, field_names, fields, data, **kwargs)
        # Populate self.message, as it is required by QiskitError.
        self.message = message
