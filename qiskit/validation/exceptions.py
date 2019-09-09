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


class ModelValidationError(QiskitError, ValidationError):
    """Raised when a sequence subscript is out of range."""
    def __init__(self, message, field_name=None, data=None, valid_data=None,
                 **kwargs):
        # Call ValidationError.__init__ manually, as the signatures differ.
        ValidationError.__init__(self, message, field_name, data, valid_data, **kwargs)
        super().__init__(str(message))
