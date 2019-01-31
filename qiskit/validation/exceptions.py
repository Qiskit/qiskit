# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

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
