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
        # pylint: disable=super-init-not-called
        # ValidationError.__init__ is called manually instead of calling super,
        # as the signatures of ValidationError and QiskitError constructors
        # differ.
        ValidationError.__init__(self, message, field_name, data, valid_data, **kwargs)
        self.message = str(message)
