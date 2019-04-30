# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Error handling for jsonschema validation."""

from qiskit.exceptions import QiskitError


class SchemaValidationError(QiskitError):
    """Represents an error during JSON Schema validation."""
    pass


class _SummaryValidationError(QiskitError):
    """Cut off the message of a jsonschema.ValidationError for compactness.

    Cut off the message of a jsonschema.ValidationError to avoid printing
    noise in the standard output. The original validation error is in the
    `validation_error` property.

    Attributes:
        validation_error (jsonschema.ValidationError): original validations
            error.
    """

    def __init__(self, validation_error):
        super().__init__(self._shorten_message(str(validation_error)))
        self.validation_error = validation_error

    @staticmethod
    def _shorten_message(message):
        if len(message) > 1000:
            return 'Original message too long to be useful: {}[...]'\
                   ''.format(message[:1000])

        return message
