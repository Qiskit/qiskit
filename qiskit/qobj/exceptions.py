# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Error handling for Qobj."""
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

    def _shorten_message(self, message):
        if len(message) > 1000:
            return 'Original message too long to be useful: {}[...]'\
                   ''.format(message[:1000])

        return message


class QobjValidationError(QiskitError):
    """Represents an error during Qobj validation."""
    pass
