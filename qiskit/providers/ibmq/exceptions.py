# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Exception for errors raised by jobs.
"""
from qiskit.exceptions import QiskitError


class IBMQAccountError(QiskitError):
    """Base class for errors raised by account management."""
    pass


class IBMQBackendError(QiskitError):
    """IBM Q Backend Errors"""
    pass


class IBMQBackendValueError(IBMQBackendError, ValueError):
    """Value errors thrown within IBMQBackend """
    pass
