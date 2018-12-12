# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Backends provided by IBM Quantum Experience."""
from qiskit.qiskiterror import QiskitError
from .ibmqprovider import IBMQProvider
from .ibmqbackend import IBMQBackend
from .ibmqjob import IBMQJob

# Global instance to be used as the entry point for convenience.
IBMQ = IBMQProvider()


def least_busy(backends):
    """
    Return the least busy available backend for those that
    have a `pending_jobs` in their `status`. Backends such as
    local backends that do not have this are not considered.

    Args:
        backends (list[BaseBackend]): backends to choose from

    Returns:
        BaseBackend: the the least busy backend

    Raises:
        QiskitError: if passing a list of backend names that is
            either empty or none have attribute ``pending_jobs``
    """
    try:
        return min([b for b in backends if b.status().operational],
                   key=lambda b: b.status().pending_jobs)
    except (ValueError, TypeError):
        raise QiskitError("Can only find least_busy backend from a non-empty list.")
