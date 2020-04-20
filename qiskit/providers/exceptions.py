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

"""Exceptions for errors raised while handling Backends and Jobs."""

from qiskit.exceptions import QiskitError


class JobError(QiskitError):
    """Base class for errors raised by Jobs."""

    pass


class JobTimeoutError(JobError):
    """Base class for timeout errors raised by jobs."""

    pass


class QiskitBackendNotFoundError(QiskitError):
    """Base class for errors raised while looking for a backend."""

    pass


class BackendPropertyError(QiskitError):
    """Base class for errors raised while looking for a backend property."""

    pass


class BackendConfigurationError(QiskitError):
    """Base class for errors raised by the BackendConfiguration."""

    pass
