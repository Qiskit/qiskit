# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
======================================
Base Objects (:mod:`qiskit.providers`)
======================================

.. currentmodule:: qiskit.providers

Base Objects
============

.. autosummary::
   :toctree: ../stubs/

   BaseProvider
   BaseBackend
   BaseJob

Job Status
==========

.. autosummary::
   :toctree: ../stubs/

   JobStatus

Exceptions
==========

.. autosummary::
   :toctree: ../stubs/

   QiskitBackendNotFoundError
   BackendPropertyError
   JobError
   JobTimeoutError
"""

import pkgutil

from .basebackend import BaseBackend
from .baseprovider import BaseProvider
from .basejob import BaseJob
from .exceptions import (JobError, JobTimeoutError, QiskitBackendNotFoundError,
                         BackendPropertyError, BackendConfigurationError)
from .jobstatus import JobStatus


# Allow extending this namespace.
__path__ = pkgutil.extend_path(__path__, __name__)
