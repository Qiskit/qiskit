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

"""
================================================
Backend Objects (:mod:`qiskit.providers.models`)
================================================

.. currentmodule:: qiskit.providers.models

Qiskit schema-conformant objects used by the backends and providers.

Backend Objects
===============

.. autosummary::
   :toctree: ../stubs/

   BackendConfiguration
   BackendProperties
   BackendStatus
   QasmBackendConfiguration
   PulseBackendConfiguration
   UchannelLO
   GateConfig
   PulseDefaults
   Command
   JobStatus
"""

from .backendconfiguration import (
    BackendConfiguration,
    PulseBackendConfiguration,
    QasmBackendConfiguration,
    UchannelLO,
    GateConfig,
)
from .backendproperties import BackendProperties
from .backendstatus import BackendStatus
from .jobstatus import JobStatus
from .pulsedefaults import PulseDefaults, Command
