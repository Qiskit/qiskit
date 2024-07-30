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

Classes
=======

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
   GateProperties
   Nduv
"""
import warnings

from .backendconfiguration import (
    BackendConfiguration,
    PulseBackendConfiguration,
    QasmBackendConfiguration,
    UchannelLO,
    GateConfig,
)
from .backendproperties import BackendProperties, GateProperties, Nduv
from .backendstatus import BackendStatus
from .jobstatus import JobStatus
from .pulsedefaults import PulseDefaults, Command


warnings.warn(
    "qiskit.providers.models is deprecated since Qiskit 1.2 and will be removed in Qiskit 2.0."
    "With the removal of Qobj, there is no need for these schema-conformant objects. If you still need"
    "to use them, it could be because you are using a BackendV1, which is also deprecated in favor"
    "of BackendV2",
    DeprecationWarning,
    2,
)
