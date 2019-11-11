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

"""
=========================
Qobj (:mod:`qiskit.qobj`)
=========================

.. currentmodule:: qiskit.qobj

Base
====

.. autosummary::
   :toctree: ../stubs/

   Qobj
   QobjInstruction
   QobjExperimentHeader
   QobjExperimentConfig
   QobjExperiment
   QobjConfig
   QobjHeader

Qasm
====

.. autosummary::
   :toctree: ../stubs/

   QasmQobj
   QasmQobjInstruction
   QasmQobjExperimentConfig
   QasmQobjExperiment
   QasmQobjConfig

Pulse
=====

.. autosummary::
   :toctree: ../stubs/

   PulseQobj
   PulseQobjInstruction
   PulseQobjExperimentConfig
   PulseQobjExperiment
   PulseQobjConfig
   QobjMeasurementOption
   PulseLibraryItem
   PulseLibraryItemSchema
   PulseQobjInstructionSchema

Validation
==========

.. autosummary::
   :toctree: ../stubs/

   validate_qobj_against_schema
"""

from .models.base import (QobjInstruction, QobjExperimentHeader, QobjExperimentConfig,
                          QobjExperiment, QobjConfig, QobjHeader)

from .models.pulse import (PulseQobjInstruction, PulseQobjExperimentConfig,
                           PulseQobjExperiment, PulseQobjConfig,
                           QobjMeasurementOption, PulseLibraryItem,
                           PulseLibraryItemSchema, PulseQobjInstructionSchema)

from .models.qasm import (QasmQobjInstruction, QasmQobjExperimentConfig,
                          QasmQobjExperiment, QasmQobjConfig)

from .qobj import Qobj, QasmQobj, PulseQobj

from .utils import validate_qobj_against_schema
