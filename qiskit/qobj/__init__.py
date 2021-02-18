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
   QobjExperimentHeader
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
   QasmExperimentCalibrations
   GateCalibration

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
"""

import warnings

from qiskit.qobj.common import QobjExperimentHeader
from qiskit.qobj.common import QobjHeader

from qiskit.qobj.pulse_qobj import PulseQobj
from qiskit.qobj.pulse_qobj import PulseQobjInstruction
from qiskit.qobj.pulse_qobj import PulseQobjExperimentConfig
from qiskit.qobj.pulse_qobj import PulseQobjExperiment
from qiskit.qobj.pulse_qobj import PulseQobjConfig
from qiskit.qobj.pulse_qobj import QobjMeasurementOption
from qiskit.qobj.pulse_qobj import PulseLibraryItem

from qiskit.qobj.qasm_qobj import GateCalibration
from qiskit.qobj.qasm_qobj import QasmExperimentCalibrations
from qiskit.qobj.qasm_qobj import QasmQobj
from qiskit.qobj.qasm_qobj import QasmQobjInstruction
from qiskit.qobj.qasm_qobj import QasmQobjExperiment
from qiskit.qobj.qasm_qobj import QasmQobjConfig
from qiskit.qobj.qasm_qobj import QasmQobjExperimentConfig

from .utils import validate_qobj_against_schema


class Qobj(QasmQobj):
    """A backwards compat alias for QasmQobj."""

    def __init__(self, qobj_id=None, config=None, experiments=None, header=None):
        """Initialize a Qobj object."""
        warnings.warn(
            "qiskit.qobj.Qobj is deprecated use either QasmQobj or "
            "PulseQobj depending on your application instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(qobj_id=qobj_id, config=config, experiments=experiments, header=header)
