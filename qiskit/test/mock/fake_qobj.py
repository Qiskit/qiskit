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
Fake Boeblingen device (20 qubit).
"""

from .fake_qasm_simulator import FakeQasmSimulator
from qiskit.qobj import (QasmQobj, QobjExperimentHeader, QobjHeader,
                         QasmQobjInstruction, QasmQobjExperimentConfig,
                         QasmQobjExperiment, QasmQobjConfig, PulseLibraryItem,
                         PulseQobjInstruction)


class FakeQobj(QasmQobj):
    """A fake `Qobj` instance."""
    def __init__(self):
        self.qobj_id = 'test_id'
        self.config = QasmQobjConfig(shots=1024, memory_slots=1, max_credits=100)
        self.header = QobjHeader(backend_name=FakeQasmSimulator().name())
        self.experiments = [QasmQobjExperiment(
                instructions=[
                        QasmQobjInstruction(name='barrier', qubits=[1])
                        ],
                header=QobjExperimentHeader(),
                config=QasmQobjExperimentConfig(seed=123456)
                )]
