# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Base Fake Qobj.
"""

from qiskit.qobj import (
    QasmQobj,
    QobjExperimentHeader,
    QobjHeader,
    QasmQobjInstruction,
    QasmQobjExperimentConfig,
    QasmQobjExperiment,
    QasmQobjConfig,
)

from .fake_qasm_simulator import FakeQasmSimulator


class FakeQobj(QasmQobj):
    """A fake `Qobj` instance."""

    def __init__(self):
        qobj_id = "test_id"
        config = QasmQobjConfig(shots=1024, memory_slots=1)
        header = QobjHeader(backend_name=FakeQasmSimulator().name())
        experiments = [
            QasmQobjExperiment(
                instructions=[QasmQobjInstruction(name="barrier", qubits=[1])],
                header=QobjExperimentHeader(),
                config=QasmQobjExperimentConfig(seed=123456),
            )
        ]
        super().__init__(qobj_id=qobj_id, config=config, experiments=experiments, header=header)
