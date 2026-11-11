# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest
from typing import Any

from qiskit import QuantumCircuit
from qiskit._accelerate import mir, mir_passes
from qiskit.passmanager import LoweringPassManager
from qiskit.transpiler import generate_preset_pbc_pass_manager
from qiskit.transpiler.passes import LitinskiTransformation
from test import QiskitTestCase
from test.python.mir.pass_adaptor import CircuitToDag, LegacyDagPass


def make_bell_circuit():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


class TestBellCircuitMIR(QiskitTestCase):
    """Unit tests for lowering Bell circuit to MIR."""

    def test_bell_circuit(self):
        """Test lowering a Bell circuit to MIR after the preset PBC passes."""
        qc = make_bell_circuit()
        pm = LoweringPassManager(
            [CircuitToDag()]
            + [
                LegacyDagPass(pass_)
                for pass_ in generate_preset_pbc_pass_manager().to_flow_controller().tasks
            ]
            + [mir_passes.DagToMirPass()]
        )
        mir_instance: Any = pm.run(qc)
        self.log.debug(f"mir_instance=\n{mir_instance}")
        self.assertIsInstance(mir_instance, mir.Mir)
        self.assertEqual(mir_instance.num_instructions(), 7)
        self.assertEqual(mir_instance.num_qids(), 2)

    def test_bell_circuit_with_litinski(self):
        """Test lowering a Bell circuit to MIR with the Litinski transformation."""
        qc = make_bell_circuit()
        pm = LoweringPassManager(
            [CircuitToDag()]
            + [
                LegacyDagPass(pass_)
                for pass_ in generate_preset_pbc_pass_manager().to_flow_controller().tasks
            ]
            + [LegacyDagPass(LitinskiTransformation(fix_clifford=False))]
            + [mir_passes.DagToMirPass()]
        )
        mir_instance: Any = pm.run(qc)
        self.log.debug(f"mir_instance=\n{mir_instance}")
        self.assertIsInstance(mir_instance, mir.Mir)
        self.assertEqual(mir_instance.num_instructions(), 2)
        self.assertEqual(mir_instance.num_qids(), 2)


if __name__ == "__main__":
    unittest.main()
