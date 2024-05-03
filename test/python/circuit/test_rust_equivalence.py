# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Rust gate definition tests"""

from math import pi

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping

from test import QiskitTestCase


SKIP_LIST = {"cy", "ccx", "rx", "ry", "ecr", "sx"}
CUSTOM_MAPPING = {"x", "rz"}


class TestRustGateEquivalence(QiskitTestCase):
    """Tests that compile time rust gate definitions is correct."""

    def setUp(self):
        super().setUp()
        self.standard_gates = get_standard_gate_name_mapping()
        # Pre-warm gate mapping cache, this is needed so rust -> py conversion
        qc = QuantumCircuit(3)
        for gate in self.standard_gates.values():
            if getattr(gate, "_standard_gate", None):
                if gate.params:
                    gate = gate.base_class(*[pi] * len(gate.params))
                qc.append(gate, list(range(gate.num_qubits)))

    def test_definitions(self):
        """Test definitions are the same in rust space."""
        for name, gate_class in self.standard_gates.items():
            standard_gate = getattr(gate_class, "_standard_gate", None)
            if name in SKIP_LIST:
                # gate does not have a rust definition yet
                continue
            if standard_gate is None:
                # gate is not in rust yet
                continue

            with self.subTest(name=name):
                print(name)
                params = [pi] * standard_gate._num_params()
                py_def = gate_class.base_class(*params).definition
                rs_def = standard_gate._get_definition(params)
                if py_def is None:
                    self.assertIsNone(rs_def)
                else:
                    rs_def = QuantumCircuit._from_circuit_data(rs_def)
                    for rs_inst, py_inst in zip(rs_def._data, py_def._data):
                        # Rust uses U but python still uses U3 and u2
                        if rs_inst.operation.name == "u":
                            if py_inst.operation.name == "u3":
                                self.assertEqual(rs_inst.operation.params, py_inst.operation.params)
                            elif py_inst.operation.name == "u2":
                                self.assertEqual(
                                    rs_inst.operation.params,
                                    [
                                        pi / 2,
                                        py_inst.operation.params[0],
                                        py_inst.operation.params[1],
                                    ],
                                )

                            self.assertEqual(
                                [py_def.find_bit(x).index for x in py_inst.qubits],
                                [rs_def.find_bit(x).index for x in rs_inst.qubits],
                            )
                        # Rust uses P but python still uses u1
                        elif rs_inst.operation.name == "p":
                            self.assertEqual(py_inst.operation.name, "u1")
                            self.assertEqual(rs_inst.operation.params, py_inst.operation.params)
                            self.assertEqual(
                                [py_def.find_bit(x).index for x in py_inst.qubits],
                                [rs_def.find_bit(x).index for x in rs_inst.qubits],
                            )
                        else:
                            self.assertEqual(py_inst.operation.name, rs_inst.operation.name)
                            self.assertEqual(rs_inst.operation.params, py_inst.operation.params)
                            self.assertEqual(
                                [py_def.find_bit(x).index for x in py_inst.qubits],
                                [rs_def.find_bit(x).index for x in rs_inst.qubits],
                            )

    def test_matrix(self):
        """Test matrices are the same in rust space."""
        for name, gate_class in self.standard_gates.items():
            standard_gate = getattr(gate_class, "_standard_gate", None)
            if standard_gate is None:
                # gate is not in rust yet
                continue

            with self.subTest(name=name):
                params = [pi] * standard_gate._num_params()
                py_def = gate_class.base_class(*params).to_matrix()
                rs_def = standard_gate._to_matrix(params)
                np.testing.assert_allclose(rs_def, py_def)
