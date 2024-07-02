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

from test import QiskitTestCase

import numpy as np

from qiskit.circuit import QuantumCircuit, CircuitInstruction
from qiskit.circuit.library.standard_gates import C3XGate, U1Gate, ZGate
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping

SKIP_LIST = {"rx", "ry", "ecr"}
CUSTOM_NAME_MAPPING = {"c3x": C3XGate()}
MATRIX_SKIP_LIST = {"c3sx"}


class TestRustGateEquivalence(QiskitTestCase):
    """Tests that compile time rust gate definitions is correct."""

    def setUp(self):
        super().setUp()
        self.standard_gates = get_standard_gate_name_mapping()
        self.standard_gates.update(CUSTOM_NAME_MAPPING)
        # Pre-warm gate mapping cache, this is needed so rust -> py conversion is done
        qc = QuantumCircuit(5)
        for gate in self.standard_gates.values():
            if getattr(gate, "_standard_gate", None):
                if gate.params:
                    gate = gate.base_class(*[pi] * len(gate.params))
                qc.append(gate, list(range(gate.num_qubits)))

    def test_gate_cross_domain_conversion(self):
        """Test the rust -> python conversion returns the right class."""
        for name, gate_class in self.standard_gates.items():
            standard_gate = getattr(gate_class, "_standard_gate", None)
            if standard_gate is None:
                # Gate not in rust yet or no constructor method
                continue
            with self.subTest(name=name):
                qc = QuantumCircuit(standard_gate.num_qubits)
                qc._append(
                    CircuitInstruction(standard_gate, qubits=qc.qubits, params=gate_class.params)
                )
                self.assertEqual(qc.data[0].operation.base_class, gate_class.base_class)
                self.assertEqual(qc.data[0].operation, gate_class)

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
                        # Rust uses P but python still uses u1/u3 in some cases
                        elif rs_inst.operation.name == "p" and not name in [
                            "cp",
                            "cs",
                            "csdg",
                            "cu",
                            "c3x",
                        ]:
                            if py_inst.operation.name == "u1":
                                self.assertEqual(py_inst.operation.name, "u1")
                                self.assertEqual(rs_inst.operation.params, py_inst.operation.params)
                                self.assertEqual(
                                    [py_def.find_bit(x).index for x in py_inst.qubits],
                                    [rs_def.find_bit(x).index for x in rs_inst.qubits],
                                )
                            else:
                                self.assertEqual(py_inst.operation.name, "u3")
                                self.assertEqual(
                                    rs_inst.operation.params[0], py_inst.operation.params[2]
                                )
                                self.assertEqual(
                                    [py_def.find_bit(x).index for x in py_inst.qubits],
                                    [rs_def.find_bit(x).index for x in rs_inst.qubits],
                                )
                        # Rust uses cp but python still uses cu1 in some cases
                        elif rs_inst.operation.name == "cp":
                            self.assertEqual(py_inst.operation.name, "cu1")
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
            if name in MATRIX_SKIP_LIST:
                # to_matrix not defined for type
                continue
            if standard_gate is None:
                # gate is not in rust yet
                continue

            with self.subTest(name=name):
                params = [0.1] * standard_gate._num_params()
                py_def = gate_class.base_class(*params).to_matrix()
                rs_def = standard_gate._to_matrix(params)
                np.testing.assert_allclose(rs_def, py_def)

    def test_name(self):
        """Test that the gate name properties match in rust space."""
        for name, gate_class in self.standard_gates.items():
            standard_gate = getattr(gate_class, "_standard_gate", None)
            if standard_gate is None or gate_class.name == "mcx":
                # gate is not in rust yet
                continue

            with self.subTest(name=name):
                self.assertEqual(gate_class.name, standard_gate.name)

    def test_num_qubits(self):
        """Test the number of qubits are the same in rust space."""
        for name, gate_class in self.standard_gates.items():
            standard_gate = getattr(gate_class, "_standard_gate", None)
            if standard_gate is None:
                # gate is not in rust yet
                continue

            with self.subTest(name=name):
                self.assertEqual(gate_class.num_qubits, standard_gate.num_qubits)

    def test_num_params(self):
        """Test the number of parameters are the same in rust space."""
        for name, gate_class in self.standard_gates.items():
            standard_gate = getattr(gate_class, "_standard_gate", None)
            if standard_gate is None:
                # gate is not in rust yet
                continue

            with self.subTest(name=name):
                self.assertEqual(
                    len(gate_class.params), standard_gate.num_params, msg=f"{name} not equal"
                )

    def test_non_default_controls(self):
        """Test that controlled gates with a non-default ctrl_state
        are not using the standard rust representation."""

        z_gate, z_term = ZGate(), -1
        u1_gate, u1_term = U1Gate(0.1), 0.99500417 + 0.09983342j

        for gate, term in zip([z_gate, u1_gate], [z_term, u1_term]):
            with self.subTest(name=gate.name):
                default_op = np.diag([1, 1, 1, term])
                non_default_op = np.diag([1, 1, term, 1])
                state_out_map = {
                    1: default_op,
                    "1": default_op,
                    None: default_op,
                    0: non_default_op,
                    "0": non_default_op,
                }
                for state, op in state_out_map.items():
                    circuit = QuantumCircuit(2)
                    circuit.append(gate.control(1, ctrl_state=state), [0, 1])
                    np.testing.assert_almost_equal(circuit.data[0].operation.to_matrix(), op)
