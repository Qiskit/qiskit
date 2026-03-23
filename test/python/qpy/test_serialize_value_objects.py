# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test serializing ParameterExpressions from qpy."""

import io
from test import QiskitTestCase
from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit
from qiskit import qpy
from qiskit.quantum_info import SparseObservable
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate


class TestQpySerializeParameterExpression(QiskitTestCase):
    """QPY serializing ParameterExpression"""

    def test_roundtrip_equal(self):
        """Test serialize deserialize with ParameterExpression in _qpy_replay"""
        a = Parameter("a")
        b = Parameter("b")
        a1 = a * 2
        a2 = a1.subs({a: 3 * b})

        qc = QuantumCircuit(1)
        qc.rz(a2, 0)

        use_symengine = True
        version = 13
        with io.BytesIO() as container:
            qpy.dump(qc, container, version=version, use_symengine=use_symengine)
            qc_qpy_str = container.getvalue()

        with io.BytesIO(qc_qpy_str) as container:
            qc_from_qpy = qpy.load(container)[0]

        self.assertEqual(qc, qc_from_qpy)


class TestPauliEvolution(QiskitTestCase):
    """QPY serializing PauliEvolutionGate with SparseObservable and SparsePauliOp"""

    def test_pauli_evolution_sparseobservable(self):
        """Test PauliEvolutionGate with SparseObservable"""
        op = SparseObservable.from_list([("XIX", 0.1), ("ZIZ", 0.3)])

        # build the evolution gate
        evo = PauliEvolutionGate(op)
        circuit = QuantumCircuit(evo.num_qubits)
        circuit.append(evo, circuit.qubits)
        version = 17

        with io.BytesIO() as container:
            qpy.dump(circuit, container, version=version)
            qc_qpy_str = container.getvalue()

        with io.BytesIO(qc_qpy_str) as container:
            qc_from_qpy = qpy.load(container)[0]

        self.assertEqual(circuit, qc_from_qpy)

    def test_pauli_evolution_sparse_pauliop(self):
        """Test PauliEvolutionGate with SparsePauliOp"""
        operator = SparsePauliOp.from_list([("ZZ", 1), ("XI", -0.1)])

        # build the evolution gate
        evo = PauliEvolutionGate(operator, time=0.2)
        circuit = QuantumCircuit(evo.num_qubits)
        circuit.append(evo, circuit.qubits)
        version = 16

        with io.BytesIO() as container:
            qpy.dump(circuit, container, version=version)
            qc_qpy_str = container.getvalue()

        with io.BytesIO(qc_qpy_str) as container:
            qc_from_qpy = qpy.load(container)[0]

        self.assertEqual(circuit, qc_from_qpy)

    def test_pauli_evolution_operator_list(self):
        """Test PauliEvolutionGate with list of operators"""
        op1 = SparseObservable.from_list([("XIX", 0.1), ("ZIZ", 0.3)])
        op2 = SparsePauliOp.from_list([("ZZI", 1), ("XIX", -0.1)])

        # build the evolution gate
        evo = PauliEvolutionGate([op1, op2], time=0.5)
        circuit = QuantumCircuit(evo.num_qubits)
        circuit.append(evo, circuit.qubits)
        version = 17

        with io.BytesIO() as container:
            qpy.dump(circuit, container, version=version)
            qc_qpy_str = container.getvalue()

        with io.BytesIO(qc_qpy_str) as container:
            qc_from_qpy = qpy.load(container)[0]

        self.assertEqual(circuit, qc_from_qpy)


class TestWriteReadValueList(QiskitTestCase):
    """Tests for the write_values / read_values Rust QPY functions."""

    def _roundtrip(self, values):
        """Helper: write then read back a list of values."""
        from qiskit._accelerate import qpy as _qpy

        buf = io.BytesIO()
        _qpy.write_values(buf, values)
        buf.seek(0)
        return _qpy.read_values(buf)

    # ------------------------------------------------------------------
    # Parameter / ParameterExpression
    # ------------------------------------------------------------------

    def test_parameter_roundtrip(self):
        """A bare Parameter survives a write/read cycle."""
        a = Parameter("a")
        result = self._roundtrip([a])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], a)

    def test_parameter_vector_element_roundtrip(self):
        """A ParameterVector element survives a write/read cycle."""
        v = ParameterVector("v", 3)
        result = self._roundtrip([v[0], v[2]])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], v[0])
        self.assertEqual(result[1], v[2])

    def test_parameter_expression_roundtrip(self):
        """A compound ParameterExpression survives a write/read cycle."""
        a = Parameter("a")
        b = Parameter("b")
        expr = a * 2 + b / 3

        result = self._roundtrip([expr])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], expr)

    def test_multiple_parameter_expressions(self):
        """Multiple ParameterExpressions in one list all round-trip correctly."""
        a = Parameter("a")
        b = Parameter("b")
        exprs = [a, b, a + b, a * b - 1.5, b**a]

        result = self._roundtrip(exprs)
        self.assertEqual(len(result), len(exprs))
        for original, recovered in zip(exprs, result):
            self.assertEqual(original, recovered)

    def test_float_roundtrip(self):
        """Float values survive a write/read cycle."""
        values = [0.0, 1.5, -3.14, float("inf")]
        result = self._roundtrip(values)
        self.assertEqual(len(result), len(values))
        for original, recovered in zip(values, result):
            self.assertEqual(original, recovered)

    def test_int_roundtrip(self):
        """Integer values survive a write/read cycle."""
        values = [0, 1, -42, 2**31]
        result = self._roundtrip(values)
        self.assertEqual(len(result), len(values))
        for original, recovered in zip(values, result):
            self.assertEqual(original, recovered)

    def test_complex_roundtrip(self):
        """Complex values survive a write/read cycle."""
        values = [1 + 2j, -3.5 + 0j, 0 - 1j]
        result = self._roundtrip(values)
        self.assertEqual(len(result), len(values))
        for original, recovered in zip(values, result):
            self.assertEqual(original, recovered)

    def test_range_roundtrip(self):
        """A Python range survives a write/read cycle."""
        r = range(2, 10, 3)
        result = self._roundtrip([r])
        self.assertEqual(len(result), 1)
        self.assertEqual(list(result[0]), list(r))

    def test_tuple_of_mixed_scalars(self):
        """A tuple of mixed scalar types survives a write/read cycle."""
        t = (1, 2.5, 3 + 4j)
        result = self._roundtrip([t])
        self.assertEqual(len(result), 1)
        recovered = result[0]
        self.assertEqual(recovered[0], 1)
        self.assertAlmostEqual(recovered[1], 2.5)
        self.assertEqual(recovered[2], 3 + 4j)

    def test_circuit_roundtrip(self):
        """A QuantumCircuit survives a write/read cycle."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        result = self._roundtrip([qc])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], qc)

    def test_parameterized_circuit_roundtrip(self):
        """A parameterized QuantumCircuit survives a write/read cycle."""
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.rz(theta, 0)

        result = self._roundtrip([qc])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], qc)

    def test_mixed_value_list(self):
        """A heterogeneous list of values all round-trip correctly."""
        a = Parameter("a")
        qc = QuantumCircuit(1)
        qc.x(0)
        values = [1, 2.5, 1 + 2j, a, a * 3, range(5), (0, 1.0), qc]

        result = self._roundtrip(values)
        self.assertEqual(len(result), len(values))
        self.assertEqual(result[0], 1)
        self.assertAlmostEqual(result[1], 2.5)
        self.assertEqual(result[2], 1 + 2j)
        self.assertEqual(result[3], a)
        self.assertEqual(result[4], a * 3)
        self.assertEqual(result[5], range(5))
        self.assertEqual(result[6], (0, 1.0))
        self.assertEqual(result[7], qc)
