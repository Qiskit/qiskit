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
from qiskit.circuit import Duration, Parameter, QuantumCircuit
from qiskit import qpy
from qiskit.qpy import type_keys
from qiskit.qpy.binary_io import value as value_io
from qiskit.qpy.exceptions import UnsupportedFeatureForVersion
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


class TestDurationValue(QiskitTestCase):
    """Serializing a ``Duration`` value, which QPY 18 gave a type key of its own.

    Up to QPY 17 a ``Duration`` shared :attr:`.Container.TUPLE`'s key, so a reader had to be told
    which of the two a payload held; there was no way to write one as a standalone value.
    """

    def test_duration_uses_its_own_key(self):
        """Every duration unit round-trips under the ``DURATION`` key from version 18."""
        for duration in (
            Duration.dt(100),
            Duration.ps(1.5),
            Duration.ns(250.0),
            Duration.us(0.5),
            Duration.ms(2.0),
            Duration.s(1.0),
        ):
            with self.subTest(duration=duration):
                type_key, data = value_io.dumps_value(duration, version=18)
                self.assertEqual(type_key, type_keys.Value.DURATION)
                self.assertEqual(value_io.loads_value(type_key, data, 18, {}), duration)

    def test_duration_key_is_distinct_from_tuple(self):
        """The whole point: the key no longer collides with a container."""
        self.assertNotEqual(type_keys.Value.DURATION, type_keys.Container.TUPLE)
        self.assertNotEqual(type_keys.Value.BIGINT, type_keys.Value.INTEGER)

    def test_duration_rejected_before_v18(self):
        """Below 18 there is no unambiguous encoding, so refuse rather than write one."""
        with self.assertRaises(UnsupportedFeatureForVersion):
            value_io.dumps_value(Duration.dt(100), version=17)
