# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the LightCone pass"""

import unittest
from test import QiskitTestCase

import ddt

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.classical import expr
from qiskit.circuit.library import (
    GlobalPhaseGate,
    HGate,
    PauliGate,
    PauliProductMeasurement,
    PauliProductRotationGate,
    UnitaryGate,
    ZGate,
    get_standard_gate_name_mapping,
    real_amplitudes,
)
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library.n_local.efficient_su2 import efficient_su2
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import (
    Pauli,
    SparseObservable,
    SparsePauliOp,
    Statevector,
    random_unitary,
)
from qiskit.transpiler.passes.optimization.light_cone import LightCone
from qiskit.transpiler.passmanager import PassManager


@ddt.ddt
class TestLightConePass(QiskitTestCase):
    """Test the LightCone pass."""

    @ddt.data("X", "Z")
    def test_nonparameterized_noncommuting(self, pauli_label):
        """Test for a non-commuting, asymmetric, weight-one Pauli."""
        bit_terms = pauli_label
        light_cone = LightCone(bit_terms=bit_terms, indices=[1])
        pm = PassManager([light_cone])

        qc = QuantumCircuit(2)
        qc.x(0)
        qc.s(1)
        qc.cy(0, 1)

        new_circuit = pm.run(qc)

        expected = qc

        self.assertEqual(expected, new_circuit)

    def test_nonparameterized_commuting(self):
        """Test for a commuting, asymmetric, weight-one Pauli."""
        light_cone = LightCone(bit_terms="Y", indices=[1])
        pm = PassManager([light_cone])

        qc = QuantumCircuit(2)
        qc.x(0)
        qc.s(1)
        qc.cy(0, 1)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(2)
        expected.s(1)

        self.assertEqual(expected, new_circuit)

    @ddt.data("Y", "Z")
    def test_parameterized_noncommuting(self, pauli_label):
        """Test for a non-commuting, asymmetric, weight-one Pauli on a parameterized circuit."""
        bit_terms = pauli_label
        light_cone = LightCone(bit_terms=bit_terms, indices=[1])
        pm = PassManager([light_cone])
        theta = Parameter("θ")

        qc = QuantumCircuit(2)
        qc.rx(theta, 0)
        qc.ry(theta, 1)
        qc.cx(0, 1)

        new_circuit = pm.run(qc)

        expected = qc

        self.assertEqual(expected, new_circuit)

    def test_parameterized_commuting(self):
        """Test for a commuting, asymmetric, weight-one Pauli on a parameterized circuit."""
        bit_terms, indices, _ = SparsePauliOp("XI").to_sparse_list()[0]
        light_cone = LightCone(bit_terms=bit_terms, indices=indices)
        pm = PassManager([light_cone])
        theta = Parameter("θ")

        qc = QuantumCircuit(2)
        qc.rx(theta, 0)
        qc.ry(theta, 1)
        qc.cx(0, 1)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(2)
        expected.ry(theta, 1)

        self.assertEqual(expected, new_circuit)

    def test_parameterized_symmetric(self):
        """Test for a double symmetric `Z` observable on a parameterized circuit."""
        bit_terms, indices, _ = SparsePauliOp("ZIIZ").to_sparse_list()[0]
        light_cone = LightCone(bit_terms=bit_terms, indices=indices)
        pm = PassManager([light_cone])

        qc = real_amplitudes(4, entanglement="pairwise", reps=1)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(4)
        theta = qc.parameters
        expected.ry(theta[0], 0)
        expected.ry(theta[1], 1)
        expected.ry(theta[2], 2)
        expected.ry(theta[3], 3)
        expected.cx(0, 1)
        expected.cx(2, 3)
        expected.ry(theta[4], 0)
        expected.ry(theta[7], 3)

        self.assertEqual(expected, new_circuit)

    def test_parameterized_asymmetric(self):
        """Test for a double asymmetric observable on a parameterized circuit."""
        bit_terms, indices, _ = SparsePauliOp("IZIX").to_sparse_list()[0]
        light_cone = LightCone(bit_terms=bit_terms, indices=indices)
        pm = PassManager([light_cone])

        qc = efficient_su2(4, entanglement="circular", reps=1)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(4)
        theta = qc.parameters
        expected.ry(theta[0], 0)
        expected.ry(theta[1], 1)
        expected.ry(theta[2], 2)
        expected.ry(theta[3], 3)
        expected.rz(theta[4], 0)
        expected.rz(theta[5], 1)
        expected.rz(theta[6], 2)
        expected.rz(theta[7], 3)
        expected.cx(3, 0)
        expected.cx(0, 1)
        expected.cx(1, 2)
        expected.cx(2, 3)
        expected.ry(theta[8], 0)
        expected.ry(theta[10], 2)
        expected.rz(theta[12], 0)

        self.assertEqual(expected, new_circuit)

    def test_all_commuting(self):
        """Test for a circuit that fully commutes with an observable."""
        bit_terms, indices, _ = SparsePauliOp("IIIZ").to_sparse_list()[0]
        light_cone = LightCone(bit_terms=bit_terms, indices=indices)
        pm = PassManager([light_cone])

        qc = QuantumCircuit(4)
        qc.s(0)
        qc.z(0)
        qc.h(2)
        qc.cx(0, 1)
        qc.cx(2, 3)

        new_circuit = pm.run(qc)

        self.assertEqual(sum(new_circuit.count_ops().values()), 0)

    def test_commuting_block(self):
        """Test for a commuting block. Currently, gates are checked
        one by one and commuting blocks are thus ignored."""
        bit_terms, indices, _ = SparsePauliOp("IIXII").to_sparse_list()[0]
        light_cone = LightCone(bit_terms=bit_terms, indices=indices)
        pm = PassManager([light_cone])

        qc = QuantumCircuit(5)
        qc.cx(2, 1)
        qc.cx(3, 4)
        qc.cx(1, 0)
        qc.cx(2, 3)
        qc.cx(2, 1)
        qc.x(1)
        qc.cx(2, 1)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(5)
        expected.cx(2, 1)
        expected.cx(3, 4)
        expected.cx(1, 0)
        expected.cx(2, 3)
        expected.cx(2, 1)
        expected.cx(2, 1)

        self.assertEqual(expected, new_circuit)

    def test_measurement_barriers(self):
        """Test for measurement and barriers."""
        light_cone = LightCone()
        pm = PassManager([light_cone])
        theta = Parameter("θ")

        qc = QuantumCircuit(2, 1)
        qc.rz(theta, 1)
        qc.ry(theta, 0)
        qc.barrier()
        qc.cx(0, 1)
        qc.barrier()
        qc.rz(theta, 1)
        qc.measure(0, 0)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(2, 1)
        expected.rz(theta, 1)
        expected.ry(theta, 0)
        expected.barrier()
        expected.cx(0, 1)
        expected.barrier()
        expected.measure(0, 0)

        self.assertEqual(expected, new_circuit)

    @ddt.data(SparsePauliOp("IX"), SparseObservable("I+"))
    def test_parameter_expression(self, sparse_object):
        """Test for Parameter expressions."""
        bit_terms, indices, _ = sparse_object.to_sparse_list()[0]
        light_cone = LightCone(bit_terms=bit_terms, indices=indices)
        pm = PassManager([light_cone])
        theta = Parameter("θ")

        qc = QuantumCircuit(2)
        qc.rz(theta + 2, 1)
        qc.ry(theta - 2, 0)
        qc.h(1)
        qc.cz(0, 1)
        qc.h(1)
        qc.rz(theta * 2, 1)
        qc.rz(theta / 2, 1)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(2)
        expected.rz(theta + 2, 1)
        expected.ry(theta - 2, 0)
        expected.h(1)
        expected.cz(0, 1)

        self.assertEqual(expected, new_circuit)

    @ddt.data(
        SparsePauliOp.from_sparse_list([("XY", [119, 2], 1)], 120),
        SparseObservable.from_sparse_list([("-l", [119, 2], 1)], 120),
    )
    def test_big_circuit(self, sparse_object):
        """Test for large circuit and observable."""
        num_qubits = sparse_object.num_qubits
        bit_terms, indices, _ = sparse_object.to_sparse_list()[0]
        light_cone = LightCone(bit_terms=bit_terms, indices=indices)
        pm = PassManager([light_cone])
        theta = Parameter("θ")

        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            qc.h(i)
        for i in range(0, num_qubits, 2):
            qc.cx(i, i + 1)
        for i in range(1, num_qubits - 1, 2):
            qc.cx(i, i + 1)
        for i in range(num_qubits):
            qc.rz(theta, i)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(num_qubits)
        expected.h(0)
        expected.h(1)
        expected.h(2)
        expected.h(3)
        expected.h(num_qubits - 2)
        expected.h(num_qubits - 1)
        expected.cx(0, 1)
        expected.cx(2, 3)
        expected.cx(1, 2)
        expected.rz(theta, 2)
        expected.cx(num_qubits - 2, num_qubits - 1)
        expected.rz(theta, num_qubits - 1)

        self.assertEqual(expected, new_circuit)

    @ddt.data(("IIIIIZXIIIIIIII", False), ("YYYYYZXYYYYYYYY", True))
    @ddt.unpack
    def test_large_observable(self, pauli_string, keeps_cx):
        """Test for a large initial observable."""
        op = SparsePauliOp([pauli_string])
        bit_terms, indices, _ = op.to_sparse_list()[0]

        light_cone = LightCone(bit_terms=bit_terms, indices=indices)
        pm = PassManager([light_cone])

        qc = QuantumCircuit(15)
        qc.cx(5, 6)

        new_circuit = pm.run(qc)

        # The first label is Z on q5 and X on q6, which commutes with CX(5, 6). The second one is
        # Y on both qubits, which does not.
        expected = qc if keeps_cx else QuantumCircuit(15)

        self.assertEqual(expected, new_circuit)

    def test_raise_error_when_indices_is_empty(self):
        """Test that `ValueError` is raised if bit_terms is given but indices is empty."""
        with self.assertRaises(
            ValueError, msg="`indices` must be non-empty when providing `bit_terms`."
        ):
            _ = LightCone(bit_terms="X", indices=[])

    def test_raise_error_for_invalid_bit_terms_characters(self):
        """Test that `ValueError` is raised if `bit_terms` has characters not in `valid_characters`."""
        with self.assertRaises(
            ValueError, msg="`bit_terms` should contain only characters in {...}."
        ):
            _ = LightCone(bit_terms="AX", indices=[0])  # 'A' is invalid

    def test_raise_error_when_indices_out_of_range(self):
        """Test that `ValueError` is raised if an index is outside the DAG qubit range."""
        qc = QuantumCircuit(1)
        dag = circuit_to_dag(qc)

        light_cone = LightCone(bit_terms="X", indices=[1])  # index 1 doesn't exist
        with self.assertRaises(
            ValueError, msg="`indices` contains values outside the qubit range."
        ):
            light_cone.run(dag)

    def test_raise_error_when_circuit_measurements_and_observable_present(self):
        """Test that `ValueError` is raised if the circuit has measurements
        and `bit_terms` is also given."""
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)  # A measurement on qubit 0
        dag = circuit_to_dag(qc)

        light_cone = LightCone(bit_terms="X", indices=[0])
        with self.assertRaises(
            ValueError, msg="The circuit contains measurements and an observable has been given"
        ):
            light_cone.run(dag)

    def test_projector_product(self):
        """``|00><00|`` is not ``ZZ``: an ``XX`` gate commutes with ``ZZ`` but not with the projector."""
        qc = QuantumCircuit(2)
        qc.append(PauliGate("XX"), [0, 1])

        self.assertEqual(QuantumCircuit(2), LightCone(bit_terms="ZZ", indices=[0, 1])(qc))
        self.assertEqual(qc, LightCone(bit_terms="00", indices=[0, 1])(qc))

    def test_projector_and_pauli(self):
        """A projector next to a Pauli: ``CZ`` commutes with ``|+><+|`` on neither qubit."""
        qc = QuantumCircuit(3)
        qc.cz(0, 1)
        qc.x(2)

        expected = QuantumCircuit(3)
        expected.cz(0, 1)
        self.assertEqual(expected, LightCone(bit_terms="+X", indices=[0, 2])(qc))

    def test_mid_circuit_measurement(self):
        """A mid-circuit measurement is kept, and so is everything it depends on."""
        qc = QuantumCircuit(2, 1)
        qc.cx(0, 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.h(0)

        expected = QuantumCircuit(2, 1)
        expected.cx(0, 1)
        expected.h(0)
        expected.measure(0, 0)
        self.assertEqual(expected, LightCone()(qc))

    def test_pauli_product_measurement(self):
        """A ``PauliProductMeasurement`` measures a Pauli, and only gates that commute with it go.

        The measured Pauli is the dense label ``"XZ"``: Z on q0 and X on q1. Right before the
        measurement, an RZ on q0 and an RX on q1 commute with it. Before those, an RX on q0 and an
        RZ on q1 do not.
        """
        qc = QuantumCircuit(2, 1)
        qc.rx(0.2, 0)
        qc.rz(0.4, 1)
        qc.rz(0.1, 0)
        qc.rx(0.3, 1)
        qc.append(PauliProductMeasurement(Pauli("XZ")), [0, 1], [0])

        expected = QuantumCircuit(2, 1)
        expected.rx(0.2, 0)
        expected.rz(0.4, 1)
        expected.append(PauliProductMeasurement(Pauli("XZ")), [0, 1], [0])
        self.assertEqual(expected, LightCone()(qc))

    def test_gate_without_pauli_generator(self):
        """A gate with no Pauli generator, such as a ``UnitaryGate``, is checked by its matrix."""
        qc = QuantumCircuit(1)
        qc.append(UnitaryGate(ZGate().to_matrix()), [0])
        qc.u(0, 0, 0.3, 0)

        self.assertEqual(QuantumCircuit(1), LightCone(bit_terms="Z", indices=[0])(qc))
        self.assertEqual(qc, LightCone(bit_terms="X", indices=[0])(qc))

    def test_unread_store_is_removed(self):
        """An operation on no qubit, such as a ``Store``, is removed if nothing reads what it writes.

        ``add_var`` with an initial value adds a ``Store`` of its own, so two are removed.
        """
        qc = QuantumCircuit(1, 1)
        flag = qc.add_var("flag", False)
        qc.h(0)
        qc.store(flag, True)
        qc.measure(0, 0)

        reduced = LightCone()(qc)
        self.assertEqual([inst.name for inst in reduced.data], ["h", "measure"])

    def test_global_phase_is_removed(self):
        """A global phase, which acts on no qubits, is removed along with gates outside the cone."""
        qc = QuantumCircuit(2, 1)
        qc.h(1)
        qc.append(GlobalPhaseGate(0.3), [])
        qc.h(0)
        qc.measure(0, 0)

        reduced = LightCone()(qc)
        self.assertEqual([inst.name for inst in reduced.data], ["h", "measure"])

    def test_store_read_later_is_kept(self):
        """A ``Store`` that a kept ``if_test`` reads is kept.

        Without it ``flag`` stays false, the ``X`` is skipped and the measurement gives 0
        instead of 1.
        """
        qc = QuantumCircuit(1, 1)
        flag = qc.add_var("flag", False)
        qc.store(flag, True)
        with qc.if_test(flag):
            qc.x(0)
        qc.measure(0, 0)
        self.assertEqual(qc, LightCone()(qc))

    def test_store_into_clbit_read_later_is_kept(self):
        """The same for a ``Store`` into a clbit."""
        qc = QuantumCircuit(1, 2)
        qc.store(qc.clbits[1], True)
        with qc.if_test((qc.clbits[1], 1)):
            qc.x(0)
        qc.measure(0, 0)
        self.assertEqual(qc, LightCone()(qc))

    def test_store_into_output_clbit_is_kept(self):
        """A ``Store`` into a clbit that nothing overwrites is part of the outcome."""
        qc = QuantumCircuit(1, 2)
        qc.store(qc.clbits[1], True)
        qc.measure(0, 0)
        self.assertEqual(qc, LightCone()(qc))

    def test_store_overwritten_by_measurement_is_kept(self):
        """A ``Store`` into a clbit that a kept measurement overwrites is still kept.

        The pass checks liveness, not whether the next item really reads the value.
        """
        qc = QuantumCircuit(1, 1)
        qc.store(qc.clbits[0], True)
        qc.x(0)
        qc.measure(0, 0)
        self.assertEqual(qc, LightCone()(qc))

    def test_store_read_only_by_removed_operations_is_removed(self):
        """A ``Store`` read only by operations outside the light cone is removed."""
        qc = QuantumCircuit(2, 1)
        flag = qc.add_var("flag", False)
        qc.store(flag, True)
        with qc.if_test(flag):
            qc.x(1)
        qc.h(0)
        qc.measure(0, 0)
        reduced = LightCone()(qc)
        self.assertEqual([inst.name for inst in reduced.data], ["h", "measure"])

    @ddt.data(([0, 1], 0, 1), ([1, 0], 1, 0))
    @ddt.unpack
    def test_bit_terms_follow_indices(self, indices, x_qubit, z_qubit):
        """``bit_terms[i]`` acts on qubit ``indices[i]``: X on ``indices[0]``, Z on ``indices[1]``.

        Only the Z gate on the X qubit anticommutes with the observable. The H gates do not
        commute with either term, so they are kept too.
        """
        qc = QuantumCircuit(2)
        qc.h([0, 1])
        qc.z(0)
        qc.z(1)

        expected = QuantumCircuit(2)
        expected.h(x_qubit)
        expected.h(z_qubit)
        expected.z(x_qubit)
        self.assertEqual(expected, LightCone(bit_terms="XZ", indices=indices)(qc))

    def test_indices_not_sorted(self):
        """X on q2 and Z on q0, given in that order. Only the Z gate on q2 is kept."""
        qc = QuantumCircuit(3)
        qc.z(0)
        qc.z(2)

        expected = QuantumCircuit(3)
        expected.z(2)
        self.assertEqual(expected, LightCone(bit_terms="XZ", indices=[2, 0])(qc))

    def test_cx_orientation(self):
        """A CX commutes with Z on its control and X on its target, but not the other way around."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)

        self.assertEqual(QuantumCircuit(2), LightCone(bit_terms="ZX", indices=[0, 1])(qc))
        self.assertEqual(qc, LightCone(bit_terms="XZ", indices=[0, 1])(qc))


def without(circuit, *names):
    """``circuit`` with every operation called one of ``names`` left out."""
    out = circuit.copy_empty_like()
    for inst in circuit.data:
        if inst.name not in names:
            out.append(inst)
    return out


@ddt.ddt
class TestLightConeMeasurements(QiskitTestCase):
    """Mid-circuit, repeated and product measurements."""

    def test_cz_before_mid_circuit_measurement(self):
        """A CZ before a mid-circuit measurement on q0 is kept, though it commutes with Z on q0.

        The open-controlled H after the measurement carries q0's later state to q1. Deleting
        the CZ moves half the probability from outcome 11 to 10.
        """
        qc = QuantumCircuit(2, 2)
        qc.h([0, 1])
        qc.cz(0, 1)
        qc.measure(0, 0)
        qc.x(0)
        qc.append(HGate().control(1, ctrl_state=0), [0, 1])
        qc.measure(1, 1)
        self.assertEqual(qc, LightCone()(qc))

    def test_measured_qubit_reused(self):
        """Gates after a mid-circuit measurement are kept when a later measurement depends on them.

        The final RZ on q1 and the H on q2 are removed.
        """
        qc = QuantumCircuit(3, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(0, 0)
        qc.rx(0.4, 0)
        qc.cx(0, 1)
        qc.h(2)
        qc.rz(0.3, 1)
        qc.measure(1, 1)

        expected = QuantumCircuit(3, 2)
        expected.h(0)
        expected.cx(0, 1)
        expected.measure(0, 0)
        expected.rx(0.4, 0)
        expected.cx(0, 1)
        expected.measure(1, 1)
        self.assertEqual(expected, LightCone()(qc))

    def test_reset(self):
        """A reset on a measured qubit is kept; a reset on an unmeasured one is removed."""
        qc = QuantumCircuit(2, 1)
        qc.h(0)
        qc.reset(0)
        qc.ry(0.3, 0)
        qc.h(1)
        qc.reset(1)
        qc.measure(0, 0)

        expected = QuantumCircuit(2, 1)
        expected.h(0)
        expected.reset(0)
        expected.ry(0.3, 0)
        expected.measure(0, 0)
        self.assertEqual(expected, LightCone()(qc))

    def test_product_measurement_after_mid_circuit_measure(self):
        """A ``measure`` on q0, then a product measurement of Z on q0 and X on q1.

        The RX on q0 between them does not commute with Z, so it and the gates it depends on are
        kept. The RZ on q0 commutes with the product measurement, and the RX on q1 commutes with
        X on q1 and with the CX target, so both are removed.
        """
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.rx(0.6, 1)
        qc.cx(0, 1)
        qc.measure(0, 0)
        qc.rx(0.4, 0)
        qc.rz(0.2, 0)
        qc.append(PauliProductMeasurement(Pauli("XZ")), [0, 1], [1])

        expected = QuantumCircuit(2, 2)
        expected.h(0)
        expected.cx(0, 1)
        expected.measure(0, 0)
        expected.rx(0.4, 0)
        expected.append(PauliProductMeasurement(Pauli("XZ")), [0, 1], [1])
        self.assertEqual(expected, LightCone()(qc))

    def test_pauli_product_rotations_and_measurement(self):
        """``PauliProductRotationGate`` s before a product measurement: only the last one, which
        rotates about the measured Pauli itself, is removed."""
        qc = QuantumCircuit(3, 1)
        for label, angle in (("XXI", 0.3), ("ZIZ", 0.5), ("YZX", 0.7), ("IIZ", 0.9)):
            qc.append(PauliProductRotationGate(Pauli(label), angle), [0, 1, 2])
        expected = qc.copy()
        qc.append(PauliProductRotationGate(Pauli("ZZX"), 0.4), [0, 1, 2])
        for circuit in (qc, expected):
            circuit.append(PauliProductMeasurement(Pauli("ZZX")), [0, 1, 2], [0])
        self.assertEqual(expected, LightCone()(qc))


BLOCKS = ("if_test", "for_loop", "while_loop", "switch")


@ddt.ddt
class TestLightConeClassicalControl(QiskitTestCase):
    """Control flow and classical data.

    An operation is kept if it writes a classical wire that something kept reads later. Every
    clbit is an output of the circuit; classical variables are not. Control flow counts as
    writing every clbit it touches.
    """

    def add_block(self, qc, kind, flag):
        """Open a block of ``kind`` that runs once while ``flag`` is true."""
        if kind == "if_test":
            return qc.if_test(flag)
        if kind == "for_loop":
            return qc.for_loop(range(1))
        if kind == "while_loop":
            return qc.while_loop(flag)
        return qc.switch(flag)

    @ddt.data(*BLOCKS)
    def test_block_on_unmeasured_qubit_writing_output_clbit(self, kind):
        """A block on q1, which is never measured outside it, measures into the output c1.

        The block is kept: dropping it changes the distribution of c1. In a ``for_loop`` the
        ``flag`` is unused, so its initial ``Store`` is removed.
        """
        qc = QuantumCircuit(2, 2)
        flag = qc.add_var("flag", True)
        block = self.add_block(qc, kind, flag)
        if kind == "switch":
            with block as case, case(True):
                qc.h(1)
                qc.measure(1, 1)
        else:
            with block:
                qc.h(1)
                qc.measure(1, 1)
                if kind == "while_loop":
                    qc.store(flag, False)
        qc.h(0)
        qc.rz(0.3, 0)
        qc.measure(0, 0)

        expected = without(qc, "rz", "store") if kind == "for_loop" else without(qc, "rz")
        self.assertEqual(expected, LightCone()(qc))

    def test_block_on_unmeasured_qubit_is_removed(self):
        """An ``if_test`` on a ``Var``, acting on a qubit that is never measured, is removed."""
        qc = QuantumCircuit(2, 1)
        flag = qc.add_var("flag", True)
        with qc.if_test(flag):
            qc.x(1)
        qc.h(0)
        qc.measure(0, 0)

        expected = qc.copy_empty_like()
        expected.h(0)
        expected.measure(0, 0)
        self.assertEqual(expected, LightCone()(qc))

    def test_for_loop(self):
        """A ``for_loop`` on the measured qubit is kept; one on another qubit is removed."""
        qc = QuantumCircuit(2, 1)
        with qc.for_loop(range(3)):
            qc.rx(0.3, 1)
        with qc.for_loop(range(2)) as i:
            qc.rx(0.3 * i, 0)
        qc.measure(0, 0)

        expected = QuantumCircuit(2, 1)
        expected.append(qc.data[1])
        expected.measure(0, 0)
        self.assertEqual(expected, LightCone()(qc))

    def test_if_else_with_expr(self):
        """An ``if_else`` with an ``expr`` condition on two mid-circuit results, feeding forward
        onto the measured q2. The RZ before the final measurement is removed."""
        qc = QuantumCircuit(3, 3)
        qc.h([0, 1])
        qc.measure([0, 1], [0, 1])
        with qc.if_test(expr.logic_and(qc.clbits[0], expr.logic_not(qc.clbits[1]))) as else_:
            qc.ry(0.4, 2)
        with else_:
            qc.h(2)
        qc.rz(0.3, 2)
        qc.measure(2, 2)
        self.assertEqual(without(qc, "rz"), LightCone()(qc))

    def test_store_from_measurement_read_by_kept_if(self):
        """A ``Var`` set from a mid-circuit result and read by a kept ``if_test`` is kept."""
        qc = QuantumCircuit(2, 2)
        flag = qc.add_var("flag", False)
        qc.h(0)
        qc.measure(0, 0)
        qc.store(flag, expr.lift(qc.clbits[0]))
        with qc.if_test(flag):
            qc.x(1)
        qc.measure(1, 1)
        self.assertEqual(qc, LightCone()(qc))

    def test_control_flow_on_clbits_only(self):
        """An ``if_test`` that only stores into c1, read later by a kept ``if_test``."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], 1)):
            qc.store(qc.clbits[1], True)
        with qc.if_test((qc.clbits[1], 1)):
            qc.x(1)
        qc.measure(1, 0)
        self.assertEqual(qc, LightCone()(qc))

    def test_nested_blocks_write_output_clbit(self):
        """A measurement into the output c1, two blocks deep on the unmeasured q1."""
        qc = QuantumCircuit(3, 2)
        flag = qc.add_var("flag", True)
        qc.h(0)
        with qc.if_test(flag):
            with qc.for_loop(range(1)):
                with qc.if_test(flag):
                    qc.h(1)
                    qc.measure(1, 1)
        qc.h(2)
        qc.measure(0, 0)
        expected = qc.copy_empty_like()
        for inst in qc.data:
            if inst.qubits != (qc.qubits[2],):  # the H on the unmeasured q2 is removed
                expected.append(inst)
        self.assertEqual(expected, LightCone()(qc))

    def test_while_body_writes_its_own_condition(self):
        """A repeat-until-zero ``while`` on q1 measures into the output c1, its own condition."""
        qc = QuantumCircuit(2, 2)
        qc.h([0, 1])
        qc.measure(1, 1)
        with qc.while_loop((qc.clbits[1], 1)):
            qc.h(1)
            qc.measure(1, 1)
        qc.measure(0, 0)
        self.assertEqual(qc, LightCone()(qc))

    def test_observable_with_measurement_inside_control_flow_raises(self):
        """A measurement inside a block counts as a measurement for the observable check."""
        qc = QuantumCircuit(2, 1)
        qc.h(0)
        with qc.for_loop(range(1)):
            qc.measure(1, 0)
        with self.assertRaisesRegex(ValueError, "contains measurements"):
            LightCone("X", [0])(qc)


class TestLightConePinnedLooseness(QiskitTestCase):
    """Operations that do not matter but that the pass keeps by design: it is sound, not minimal.

    If a future change makes the pass tighter here, update these tests deliberately.
    """

    def test_predictable_feed_forward_is_kept(self):
        """LightCone does not predict classical values, so it keeps this redundant feed-forward.

        A Bell pair on q0, q1 always gives ``c0 ^ c1 = 0``, so ``p`` is always false and the ``X``
        on q2 never runs. The pass keeps the ``if``, the ``store``, the measurements and the Bell
        gates anyway.
        """
        qc = QuantumCircuit(3, 3)
        p = qc.add_var("p", False)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        qc.store(p, expr.bit_xor(qc.clbits[0], qc.clbits[1]))
        with qc.if_test(p):
            qc.x(2)
        qc.measure(2, 2)
        self.assertEqual(qc, LightCone()(qc))

    def test_if_test_reading_output_clbit_is_kept(self):
        """An ``if_test`` on the unmeasured q1 is kept because it reads c0, which is an output.

        Control flow counts as writing every clbit it touches, including its condition.
        """
        qc = QuantumCircuit(2, 1)
        qc.h(0)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], 1)):
            qc.x(1)
        qc.h(0)
        qc.measure(0, 0)
        self.assertEqual(qc, LightCone()(qc))

    def test_barrier_is_kept(self):
        """Nothing commutes with a barrier, so gates before one are kept.

        The RZ and the delay before the barrier of ``measure_all`` would otherwise be removed,
        and barriers on their own are kept too.
        """
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.3, 1)
        qc.delay(10, 1)
        qc.measure_all()
        self.assertEqual(qc, LightCone()(qc))

        qc = QuantumCircuit(3)
        qc.barrier()
        qc.barrier(1)
        self.assertEqual(qc, LightCone("Z", [1])(qc))

    def test_cancelling_pair_is_kept(self):
        """``CX; CX`` is the identity, but neither CX commutes with ``X`` on its control."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(0, 1)
        self.assertEqual(qc, LightCone("X", [0])(qc))


STANDARD_GATES = sorted(
    name
    for name, gate in get_standard_gate_name_mapping().items()
    if name not in ("measure", "reset", "delay", "global_phase")
)


def sandwich(gate, seed):
    """``gate`` on every wire after random single-qubit layers, so that it is not trivial."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(gate.num_qubits)
    for qubit in range(gate.num_qubits):
        qc.u(*rng.uniform(0, np.pi, 3), qubit)
    qc.append(gate, range(gate.num_qubits))
    return qc


@ddt.ddt
class TestLightConeExpectationValues(QiskitTestCase):
    """In observable mode the reduced circuit must give the same expectation value."""

    def assertSameExpectation(self, circuit, bit_terms, indices, bind=None):
        """``<O>`` is the same before and after the pass, with ``bind`` assigned afterwards."""
        reduced = LightCone(bit_terms, indices)(circuit)
        if bind is not None:
            circuit, reduced = circuit.assign_parameters(bind), reduced.assign_parameters(
                bind, strict=False
            )
        observable = SparsePauliOp.from_sparse_observable(
            SparseObservable.from_sparse_list([(bit_terms, indices, 1.0)], circuit.num_qubits)
        )
        self.assertAlmostEqual(
            Statevector(circuit).expectation_value(observable),
            Statevector(reduced).expectation_value(observable),
            msg=f"\noriginal:\n{circuit}\nreduced:\n{reduced}",
        )

    def assertSameForAllTerms(self, circuit):
        """Same expectation value for X, Y, Z and every projector on each qubit."""
        for qubit in range(circuit.num_qubits):
            for term in "XYZ01+-lr":
                with self.subTest(term=term, qubit=qubit):
                    self.assertSameExpectation(circuit, term, [qubit])

    @ddt.data(*STANDARD_GATES)
    def test_standard_gate(self, name):
        """Every standard gate with random parameters, up to the 4-qubit ``C3SX`` and ``RC3X``."""
        gate = get_standard_gate_name_mapping()[name]
        rng = np.random.default_rng(STANDARD_GATES.index(name))
        params = list(rng.uniform(0.1, 2 * np.pi - 0.1, len(gate.params)))
        self.assertSameForAllTerms(sandwich(type(gate)(*params) if params else gate, seed=1))

    @ddt.data(
        UnitaryGate(random_unitary(4, seed=2)),
        PauliGate("IXZ"),
        HGate().control(1, ctrl_state=0),
    )
    def test_non_standard_gate(self, gate):
        """A ``UnitaryGate``, a ``PauliGate`` with an identity factor and an open control."""
        self.assertSameForAllTerms(sandwich(gate, seed=3))

    def test_parameters_bound_after_the_pass(self):
        """Parameter expressions, also ones that are zero once bound, bound after the pass."""
        theta = Parameter("θ")
        qc = QuantumCircuit(2)
        qc.h([0, 1])
        qc.rx(2 * theta - 1, 0)
        qc.rz(theta**2, 1)
        qc.rzx(theta / 3, 0, 1)
        for value in (0.0, 0.5, 2.3):
            for term in "XYZ":
                with self.subTest(value=value, term=term):
                    self.assertSameExpectation(qc, term, [1], bind={theta: value})

    @ddt.data(
        ("ZZZZ", [0, 1, 2, 3]),
        ("YXZY", [4, 1, 3, 0]),
        ("0+", [2, 0]),
        ("1-lr", [0, 1, 2, 3]),
        ("Z0X+", [4, 0, 2, 3]),
    )
    @ddt.unpack
    def test_observable(self, bit_terms, indices):
        """Weight >= 4, projectors, mixed terms and unsorted indices on a random circuit."""
        self.assertSameExpectation(random_circuit(5, depth=6, seed=4), bit_terms, indices)

    @ddt.data(*range(10))
    def test_random_circuit(self, seed):
        """A random observable on a random circuit."""
        rng = np.random.default_rng(seed)
        weight = int(rng.integers(1, 6))
        indices = [int(i) for i in rng.choice(5, weight, replace=False)]
        bit_terms = "".join(rng.choice(list("XYZ01+-lr"), weight))
        self.assertSameExpectation(random_circuit(5, depth=6, seed=seed), bit_terms, indices)

    @ddt.data(
        ("ry", [0.3] * 10),
        ("ry", [0.2 * np.pi] * 10),
        ("rx", [0.4, -0.1, -0.3]),
    )
    @ddt.unpack
    def test_run_of_one_rotation(self, name, angles):
        """A run of the same rotation, also one that sums to 0 or 2π."""
        qc = QuantumCircuit(1)
        qc.h(0)
        for angle in angles:
            getattr(qc, name)(angle, 0)
        self.assertSameForAllTerms(qc)

    def test_circuit_and_its_inverse(self):
        """``qc.compose(qc.inverse())`` next to a gate on another qubit."""
        inner = random_circuit(2, depth=4, seed=5)
        qc = QuantumCircuit(3)
        qc.compose(inner, [0, 1], inplace=True)
        qc.compose(inner.inverse(), [0, 1], inplace=True)
        qc.h(2)
        self.assertSameForAllTerms(qc)


class TestLightConeTrivialCircuits(QiskitTestCase):
    """Empty and degenerate circuits."""

    def test_empty_circuit(self):
        """An empty circuit, with and without clbits, is returned as it is."""
        self.assertEqual(QuantumCircuit(2), LightCone("X", [0])(QuantumCircuit(2)))
        self.assertEqual(QuantumCircuit(2, 2), LightCone()(QuantumCircuit(2, 2)))

    def test_only_measurements(self):
        """A circuit made of measurements is kept as it is."""
        qc = QuantumCircuit(2, 3)
        qc.measure([0, 1, 0], [0, 1, 2])
        self.assertEqual(qc, LightCone()(qc))

    def test_observable_on_untouched_qubit(self):
        """An observable on a qubit no gate touches removes every gate."""
        qc = QuantumCircuit(3).compose(random_circuit(2, depth=3, seed=6), [0, 1])
        for term in "XZ+1":
            with self.subTest(term=term):
                self.assertEqual(QuantumCircuit(3), LightCone(term, [2])(qc))


if __name__ == "__main__":
    unittest.main()
