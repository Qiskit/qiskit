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


"""Test hardcoded decomposition rules and matrix definitions for standard gates."""

from inspect import signature

from ddt import ddt, data

from qiskit import QuantumCircuit, BasicAer, execute
from qiskit.quantum_info import Operator
from qiskit.test import QiskitTestCase
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.circuit import ParameterVector


from qiskit.extensions.standard import (
    HGate, CHGate, IGate, RGate, RXGate, CRXGate, RYGate, CRYGate, RZGate,
    CRZGate, SGate, SdgGate, CSwapGate, TGate, TdgGate, U1Gate, CU1Gate,
    U2Gate, U3Gate, CU3Gate, XGate, CXGate, CCXGate, YGate, CYGate,
    ZGate, CZGate
)

from qiskit.extensions.standard.equivalence_library import StandardEquivalenceLibrary as std_eqlib


class TestGateDefinitions(QiskitTestCase):
    """Test the decomposition of a gate in terms of other gates
    yields the same matrix as the hardcoded matrix definition."""

    def test_ch_definition(self):  # TODO: expand this to all gates
        """Test ch gate matrix and definition.
        """
        circ = QuantumCircuit(2)
        circ.ch(0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_ccx_definition(self):
        """Test ccx gate matrix and definition.
        """
        circ = QuantumCircuit(3)
        circ.ccx(0, 1, 2)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_crz_definition(self):
        """Test crz gate matrix and definition.
        """
        circ = QuantumCircuit(2)
        circ.crz(1, 0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_cry_definition(self):
        """Test cry gate matrix and definition.
        """
        circ = QuantumCircuit(2)
        circ.cry(1, 0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_crx_definition(self):
        """Test crx gate matrix and definition.
        """
        circ = QuantumCircuit(2)
        circ.crx(1, 0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_cswap_definition(self):
        """Test cswap gate matrix and definition.
        """
        circ = QuantumCircuit(3)
        circ.cswap(0, 1, 2)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_cu1_definition(self):
        """Test cu1 gate matrix and definition.
        """
        circ = QuantumCircuit(2)
        circ.cu1(1, 0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_cu3_definition(self):
        """Test cu3 gate matrix and definition.
        """
        circ = QuantumCircuit(2)
        circ.cu3(1, 1, 1, 0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_cx_definition(self):
        """Test cx gate matrix and definition.
        """
        circ = QuantumCircuit(2)
        circ.cx(0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_ryy_matrix_representation(self):
        """Test the matrix representation of the RYY gate.
        """
        from qiskit.extensions.standard.ryy import RYYGate
        theta = 0.991283
        expected = RYYGate(theta).to_matrix()

        circuit = QuantumCircuit(2)
        circuit.ryy(theta, 0, 1)
        backend = BasicAer.get_backend('unitary_simulator')
        simulated = execute(circuit, backend).result().get_unitary()

        self.assertTrue(matrix_equal(expected, simulated))


@ddt
class TestStandardEquivalenceLibrary(QiskitTestCase):
    """Standard Extension Test."""

    @data(
        HGate, CHGate, IGate, RGate, RXGate, CRXGate, RYGate, CRYGate, RZGate,
        CRZGate, SGate, SdgGate, CSwapGate, TGate, TdgGate, U1Gate, CU1Gate,
        U2Gate, U3Gate, CU3Gate, XGate, CXGate, CCXGate, YGate, CYGate,
        ZGate, CZGate
    )
    def test_definition_parameters(self, gate_class):
        """Verify decompositions from standard equivalence library match definitions."""

        non_param_attributes = {'self', 'label', 'num_ctrl_qubits'}
        n_params = len(set(signature(gate_class.__init__).parameters) - non_param_attributes)
        param_vector = ParameterVector('th', n_params)
        float_vector = [0.1 * i for i in range(n_params)]

        param_gate = gate_class(*param_vector)
        float_gate = gate_class(*float_vector)

        param_entry = std_eqlib.get_entry(param_gate)
        float_entry = std_eqlib.get_entry(float_gate)

        if not param_gate.definition:
            self.assertEqual(len(param_entry), 0)
            self.assertEqual(len(float_entry), 0)
            return

        if gate_class is CXGate:
            # CXGate currently has a definition in terms of CXGate.
            self.assertEqual(len(param_entry), 0)
            self.assertEqual(len(float_entry), 0)
            return

        self.assertEqual(len(param_entry), 1)
        self.assertEqual(len(float_entry), 1)

        param_qc = QuantumCircuit(param_gate.num_qubits)
        float_qc = QuantumCircuit(float_gate.num_qubits)

        param_qc.append(param_gate, param_qc.qregs[0])
        float_qc.append(float_gate, float_qc.qregs[0])

        self.assertEqual(param_entry[0], param_qc.decompose())
        self.assertEqual(float_entry[0], float_qc.decompose())
