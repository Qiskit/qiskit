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

import numpy as np
from ddt import ddt, data

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.test import QiskitTestCase
from qiskit.circuit import ParameterVector, Gate, ControlledGate


from qiskit.circuit.library import (
    HGate, CHGate, IGate, RGate, RXGate, CRXGate, RYGate, CRYGate, RZGate,
    CRZGate, SGate, SdgGate, CSwapGate, TGate, TdgGate, U1Gate, CU1Gate,
    U2Gate, U3Gate, CU3Gate, XGate, CXGate, CCXGate, YGate, CYGate,
    ZGate, CZGate, RYYGate
)

from qiskit.circuit.library.standard_gates.equivalence_library import (
    StandardEquivalenceLibrary as std_eqlib
)

from .gate_utils import _get_free_params


class TestGateDefinitions(QiskitTestCase):
    """Test the decomposition of a gate in terms of other gates
    yields the equivalent matrix as the hardcoded matrix definition
    up to a global phase."""

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


class TestGateEquivalenceEqual(QiskitTestCase):
    """Test the decomposition of a gate in terms of other gates
    yields the same matrix as the hardcoded matrix definition."""

    @classmethod
    def setUpClass(cls):
        class_list = Gate.__subclasses__() + ControlledGate.__subclasses__()
        exclude = {'ControlledGate', 'DiagonalGate', 'UCGate', 'MCGupDiag',
                   'MCU1Gate', 'UnitaryGate', 'HamiltonianGate',
                   'UCPauliRotGate', 'SingleQubitUnitary', 'MCXGate',
                   'VariadicZeroParamGate'}
        cls._gate_classes = []
        for aclass in class_list:
            if aclass.__name__ not in exclude:
                cls._gate_classes.append(aclass)

    def test_equivalence_phase(self):
        """Test that the equivalent circuits from the equivalency_library
        have equal matrix representations"""
        for gate_class in self._gate_classes:
            with self.subTest(i=gate_class):
                n_params = len(_get_free_params(gate_class))
                params = [0.1 * i for i in range(1, n_params+1)]
                if gate_class.__name__ == 'RXXGate':
                    params = [np.pi/2]
                params = [-np.pi/2 * i for i in range(1, n_params+1)]
                if gate_class.__name__ in ['MSGate']:
                    params[0] = 2
                elif gate_class in ['MCU1Gate']:
                    params[1] = 2
                gate = gate_class(*params)
                equiv_lib_list = std_eqlib.get_entry(gate)
                for ieq, equivalency in enumerate(equiv_lib_list):
                    with self.subTest(msg=gate.name + '_' + str(ieq)):
                        op1 = Operator(gate)
                        op2 = Operator(equivalency)
                        self.assertEqual(op1, op2)


@ddt
class TestStandardEquivalenceLibrary(QiskitTestCase):
    """Standard Extension Test."""

    @data(
        HGate, CHGate, IGate, RGate, RXGate, CRXGate, RYGate, CRYGate, RZGate,
        CRZGate, SGate, SdgGate, CSwapGate, TGate, TdgGate, U1Gate, CU1Gate,
        U2Gate, U3Gate, CU3Gate, XGate, CXGate, CCXGate, YGate, CYGate,
        ZGate, CZGate, RYYGate
    )
    def test_definition_parameters(self, gate_class):
        """Verify decompositions from standard equivalence library match definitions."""
        n_params = len(_get_free_params(gate_class))
        param_vector = ParameterVector('th', n_params)
        float_vector = [0.1 * i for i in range(n_params)]

        param_gate = gate_class(*param_vector)
        float_gate = gate_class(*float_vector)

        param_entry = std_eqlib.get_entry(param_gate)
        float_entry = std_eqlib.get_entry(float_gate)

        if not param_gate.definition or not param_gate.definition.data:
            return

        self.assertGreaterEqual(len(param_entry), 1)
        self.assertGreaterEqual(len(float_entry), 1)

        param_qc = QuantumCircuit(param_gate.num_qubits)
        float_qc = QuantumCircuit(float_gate.num_qubits)

        param_qc.append(param_gate, param_qc.qregs[0])
        float_qc.append(float_gate, float_qc.qregs[0])

        self.assertTrue(any(equiv == param_qc.decompose()
                            for equiv in param_entry))
        self.assertTrue(any(equiv == float_qc.decompose()
                            for equiv in float_entry))
