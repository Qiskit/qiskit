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
                   'UCPauliRotGate', 'SingleQubitUnitary', 'MCXGate'}
        cls._gate_classes = []
        for aclass in class_list:
            if aclass.__name__ not in exclude:
                cls._gate_classes.append(aclass)

    def pgdef(self, gate):
        for rule in gate.definition:
            qubit_inds = [qubit.index for qubit in rule[1]]
            params = np.array(rule[0].params)
            print(f'{rule[0].name:5s} {str(qubit_inds):8s} {params} ')

    def test_equivalence_phase(self):
        """Test that the equivalent circuits from the equivalency_library
        have equal matrix representations"""
        for gate_class in self._gate_classes:
        #for gate_class in [RYYGate]:
            with self.subTest(i=gate_class):
                n_params = len(_get_free_params(gate_class))
                params = [0.1 * i for i in range(1, n_params+1)]
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
                        print(gate.name)
                        # np.set_printoptions(precision=4, linewidth=200, suppress=True)
                        # print('')
                        # print(gate.to_matrix_hide())
                        # print('')                        
                        # print(op1.data)
                        # print('')                        
                        # print(op2.data)
                        # import ipdb;ipdb.set_trace()
                        self.assertEqual(op1, op2)
                        try:
                            tomat = gate.to_matrix()
                        except:
                            try:
                                tomat = gate.to_matrix_hide()
                            except:
                                pass
                            else:
                                self.assertTrue(np.allclose(op1.data, tomat))
                                print('ok, HIDDEN')
                        else:
                            self.assertTrue(np.allclose(op1.data, tomat))
                            print('ok')

    def test_phase_gate(self):
        from qiskit import transpile, QuantumRegister
        np.set_printoptions(precision=2, linewidth=200, suppress=True)
        def show_gates(gate):
            print(gate.to_matrix())
            for o, qreg, _ in gate.definition:
                print(o.name, [qubit.index for qubit in qreg], o.params)
                
        class PhaseGate(Gate):
            def __init__(self, theta):
                """Create new r single-qubit gate."""
                super().__init__('gph', 1, [theta])

            def _define(self):
                theta = self.params[0]
                q = QuantumRegister(1, 'q')
                self.definition = [
                    (U3Gate(np.pi, theta, np.pi+theta), [q[0]], []),
                    (XGate(),  [q[0]], []),
                    ]

            def inverse(self):
                return PhaseGate(-self.params[0])

        basis = ['id', 'u1', 'u3', 'cx']
        from qiskit.quantum_info import ScalarOp, Operator
        n_qubits = 2
        dim = 2**n_qubits
        op = Operator(np.eye(dim))
        op *= ScalarOp(dim, np.exp(1j * np.pi/4))
        instr = op.to_instruction()
        show_gates(instr)
        
        op2 = Operator(np.eye(dim))
        op2 *= ScalarOp(dim, np.exp(1j * np.pi/2))
        instr2 = op2.to_instruction()
        show_gates(instr2)
        
        op3 = Operator(np.eye(dim))
        op3 *= ScalarOp(dim, np.exp(1j * np.pi/2))
        instr3 = op3.to_instruction()
        show_gates(instr3)

        phase_gate = PhaseGate(np.pi)
        op4 = Operator(phase_gate)
        show_gates(op4.to_instruction())

        qc = QuantumCircuit(n_qubits)
        qc.append(PhaseGate(np.pi/8), qc.qregs)
        
        print(qc)
        print(Operator(qc).data)
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
