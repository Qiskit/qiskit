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

import inspect

import numpy as np
import scipy
from ddt import ddt, data, unpack

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator
from qiskit.test import QiskitTestCase
from qiskit.circuit import Parameter, ParameterVector, Gate, ControlledGate

from qiskit.circuit.library import standard_gates
from qiskit.circuit.library import (
    HGate, CHGate, IGate, RGate, RXGate, CRXGate, RYGate, CRYGate, RZGate,
    CRZGate, SGate, SdgGate, CSwapGate, TGate, TdgGate, U1Gate, CU1Gate,
    U2Gate, U3Gate, CU3Gate, XGate, CXGate, CCXGate, YGate, CYGate,
    ZGate, CZGate, RYYGate, PhaseGate, CPhaseGate, UGate, CUGate,
    SXGate, SXdgGate, CSXGate, RVGate
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

    def test_rv_definition(self):
        """Test R(v) gate to_matrix and definition.
        """
        qreg = QuantumRegister(1)
        circ = QuantumCircuit(qreg)
        vec = np.array([0.1, 0.2, 0.3], dtype=float)
        rvgate = RVGate(*vec)
        circ.rv(*vec, 0)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_rv_r_equiv(self):
        """Test R(v) gate is equivalent to R gate.
        """
        import math
        import numpy
        from scipy.spatial.transform import Rotation
        theta = np.pi / 5
        phi = np.pi / 3
        rgate = RGate(theta, phi)
        axis = np.array([math.cos(phi), math.sin(phi), 0])  # RGate axis
        rotvec = theta * axis
        rv = RVGate(*rotvec)
        np.set_printoptions(linewidth=250, precision=3, suppress=True)
        print('')
        print(rgate.to_matrix())
        print(rv.to_matrix())
        print(Operator(rv.definition).data)
        # self.assertTrue(numpy.array_equal(
        #     self._su2so3(rgate.to_matrix()), rv._rot.as_matrix()))
        self.assertTrue(numpy.array_equal(rgate.to_matrix(), rv.to_matrix()))
        #self.assertTrue(numpy.array_equal(rgate.to_matrix(), rv.to_matrix2()))

    def _su2so3(self, su2):
        """Convert su2 matrix to so3.

        see 'Topology, Geometry and Gauge fields', Gregory Naber
        """
        a, b = su2[0, 0].real, su2[0, 0].imag
        c, d = su2[0, 1].real, su2[0, 1].imag
        so3 = np.zeros([3, 3], dtype=float)
        so3[0, 0] = a**2 - b**2 - c**2 + d**2
        so3[0, 1] = 2 * (a*b + c*d)
        so3[0, 2] = 2 * (-a*c + b*d)
        so3[1, 0] = 2 * (-a*b + c*d)
        so3[1, 1] = a**2 - b**2 + c**2 - d**2
        so3[1, 2] = 2 * (a*d + b*c)
        so3[2, 0] = 2 * (a*c - b*d)
        so3[2, 1] = 2 * (b*c - a*d)
        so3[2, 2] = a**2 + b**2 - c**2 - d**2
        return so3

    def test_rv_param_def(self):
        """Test parameter expression definition agrees with numeric."""
        from scipy.linalg import expm, norm
        #θ, φ, ψ = Parameter('θ'), Parameter('φ'), Parameter('ψ')
        np.set_printoptions(linewidth=250, precision=3, suppress=True)
        # theta = np.pi / 5
        # phi = np.pi / 3
        thetar = np.pi / 2
        phi = np.pi
        rgate = RGate(thetar, phi)
        #axis = np.array([np.cos(phi), np.sin(phi), 0])  # RGate axis
        axis = np.array([0,0,-1])
        axis = axis/norm(axis)
        rotvec = thetar * axis

        print('\nrotvec: ', rotvec)
        #vec_num = np.array([0.1, 0.2, 0.3])
        vec_num = rotvec
        #vec_num = np.array([1, 0, 0])
        #vec_num = np.array([0, 1, 0])

        from scipy.spatial.transform import Rotation
        rot = Rotation.from_rotvec(vec_num)
        euler = rot.as_euler('zyz')
        
        pm = np.array((
            ((0, 1), (1, 0)),
            ((0, -1j), (1j, 0)),
            ((1, 0), (0, -1))
        ))
        vx, vy, vz = vec_num
        from numpy import sqrt, tan, cos, sin, arctan2, arccos, arcsin
        L = sqrt(vx**2 + vy**2 + vz**2)
        tanL = tan(L/2)
        Rv = expm(-1j * np.einsum('i,ijk', vec_num, pm)/2)
        Rv2 = cos(L/2) * np.eye(2) - 1j*sin(L/2)*np.einsum('i,ijk', vec_num/L, pm)

        if vy or vz
            theta = arctan2(vy*vz*(1-cos(L)) - vx*L*sin(L), vx*vz*(1-cos(L)) + vy*L*sin(L))
        else:
            #theta = -thetar
            theta = arctan2(vy*vz*(1-cos(L)) - vx*L*sin(L), vx*vz*(1-cos(L)) + vy*L*sin(L))
        if vy or vz:
            phi = 2*arctan2(vx*vz*(1-cos(L)) + vy*L*sin(L),
                            (vx**2 * cos(L) + vx**2 + vy**2 * cos(L) + vy**2 + 2*vz**2) * cos(theta))
        else:
            phi = -thetar
        qc2 = QuantumCircuit(1)
        if vx or vy:
            lam = 2*arccos(np.round(L*sin(phi/2) * (vx*sin(theta/2) + vy*cos(theta/2)) / ((vx**2 + vy**2)*sin(L/2)),
                                    decimals=8))
        else:
            lam = thetar
        # if not (vy and vz):
        #     qc2.global_phase = thetar
        # numerator = vy*vz*tanL - L*vx - sqrt(vx**4 + 2*(vx*vy)**2 + vy**4 +
        #                                      ((vx*vz)**2 + (vy*vz)**2)/cos(L/2)**2)
        # denominator = vx*vz*tanL + L*vy
        # theta = 2*arctan2(numerator, denominator)
        # lam = 2*arctan2(vx + tan(theta/2) - vy, vx+vy*tan(theta/2))
        # phi = 2 * arcsin((vx * sin((lam-theta)/2) + vy * cos((lam-theta)/2)) / L)
        
        if theta:
            qc2.rz(theta,0)
        if phi:
            qc2.ry(phi, 0)
        if lam:
            qc2.rz(lam, 0)
        print(theta, phi, lam)
        print('qc2:\n', Operator(qc2).data)
        #print(Operator(qc2).data @ np.linalg.inv(Rv))
#        import ipdb; ipdb.set_trace()
        
        # vec_sym = ParameterVector('vec', length=3)
        # rv_sym = RVGate(*vec_sym)
        # qc_sym = rv_sym.definition

        rv_num = RVGate(*vec_num)
        qc_num = rv_num.definition

 #       binding = {key:value for key, value in zip(vec_sym, vec_num)}
  #      qc_sym2 = qc_sym.bind_parameters(binding)
        
        from scipy.spatial.transform import Rotation
        print('')
        print('Rv\n', Rv)
        print('Rv2\n', Rv2)
        print('unitary check Rv\n', Rv @ np.linalg.inv(Rv))
        print('rgate.to_matrix()\n', rgate.to_matrix())
        return
        import ipdb; ipdb.set_trace()
        print(qc_sym2)
        print(qc_num)
        print('rv_num.to_matrix()\n', rv_num.to_matrix())
        import ipdb; ipdb.set_trace()
        print(Operator(qc_sym2).data)
        print(Operator(qc_num).data)
 #      import ipdb; ipdb.set_trace()

@ddt
class TestStandardGates(QiskitTestCase):
    """Standard Extension Test."""
    @unpack
    @data(
        *inspect.getmembers(
            standard_gates,
            predicate=lambda value: (inspect.isclass(value)
                                     and issubclass(value, Gate)))
    )
    def test_definition_parameters(self, class_name, gate_class):
        """Verify definitions from standard library include correct parameters."""

        free_params = _get_free_params(gate_class)
        n_params = len(free_params)
        param_vector = ParameterVector('th', n_params)

        if class_name in ('MCPhaseGate', 'MCU1Gate'):
            param_vector = param_vector[:-1]
            gate = gate_class(*param_vector, num_ctrl_qubits=2)
        elif class_name in ('MCXGate', 'MCXGrayCode', 'MCXRecursive', 'MCXVChain'):
            num_ctrl_qubits = 2
            param_vector = param_vector[:-1]
            gate = gate_class(num_ctrl_qubits, *param_vector)
        elif class_name == 'MSGate':
            num_qubits = 2
            param_vector = param_vector[:-1]
            gate = gate_class(num_qubits, *param_vector)
        else:
            gate = gate_class(*param_vector)

        if gate.definition is not None:
            self.assertEqual(gate.definition.parameters, set(param_vector))

    @unpack
    @data(
        *inspect.getmembers(
            standard_gates,
            predicate=lambda value: (inspect.isclass(value)
                                     and issubclass(value, Gate)))
    )
    def test_inverse(self, class_name, gate_class):
        """Verify self-inverse pair yield identity for all standard gates."""

        free_params = _get_free_params(gate_class)
        n_params = len(free_params)
        float_vector = [0.1 + 0.1*i for i in range(n_params)]

        if class_name in ('MCPhaseGate', 'MCU1Gate'):
            float_vector = float_vector[:-1]
            gate = gate_class(*float_vector, num_ctrl_qubits=2)
        elif class_name in ('MCXGate', 'MCXGrayCode', 'MCXRecursive', 'MCXVChain'):
            num_ctrl_qubits = 3
            float_vector = float_vector[:-1]
            gate = gate_class(num_ctrl_qubits, *float_vector)
        elif class_name == 'MSGate':
            num_qubits = 3
            float_vector = float_vector[:-1]
            gate = gate_class(num_qubits, *float_vector)
        else:
            gate = gate_class(*float_vector)

        from qiskit.quantum_info.operators.predicates import is_identity_matrix

        self.assertTrue(is_identity_matrix(Operator(gate).dot(gate.inverse()).data))

        if gate.definition is not None:
            self.assertTrue(is_identity_matrix(Operator(gate).dot(gate.definition.inverse()).data))
            self.assertTrue(is_identity_matrix(Operator(gate).dot(gate.inverse().definition).data))


class TestGateEquivalenceEqual(QiskitTestCase):
    """Test the decomposition of a gate in terms of other gates
    yields the same matrix as the hardcoded matrix definition."""

    @classmethod
    def setUpClass(cls):
        class_list = Gate.__subclasses__() + ControlledGate.__subclasses__()
        exclude = {'ControlledGate', 'DiagonalGate', 'UCGate', 'MCGupDiag',
                   'MCU1Gate', 'UnitaryGate', 'HamiltonianGate', 'MCPhaseGate',
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
                if gate_class.__name__ in ['MSGate']:
                    params[0] = 2
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
        ZGate, CZGate, RYYGate, PhaseGate, CPhaseGate, UGate, CUGate,
        SXGate, SXdgGate, CSXGate
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
