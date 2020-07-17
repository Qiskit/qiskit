# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test circuits with variable parameters."""

import pickle
from operator import add, sub, mul, truediv

from test import combine
import numpy

from ddt import ddt, data

import qiskit
from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Gate, Instruction
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.exceptions import CircuitError
from qiskit.compiler import assemble, transpile
from qiskit.execute import execute
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOurense
from qiskit.tools import parallel_map


def raise_if_parameter_table_invalid(circuit):  # pylint: disable=invalid-name
    """Validates the internal consistency of a ParameterTable and its
    containing QuantumCircuit. Intended for use in testing.

    Raises:
       CircuitError: if QuantumCircuit and ParameterTable are inconsistent.
    """

    table = circuit._parameter_table

    # Assert parameters present in circuit match those in table.
    circuit_parameters = {parameter
                          for instr, qargs, cargs in circuit._data
                          for param in instr.params
                          for parameter in param.parameters
                          if isinstance(param, ParameterExpression)}
    table_parameters = set(table._table.keys())

    if circuit_parameters != table_parameters:
        raise CircuitError('Circuit/ParameterTable Parameter mismatch. '
                           'Circuit parameters: {}. '
                           'Table parameters: {}.'.format(
                               circuit_parameters,
                               table_parameters))

    # Assert parameter locations in table are present in circuit.
    circuit_instructions = [instr
                            for instr, qargs, cargs in circuit._data]

    for parameter, instr_list in table.items():
        for instr, param_index in instr_list:
            if instr not in circuit_instructions:
                raise CircuitError('ParameterTable instruction not present '
                                   'in circuit: {}.'.format(instr))

            if not isinstance(instr.params[param_index], ParameterExpression):
                raise CircuitError('ParameterTable instruction does not have a '
                                   'ParameterExpression at param_index {}: {}.'
                                   ''.format(param_index, instr))

            if parameter not in instr.params[param_index].parameters:
                raise CircuitError('ParameterTable instruction parameters does '
                                   'not match ParameterTable key. Instruction '
                                   'parameters: {} ParameterTable key: {}.'
                                   ''.format(instr.params[param_index].parameters,
                                             parameter))

    # Assert circuit has no other parameter locations other than those in table.
    for instr, qargs, cargs in circuit._data:
        for param_index, param in enumerate(instr.params):
            if isinstance(param, ParameterExpression):
                parameters = param.parameters

                for parameter in parameters:
                    if (instr, param_index) not in table[parameter]:
                        raise CircuitError('Found parameterized instruction not '
                                           'present in table. Instruction: {} '
                                           'param_index: {}'.format(instr,
                                                                    param_index))


@ddt
class TestParameters(QiskitTestCase):
    """QuantumCircuit Operations tests."""

    def test_gate(self):
        """Test instantiating gate with variable parameters"""
        theta = Parameter('θ')
        theta_gate = Gate('test', 1, params=[theta])
        self.assertEqual(theta_gate.name, 'test')
        self.assertIsInstance(theta_gate.params[0], Parameter)

    def test_compile_quantum_circuit(self):
        """Test instantiating gate with variable parameters"""
        theta = Parameter('θ')
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        backend = BasicAer.get_backend('qasm_simulator')
        qc_aer = transpile(qc, backend)
        self.assertIn(theta, qc_aer.parameters)

    def test_duplicate_name_on_append(self):
        """Test adding a second parameter object with the same name fails."""
        param_a = Parameter('a')
        param_a_again = Parameter('a')
        qc = QuantumCircuit(1)
        qc.rx(param_a, 0)
        self.assertRaises(CircuitError, qc.rx, param_a_again, 0)

    def test_get_parameters(self):
        """Test instantiating gate with variable parameters"""
        from qiskit.circuit.library.standard_gates.rx import RXGate
        theta = Parameter('θ')
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        rxg = RXGate(theta)
        qc.append(rxg, [qr[0]], [])
        vparams = qc._parameter_table
        self.assertEqual(len(vparams), 1)
        self.assertIs(theta, next(iter(vparams)))
        self.assertIs(rxg, vparams[theta][0][0])

    def test_is_parameterized(self):
        """Test checking if a gate is parameterized (bound/unbound)"""
        from qiskit.circuit.library.standard_gates.h import HGate
        from qiskit.circuit.library.standard_gates.rx import RXGate
        theta = Parameter('θ')
        rxg = RXGate(theta)
        self.assertTrue(rxg.is_parameterized())
        theta_bound = theta.bind({theta: 3.14})
        rxg = RXGate(theta_bound)
        self.assertFalse(rxg.is_parameterized())
        h_gate = HGate()
        self.assertFalse(h_gate.is_parameterized())

    def test_fix_variable(self):
        """Test setting a variable to a constant value"""
        theta = Parameter('θ')
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u3(0, theta, 0, qr)

        # test for both `bind_parameters` and `assign_parameters`
        for assign_fun in ['bind_parameters', 'assign_parameters']:
            with self.subTest(assign_fun=assign_fun):
                bqc = getattr(qc, assign_fun)({theta: 0.5})
                self.assertEqual(float(bqc.data[0][0].params[0]), 0.5)
                self.assertEqual(float(bqc.data[1][0].params[1]), 0.5)
                bqc = getattr(qc, assign_fun)({theta: 0.6})
                self.assertEqual(float(bqc.data[0][0].params[0]), 0.6)
                self.assertEqual(float(bqc.data[1][0].params[1]), 0.6)

    def test_multiple_parameters(self):
        """Test setting multiple parameters"""
        theta = Parameter('θ')
        x = Parameter('x')
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u3(0, theta, x, qr)
        self.assertEqual(qc.parameters, {theta, x})

    def test_multiple_named_parameters(self):
        """Test setting multiple named/keyword argument based parameters"""
        theta = Parameter(name='θ')
        x = Parameter(name='x')
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u3(0, theta, x, qr)
        self.assertEqual(theta.name, 'θ')
        self.assertEqual(qc.parameters, {theta, x})

    def test_partial_binding(self):
        """Test that binding a subset of circuit parameters returns a new parameterized circuit."""
        theta = Parameter('θ')
        x = Parameter('x')
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u3(0, theta, x, qr)

        # test for both `bind_parameters` and `assign_parameters`
        for assign_fun in ['bind_parameters', 'assign_parameters']:
            with self.subTest(assign_fun=assign_fun):
                pqc = getattr(qc, assign_fun)({theta: 2})

                self.assertEqual(pqc.parameters, {x})

                self.assertEqual(float(pqc.data[0][0].params[0]), 2)
                self.assertEqual(float(pqc.data[1][0].params[1]), 2)

    @data(True, False)
    def test_mixed_binding(self, inplace):
        """Test we can bind a mixed dict with Parameter objects and floats."""
        theta = Parameter('θ')
        x, new_x = Parameter('x'), Parameter('new_x')
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u3(0, theta, x, qr)

        pqc = qc.assign_parameters({theta: 2, x: new_x}, inplace=inplace)
        if inplace:
            self.assertEqual(qc.parameters, {new_x})
        else:
            self.assertEqual(pqc.parameters, {new_x})

    def test_expression_partial_binding(self):
        """Test that binding a subset of expression parameters returns a new
        parameterized circuit."""
        theta = Parameter('θ')
        phi = Parameter('phi')

        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta + phi, qr)

        # test for both `bind_parameters` and `assign_parameters`
        for assign_fun in ['bind_parameters', 'assign_parameters']:
            with self.subTest(assign_fun=assign_fun):
                pqc = getattr(qc, assign_fun)({theta: 2})

                self.assertEqual(pqc.parameters, {phi})

                self.assertTrue(isinstance(pqc.data[0][0].params[0], ParameterExpression))
                self.assertEqual(str(pqc.data[0][0].params[0]), 'phi + 2')

                fbqc = getattr(pqc, assign_fun)({phi: 1})

                self.assertEqual(fbqc.parameters, set())
                self.assertTrue(isinstance(fbqc.data[0][0].params[0], ParameterExpression))
                self.assertEqual(float(fbqc.data[0][0].params[0]), 3)

    def test_expression_partial_binding_zero(self):
        """Verify that binding remains possible even if a previous partial bind
        would reduce the expression to zero.
        """
        theta = Parameter('theta')
        phi = Parameter('phi')

        qc = QuantumCircuit(1)
        qc.u1(theta * phi, 0)

        # test for both `bind_parameters` and `assign_parameters`
        for assign_fun in ['bind_parameters', 'assign_parameters']:
            with self.subTest(assign_fun=assign_fun):
                pqc = getattr(qc, assign_fun)({theta: 0})

                self.assertEqual(pqc.parameters, {phi})

                self.assertTrue(isinstance(pqc.data[0][0].params[0], ParameterExpression))
                self.assertEqual(str(pqc.data[0][0].params[0]), '0')

                fbqc = getattr(pqc, assign_fun)({phi: 1})

                self.assertEqual(fbqc.parameters, set())
                self.assertTrue(isinstance(fbqc.data[0][0].params[0], ParameterExpression))
                self.assertEqual(float(fbqc.data[0][0].params[0]), 0)

    def test_raise_if_assigning_params_not_in_circuit(self):
        """Verify binding parameters which are not present in the circuit raises an error."""
        x = Parameter('x')
        y = Parameter('y')
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        # test for both `bind_parameters` and `assign_parameters`
        for assign_fun in ['bind_parameters', 'assign_parameters']:
            qc = QuantumCircuit(qr)
            with self.subTest(assign_fun=assign_fun):
                qc.u1(0.1, qr[0])
                self.assertRaises(CircuitError, getattr(qc, assign_fun), {x: 1})

                qc.u1(x, qr[0])
                self.assertRaises(CircuitError, getattr(qc, assign_fun), {x: 1, y: 2})

    def test_gate_multiplicity_binding(self):
        """Test binding when circuit contains multiple references to same gate"""
        from qiskit.circuit.library.standard_gates.rz import RZGate
        qc = QuantumCircuit(1)
        theta = Parameter('theta')
        gate = RZGate(theta)
        qc.append(gate, [0], [])
        qc.append(gate, [0], [])
        # test for both `bind_parameters` and `assign_parameters`
        for assign_fun in ['bind_parameters', 'assign_parameters']:
            with self.subTest(assign_fun=assign_fun):
                qc2 = getattr(qc, assign_fun)({theta: 1.0})
                self.assertEqual(len(qc2._parameter_table), 0)
                for gate, _, _ in qc2.data:
                    self.assertEqual(float(gate.params[0]), 1.0)

    def test_circuit_generation(self):
        """Test creating a series of circuits parametrically"""
        theta = Parameter('θ')
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        backend = BasicAer.get_backend('qasm_simulator')
        qc_aer = transpile(qc, backend)

        # generate list of circuits
        for assign_fun in ['bind_parameters', 'assign_parameters']:
            with self.subTest(assign_fun=assign_fun):
                circs = []
                theta_list = numpy.linspace(0, numpy.pi, 20)
                for theta_i in theta_list:
                    circs.append(getattr(qc_aer, assign_fun)({theta: theta_i}))
                qobj = assemble(circs)
                for index, theta_i in enumerate(theta_list):
                    self.assertEqual(float(qobj.experiments[index].instructions[0].params[0]),
                                     theta_i)

    def test_circuit_composition(self):
        """Test preservation of parameters when combining circuits."""
        theta = Parameter('θ')
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc1 = QuantumCircuit(qr)
        qc1.rx(theta, qr)

        phi = Parameter('phi')
        qc2 = QuantumCircuit(qr, cr)
        qc2.ry(phi, qr)
        qc2.h(qr)
        qc2.measure(qr, cr)

        qc3 = qc1 + qc2
        self.assertEqual(qc3.parameters, {theta, phi})

    def test_composite_instruction(self):
        """Test preservation of parameters via parameterized instructions."""
        theta = Parameter('θ')
        qr1 = QuantumRegister(1, name='qr1')
        qc1 = QuantumCircuit(qr1)
        qc1.rx(theta, qr1)
        qc1.rz(numpy.pi/2, qr1)
        qc1.ry(theta, qr1)
        gate = qc1.to_instruction()
        self.assertEqual(gate.params, [theta])

        phi = Parameter('phi')
        qr2 = QuantumRegister(3, name='qr2')
        qc2 = QuantumCircuit(qr2)
        qc2.ry(phi, qr2[0])
        qc2.h(qr2)
        qc2.append(gate, qargs=[qr2[1]])
        self.assertEqual(qc2.parameters, {theta, phi})

    def test_parameter_name_conflicts_raises(self):
        """Verify attempting to add different parameters with matching names raises an error."""
        theta1 = Parameter('theta')
        theta2 = Parameter('theta')

        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        qc.u1(theta1, 0)

        self.assertRaises(CircuitError, qc.u1, theta2, 0)

    def test_bind_ryrz_vector(self):
        """Test binding a list of floats to a ParameterVector"""
        qc = QuantumCircuit(4)
        depth = 4
        theta = ParameterVector('θ', length=len(qc.qubits) * depth * 2)
        theta_iter = iter(theta)
        for _ in range(depth):
            for q in qc.qubits:
                qc.ry(next(theta_iter), q)
                qc.rz(next(theta_iter), q)
            for i, q in enumerate(qc.qubits[:-1]):
                qc.cx(qc.qubits[i], qc.qubits[i+1])
            qc.barrier()
        theta_vals = numpy.linspace(0, 1, len(theta)) * numpy.pi
        self.assertEqual(set(qc.parameters), set(theta.params))
        for assign_fun in ['bind_parameters', 'assign_parameters']:
            with self.subTest(assign_fun=assign_fun):
                bqc = getattr(qc, assign_fun)({theta: theta_vals})
                for gate_tuple in bqc.data:
                    if hasattr(gate_tuple[0], 'params') and gate_tuple[0].params:
                        self.assertIn(float(gate_tuple[0].params[0]), theta_vals)

    def test_compile_vector(self):
        """Test compiling a circuit with an unbound ParameterVector"""
        qc = QuantumCircuit(4)
        depth = 4
        theta = ParameterVector('θ', length=len(qc.qubits)*depth*2)
        theta_iter = iter(theta)
        for _ in range(depth):
            for q in qc.qubits:
                qc.ry(next(theta_iter), q)
                qc.rz(next(theta_iter), q)
            for i, q in enumerate(qc.qubits[:-1]):
                qc.cx(qc.qubits[i], qc.qubits[i+1])
            qc.barrier()
        backend = BasicAer.get_backend('qasm_simulator')
        qc_aer = transpile(qc, backend)
        for param in theta:
            self.assertIn(param, qc_aer.parameters)

    def test_instruction_ryrz_vector(self):
        """Test constructing a circuit from instructions with remapped ParameterVectors"""
        qubits = 5
        depth = 4
        ryrz = QuantumCircuit(qubits, name='ryrz')
        theta = ParameterVector('θ0', length=len(ryrz.qubits) * 2)
        theta_iter = iter(theta)
        for q in ryrz.qubits:
            ryrz.ry(next(theta_iter), q)
            ryrz.rz(next(theta_iter), q)

        cxs = QuantumCircuit(qubits-1, name='cxs')
        for i, _ in enumerate(cxs.qubits[:-1:2]):
            cxs.cx(cxs.qubits[2*i], cxs.qubits[2*i+1])

        paramvecs = []
        qc = QuantumCircuit(qubits)
        for i in range(depth):
            theta_l = ParameterVector('θ{}'.format(i+1), length=len(ryrz.qubits) * 2)
            ryrz_inst = ryrz.to_instruction(parameter_map={theta: theta_l})
            paramvecs += [theta_l]
            qc.append(ryrz_inst, qargs=qc.qubits)
            qc.append(cxs, qargs=qc.qubits[1:])
            qc.append(cxs, qargs=qc.qubits[:-1])
            qc.barrier()

        backend = BasicAer.get_backend('qasm_simulator')
        qc_aer = transpile(qc, backend)
        for vec in paramvecs:
            for param in vec:
                self.assertIn(param, qc_aer.parameters)

    def test_parameter_equality_through_serialization(self):
        """Verify parameters maintain their equality after serialization."""

        # pylint: disable=invalid-name
        x = Parameter('x')
        x1 = Parameter('x')

        x_p = pickle.loads(pickle.dumps(x))
        x1_p = pickle.loads(pickle.dumps(x1))

        self.assertEqual(x, x_p)
        self.assertEqual(x1, x1_p)

        self.assertNotEqual(x, x1_p)
        self.assertNotEqual(x1, x_p)

    def test_binding_parameterized_circuits_built_in_multiproc(self):
        """Verify subcircuits built in a subprocess can still be bound."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/2429

        num_processes = 4

        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)

        circuit = QuantumCircuit(qr, cr)
        parameters = [Parameter('x{}'.format(i))
                      for i in range(num_processes)]

        results = parallel_map(_construct_circuit,
                               parameters,
                               task_args=(qr,),
                               num_processes=num_processes)

        for qc in results:
            circuit += qc

        parameter_values = [{x: 1 for x in parameters}]

        qobj = assemble(circuit,
                        backend=BasicAer.get_backend('qasm_simulator'),
                        parameter_binds=parameter_values)

        self.assertEqual(len(qobj.experiments), 1)
        self.assertEqual(len(qobj.experiments[0].instructions), 4)
        self.assertTrue(all(len(inst.params) == 1
                            and isinstance(inst.params[0], ParameterExpression)
                            and float(inst.params[0]) == 1
                            for inst in qobj.experiments[0].instructions))

    def test_transpiling_multiple_parameterized_circuits(self):
        """Verify several parameterized circuits can be transpiled at once."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/2864

        qr = QuantumRegister(1)
        qc1 = QuantumCircuit(qr)
        qc2 = QuantumCircuit(qr)

        theta = Parameter('theta')

        qc1.u3(theta, 0, 0, qr[0])
        qc2.u3(theta, 3.14, 0, qr[0])

        circuits = [qc1, qc2]

        job = execute(circuits,
                      BasicAer.get_backend('unitary_simulator'),
                      shots=512,
                      parameter_binds=[{theta: 1}])

        self.assertTrue(len(job.result().results), 2)

    def test_transpile_across_optimization_levels(self):
        """Verify parameterized circuits can be transpiled with all default pass managers."""

        qc = QuantumCircuit(5, 5)

        theta = Parameter('theta')
        phi = Parameter('phi')

        qc.rx(theta, 0)
        qc.x(0)
        for i in range(5-1):
            qc.rxx(phi, i, i+1)

        qc.measure(range(5-1), range(5-1))

        for i in [0, 1, 2, 3]:
            transpile(qc, FakeOurense(), optimization_level=i)

    def test_repeated_gates_to_dag_and_back(self):
        """Verify circuits with repeated parameterized gates can be converted
        to DAG and back, maintaining consistency of circuit._parameter_table."""

        from qiskit.converters import circuit_to_dag, dag_to_circuit

        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        theta = Parameter('theta')

        qc.u1(theta, qr[0])

        double_qc = qc + qc
        test_qc = dag_to_circuit(circuit_to_dag(double_qc))

        for assign_fun in ['bind_parameters', 'assign_parameters']:
            with self.subTest(assign_fun=assign_fun):
                bound_test_qc = getattr(test_qc, assign_fun)({theta: 1})
                self.assertEqual(len(bound_test_qc.parameters), 0)

    def test_rebinding_instruction_copy(self):
        """Test rebinding a copied instruction does not modify the original."""

        theta = Parameter('th')

        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        instr = qc.to_instruction()

        qc1 = QuantumCircuit(1)
        qc1.append(instr, [0])

        for assign_fun in ['bind_parameters', 'assign_parameters']:
            with self.subTest(assign_fun=assign_fun):
                output1 = getattr(qc1, assign_fun)({theta: 0.1}).decompose()
                output2 = getattr(qc1, assign_fun)({theta: 0.2}).decompose()

                expected1 = QuantumCircuit(1)
                expected1.rx(0.1, 0)

                expected2 = QuantumCircuit(1)
                expected2.rx(0.2, 0)

                self.assertEqual(expected1, output1)
                self.assertEqual(expected2, output2)

    @combine(target_type=['gate', 'instruction'], parameter_type=['numbers', 'parameters'])
    def test_decompose_propagates_bound_parameters(self, target_type, parameter_type):
        """Verify bind-before-decompose preserves bound values."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/2482
        theta = Parameter('th')
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        if target_type == 'gate':
            inst = qc.to_gate()
        elif target_type == 'instruction':
            inst = qc.to_instruction()

        qc2 = QuantumCircuit(1)
        qc2.append(inst, [0])

        if parameter_type == 'numbers':
            bound_qc2 = qc2.assign_parameters({theta: 0.5})
            expected_parameters = set()
            expected_qc2 = QuantumCircuit(1)
            expected_qc2.rx(0.5, 0)
        else:
            phi = Parameter('ph')
            bound_qc2 = qc2.assign_parameters({theta: phi})
            expected_parameters = {phi}
            expected_qc2 = QuantumCircuit(1)
            expected_qc2.rx(phi, 0)

        decomposed_qc2 = bound_qc2.decompose()

        with self.subTest(msg='testing parameters of initial circuit'):
            self.assertEqual(qc2.parameters, {theta})

        with self.subTest(msg='testing parameters of bound circuit'):
            self.assertEqual(bound_qc2.parameters, expected_parameters)

        with self.subTest(msg='testing parameters of deep decomposed bound circuit'):
            self.assertEqual(decomposed_qc2.parameters, expected_parameters)

        with self.subTest(msg='testing deep decomposed circuit'):
            self.assertEqual(decomposed_qc2, expected_qc2)

    @combine(target_type=['gate', 'instruction'], parameter_type=['numbers', 'parameters'])
    def test_decompose_propagates_deeply_bound_parameters(self, target_type, parameter_type):
        """Verify bind-before-decompose preserves deeply bound values."""
        theta = Parameter('th')
        qc1 = QuantumCircuit(1)
        qc1.rx(theta, 0)

        if target_type == 'gate':
            inst = qc1.to_gate()
        elif target_type == 'instruction':
            inst = qc1.to_instruction()

        qc2 = QuantumCircuit(1)
        qc2.append(inst, [0])

        if target_type == 'gate':
            inst = qc2.to_gate()
        elif target_type == 'instruction':
            inst = qc2.to_instruction()

        qc3 = QuantumCircuit(1)
        qc3.append(inst, [0])

        if parameter_type == 'numbers':
            bound_qc3 = qc3.assign_parameters({theta: 0.5})
            expected_parameters = set()
            expected_qc3 = QuantumCircuit(1)
            expected_qc3.rx(0.5, 0)
        else:
            phi = Parameter('ph')
            bound_qc3 = qc3.assign_parameters({theta: phi})
            expected_parameters = {phi}
            expected_qc3 = QuantumCircuit(1)
            expected_qc3.rx(phi, 0)

        deep_decomposed_qc3 = bound_qc3.decompose().decompose()

        with self.subTest(msg='testing parameters of initial circuit'):
            self.assertEqual(qc3.parameters, {theta})

        with self.subTest(msg='testing parameters of bound circuit'):
            self.assertEqual(bound_qc3.parameters, expected_parameters)

        with self.subTest(msg='testing parameters of deep decomposed bound circuit'):
            self.assertEqual(deep_decomposed_qc3.parameters, expected_parameters)

        with self.subTest(msg='testing deep decomposed circuit'):
            self.assertEqual(deep_decomposed_qc3, expected_qc3)

    @data('gate', 'instruction')
    def test_executing_parameterized_instruction_bound_early(self, target_type):
        """Verify bind-before-execute preserves bound values."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/2482

        theta = Parameter('theta')

        sub_qc = QuantumCircuit(2)
        sub_qc.h(0)
        sub_qc.cx(0, 1)
        sub_qc.rz(theta, [0, 1])
        sub_qc.cx(0, 1)
        sub_qc.h(0)

        if target_type == 'gate':
            sub_inst = sub_qc.to_gate()
        elif target_type == 'instruction':
            sub_inst = sub_qc.to_instruction()

        unbound_qc = QuantumCircuit(2, 1)
        unbound_qc.append(sub_inst, [0, 1], [])
        unbound_qc.measure(0, 0)

        for assign_fun in ['bind_parameters', 'assign_parameters']:
            with self.subTest(assign_fun=assign_fun):
                bound_qc = getattr(unbound_qc, assign_fun)({theta: numpy.pi/2})

                shots = 1024
                job = execute(bound_qc, backend=BasicAer.get_backend('qasm_simulator'), shots=shots)
                self.assertDictAlmostEqual(job.result().get_counts(), {'1': shots}, 0.05 * shots)

    def test_num_parameters(self):
        """Test the num_parameters property."""
        with self.subTest(msg='standard case'):
            theta = Parameter('θ')
            x = Parameter('x')
            qc = QuantumCircuit(1)
            qc.rx(theta, 0)
            qc.u3(0, theta, x, 0)
            self.assertEqual(qc.num_parameters, 2)

        with self.subTest(msg='parameter vector'):
            params = ParameterVector('x', length=3)
            qc = QuantumCircuit(4)
            qc.rx(params[0], 2)
            qc.ry(params[1], 1)
            qc.rz(params[2], 3)
            self.assertEqual(qc.num_parameters, 3)

        with self.subTest(msg='no params'):
            qc = QuantumCircuit(1)
            qc.x(0)
            self.assertEqual(qc.num_parameters, 0)

    def test_to_instruction_after_inverse(self):
        """Verify converting an inverse generates a valid ParameterTable"""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/4235
        qc = QuantumCircuit(1)
        theta = Parameter('theta')
        qc.rz(theta, 0)

        inv_instr = qc.inverse().to_instruction()
        self.assertIsInstance(inv_instr, Instruction)

    def test_repeated_circuit(self):
        """Test repeating a circuit maintains the parameters."""
        qc = QuantumCircuit(1)
        theta = Parameter('theta')
        qc.rz(theta, 0)
        rep = qc.repeat(3)

        self.assertEqual(rep.parameters, {theta})

    def test_copy_after_inverse(self):
        """Verify circuit.inverse generates a valid ParameterTable."""
        qc = QuantumCircuit(1)
        theta = Parameter('theta')
        qc.rz(theta, 0)

        inverse = qc.inverse()
        self.assertIn(theta, inverse.parameters)
        raise_if_parameter_table_invalid(inverse)

    def test_copy_after_reverse(self):
        """Verify circuit.reverse generates a valid ParameterTable."""
        qc = QuantumCircuit(1)
        theta = Parameter('theta')
        qc.rz(theta, 0)

        reverse = qc.reverse_ops()
        self.assertIn(theta, reverse.parameters)
        raise_if_parameter_table_invalid(reverse)

    def test_copy_after_dot_data_setter(self):
        """Verify setting circuit.data generates a valid ParameterTable."""
        qc = QuantumCircuit(1)
        theta = Parameter('theta')
        qc.rz(theta, 0)

        qc.data = []
        self.assertEqual(qc.parameters, set())
        raise_if_parameter_table_invalid(qc)


def _construct_circuit(param, qr):
    qc = QuantumCircuit(qr)
    qc.ry(param, qr[0])
    return qc


@ddt
class TestParameterExpressions(QiskitTestCase):
    """Test expressions of Parameters."""

    supported_operations = [add, sub, mul, truediv]

    def test_raise_if_sub_unknown_parameters(self):
        """Verify we raise if asked to sub a parameter not in self."""
        x = Parameter('x')
        expr = x + 2

        y = Parameter('y')
        z = Parameter('z')

        with self.assertRaisesRegex(CircuitError, 'not present'):
            expr.subs({y: z})

    def test_raise_if_subbing_in_parameter_name_conflict(self):
        """Verify we raise if substituting in conflicting parameter names."""
        x = Parameter('x')
        y_first = Parameter('y')

        expr = x + y_first

        y_second = Parameter('y')

        # Replacing an existing name is okay.
        expr.subs({y_first: y_second})

        with self.assertRaisesRegex(CircuitError, 'Name conflict'):
            expr.subs({x: y_second})

    def test_expressions_of_parameter_with_constant(self):
        """Verify operating on a Parameter with a constant."""

        good_constants = [2, 1.3, 0, -1, -1.0, numpy.pi]

        x = Parameter('x')

        for op in self.supported_operations:
            for const in good_constants:
                expr = op(const, x)
                bound_expr = expr.bind({x: 2.3})

                self.assertEqual(float(bound_expr),
                                 op(const, 2.3))

                # Division by zero will raise. Tested elsewhere.
                if const == 0 and op == truediv:
                    continue

                # Repeat above, swapping position of Parameter and constant.
                expr = op(x, const)
                bound_expr = expr.bind({x: 2.3})

                self.assertEqual(float(bound_expr),
                                 op(2.3, const))

    def test_operating_on_a_parameter_with_a_non_float_will_raise(self):
        """Verify operations between a Parameter and a non-float will raise."""

        bad_constants = [1j, '1', numpy.Inf, numpy.NaN, None, {}, []]

        x = Parameter('x')

        for op in self.supported_operations:
            for const in bad_constants:
                with self.assertRaises(TypeError):
                    _ = op(const, x)

                with self.assertRaises(TypeError):
                    _ = op(x, const)

    def test_expressions_division_by_zero(self):
        """Verify dividing a Parameter by 0, or binding 0 as a denominator raises."""

        x = Parameter('x')

        with self.assertRaises(ZeroDivisionError):
            _ = x / 0

        with self.assertRaises(ZeroDivisionError):
            _ = x / 0.0

        expr = 2 / x

        with self.assertRaises(ZeroDivisionError):
            _ = expr.bind({x: 0})

        with self.assertRaises(ZeroDivisionError):
            _ = expr.bind({x: 0.0})

    def test_expressions_of_parameter_with_parameter(self):
        """Verify operating on two Parameters."""

        x = Parameter('x')
        y = Parameter('y')

        for op in self.supported_operations:
            expr = op(x, y)

            partially_bound_expr = expr.bind({x: 2.3})

            self.assertEqual(partially_bound_expr.parameters, {y})

            fully_bound_expr = partially_bound_expr.bind({y: -numpy.pi})

            self.assertEqual(fully_bound_expr.parameters, set())
            self.assertEqual(float(fully_bound_expr),
                             op(2.3, -numpy.pi))

            bound_expr = expr.bind({x: 2.3, y: -numpy.pi})

            self.assertEqual(bound_expr.parameters, set())
            self.assertEqual(float(bound_expr),
                             op(2.3, -numpy.pi))

    def test_expressions_operation_order(self):
        """Verify ParameterExpressions respect order of operations."""

        x = Parameter('x')
        y = Parameter('y')
        z = Parameter('z')

        # Parenthesis before multiplication/division
        expr = (x + y) * z
        bound_expr = expr.bind({x: 1, y: 2, z: 3})

        self.assertEqual(float(bound_expr), 9)

        expr = x * (y + z)
        bound_expr = expr.bind({x: 1, y: 2, z: 3})

        self.assertEqual(float(bound_expr), 5)

        # Multiplication/division before addition/subtraction
        expr = x + y * z
        bound_expr = expr.bind({x: 1, y: 2, z: 3})

        self.assertEqual(float(bound_expr), 7)

        expr = x * y + z
        bound_expr = expr.bind({x: 1, y: 2, z: 3})

        self.assertEqual(float(bound_expr), 5)

    def test_nested_expressions(self):
        """Verify ParameterExpressions can also be the target of operations."""

        x = Parameter('x')
        y = Parameter('y')
        z = Parameter('z')

        expr1 = x * y
        expr2 = expr1 + z
        bound_expr2 = expr2.bind({x: 1, y: 2, z: 3})

        self.assertEqual(float(bound_expr2), 5)

    def test_negated_expression(self):
        """Verify ParameterExpressions can be negated."""

        x = Parameter('x')
        y = Parameter('y')
        z = Parameter('z')

        expr1 = -x + y
        expr2 = -expr1 * (-z)
        bound_expr2 = expr2.bind({x: 1, y: 2, z: 3})

        self.assertEqual(float(bound_expr2), 3)

    def test_standard_cu3(self):
        """This tests parameter negation in standard extension gate cu3."""
        x = Parameter('x')
        y = Parameter('y')
        z = Parameter('z')
        qc = qiskit.QuantumCircuit(2)
        qc.cu3(x, y, z, 0, 1)
        try:
            qc.decompose()
        except TypeError:
            self.fail('failed to decompose cu3 gate with negated parameter '
                      'expression')

    def test_name_collision(self):
        """Verify Expressions of distinct Parameters of shared name raises."""

        x = Parameter('p')
        y = Parameter('p')

        # Expression of the same Parameter are valid.
        _ = x + x
        _ = x - x
        _ = x * x
        _ = x / x

        with self.assertRaises(CircuitError):
            _ = x + y
        with self.assertRaises(CircuitError):
            _ = x - y
        with self.assertRaises(CircuitError):
            _ = x * y
        with self.assertRaises(CircuitError):
            _ = x / y

    @combine(target_type=['gate', 'instruction'],
             order=['bind-decompose', 'decompose-bind'])
    def test_to_instruction_with_expression(self, target_type, order):
        """Test preservation of expressions via parameterized instructions.

                  ┌───────┐┌──────────┐┌───────────┐
        qr1_0: |0>┤ Rx(θ) ├┤ Rz(pi/2) ├┤ Ry(phi*θ) ├
                  └───────┘└──────────┘└───────────┘

                     ┌───────────┐
        qr2_0: |0>───┤ Ry(delta) ├───
                  ┌──┴───────────┴──┐
        qr2_1: |0>┤ Circuit0(phi,θ) ├
                  └─────────────────┘
        qr2_2: |0>───────────────────
        """

        theta = Parameter('θ')
        phi = Parameter('phi')
        qr1 = QuantumRegister(1, name='qr1')
        qc1 = QuantumCircuit(qr1)

        qc1.rx(theta, qr1)
        qc1.rz(numpy.pi/2, qr1)
        qc1.ry(theta * phi, qr1)

        if target_type == 'gate':
            gate = qc1.to_gate()
        elif target_type == 'instruction':
            gate = qc1.to_instruction()

        self.assertEqual(gate.params, [phi, theta])

        delta = Parameter('delta')
        qr2 = QuantumRegister(3, name='qr2')
        qc2 = QuantumCircuit(qr2)
        qc2.ry(delta, qr2[0])
        qc2.append(gate, qargs=[qr2[1]])

        self.assertEqual(qc2.parameters, {delta, theta, phi})

        binds = {delta: 1, theta: 2, phi: 3}
        expected_qc = QuantumCircuit(qr2)
        expected_qc.rx(2, 1)
        expected_qc.rz(numpy.pi/2, 1)
        expected_qc.ry(3 * 2, 1)
        expected_qc.r(1, numpy.pi/2, 0)

        if order == 'bind-decompose':
            decomp_bound_qc = qc2.assign_parameters(binds).decompose()
        elif order == 'decompose-bind':
            decomp_bound_qc = qc2.decompose().assign_parameters(binds)

        self.assertEqual(decomp_bound_qc.parameters, set())
        self.assertEqual(decomp_bound_qc, expected_qc)

    @combine(target_type=['gate', 'instruction'],
             order=['bind-decompose', 'decompose-bind'])
    def test_to_instruction_expression_parameter_map(self, target_type, order):
        """Test preservation of expressions via instruction parameter_map."""

        theta = Parameter('θ')
        phi = Parameter('phi')
        qr1 = QuantumRegister(1, name='qr1')
        qc1 = QuantumCircuit(qr1)

        qc1.rx(theta, qr1)
        qc1.rz(numpy.pi/2, qr1)
        qc1.ry(theta * phi, qr1)

        theta_p = Parameter('theta')
        phi_p = Parameter('phi')

        if target_type == 'gate':
            gate = qc1.to_gate(parameter_map={theta: theta_p, phi: phi_p})
        elif target_type == 'instruction':
            gate = qc1.to_instruction(parameter_map={theta: theta_p, phi: phi_p})

        self.assertEqual(gate.params, [phi_p, theta_p])

        delta = Parameter('delta')
        qr2 = QuantumRegister(3, name='qr2')
        qc2 = QuantumCircuit(qr2)
        qc2.ry(delta, qr2[0])
        qc2.append(gate, qargs=[qr2[1]])

        self.assertEqual(qc2.parameters, {delta, theta_p, phi_p})

        binds = {delta: 1, theta_p: 2, phi_p: 3}
        expected_qc = QuantumCircuit(qr2)
        expected_qc.rx(2, 1)
        expected_qc.rz(numpy.pi/2, 1)
        expected_qc.ry(3 * 2, 1)
        expected_qc.r(1, numpy.pi/2, 0)

        if order == 'bind-decompose':
            decomp_bound_qc = qc2.assign_parameters(binds).decompose()
        elif order == 'decompose-bind':
            decomp_bound_qc = qc2.decompose().assign_parameters(binds)

        self.assertEqual(decomp_bound_qc.parameters, set())
        self.assertEqual(decomp_bound_qc, expected_qc)

    def test_binding_across_broadcast_instruction(self):
        """Bind a parameter which was included via a broadcast instruction."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/3008

        from qiskit.circuit.library.standard_gates.rz import RZGate
        theta = Parameter('θ')
        n = 5

        qc = QuantumCircuit(n, 1)

        qc.h(0)
        for i in range(n-1):
            qc.cx(i, i+1)

        qc.barrier()
        qc.rz(theta, range(n))
        qc.barrier()

        for i in reversed(range(n-1)):
            qc.cx(i, i+1)
        qc.h(0)
        qc.measure(0, 0)

        theta_range = numpy.linspace(0, 2 * numpy.pi, 128)
        circuits = [qc.assign_parameters({theta: theta_val})
                    for theta_val in theta_range]

        self.assertEqual(len(circuits), len(theta_range))
        for theta_val, bound_circ in zip(theta_range, circuits):
            rz_gates = [inst for inst, qargs, cargs in bound_circ.data
                        if isinstance(inst, RZGate)]

            self.assertEqual(len(rz_gates), n)
            self.assertTrue(all(float(gate.params[0]) == theta_val
                                for gate in rz_gates))

    def test_substituting_parameter_with_simple_expression(self):
        """Substitute a simple parameter expression for a parameter."""
        x = Parameter('x')

        y = Parameter('y')
        sub_ = y / 2

        updated_expr = x.subs({x: sub_})

        expected = y / 2

        self.assertEqual(updated_expr, expected)

    def test_substituting_parameter_with_compound_expression(self):
        """Substitute a simple parameter expression for a parameter."""
        x = Parameter('x')

        y = Parameter('y')
        z = Parameter('z')
        sub_ = y * z

        updated_expr = x.subs({x: sub_})

        expected = y * z

        self.assertEqual(updated_expr, expected)

    def test_substituting_simple_with_simple_expression(self):
        """Substitute a simple parameter expression in a parameter expression."""
        x = Parameter('x')
        expr = x * x

        y = Parameter('y')
        sub_ = y / 2

        updated_expr = expr.subs({x: sub_})

        expected = y*y / 4

        self.assertEqual(updated_expr, expected)

    def test_substituting_compound_expression(self):
        """Substitute a compound parameter expression in a parameter expression."""
        x = Parameter('x')
        expr = x*x

        y = Parameter('y')
        z = Parameter('z')
        sub_ = y + z

        updated_expr = expr.subs({x: sub_})

        expected = (y + z) * (y + z)

        self.assertEqual(updated_expr, expected)

    def test_conjugate(self):
        """Test calling conjugate on a ParameterExpression."""
        x = Parameter('x')
        self.assertEqual(x, x.conjugate())  # Parameters are real, therefore conjugate returns self


class TestParameterEquality(QiskitTestCase):
    """Test equality of Parameters and ParameterExpressions."""

    def test_parameter_equal_self(self):
        """Verify a parameter is equal to it self."""
        theta = Parameter('theta')
        self.assertEqual(theta, theta)

    def test_parameter_not_equal_param_of_same_name(self):
        """Verify a parameter is not equal to a Parameter of the same name."""
        theta1 = Parameter('theta')
        theta2 = Parameter('theta')
        self.assertNotEqual(theta1, theta2)

    def test_parameter_expression_equal_to_self(self):
        """Verify an expression is equal to itself."""
        theta = Parameter('theta')
        expr = 2 * theta

        self.assertEqual(expr, expr)

    def test_parameter_expression_equal_to_identical(self):
        """Verify an expression is equal an identical expression."""
        theta = Parameter('theta')
        expr1 = 2 * theta
        expr2 = 2 * theta

        self.assertEqual(expr1, expr2)

    def test_parameter_expression_not_equal_if_params_differ(self):
        """Verify expressions not equal if parameters are different."""
        theta1 = Parameter('theta')
        theta2 = Parameter('theta')
        expr1 = 2 * theta1
        expr2 = 2 * theta2

        self.assertNotEqual(expr1, expr2)

    def test_parameter_equal_to_identical_expression(self):
        """Verify parameters and ParameterExpressions can be equal if identical."""
        theta = Parameter('theta')
        phi = Parameter('phi')

        expr = (theta + phi).bind({phi: 0})

        self.assertEqual(expr, theta)
        self.assertEqual(theta, expr)
