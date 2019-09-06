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

import numpy

from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Gate, Parameter, ParameterVector, ParameterExpression
from qiskit.compiler import assemble, transpile
from qiskit.execute import execute
from qiskit.test import QiskitTestCase
from qiskit.tools import parallel_map
from qiskit.exceptions import QiskitError


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

    def test_get_parameters(self):
        """Test instantiating gate with variable parameters"""
        from qiskit.extensions.standard.rx import RXGate
        theta = Parameter('θ')
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        rxg = RXGate(theta)
        qc.append(rxg, [qr[0]], [])
        vparams = qc._parameter_table
        self.assertEqual(len(vparams), 1)
        self.assertIs(theta, next(iter(vparams)))
        self.assertIs(rxg, vparams[theta][0][0])

    def test_fix_variable(self):
        """Test setting a variable to a constant value"""
        theta = Parameter('θ')
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u3(0, theta, 0, qr)
        bqc = qc.bind_parameters({theta: 0.5})
        self.assertEqual(float(bqc.data[0][0].params[0]), 0.5)
        self.assertEqual(float(bqc.data[1][0].params[1]), 0.5)
        bqc = qc.bind_parameters({theta: 0.6})
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

    def test_partial_binding(self):
        """Test that binding a subset of circuit parameters returns a new parameterized circuit."""
        theta = Parameter('θ')
        x = Parameter('x')
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u3(0, theta, x, qr)

        pqc = qc.bind_parameters({theta: 2})

        self.assertEqual(pqc.parameters, {x})

        self.assertEqual(float(pqc.data[0][0].params[0]), 2)
        self.assertEqual(float(pqc.data[1][0].params[1]), 2)

    def test_expression_partial_binding(self):
        """Test that binding a subset of expression parameters returns a new
        parameterized circuit."""
        theta = Parameter('θ')
        phi = Parameter('phi')

        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta + phi, qr)

        pqc = qc.bind_parameters({theta: 2})

        self.assertEqual(pqc.parameters, {phi})

        self.assertTrue(isinstance(pqc.data[0][0].params[0], ParameterExpression))
        self.assertEqual(str(pqc.data[0][0].params[0]), 'phi + 2')

        fbqc = pqc.bind_parameters({phi: 1})

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

        pqc = qc.bind_parameters({theta: 0})

        self.assertEqual(pqc.parameters, {phi})

        self.assertTrue(isinstance(pqc.data[0][0].params[0], ParameterExpression))
        self.assertEqual(str(pqc.data[0][0].params[0]), '0')

        fbqc = pqc.bind_parameters({phi: 1})

        self.assertEqual(fbqc.parameters, set())
        self.assertTrue(isinstance(fbqc.data[0][0].params[0], ParameterExpression))
        self.assertEqual(float(fbqc.data[0][0].params[0]), 0)

    def test_raise_if_assigning_params_not_in_circuit(self):
        """Verify binding parameters which are not present in the circuit raises an error."""
        x = Parameter('x')
        y = Parameter('y')
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        qc.u1(0.1, qr[0])
        self.assertRaises(QiskitError, qc.bind_parameters, {x: 1})

        qc.u1(x, qr[0])
        self.assertRaises(QiskitError, qc.bind_parameters, {x: 1, y: 2})

    def test_gate_multiplicity_binding(self):
        """Test binding when circuit contains multiple references to same gate"""
        from qiskit.extensions.standard import RZGate
        qc = QuantumCircuit(1)
        theta = Parameter('theta')
        gate = RZGate(theta)
        qc.append(gate, [0], [])
        qc.append(gate, [0], [])
        qc2 = qc.bind_parameters({theta: 1.0})
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
        circs = []
        theta_list = numpy.linspace(0, numpy.pi, 20)
        for theta_i in theta_list:
            circs.append(qc_aer.bind_parameters({theta: theta_i}))
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

        self.assertRaises(QiskitError, qc.u1, theta2, 0)

    def test_bind_ryrz_vector(self):
        """Test binding a list of floats to a ParamterVector"""
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
        bqc = qc.bind_parameters({theta: theta_vals})
        for gate_tuple in bqc.data:
            if hasattr(gate_tuple[0], 'params') and gate_tuple[0].params:
                self.assertIn(float(gate_tuple[0].params[0]), theta_vals)

    def test_compile_vector(self):
        """Test compiling a circuit with an unbound ParamterVector"""
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
        """Test constructing a circuit from instructions with remapped ParamterVectors"""
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
                               [(param) for param in parameters],
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


def _construct_circuit(param, qr):
    qc = QuantumCircuit(qr)
    qc.ry(param, qr[0])
    return qc


class TestParameterExpressions(QiskitTestCase):
    """Test expressions of Parameters."""

    supported_operations = [add, sub, mul, truediv]

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
        """Verify divding a Parameter by 0, or binding 0 as a denominator raises."""

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

    def test_name_collision(self):
        """Verify Expressions of distinct Parameters of shared name raises."""

        x = Parameter('p')
        y = Parameter('p')

        # Expression of the same Parameter are valid.
        _ = x + x
        _ = x - x
        _ = x * x
        _ = x / x

        with self.assertRaises(QiskitError):
            _ = x + y
        with self.assertRaises(QiskitError):
            _ = x - y
        with self.assertRaises(QiskitError):
            _ = x * y
        with self.assertRaises(QiskitError):
            _ = x / y

    def test_to_instruction_with_expresion(self):
        """Test preservation of expressions via parameterized instructions."""

        theta = Parameter('θ')
        phi = Parameter('phi')
        qr1 = QuantumRegister(1, name='qr1')
        qc1 = QuantumCircuit(qr1)
        qc1.rx(theta, qr1)
        qc1.rz(numpy.pi/2, qr1)
        qc1.ry(theta * phi, qr1)
        gate = qc1.to_instruction()

        self.assertEqual(gate.params, [phi, theta])

        delta = Parameter('delta')
        qr2 = QuantumRegister(3, name='qr2')
        qc2 = QuantumCircuit(qr2)
        qc2.ry(delta, qr2[0])
        qc2.append(gate, qargs=[qr2[1]])
        self.assertEqual(qc2.parameters, {delta, theta, phi})

        bound_qc = qc2.decompose().bind_parameters({delta: 1, theta: 2, phi: 3})
        self.assertEqual(float(bound_qc.data[0][0].params[0]), 1)
        self.assertEqual(float(bound_qc.data[1][0].params[0]), 2)
        self.assertEqual(float(bound_qc.data[2][0].params[0]), numpy.pi/2)
        self.assertEqual(float(bound_qc.data[3][0].params[0]), 2 * 3)

    def test_to_instruction_expression_parameter_map(self):
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

        gate = qc1.to_instruction(parameter_map={theta: theta_p, phi: phi_p})

        self.assertEqual(gate.params, [phi_p, theta_p])

        delta = Parameter('delta')
        qr2 = QuantumRegister(3, name='qr2')
        qc2 = QuantumCircuit(qr2)
        qc2.ry(delta, qr2[0])
        qc2.append(gate, qargs=[qr2[1]])
        self.assertEqual(qc2.parameters, {delta, theta_p, phi_p})

        bound_qc = qc2.decompose().bind_parameters({delta: 1, theta_p: 2, phi_p: 3})
        self.assertEqual(float(bound_qc.data[0][0].params[0]), 1)
        self.assertEqual(float(bound_qc.data[1][0].params[0]), 2)
        self.assertEqual(float(bound_qc.data[2][0].params[0]), numpy.pi/2)
        self.assertEqual(float(bound_qc.data[3][0].params[0]), 2 * 3)

    def test_binding_across_broadcast_instruction(self):
        """Bind a parameter which was included via a broadcast instruction."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/3008

        from qiskit.extensions.standard import RZGate
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
        circuits = [qc.bind_parameters({theta: theta_val})
                    for theta_val in theta_range]

        self.assertEqual(len(circuits), len(theta_range))
        for theta_val, bound_circ in zip(theta_range, circuits):
            rz_gates = [inst for inst, qargs, cargs in bound_circ.data
                        if isinstance(inst, RZGate)]

            self.assertEqual(len(rz_gates), n)
            self.assertTrue(all(float(gate.params[0]) == theta_val
                                for gate in rz_gates))
