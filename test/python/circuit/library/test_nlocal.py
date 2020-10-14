# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test library of n-local circuits."""

import unittest

import numpy as np

from ddt import ddt, data, unpack

from qiskit.test.base import QiskitTestCase
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.library import (
    NLocal, TwoLocal, RealAmplitudes, ExcitationPreserving, XGate, CRXGate, CCXGate,
    SwapGate, RXGate, RYGate, EfficientSU2, RZGate, RXXGate, RYYGate
)
from qiskit.circuit.random.utils import random_circuit
from qiskit.converters.circuit_to_dag import circuit_to_dag


@ddt
class TestNLocal(QiskitTestCase):
    """Test the n-local circuit class."""

    def assertCircuitEqual(self, qc1, qc2, visual=False, transpiled=True):
        """An equality test specialized to circuits."""
        if transpiled:
            basis_gates = ['id', 'u1', 'u3', 'cx']
            qc1_transpiled = transpile(qc1, basis_gates=basis_gates, optimization_level=0)
            qc2_transpiled = transpile(qc2, basis_gates=basis_gates, optimization_level=0)
            qc1, qc2 = qc1_transpiled, qc2_transpiled

        if visual:
            self.assertEqual(qc1.draw(), qc2.draw())
        else:
            self.assertEqual(qc1, qc2)

    def test_empty_nlocal(self):
        """Test the creation of an empty NLocal."""
        nlocal = NLocal()
        self.assertEqual(nlocal.num_qubits, 0)
        self.assertEqual(nlocal.num_parameters_settable, 0)
        self.assertEqual(nlocal.reps, 1)

        self.assertEqual(nlocal, QuantumCircuit())

        for attribute in [nlocal.rotation_blocks, nlocal.entanglement_blocks]:
            self.assertEqual(len(attribute), 0)

    @data((XGate(), [[0], [2], [1]]),
          (XGate(), [[0]]),
          (CRXGate(-0.2), [[2, 0], [1, 3]]),
          )
    @unpack
    def test_add_layer_to_empty_nlocal(self, block, entangler_map):
        """Test appending gates to an empty nlocal."""
        nlocal = NLocal()
        nlocal.add_layer(block, entangler_map)

        max_num_qubits = max(max(indices) for indices in entangler_map)
        reference = QuantumCircuit(max_num_qubits + 1)
        for indices in entangler_map:
            reference.append(block, indices)

        self.assertCircuitEqual(nlocal, reference)

    @data([5, 3], [1, 5], [1, 1], [1, 2, 3, 10])
    def test_append_circuit(self, num_qubits):
        """Test appending circuits to an nlocal works normally."""
        # fixed depth of 3 gates per circuit
        depth = 3

        # keep track of a reference circuit
        reference = QuantumCircuit(max(num_qubits))

        # construct the NLocal from the first circuit
        first_circuit = random_circuit(num_qubits[0], depth)
        # TODO Terra bug: if this is to_gate it fails, since the QC adds an instruction not gate
        nlocal = NLocal(max(num_qubits), entanglement_blocks=first_circuit.to_instruction(), reps=1)
        reference.append(first_circuit, list(range(num_qubits[0])))

        # append the rest
        for num in num_qubits[1:]:
            circuit = random_circuit(num, depth)
            nlocal.append(circuit, list(range(num)))
            reference.append(circuit, list(range(num)))

        self.assertCircuitEqual(nlocal, reference)

    @data([5, 3], [1, 5], [1, 1], [1, 2, 3, 10])
    def test_add_nlocal(self, num_qubits):
        """Test adding an nlocal to an nlocal (using add_layer)."""
        # fixed depth of 3 gates per circuit
        depth = 3

        # keep track of a reference circuit
        reference = QuantumCircuit(max(num_qubits))

        # construct the NLocal from the first circuit
        first_circuit = random_circuit(num_qubits[0], depth)
        # TODO Terra bug: if this is to_gate it fails, since the QC adds an instruction not gate
        nlocal = NLocal(max(num_qubits), entanglement_blocks=first_circuit.to_instruction(), reps=1)
        reference.append(first_circuit, list(range(num_qubits[0])))

        # append the rest
        for num in num_qubits[1:]:
            circuit = random_circuit(num, depth)
            nlocal.add_layer(NLocal(num, entanglement_blocks=circuit, reps=1))
            reference.append(circuit, list(range(num)))

        self.assertCircuitEqual(nlocal, reference)

    @unittest.skip('Feature missing')
    def test_iadd_overload(self):
        """Test the overloaded + operator."""
        num_qubits, depth = 2, 2

        # construct two circuits for adding
        first_circuit = random_circuit(num_qubits, depth)
        circuit = random_circuit(num_qubits, depth)

        # get a reference
        reference = first_circuit + circuit

        # convert the object to be appended to different types
        others = [circuit, circuit.to_instruction(), circuit.to_gate(), NLocal(circuit)]

        # try adding each type
        for other in others:
            nlocal = NLocal(num_qubits, entanglement_blocks=first_circuit, reps=1)
            nlocal += other
            with self.subTest(msg='type: {}'.format(type(other))):
                self.assertCircuitEqual(nlocal, reference)

    def test_parameter_getter_from_automatic_repetition(self):
        """Test getting and setting of the nlocal parameters."""
        circuit = QuantumCircuit(2)
        circuit.ry(Parameter('a'), 0)
        circuit.crx(Parameter('b'), 0, 1)

        # repeat circuit and check that parameters are duplicated
        reps = 3
        nlocal = NLocal(2, entanglement_blocks=circuit, reps=reps)
        self.assertTrue(nlocal.num_parameters, 6)
        self.assertTrue(len(nlocal.parameters), 6)

    @data(list(range(6)), ParameterVector('θ', length=6), [0, 1, Parameter('theta'), 3, 4, 5])
    def test_parameter_setter_from_automatic_repetition(self, params):
        """Test getting and setting of the nlocal parameters."""
        circuit = QuantumCircuit(2)
        circuit.ry(Parameter('a'), 0)
        circuit.crx(Parameter('b'), 0, 1)

        # repeat circuit and check that parameters are duplicated
        reps = 3
        nlocal = NLocal(2, entanglement_blocks=circuit, reps=reps)
        nlocal.assign_parameters(params, inplace=True)

        param_set = set(p for p in params if isinstance(p, ParameterExpression))
        with self.subTest(msg='Test the parameters of the non-transpiled circuit'):
            # check the parameters of the final circuit
            self.assertEqual(nlocal.parameters, param_set)

        with self.subTest(msg='Test the parameters of the transpiled circuit'):
            basis_gates = ['id', 'u1', 'u2', 'u3', 'cx']
            transpiled_circuit = transpile(nlocal, basis_gates=basis_gates)
            self.assertEqual(transpiled_circuit.parameters, param_set)

    @data(list(range(6)), ParameterVector('θ', length=6), [0, 1, Parameter('theta'), 3, 4, 5])
    def test_parameters_setter(self, params):
        """Test setting the parameters via list."""
        # construct circuit with some parameters
        initial_params = ParameterVector('p', length=6)
        circuit = QuantumCircuit(1)
        for i, initial_param in enumerate(initial_params):
            circuit.ry(i * initial_param, 0)

        # create an NLocal from the circuit and set the new parameters
        nlocal = NLocal(1, entanglement_blocks=circuit, reps=1)
        nlocal.assign_parameters(params, inplace=True)

        param_set = set(p for p in params if isinstance(p, ParameterExpression))
        with self.subTest(msg='Test the parameters of the non-transpiled circuit'):
            # check the parameters of the final circuit
            self.assertEqual(nlocal.parameters, param_set)

        with self.subTest(msg='Test the parameters of the transpiled circuit'):
            basis_gates = ['id', 'u1', 'u2', 'u3', 'cx']
            transpiled_circuit = transpile(nlocal, basis_gates=basis_gates)
            self.assertEqual(transpiled_circuit.parameters, param_set)

    def test_repetetive_parameter_setting(self):
        """Test alternate setting of parameters and circuit construction."""
        x = Parameter('x')
        circuit = QuantumCircuit(1)
        circuit.rx(x, 0)

        nlocal = NLocal(1, entanglement_blocks=circuit, reps=3, insert_barriers=True)
        with self.subTest(msg='immediately after initialization'):
            self.assertEqual(len(nlocal.parameters), 3)

        with self.subTest(msg='after circuit construction'):
            self.assertEqual(len(nlocal.parameters), 3)

        q = Parameter('q')
        nlocal.assign_parameters([x, q, q], inplace=True)
        with self.subTest(msg='setting parameter to Parameter objects'):
            self.assertEqual(nlocal.parameters, set({x, q}))

        nlocal.assign_parameters([0, -1], inplace=True)
        with self.subTest(msg='setting parameter to numbers'):
            self.assertEqual(nlocal.parameters, set())

    def test_skip_unentangled_qubits(self):
        """Test skipping the unentangled qubits."""
        num_qubits = 6
        entanglement_1 = [[0, 1, 3], [1, 3, 5], [0, 1, 5]]
        skipped_1 = [2, 4]

        entanglement_2 = [
            entanglement_1,
            [[0, 1, 2], [2, 3, 5]]
        ]
        skipped_2 = [4]

        for entanglement, skipped in zip([entanglement_1, entanglement_2], [skipped_1, skipped_2]):
            with self.subTest(entanglement=entanglement, skipped=skipped):
                nlocal = NLocal(num_qubits, rotation_blocks=XGate(), entanglement_blocks=CCXGate(),
                                entanglement=entanglement, reps=3, skip_unentangled_qubits=True)

                skipped_set = set(nlocal.qubits[i] for i in skipped)
                dag = circuit_to_dag(nlocal)
                idle = set(dag.idle_wires())
                self.assertEqual(skipped_set, idle)

    @data('linear', 'full', 'circular', 'sca',
          ['linear', 'full'],
          ['circular', 'linear', 'sca'])
    def test_entanglement_by_str(self, entanglement):
        """Test setting the entanglement of the layers by str."""
        reps = 3
        nlocal = NLocal(5, rotation_blocks=XGate(), entanglement_blocks=CCXGate(),
                        entanglement=entanglement, reps=reps)

        def get_expected_entangler_map(rep_num, mode):
            if mode == 'linear':
                return [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
            elif mode == 'full':
                return [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4), (0, 3, 4),
                        (1, 2, 3), (1, 2, 4), (1, 3, 4),
                        (2, 3, 4)]
            else:
                circular = [(3, 4, 0), (0, 1, 2), (1, 2, 3), (2, 3, 4)]
                if mode == 'circular':
                    return circular
                sca = circular[-rep_num:] + circular[:-rep_num]
                if rep_num % 2 == 1:
                    sca = [tuple(reversed(indices)) for indices in sca]
                return sca

        for rep_num in range(reps):
            entangler_map = nlocal.get_entangler_map(rep_num, 0, 3)
            if isinstance(entanglement, list):
                mode = entanglement[rep_num % len(entanglement)]
            else:
                mode = entanglement
            expected = get_expected_entangler_map(rep_num, mode)

            with self.subTest(rep_num=rep_num):
                # using a set here since the order does not matter
                self.assertEqual(set(entangler_map), set(expected))

    def test_entanglement_by_list(self):
        """Test setting the entanglement by list.

        This is the circuit we test (times 2, with final X layer)
                ┌───┐                ┌───┐┌───┐                  ┌───┐
        q_0: |0>┤ X ├──■────■───X────┤ X ├┤ X ├──■───X─────── .. ┤ X ├
                ├───┤  │    │   │    ├───┤└─┬─┘  │   │           ├───┤
        q_1: |0>┤ X ├──■────┼───┼──X─┤ X ├──■────┼───X──X──── .. ┤ X ├
                ├───┤┌─┴─┐  │   │  │ ├───┤  │    │      │     x2 ├───┤
        q_2: |0>┤ X ├┤ X ├──■───┼──X─┤ X ├──■────■──────X──X─ .. ┤ X ├
                ├───┤└───┘┌─┴─┐ │    ├───┤     ┌─┴─┐       │     ├───┤
        q_3: |0>┤ X ├─────┤ X ├─X────┤ X ├─────┤ X ├───────X─ .. ┤ X ├
                └───┘     └───┘      └───┘     └───┘             └───┘
        """
        circuit = QuantumCircuit(4)
        for _ in range(2):
            circuit.x([0, 1, 2, 3])
            circuit.barrier()
            circuit.ccx(0, 1, 2)
            circuit.ccx(0, 2, 3)
            circuit.swap(0, 3)
            circuit.swap(1, 2)
            circuit.barrier()
            circuit.x([0, 1, 2, 3])
            circuit.barrier()
            circuit.ccx(2, 1, 0)
            circuit.ccx(0, 2, 3)
            circuit.swap(0, 1)
            circuit.swap(1, 2)
            circuit.swap(2, 3)
            circuit.barrier()
        circuit.x([0, 1, 2, 3])

        layer_1_ccx = [(0, 1, 2), (0, 2, 3)]
        layer_1_swap = [(0, 3), (1, 2)]
        layer_1 = [layer_1_ccx, layer_1_swap]

        layer_2_ccx = [(2, 1, 0), (0, 2, 3)]
        layer_2_swap = [(0, 1), (1, 2), (2, 3)]
        layer_2 = [layer_2_ccx, layer_2_swap]

        entanglement = [layer_1, layer_2]

        nlocal = NLocal(4, rotation_blocks=XGate(), entanglement_blocks=[CCXGate(), SwapGate()],
                        reps=4, entanglement=entanglement, insert_barriers=True)

        self.assertCircuitEqual(nlocal, circuit)


@ddt
class TestTwoLocal(QiskitTestCase):
    """Tests for the TwoLocal circuit."""

    def assertCircuitEqual(self, qc1, qc2, visual=False, transpiled=True):
        """An equality test specialized to circuits."""
        if transpiled:
            basis_gates = ['id', 'u1', 'u3', 'cx']
            qc1_transpiled = transpile(qc1, basis_gates=basis_gates, optimization_level=0)
            qc2_transpiled = transpile(qc2, basis_gates=basis_gates, optimization_level=0)
            qc1, qc2 = qc1_transpiled, qc2_transpiled

        if visual:
            self.assertEqual(qc1.draw(), qc2.draw())
        else:
            self.assertEqual(qc1, qc2)

    def test_skip_final_rotation_layer(self):
        """Test skipping the final rotation layer works."""
        two = TwoLocal(3, ['ry', 'h'], ['cz', 'cx'], reps=2, skip_final_rotation_layer=True)
        self.assertEqual(two.num_parameters, 6)  # would be 9 with a final rotation layer

    @data((5, 'rx', 'cx', 'full', 2, 15),
          (3, 'x', 'z', 'linear', 1, 0),
          (3, ['rx', 'ry'], ['cry', 'cx'], 'circular', 2, 24),
          )
    @unpack
    def test_num_parameters(self, num_qubits, rot, ent, ent_mode, reps, expected):
        """Test the number of parameters."""
        two = TwoLocal(num_qubits, rotation_blocks=rot, entanglement_blocks=ent,
                       entanglement=ent_mode, reps=reps)

        with self.subTest(msg='num_parameters_settable'):
            self.assertEqual(two.num_parameters_settable, expected)

        with self.subTest(msg='num_parameters'):
            self.assertEqual(two.num_parameters, expected)

    def test_empty_two_local(self):
        """Test the setup of an empty two-local circuit."""
        two = TwoLocal()

        with self.subTest(msg='0 qubits'):
            self.assertEqual(two.num_qubits, 0)

        with self.subTest(msg='no blocks are set'):
            self.assertListEqual(two.rotation_blocks, [])
            self.assertListEqual(two.entanglement_blocks, [])

        with self.subTest(msg='equal to empty circuit'):
            self.assertEqual(two, QuantumCircuit())

    @data('rx', RXGate(Parameter('p')), RXGate, 'circuit')
    def test_various_block_types(self, rot):
        """Test setting the rotation blocks to various type and assert the output type is RX."""
        if rot == 'circuit':
            rot = QuantumCircuit(1)
            rot.rx(Parameter('angle'), 0)

        two = TwoLocal(3, rot, 'cz', reps=1)
        self.assertEqual(len(two.rotation_blocks), 1)
        rotation = two.rotation_blocks[0]

        # decompose
        self.assertIsInstance(rotation.data[0][0], RXGate)

    def test_parameter_setters(self):
        """Test different possibilities to set parameters."""
        two = TwoLocal(3, rotation_blocks='rx', entanglement='cz', reps=2)
        params = [0, 1, 2, Parameter('x'), Parameter('y'), Parameter('z'), 6, 7, 0]
        params_set = set(param for param in params if isinstance(param, Parameter))

        with self.subTest(msg='dict assign and copy'):
            ordered = two.ordered_parameters
            bound = two.assign_parameters(dict(zip(ordered, params)), inplace=False)
            self.assertEqual(bound.parameters, params_set)
            self.assertEqual(two.num_parameters, 9)

        with self.subTest(msg='list assign and copy'):
            ordered = two.ordered_parameters
            bound = two.assign_parameters(params, inplace=False)
            self.assertEqual(bound.parameters, params_set)
            self.assertEqual(two.num_parameters, 9)

        with self.subTest(msg='list assign inplace'):
            ordered = two.ordered_parameters
            two.assign_parameters(params, inplace=True)
            self.assertEqual(two.parameters, params_set)
            self.assertEqual(two.num_parameters, 3)
            self.assertEqual(two.num_parameters_settable, 9)

    def test_parameters_settable_is_constant(self):
        """Test the attribute num_parameters_settable does not change on parameter change."""
        two = TwoLocal(3, rotation_blocks='rx', entanglement='cz', reps=2)
        ordered_params = two.ordered_parameters

        x = Parameter('x')
        two.assign_parameters(dict(zip(ordered_params, [x] * two.num_parameters)), inplace=True)

        with self.subTest(msg='num_parameters collapsed to 1'):
            self.assertEqual(two.num_parameters, 1)

        with self.subTest(msg='num_parameters_settable remained constant'):
            self.assertEqual(two.num_parameters_settable, len(ordered_params))

    def test_iadd_to_circuit(self):
        """Test adding a two-local to an existing circuit."""
        two = TwoLocal(3, ['ry', 'rz'], 'cz', 'full', reps=1, insert_barriers=True)
        circuit = QuantumCircuit(3)
        circuit += two

        reference = QuantumCircuit(3)
        param_iter = iter(two.ordered_parameters)
        for i in range(3):
            reference.ry(next(param_iter), i)
        for i in range(3):
            reference.rz(next(param_iter), i)
        reference.barrier()
        reference.cz(0, 1)
        reference.cz(0, 2)
        reference.cz(1, 2)
        reference.barrier()
        for i in range(3):
            reference.ry(next(param_iter), i)
        for i in range(3):
            reference.rz(next(param_iter), i)

        self.assertCircuitEqual(circuit, reference)

    def test_adding_two(self):
        """Test adding two two-local circuits."""
        entangler_map = [[0, 3], [0, 2]]
        two = TwoLocal(4, [], 'cry', entangler_map, reps=1)
        circuit = two + two

        reference = QuantumCircuit(4)
        params = two.ordered_parameters
        for _ in range(2):
            reference.cry(params[0], 0, 3)
            reference.cry(params[1], 0, 2)

        self.assertCircuitEqual(reference, circuit)

    def test_ry_blocks(self):
        """Test that the RealAmplitudes circuit is instantiated correctly."""
        two = RealAmplitudes(4)
        with self.subTest(msg='test rotation gate'):
            self.assertEqual(len(two.rotation_blocks), 1)
            self.assertIsInstance(two.rotation_blocks[0].data[0][0], RYGate)

        with self.subTest(msg='test parameter bounds'):
            expected = [(-np.pi, np.pi)] * two.num_parameters
            np.testing.assert_almost_equal(two.parameter_bounds, expected)

    def test_ry_circuit(self):
        """Test an RealAmplitudes circuit."""
        num_qubits = 3
        reps = 2
        entanglement = 'full'
        parameters = ParameterVector('theta', num_qubits * (reps + 1))
        param_iter = iter(parameters)

        expected = QuantumCircuit(3)
        for _ in range(reps):
            for i in range(num_qubits):
                expected.ry(next(param_iter), i)
            expected.cx(0, 1)
            expected.cx(0, 2)
            expected.cx(1, 2)
        for i in range(num_qubits):
            expected.ry(next(param_iter), i)

        library = RealAmplitudes(num_qubits, reps=reps,
                                 entanglement=entanglement).assign_parameters(parameters)

        self.assertCircuitEqual(library, expected)

    def test_ryrz_blocks(self):
        """Test that the EfficientSU2 circuit is instantiated correctly."""
        two = EfficientSU2(3)
        with self.subTest(msg='test rotation gate'):
            self.assertEqual(len(two.rotation_blocks), 2)
            self.assertIsInstance(two.rotation_blocks[0].data[0][0], RYGate)
            self.assertIsInstance(two.rotation_blocks[1].data[0][0], RZGate)

        with self.subTest(msg='test parameter bounds'):
            expected = [(-np.pi, np.pi)] * two.num_parameters
            np.testing.assert_almost_equal(two.parameter_bounds, expected)

    def test_ryrz_circuit(self):
        """Test an EfficientSU2 circuit."""
        num_qubits = 3
        reps = 2
        entanglement = 'circular'
        parameters = ParameterVector('theta', 2 * num_qubits * (reps + 1))
        param_iter = iter(parameters)

        expected = QuantumCircuit(3)
        for _ in range(reps):
            for i in range(num_qubits):
                expected.ry(next(param_iter), i)
            for i in range(num_qubits):
                expected.rz(next(param_iter), i)
            expected.cx(2, 0)
            expected.cx(0, 1)
            expected.cx(1, 2)
        for i in range(num_qubits):
            expected.ry(next(param_iter), i)
        for i in range(num_qubits):
            expected.rz(next(param_iter), i)

        library = EfficientSU2(num_qubits, reps=reps, entanglement=entanglement).assign_parameters(
            parameters
        )

        self.assertCircuitEqual(library, expected)

    def test_swaprz_blocks(self):
        """Test that the ExcitationPreserving circuit is instantiated correctly."""
        two = ExcitationPreserving(5)
        with self.subTest(msg='test rotation gate'):
            self.assertEqual(len(two.rotation_blocks), 1)
            self.assertIsInstance(two.rotation_blocks[0].data[0][0], RZGate)

        with self.subTest(msg='test entanglement gate'):
            self.assertEqual(len(two.entanglement_blocks), 1)
            block = two.entanglement_blocks[0]
            self.assertEqual(len(block.data), 2)
            self.assertIsInstance(block.data[0][0], RXXGate)
            self.assertIsInstance(block.data[1][0], RYYGate)

        with self.subTest(msg='test parameter bounds'):
            expected = [(-np.pi, np.pi)] * two.num_parameters
            np.testing.assert_almost_equal(two.parameter_bounds, expected)

    def test_swaprz_circuit(self):
        """Test a ExcitationPreserving circuit in iswap mode."""
        num_qubits = 3
        reps = 2
        entanglement = 'linear'
        parameters = ParameterVector('theta', num_qubits * (reps + 1) + reps * (num_qubits - 1))
        param_iter = iter(parameters)

        expected = QuantumCircuit(3)
        for _ in range(reps):
            for i in range(num_qubits):
                expected.rz(next(param_iter), i)
            shared_param = next(param_iter)
            expected.rxx(shared_param, 0, 1)
            expected.ryy(shared_param, 0, 1)
            shared_param = next(param_iter)
            expected.rxx(shared_param, 1, 2)
            expected.ryy(shared_param, 1, 2)
        for i in range(num_qubits):
            expected.rz(next(param_iter), i)

        library = ExcitationPreserving(num_qubits, reps=reps,
                                       entanglement=entanglement).assign_parameters(parameters)

        self.assertCircuitEqual(library, expected)

    def test_fsim_circuit(self):
        """Test a ExcitationPreserving circuit in fsim mode."""
        num_qubits = 3
        reps = 2
        entanglement = 'linear'
        # need the parameters in the entanglement blocks to be the same because the order
        # can get mixed up in ExcitationPreserving (since parameters are not ordered in circuits)
        parameters = [1] * (num_qubits * (reps + 1) + reps * (1 + num_qubits))
        param_iter = iter(parameters)

        expected = QuantumCircuit(3)
        for _ in range(reps):
            for i in range(num_qubits):
                expected.rz(next(param_iter), i)
            shared_param = next(param_iter)
            expected.rxx(shared_param, 0, 1)
            expected.ryy(shared_param, 0, 1)
            expected.cp(next(param_iter), 0, 1)
            shared_param = next(param_iter)
            expected.rxx(shared_param, 1, 2)
            expected.ryy(shared_param, 1, 2)
            expected.cp(next(param_iter), 1, 2)
        for i in range(num_qubits):
            expected.rz(next(param_iter), i)

        library = ExcitationPreserving(num_qubits, reps=reps, mode='fsim',
                                       entanglement=entanglement).assign_parameters(parameters)

        self.assertCircuitEqual(library, expected)


if __name__ == '__main__':
    unittest.main()
