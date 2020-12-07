# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" test Grover """

import itertools
import unittest
import warnings
from test.aqua import QiskitAquaTestCase

from ddt import ddt, idata, unpack
from qiskit import BasicAer, QuantumCircuit, QuantumRegister
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import Grover
from qiskit.aqua.components.initial_states import Zero, Custom
from qiskit.aqua.components.oracles import LogicalExpressionOracle as LEO
from qiskit.aqua.components.oracles import TruthTableOracle as TTO
from qiskit.aqua.components.oracles import CustomCircuitOracle
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Operator, Statevector

TESTS = [
    ['p cnf 3 5 \n -1 -2 -3 0 \n 1 -2 3 0 \n 1 2 -3 0 \n 1 -2 -3 0 \n -1 2 3 0',
     ['101', '000', '011'], LEO],
    ['p cnf 2 2 \n 1  0 \n -2  0', ['01'], LEO],
    ['p cnf 2 4 \n 1  0 \n -1 0 \n 2  0 \n -2 0', [], LEO],
    ['a & b & c', ['111'], LEO],
    ['(a ^ b) & a & b', [], LEO],
    ['a & b | c & d', ['0011', '1011', '0111', '1100', '1101', '1110', '1111'], LEO],
    ['1000000000000001', ['0000', '1111'], TTO],
    ['00000000', [], TTO],
    ['0001', ['11'], TTO],
]

MCT_MODES = ['basic', 'basic-dirty-ancilla', 'advanced', 'noancilla']
SIMULATORS = ['statevector_simulator', 'qasm_simulator']
OPTIMIZATIONS = [True, False]
LAMBDA = [1.44, 8/7]
ROTATION_COUNTS = [
    None,
    [0, 0, 0, 1, 1, 0, 1, 1, 2, 1, 2, 3, 1, 4, 5, 1, 6, 2, 7, 9,
     11, 13, 16, 5, 20, 24, 28, 34, 2, 41, 49, 4, 60]
]


@ddt
class TestGrover(QiskitAquaTestCase):
    """ Grover test """

    def setUp(self):
        super().setUp()
        warnings.filterwarnings('ignore', category=DeprecationWarning)

    def tearDown(self):
        super().tearDown()
        warnings.filterwarnings('always', category=DeprecationWarning)

    @idata(
        [x[0] + list(x[1:]) for x in list(itertools.product(TESTS, MCT_MODES, SIMULATORS,
                                                            OPTIMIZATIONS, LAMBDA,
                                                            ROTATION_COUNTS))]
    )
    @unpack
    def test_grover(self, input_test, sol, oracle_cls, mct_mode,
                    simulator, optimization, lam, rotation_counts):
        """ grover test """
        groundtruth = sol
        oracle = oracle_cls(input_test, optimization=optimization)
        grover = Grover(oracle, incremental=True, lam=lam,
                        rotation_counts=rotation_counts, mct_mode=mct_mode)
        backend = BasicAer.get_backend(simulator)
        quantum_instance = QuantumInstance(backend, shots=1000)

        ret = grover.run(quantum_instance)

        self.log.debug('Ground-truth Solutions: %s.', groundtruth)
        self.log.debug('Top measurement:        %s.', ret.top_measurement)
        if ret.oracle_evaluation:
            self.assertIn(ret.top_measurement, groundtruth)
            self.log.debug('Search Result:          %s.', ret.assignment)
        else:
            self.assertEqual(groundtruth, [])
            self.log.debug('Nothing found.')

    def test_old_signature(self):
        """Test the old signature without naming arguments works."""
        oracle = TTO('0001')
        circuit = QuantumCircuit(2)
        circuit.h([0, 1])
        init_state = Custom(2, circuit=circuit)
        backend = BasicAer.get_backend('statevector_simulator')
        grover = Grover(oracle, init_state, True, 10, 1.44, ROTATION_COUNTS[1],
                        'noancilla', backend)
        ret = grover.run()
        self.assertEqual(ret.top_measurement, '11')


class TestGroverConstructor(QiskitAquaTestCase):
    """Test for the constructor of Grover"""

    def setUp(self):
        super().setUp()
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        self._expected_grover_op = GroverOperator(oracle=oracle)

    def test_oracle_quantumcircuit(self):
        """Test QuantumCircuit oracle"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        grover = Grover(oracle=oracle, good_state=["11"])
        grover_op = grover._grover_operator
        self.assertTrue(Operator(grover_op).equiv(Operator(self._expected_grover_op)))

    def test_oracle_statevector(self):
        """Test StateVector oracle"""
        mark_state = Statevector.from_label('11')
        grover = Grover(oracle=mark_state, good_state=['11'])
        grover_op = grover._grover_operator
        self.assertTrue(Operator(grover_op).equiv(Operator(self._expected_grover_op)))

    def test_state_preparation_quantumcircuit(self):
        """Test QuantumCircuit state_preparation"""
        state_preparation = QuantumCircuit(2)
        state_preparation.h(0)
        oracle = QuantumCircuit(3)
        oracle.cz(0, 1)
        grover = Grover(oracle=oracle, state_preparation=state_preparation,
                        good_state=["011"])
        grover_op = grover._grover_operator
        expected_grover_op = GroverOperator(oracle, state_preparation=state_preparation)
        self.assertTrue(Operator(grover_op).equiv(Operator(expected_grover_op)))

    def test_state_preparation_type_error(self):
        """Test InitialState state_preparation with QuantumCircuit oracle"""
        init_state = Zero(2)
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        # filtering the following:
        # DeprecationWarning: Passing an InitialState component is deprecated as of 0.8.0,
        # and will be removed no earlier than 3 months after the release date.
        # You should pass a QuantumCircuit instead.
        try:
            warnings.filterwarnings(action="ignore", category=DeprecationWarning)
            with self.assertRaises(TypeError):
                Grover(oracle=oracle, state_preparation=init_state)
        finally:
            warnings.filterwarnings(action="always", category=DeprecationWarning)

    def test_is_good_state_list(self):
        """Test List is_good_state"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        is_good_state = ["11", "00"]
        grover = Grover(oracle=oracle, good_state=is_good_state)
        self.assertListEqual(grover._is_good_state, ["11", "00"])

    def test_is_good_state_statevector(self):
        """Test StateVector is_good_state"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        is_good_state = Statevector.from_label('11')
        grover = Grover(oracle=oracle, good_state=is_good_state)
        self.assertTrue(grover._is_good_state.equiv(Statevector.from_label('11')))

    def test_grover_operator(self):
        """Test GroverOperator"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        grover_op = GroverOperator(oracle)
        grover = Grover(oracle=grover_op.oracle,
                        grover_operator=grover_op, good_state=["11"])
        grover_op = grover._grover_operator
        self.assertTrue(Operator(grover_op).equiv(Operator(self._expected_grover_op)))


class TestGroverPublicMethods(QiskitAquaTestCase):
    """Test for the public methods of Grover"""

    def test_is_good_state(self):
        """Test is_good_state"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        list_str_good_state = ["11"]
        grover = Grover(oracle=oracle, good_state=list_str_good_state)
        self.assertTrue(grover.is_good_state("11"))

        statevector_good_state = Statevector.from_label('11')
        grover = Grover(oracle=oracle, good_state=statevector_good_state)
        self.assertTrue(grover.is_good_state("11"))

        list_int_good_state = [0, 1]
        grover = Grover(oracle=oracle, good_state=list_int_good_state)
        self.assertTrue(grover.is_good_state("11"))

        def _callable_good_state(bitstr):
            if bitstr == "11":
                return True, bitstr
            else:
                return False, bitstr
        grover = Grover(oracle=oracle, good_state=_callable_good_state)
        self.assertTrue(grover.is_good_state("11"))

    def test_construct_circuit(self):
        """Test construct_circuit"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        grover = Grover(oracle=oracle, good_state=["11"])
        constructed = grover.construct_circuit(1)
        grover_op = GroverOperator(oracle)
        expected = QuantumCircuit(2)
        expected.compose(grover_op.state_preparation, inplace=True)
        expected.compose(grover_op, inplace=True)
        self.assertTrue(Operator(constructed).equiv(Operator(expected)))

    def test_post_processing(self):
        """Test post_processing"""
        # For the Oracle class
        q_v = QuantumRegister(2, name='v')
        q_o = QuantumRegister(1, name='o')
        circuit = QuantumCircuit(q_v, q_o)
        circuit.ccx(q_v[0], q_v[1], q_o[0])
        oracle = CustomCircuitOracle(variable_register=q_v, output_register=q_o, circuit=circuit,
                                     evaluate_classically_callback=lambda m: (m == '11', [1, 2]))
        grover = Grover(oracle)
        self.assertListEqual(grover.post_processing("11"), [1, 2])
        # For the specified post_processing
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        grover = Grover(oracle, good_state=["11"],
                        post_processing=lambda bitstr: [idx for idx, x_i in enumerate(bitstr)
                                                        if x_i == '1'])
        self.assertEqual(grover.post_processing("11"), [0, 1])
        # When Not specified
        grover = Grover(oracle, good_state=["11"])
        self.assertEqual(grover.post_processing("11"), "11")

    def test_grover_operator_getter(self):
        """Test the getter of grover_operator"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        grover = Grover(oracle=oracle, good_state=["11"])
        constructed = grover.grover_operator
        expected = GroverOperator(oracle)
        self.assertTrue(Operator(constructed).equiv(Operator(expected)))


class TestGroverFunctionality(QiskitAquaTestCase):
    """Test for the functionality of Grover"""

    def setUp(self):
        super().setUp()
        self._oracle = Statevector.from_label('111')
        self._expected_grover_op = GroverOperator(oracle=self._oracle)
        self._expected = QuantumCircuit(self._expected_grover_op.num_qubits)
        self._expected.compose(self._expected_grover_op.state_preparation, inplace=True)
        self._expected.compose(self._expected_grover_op.power(2), inplace=True)
        backend = BasicAer.get_backend('statevector_simulator')
        self._sv = QuantumInstance(backend)

    def test_num_iteration(self):
        """Test specified num_iterations"""
        # filtering the following:
        # DeprecationWarning: The num_iterations argument is deprecated as of 0.8.0,
        # and will be removed no earlier than 3 months after the release date.
        # If you want to use the num_iterations argument you should use the iterations
        # argument instead and pass an integer for the number of iterations.
        try:
            warnings.filterwarnings(action="ignore", category=DeprecationWarning)
            grover = Grover(oracle=self._oracle, good_state=['111'], num_iterations=2)
        finally:
            warnings.filterwarnings(action="always", category=DeprecationWarning)
        ret = grover.run(self._sv)
        self.assertTrue(Operator(ret['circuit']).equiv(Operator(self._expected)))

    def test_iterations(self):
        """Test the iterations argument"""
        grover = Grover(oracle=self._oracle, good_state=['111'], iterations=2)
        ret = grover.run(self._sv)
        self.assertTrue(Operator(ret['circuit']).equiv(Operator(self._expected)))

        grover = Grover(oracle=self._oracle, good_state=['111'], iterations=[1, 2, 3])
        ret = grover.run(self._sv)
        self.assertTrue(ret.oracle_evaluation)
        self.assertIn(ret.top_measurement, ['111'])


class TestGroverExecution(QiskitAquaTestCase):
    """Test for the execution of Grover"""

    def setUp(self):
        super().setUp()
        backend = BasicAer.get_backend('qasm_simulator')
        self._qasm = QuantumInstance(backend)

    def test_run_circuit_oracle(self):
        """Test execution with a quantum circuit oracle"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        list_good_state = ["11"]
        grover = Grover(oracle=oracle, good_state=list_good_state)
        ret = grover.run(self._qasm)
        self.assertIn(ret['top_measurement'], list_good_state)

    def test_run_state_vector_oracle(self):
        """Test execution with a state vector oracle"""
        mark_state = Statevector.from_label('11')
        grover = Grover(oracle=mark_state, good_state=['11'])
        ret = grover.run(self._qasm)
        self.assertIn(ret['top_measurement'], ['11'])

    def test_run_grover_operator_oracle(self):
        """Test execution with a grover operator oracle"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        grover_op = GroverOperator(oracle)
        grover = Grover(oracle=grover_op.oracle,
                        grover_operator=grover_op, good_state=["11"])
        ret = grover.run(self._qasm)
        self.assertIn(ret['top_measurement'], ['11'])


if __name__ == '__main__':
    unittest.main()
