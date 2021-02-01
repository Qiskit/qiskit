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

import unittest

from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit import BasicAer, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.algorithms import Grover
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Operator, Statevector


class TestGroverConstructor(QiskitAlgorithmsTestCase):
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


class TestGroverPublicMethods(QiskitAlgorithmsTestCase):
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

    def test_grover_operator_getter(self):
        """Test the getter of grover_operator"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        grover = Grover(oracle=oracle, good_state=["11"])
        constructed = grover.grover_operator
        expected = GroverOperator(oracle)
        self.assertTrue(Operator(constructed).equiv(Operator(expected)))


class TestGroverFunctionality(QiskitAlgorithmsTestCase):
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

    def test_iterations(self):
        """Test the iterations argument"""
        grover = Grover(oracle=self._oracle, good_state=['111'], iterations=2)
        ret = grover.run(self._sv)
        self.assertTrue(Operator(ret.circuit).equiv(Operator(self._expected)))

        grover = Grover(oracle=self._oracle, good_state=['111'], iterations=[1, 2, 3])
        ret = grover.run(self._sv)
        self.assertTrue(ret.oracle_evaluation)
        self.assertIn(ret.top_measurement, ['111'])


class TestGroverExecution(QiskitAlgorithmsTestCase):
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
        self.assertIn(ret.top_measurement, list_good_state)

    def test_run_state_vector_oracle(self):
        """Test execution with a state vector oracle"""
        mark_state = Statevector.from_label('11')
        grover = Grover(oracle=mark_state, good_state=['11'])
        ret = grover.run(self._qasm)
        self.assertIn(ret.top_measurement, ['11'])

    def test_run_grover_operator_oracle(self):
        """Test execution with a grover operator oracle"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        grover_op = GroverOperator(oracle)
        grover = Grover(oracle=grover_op.oracle,
                        grover_operator=grover_op, good_state=["11"])
        ret = grover.run(self._qasm)
        self.assertIn(ret.top_measurement, ['11'])


if __name__ == '__main__':
    unittest.main()
