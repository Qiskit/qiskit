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

"""Test the Solovay Kitaev transpilation pass."""

import unittest
import math
import numpy as np

from scipy.optimize import minimize
import qiskit.circuit.library as gates

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.library import TGate, RXGate, RYGate, HGate, SGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import SolovayKitaevDecomposition
from qiskit.test import QiskitTestCase
from qiskit.quantum_info import Operator


# pylint: disable=invalid-name, missing-class-docstring

class H(Gate):
    def __init__(self):
        super().__init__('H', 1, [])

    def _define(self):
        definition = QuantumCircuit(1)
        definition.h(0)
        definition.global_phase = np.pi / 2
        self.definition = definition

    def to_matrix(self):
        return 1j * gates.HGate().to_matrix()

    def inverse(self):
        return H_dg()


class H_dg(Gate):
    def __init__(self):
        super().__init__('iH_dg', 1, [])

    def _define(self):
        definition = QuantumCircuit(1)
        definition.h(0)
        definition.global_phase = -np.pi / 2
        self.definition = definition

    def to_matrix(self):
        return -1j * gates.HGate().to_matrix()

    def inverse(self):
        return H()


class T(Gate):
    def __init__(self):
        super().__init__('T', 1, [])

    def _define(self):
        definition = QuantumCircuit(1)
        definition.t(0)
        definition.global_phase = -np.pi / 8
        self.definition = definition

    def to_matrix(self):
        return np.exp(-1j * np.pi / 8) * gates.TGate().to_matrix()

    def inverse(self):
        return T_dg()


class T_dg(Gate):
    def __init__(self):
        super().__init__('T_dg', 1, [])

    def _define(self):
        definition = QuantumCircuit(1)
        definition.tdg(0)
        definition.global_phase = np.pi / 8
        self.definition = definition

    def to_matrix(self):
        return np.exp(1j * np.pi / 8) * gates.TdgGate().to_matrix()

    def inverse(self):
        return T()


class S(Gate):
    def __init__(self):
        super().__init__('S', 1, [])

    def _define(self):
        definition = QuantumCircuit(1)
        definition.s(0)
        definition.global_phase = -np.pi / 4
        self.definition = definition

    def to_matrix(self):
        return np.exp(-1j * np.pi / 4) * gates.SGate().to_matrix()

    def inverse(self):
        return S_dg()


class S_dg(Gate):
    def __init__(self):
        super().__init__('S_dg', 1, [])

    def _define(self):
        definition = QuantumCircuit(1)
        definition.sdg(0)
        definition.global_phase = np.pi / 4
        self.definition = definition

    def to_matrix(self):
        return np.exp(1j * np.pi / 4) * gates.SdgGate().to_matrix()

    def inverse(self):
        return S()


def distance(A, B):
    """Find the distance in norm of A and B, ignoring global phase."""

    def objective(global_phase):
        return np.linalg.norm(A - np.exp(1j * global_phase) * B)
    result1 = minimize(objective, [1], bounds=[(-np.pi, np.pi)])
    result2 = minimize(objective, [0.5], bounds=[(-np.pi, np.pi)])
    return min(result1.fun, result2.fun)


class TestSolovayKitaev(QiskitTestCase):
    """Test the Solovay Kitaev algorithm and transformation pass."""

    def test_example(self):
        """@Lisa Example to show how to call the pass."""
        circuit = QuantumCircuit(1)
        circuit.rx(0.2, 0)

        basic_gates = [H(), T(), S(), gates.IGate(), H_dg(), T_dg(),
                       S_dg(), RXGate(math.pi), RYGate(math.pi)]
        synth = SolovayKitaevDecomposition(3, basic_gates)

        dag = circuit_to_dag(circuit)
        decomposed_dag = synth.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)

        print(decomposed_circuit.draw())

    def test_example_2(self):
        """@Lisa Example to show how to call the pass."""
        circuit = QuantumCircuit(1)
        circuit.rx(0.8, 0)

        basic_gates = [H(), T(), S(), T_dg(), S_dg()]
        synth = SolovayKitaevDecomposition(2, basic_gates)

        dag = circuit_to_dag(circuit)
        decomposed_dag = synth.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)

        print(decomposed_circuit.draw())
        print('Original')
        print(Operator(circuit))
        print('Synthesized')
        print(Operator(decomposed_circuit))

    def test_example_non_su2(self):
        """@Lisa Example to show how to call the pass."""
        circuit = QuantumCircuit(1)
        circuit.rx(0.8, 0)

        basic_gates = [HGate(), TGate(), SGate()]
        synth = SolovayKitaevDecomposition(2, basic_gates)

        dag = circuit_to_dag(circuit)
        decomposed_dag = synth.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)

        print(decomposed_circuit.draw())
        print('Original')
        print(Operator(circuit))
        print('Synthesized')
        print(Operator(decomposed_circuit))
        self.assertLess(distance(Operator(circuit).data, Operator(decomposed_circuit).data), 0.1)


if __name__ == '__main__':
    unittest.main()


class TestSolovayKitaevUtils(QiskitTestCase):
    """Test the algebra utils."""
    pass
