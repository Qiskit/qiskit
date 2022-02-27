# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test Qiskit's Arguments Broadcaster."""

from qiskit.circuit import QuantumCircuit, Barrier
from qiskit.circuit.argumentsbroadcaster import ArgumentsBroadcasterBarrier
from qiskit.quantum_info.operators import Clifford


def test_barrier():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.barrier([0, 1, 2])
    qc.s(1)
    print(qc)

def test_delay():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.delay(1)
    qc.s(1)
    print(qc)


def test_gates():
    qc = QuantumCircuit(5)
    qc.h(0)
    qc.s([0, 1, 2])
    qc.cx(2, 3)
    qc.cx([2, 3, 4], [1])
    qc.cx([1], [2, 3, 4])
    qc.cx([1, 2], [3, 4])
    print(qc)


def test_measure():
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.s(1)
    qc.cx(0, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    print(qc)


def test_reset():
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.reset(0)
    qc.cx(0, 2)
    print(qc)


def test_clifford():
    qc1 = QuantumCircuit(3)
    qc1.h(0)
    qc1.s(1)
    qc1.cx(0, 2)
    cliff1 = Clifford(qc1)

    qc2 = QuantumCircuit(5)
    qc2.append(cliff1, [1, 3, 4])
    print(qc2)


# appending one qc to another
def test_quantum_circuit():
    qc1 = QuantumCircuit(3)
    qc1.h(0)
    qc1.s(1)
    qc1.cx(0, 2)

    qc2 = QuantumCircuit(5)
    qc2.append(qc1, [1, 3, 4])
    print(qc2)


if __name__ == "__main__":
    test_barrier()
    #test_delay()
    #test_gates()
    #test_measure()
    #test_reset()
    #test_clifford()
    #test_quantum_circuit()
