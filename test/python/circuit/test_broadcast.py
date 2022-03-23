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

import math
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
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

    # variants for 1q
    qc.h(0)
    qc.s([0, 1, 2])
    qc.y([4])

    # variants for 2q
    qc.cx(2, 3)
    qc.cx([2, 3, 4], [1])
    qc.cx([1], [2, 3, 4])
    qc.cx([1, 2], [3, 4])
    qc.ccx(1, 2, 3)
    qc.cx([0, 0, 0], [1, 1, 1])

    # variants for 3q
    qc.ccx([1], [2], [3])
    qc.ccx([0, 0, 0], [1, 1, 1], [2, 3, 4])

    print(qc)


def test_measure():
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.s(1)
    qc.cx(0, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    print(qc)


def test_measure2():
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.s(1)
    qc.cx(0, 2)
    qc.measure(0, [0, 1, 2])
    qc.measure([1], [0, 1, 2])
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


def test_clifford2():
    # Do we really need this functionality?

    qc1 = QuantumCircuit(3)
    qc1.h(0)
    qc1.s(1)
    qc1.cx(0, 2)
    cliff1 = Clifford(qc1)

    qc3 = QuantumCircuit(6)
    qc3.append(cliff1, [[0, 3], [1, 4], [2, 5]])
    print(qc3)


# appending one qc to another
def test_quantum_circuit():
    qc1 = QuantumCircuit(3)
    qc1.h(0)
    qc1.s(1)
    qc1.cx(0, 2)

    qc2 = QuantumCircuit(6)
    qc2.append(qc1, [0, 2, 4])
    print(qc2)


def test_initializer():
    """Combining two circuits containing initialize."""
    desired_vector_1 = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]
    qr = QuantumRegister(1, "qr")
    cr = ClassicalRegister(1, "cr")
    qc1 = QuantumCircuit(qr, cr)
    qc1.initialize(desired_vector_1, [qr[0]])
    print(qc1)


if __name__ == "__main__":
    test_barrier()
    test_delay()
    test_gates()
    test_measure()
    test_measure2()
    test_reset()
    test_clifford()
    test_clifford2()
    test_quantum_circuit()
    test_initializer()
