# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit synthesis for the CNOTDihedral class.
"""

import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit


def decompose_cnotdihedral(elem):
    """Decompose a CNOTDihedral element into a QuantumCircuit.

    Args:
        elem (CNOTDihedral): a CNOTDihedral element.
    Return:
        QuantumCircuit: a circuit implementation of the CNOTDihedral element.

    References:
        1. Shelly Garion and Andrew W. Cross, *Synthesis of CNOT-Dihedral circuits
           with optimal number of two qubit gates*, `Quantum 4(369), 2020
           <https://quantum-journal.org/papers/q-2020-12-07-369/>`_
        2. Andrew W. Cross, Easwar Magesan, Lev S. Bishop, John A. Smolin and Jay M. Gambetta,
           *Scalable randomised benchmarking of non-Clifford gates*,
           npj Quantum Inf 2, 16012 (2016).
    """

    num_qubits = elem.num_qubits

    if num_qubits < 3:
        return decompose_cnotdihedral_2_qubits(elem)

    return decompose_cnotdihedral_general(elem)


def decompose_cnotdihedral_2_qubits(elem):
    """Decompose a CNOTDihedral element into a QuantumCircuit.

    Args:
        elem (CNOTDihedral): a CNOTDihedral element.
    Return:
        QuantumCircuit: a circuit implementation of the CNOTDihedral element.
    Remark:
        Decompose 1 and 2-qubit CNOTDihedral elements.
    Raises:
        QiskitError: if the element in not 1 or 2-qubit CNOTDihedral.

    References:
        1. Shelly Garion and Andrew W. Cross, *On the structure of the CNOT-Dihedral group*,
           `arXiv:2006.12042 [quant-ph] <https://arxiv.org/abs/2006.12042>`_
        2. Andrew W. Cross, Easwar Magesan, Lev S. Bishop, John A. Smolin and Jay M. Gambetta,
           *Scalable randomised benchmarking of non-Clifford gates*,
           npj Quantum Inf 2, 16012 (2016).
    """

    circuit = QuantumCircuit(elem.num_qubits)

    if elem.num_qubits > 2:
        raise QiskitError(
            "Cannot decompose a CNOT-Dihedral element with more than 2 qubits. "
            "use decompose_cnotdihedral_general function instead."
        )

    if elem.num_qubits == 1:
        if elem.poly.weight_0 != 0 or elem.linear != [[1]]:
            raise QiskitError("1-qubit element in not CNOT-Dihedral .")
        tpow0 = elem.poly.weight_1[0]
        xpow0 = elem.shift[0]
        if tpow0 > 0:
            circuit.p((tpow0 * np.pi / 4), [0])
        if xpow0 == 1:
            circuit.x(0)
        if tpow0 == 0 and xpow0 == 0:
            circuit.id(0)
        return circuit

    # case elem.num_qubits == 2:
    if elem.poly.weight_0 != 0:
        raise QiskitError("2-qubit element in not CNOT-Dihedral .")
    weight_1 = elem.poly.weight_1
    weight_2 = elem.poly.weight_2
    linear = elem.linear
    shift = elem.shift

    # CS subgroup
    if (linear == [[1, 0], [0, 1]]).all():
        [xpow0, xpow1] = shift

        # Dihedral class
        if weight_2 == [0]:
            [tpow0, tpow1] = weight_1
            if tpow0 > 0:
                circuit.p((tpow0 * np.pi / 4), [0])
            if xpow0 == 1:
                circuit.x(0)
            if tpow1 > 0:
                circuit.p((tpow1 * np.pi / 4), [1])
            if xpow1 == 1:
                circuit.x(1)
            if tpow0 == 0 and tpow1 == 0 and xpow0 == 0 and xpow1 == 0:
                circuit.id(0)
                circuit.id(1)

        # CS-like class
        if (weight_2 == [2] and xpow0 == xpow1) or (weight_2 == [6] and xpow0 != xpow1):
            tpow0 = (weight_1[0] - 2 * xpow1 - 4 * xpow0 * xpow1) % 8
            tpow1 = (weight_1[1] - 2 * xpow0 - 4 * xpow0 * xpow1) % 8
            if tpow0 > 0:
                circuit.p((tpow0 * np.pi / 4), [0])
            if xpow0 == 1:
                circuit.x(0)
            if tpow1 > 0:
                circuit.p((tpow1 * np.pi / 4), [1])
            if xpow1 == 1:
                circuit.x(1)
            # CS gate is implemented using 2 CX gates
            circuit.p((np.pi / 4), [0])
            circuit.p((np.pi / 4), [1])
            circuit.cx(0, 1)
            circuit.p((7 * np.pi / 4), [1])
            circuit.cx(0, 1)

        # CSdg-like class
        if (weight_2 == [6] and xpow0 == xpow1) or (weight_2 == [2] and xpow0 != xpow1):
            tpow0 = (weight_1[0] - 6 * xpow1 - 4 * xpow0 * xpow1) % 8
            tpow1 = (weight_1[1] - 6 * xpow0 - 4 * xpow0 * xpow1) % 8
            if tpow0 > 0:
                circuit.p((tpow0 * np.pi / 4), [0])
            if xpow0 == 1:
                circuit.x(0)
            if tpow1 > 0:
                circuit.p((tpow1 * np.pi / 4), [1])
            if xpow1 == 1:
                circuit.x(1)
            # CSdg gate is implemented using 2 CX gates
            circuit.p((7 * np.pi / 4), [0])
            circuit.p((7 * np.pi / 4), [1])
            circuit.cx(0, 1)
            circuit.p((np.pi / 4), [1])
            circuit.cx(0, 1)

        # CZ-like class
        if weight_2 == [4]:
            tpow0 = (weight_1[0] - 4 * xpow1) % 8
            tpow1 = (weight_1[1] - 4 * xpow0) % 8
            if tpow0 > 0:
                circuit.p((tpow0 * np.pi / 4), [0])
            if xpow0 == 1:
                circuit.x(0)
            if tpow1 > 0:
                circuit.p((tpow1 * np.pi / 4), [1])
            if xpow1 == 1:
                circuit.x(1)
            # CZ gate is implemented using 2 CX gates
            circuit.cz(1, 0)

    # CX01-like class
    if (linear == [[1, 0], [1, 1]]).all():
        xpow0 = shift[0]
        xpow1 = (shift[1] + xpow0) % 2
        if xpow0 == xpow1:
            m = ((8 - weight_2[0]) / 2) % 4
            tpow0 = (weight_1[0] - m) % 8
            tpow1 = (weight_1[1] - m) % 8
        else:
            m = (weight_2[0] / 2) % 4
            tpow0 = (weight_1[0] + m) % 8
            tpow1 = (weight_1[1] + m) % 8
        if tpow0 > 0:
            circuit.p((tpow0 * np.pi / 4), [0])
        if xpow0 == 1:
            circuit.x(0)
        if tpow1 > 0:
            circuit.p((tpow1 * np.pi / 4), [1])
        if xpow1 == 1:
            circuit.x(1)
        circuit.cx(0, 1)
        if m > 0:
            circuit.p((m * np.pi / 4), [1])

    # CX10-like class
    if (linear == [[1, 1], [0, 1]]).all():
        xpow1 = shift[1]
        xpow0 = (shift[0] + xpow1) % 2
        if xpow0 == xpow1:
            m = ((8 - weight_2[0]) / 2) % 4
            tpow0 = (weight_1[0] - m) % 8
            tpow1 = (weight_1[1] - m) % 8
        else:
            m = (weight_2[0] / 2) % 4
            tpow0 = (weight_1[0] + m) % 8
            tpow1 = (weight_1[1] + m) % 8
        if tpow0 > 0:
            circuit.p((tpow0 * np.pi / 4), [0])
        if xpow0 == 1:
            circuit.x(0)
        if tpow1 > 0:
            circuit.p((tpow1 * np.pi / 4), [1])
        if xpow1 == 1:
            circuit.x(1)
        circuit.cx(1, 0)
        if m > 0:
            circuit.p((m * np.pi / 4), [0])

    # CX01*CX10-like class
    if (linear == [[0, 1], [1, 1]]).all():
        xpow1 = shift[0]
        xpow0 = (shift[1] + xpow1) % 2
        if xpow0 == xpow1:
            m = ((8 - weight_2[0]) / 2) % 4
            tpow0 = (weight_1[0] - m) % 8
            tpow1 = (weight_1[1] - m) % 8
        else:
            m = (weight_2[0] / 2) % 4
            tpow0 = (weight_1[0] + m) % 8
            tpow1 = (weight_1[1] + m) % 8
        if tpow0 > 0:
            circuit.p((tpow0 * np.pi / 4), [0])
        if xpow0 == 1:
            circuit.x(0)
        if tpow1 > 0:
            circuit.p((tpow1 * np.pi / 4), [1])
        if xpow1 == 1:
            circuit.x(1)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        if m > 0:
            circuit.p((m * np.pi / 4), [1])

    # CX10*CX01-like class
    if (linear == [[1, 1], [1, 0]]).all():
        xpow0 = shift[1]
        xpow1 = (shift[0] + xpow0) % 2
        if xpow0 == xpow1:
            m = ((8 - weight_2[0]) / 2) % 4
            tpow0 = (weight_1[0] - m) % 8
            tpow1 = (weight_1[1] - m) % 8
        else:
            m = (weight_2[0] / 2) % 4
            tpow0 = (weight_1[0] + m) % 8
            tpow1 = (weight_1[1] + m) % 8
        if tpow0 > 0:
            circuit.p((tpow0 * np.pi / 4), [0])
        if xpow0 == 1:
            circuit.x(0)
        if tpow1 > 0:
            circuit.p((tpow1 * np.pi / 4), [1])
        if xpow1 == 1:
            circuit.x(1)
        circuit.cx(1, 0)
        circuit.cx(0, 1)
        if m > 0:
            circuit.p((m * np.pi / 4), [0])

    # CX01*CX10*CX01-like class
    if (linear == [[0, 1], [1, 0]]).all():
        xpow0 = shift[1]
        xpow1 = shift[0]
        if xpow0 == xpow1:
            m = ((8 - weight_2[0]) / 2) % 4
            tpow0 = (weight_1[0] - m) % 8
            tpow1 = (weight_1[1] - m) % 8
        else:
            m = (weight_2[0] / 2) % 4
            tpow0 = (weight_1[0] + m) % 8
            tpow1 = (weight_1[1] + m) % 8
        if tpow0 > 0:
            circuit.p((tpow0 * np.pi / 4), [0])
        if xpow0 == 1:
            circuit.x(0)
        if tpow1 > 0:
            circuit.p((tpow1 * np.pi / 4), [1])
        if xpow1 == 1:
            circuit.x(1)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        if m > 0:
            circuit.p((m * np.pi / 4), [1])
        circuit.cx(0, 1)

    return circuit


def decompose_cnotdihedral_general(elem):
    """Decompose a CNOTDihedral element into a QuantumCircuit.

    Args:
        elem (CNOTDihedral): a CNOTDihedral element.
    Return:
        QuantumCircuit: a circuit implementation of the CNOTDihedral element.
    Remark:
        Decompose general CNOTDihedral elements.
        The number of CNOT gates is not necessarily optimal.
    Raises:
        QiskitError: if the element could not be decomposed into a circuit.

    References:
        1. Andrew W. Cross, Easwar Magesan, Lev S. Bishop, John A. Smolin and Jay M. Gambetta,
           *Scalable randomised benchmarking of non-Clifford gates*,
           npj Quantum Inf 2, 16012 (2016).
    """

    num_qubits = elem.num_qubits
    circuit = QuantumCircuit(num_qubits)

    # Make a copy of the CNOTDihedral element as we are going to
    # reduce it to an identity
    elem_cpy = elem.copy()

    if not np.allclose((np.linalg.det(elem_cpy.linear) % 2), 1):
        raise QiskitError("Linear part is not invertible.")

    # Do x gate for each qubit i where shift[i]=1
    for i in range(num_qubits):
        if elem.shift[i]:
            circuit.x(i)
            elem_cpy._append_x(i)

    # Do Gauss elimination on the linear part by adding cx gates
    for i in range(num_qubits):
        # set i-th element to be 1
        if not elem_cpy.linear[i][i]:
            for j in range(i + 1, num_qubits):
                if elem_cpy.linear[j][i]:  # swap qubits i and j
                    circuit.cx(j, i)
                    circuit.cx(i, j)
                    circuit.cx(j, i)
                    elem_cpy._append_cx(j, i)
                    elem_cpy._append_cx(i, j)
                    elem_cpy._append_cx(j, i)
                    break
        # make all the other elements in column i zero
        for j in range(num_qubits):
            if j != i:
                if elem_cpy.linear[j][i]:
                    circuit.cx(i, j)
                    elem_cpy._append_cx(i, j)

    if (
        not (elem_cpy.shift == np.zeros(num_qubits)).all()
        or not (elem_cpy.linear == np.eye(num_qubits)).all()
    ):
        raise QiskitError("Cannot do Gauss elimination on linear part.")

    # Initialize new_elem to an identity CNOTDihderal element
    new_elem = elem_cpy.copy()
    new_elem.poly.weight_0 = 0
    new_elem.poly.weight_1 = np.zeros(num_qubits, dtype=np.int8)
    new_elem.poly.weight_2 = np.zeros(int(num_qubits * (num_qubits - 1) / 2), dtype=np.int8)
    new_elem.poly.weight_3 = np.zeros(
        int(num_qubits * (num_qubits - 1) * (num_qubits - 2) / 6), dtype=np.int8
    )

    new_circuit = QuantumCircuit(num_qubits)

    # Do cx and phase gates to construct all monomials of weight 3
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            for k in range(j + 1, num_qubits):
                if elem_cpy.poly.get_term([i, j, k]) != 0:
                    new_elem._append_cx(i, k)
                    new_elem._append_cx(j, k)
                    new_elem._append_phase(1, k)
                    new_elem._append_cx(i, k)
                    new_elem._append_cx(j, k)
                    new_circuit.cx(i, k)
                    new_circuit.cx(j, k)
                    new_circuit.p((np.pi / 4), [k])
                    new_circuit.cx(i, k)
                    new_circuit.cx(j, k)

    # Do cx and phase gates to construct all monomials of weight 2
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            tpow1 = elem_cpy.poly.get_term([i, j])
            tpow2 = new_elem.poly.get_term([i, j])
            tpow = ((tpow2 - tpow1) / 2) % 4
            if tpow != 0:
                new_elem._append_cx(i, j)
                new_elem._append_phase(tpow, j)
                new_elem._append_cx(i, j)
                new_circuit.cx(i, j)
                new_circuit.p((tpow * np.pi / 4), [j])
                new_circuit.cx(i, j)

    # Do phase gates to construct all monomials of weight 1
    for i in range(num_qubits):
        tpow1 = elem_cpy.poly.get_term([i])
        tpow2 = new_elem.poly.get_term([i])
        tpow = (tpow1 - tpow2) % 8
        if tpow != 0:
            new_elem._append_phase(tpow, i)
            new_circuit.p((tpow * np.pi / 4), [i])

    if elem.poly != new_elem.poly:
        raise QiskitError("Could not recover phase polynomial.")

    inv_circuit = circuit.inverse()
    return new_circuit.compose(inv_circuit)
