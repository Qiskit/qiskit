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

"""Algebra utilities and the ``GateSequence`` class."""

from typing import List, Optional, Tuple

import math
import numpy as np
import scipy

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.library import IGate


class GateSequence():
    """A class implementing a sequence of gates.

    This class stores the sequence of gates along with the unitary they implement.
    """

    def __init__(self, gates: Optional[List[Gate]] = None) -> None:
        """Create a new sequence of gates.

        Args:
            gates: The gates in the sequence. The default is [].
        """
        if gates is None:
            gates = []

        # store the gates
        self.gates = gates

        # get U(2) representation of the gate sequence
        u2_matrix = np.identity(2)
        for gate in gates:
            u2_matrix = gate.to_matrix().dot(u2_matrix)

        # convert to SU(2)
        su2_matrix, global_phase = _convert_u2_to_su2(u2_matrix)

        # convert to SO(3), that's what the Solovay Kitaev algorithm uses
        so3_matrix = convert_su2_to_so3(su2_matrix)

        # store the matrix and the global phase
        self.global_phase = global_phase
        self.product = so3_matrix

    def __eq__(self, other: 'GateSequence') -> bool:
        """Check if this GateSequence is the same as the other GateSequence.

        Args:
            other: The GateSequence that will be compared to ``self``.

        Returns:
            True if ``other`` is equivalent to ``self``, false otherwise.

        """
        if not len(self.gates) == len(other.gates):
            return False

        for gate1, gate2 in zip(self.gates, other.gates):
            if gate1 != gate2:
                return False

        if self.global_phase != other.global_phase:
            return False

        return True

    def to_circuit(self):
        """Convert to a circuit.

        If no gates set but the product is not the identity, returns a circuit with a
        unitary operation to implement the matrix.
        """
        if len(self.gates) == 0 and not np.allclose(self.product, np.identity(3)):
            circuit = QuantumCircuit(1, global_phase=self.global_phase)
            su2 = convert_so3_to_su2(self.product)
            circuit.unitary(su2, [0])
            return circuit

        circuit = QuantumCircuit(1, global_phase=self.global_phase)
        for gate in self.gates:
            circuit.append(gate, [0])

        return circuit

    def represents_same_gate(self, other: 'GateSequence', precision: float = 0.0) -> bool:
        """Returns whether to ``self`` represents the same gate as ``other`` up to ``precision``.

        Args:
            other: The GateSequence compared to ``self``.
            precision: The error tolerated when comparing the fields of the matrices of the gates.

        Returns:
            True when ``self`` represents the same gate as ``other`` up to ``precision``,
            False otherwise.
        """
        return np.allclose(self.product, other.product, atol=precision)

    def append(self, gate: Gate) -> 'GateSequence':
        """Append gate to the sequence of gates.

        Args:
            gate: The gate to be appended.

        Returns:
            GateSequence with ``gate`` appended.
        """
        # TODO: this recomputes the product whenever we append something, which could be more
        # efficient by storing the current matrix and just multiplying the input gate to it
        # self.product = convert_su2_to_so3(self._compute_product(self.gates))
        su2, phase = _convert_u2_to_su2(gate.to_matrix())
        so3 = convert_su2_to_so3(su2)

        self.product = so3.dot(self.product)
        self.global_phase = self.global_phase + phase
        self.gates.append(gate)

        return self

    def adjoint(self) -> 'GateSequence':
        """Get the complex conjugate."""
        adjoint = GateSequence()
        adjoint.gates = [gate.inverse() for gate in reversed(self.gates)]
        adjoint.product = np.matrix.getH(self.product)
        adjoint.global_phase = -self.global_phase

        return adjoint

    def count(self, gate: Gate) -> int:
        """Count the number of times the argument ``gate`` occurs in the sequence.

        Args:
            gate: The gate to be counted in the sequence.

        Returns:
            The number of times the argument ``gate`` occurs in the sequence.
        """
        return self.gates.count(gate)

    def simplify(self, precision: float = 0.0) -> 'GateSequence':
        """Returns GateSequence with less gates that represents the same gate up to ``precision``.

        Args:
            precision: The required precision for letting two sequences represent the same gate.

        Returns:
            The simplified GateSequence.
        """
        gslist = [GateSequence([])]
        for gate in self.gates:
            new_sequence = gslist[len(gslist)-1].copy()
            new_sequence.append(gate)
            same_as_new_sequences = [
                s for s in gslist if s.represents_same_gate(new_sequence, precision)]
            if len(same_as_new_sequences) > 1 or gate == IGate():
                index = gslist.index(same_as_new_sequences[0])
                gslist = gslist[:index+1]
            else:
                gslist.append(new_sequence)
        self.gates = gslist[len(gslist)-1].gates
        return self

    def copy(self) -> 'GateSequence':
        """Create copy of the sequence of gates.

        Returns:
            A new ``GateSequence`` containing copy of list of gates.

        """
        return GateSequence(self.gates.copy())

    def __len__(self) -> int:
        """Return length of sequence of gates.

        Returns:
            Length of list containing gates.
        """
        return len(self.gates)

    def __getitem__(self, index: int) -> Gate:
        """Returns the gate at ``index`` from the list of gates.

        Args
            index: Index of gate in list that will be returned.

        Returns:
            The gate at ``index`` in the list of gates.
        """
        return self.gates[index]

    def __repr__(self) -> str:
        """Return string representation of this object.

        Returns:
            Representation of this sequence of gates.
        """
        out = '['
        for gate in self.gates:
            out += gate.name
            out += ', '
        out += ']'
        out += ', product: '
        out += str(self.product)
        return out

    def __str__(self) -> str:
        """Return string representation of this object.

        Returns:
            Representation of this sequence of gates.
        """
        out = '['
        for gate in self.gates:
            out += gate.name
            out += ', '
        out += ']'
        out += ', product: \n'
        out += str(self.product)
        return out

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'GateSequence':
        """Initialize the gate sequence from a matrix, without a gate sequence.

        Args:
            matrix: The matrix, can be SU(2) or SO(3).

        Returns:
            A ``GateSequence`` initialized from the input matrix.

        Raises:
            ValueError: If the matrix has an invalid shape.
        """
        instance = cls()
        if matrix.shape == (2, 2):
            instance.product = convert_su2_to_so3(matrix)
        elif matrix.shape == (3, 3):
            instance.product = matrix
        else:
            raise ValueError('Matrix has an invalid shape.')

        instance.gates = []
        return instance

    def dot(self, other: 'GateSequence') -> 'GateSequence':
        """Compute the dot-product with another gate sequence.

        Args:
            other: The other gate sequence.

        Returns:
            The dot-product as gate sequence.
        """
        composed = GateSequence()
        composed.gates = other.gates + self.gates
        composed.product = np.dot(self.product, other.product)
        composed.global_phase = self.global_phase + other.global_phase

        return composed


def _convert_u2_to_su2(u2_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    z = 1 / np.sqrt(np.linalg.det(u2_matrix))
    su2_matrix = z * u2_matrix
    phase = np.arctan2(np.imag(z), np.real(z))

    return su2_matrix, phase


def compute_euler_angles_from_so3(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Computes the Euler angles from the SO(3)-matrix u.

    Uses the algorithm from Gregory Slabaugh,
    see `here <https://www.gregslabaugh.net/publications/euler.pdf>`_.

    Args:
        matrix: The SO(3)-matrix for which the Euler angles need to be computed.

    Returns:
        Tuple phi, theta, psi\n
        where phi is rotation about z-axis, theta rotation about y-axis\n
        and psi rotation about x-axis.
    """
    matrix = np.round(matrix, decimals=7)
    if (matrix[2][0] != 1 and matrix[2][1] != -1):
        theta = -math.asin(matrix[2][0])
        psi = math.atan2(matrix[2][1] / math.cos(theta),
                         matrix[2][2] / math.cos(theta))
        phi = math.atan2(matrix[1][0] / math.cos(theta),
                         matrix[0][0] / math.cos(theta))
        return phi, theta, psi
    else:
        phi = 0
        if matrix[2][0] == 1:
            theta = math.pi/2
            psi = phi + math.atan2(matrix[0][1], matrix[0][2])
        else:
            theta = -math.pi/2
            psi = -phi + math.atan2(-matrix[0][1], -matrix[0][2])
        return phi, theta, psi


def compute_su2_from_euler_angles(angles: Tuple[float, float, float]) -> np.ndarray:
    """Computes SU(2)-matrix from Euler angles.

    Args:
        angles: The tuple containing the Euler angles for which the corresponding SU(2)-matrix
            needs to be computed.

    Returns:
        The SU(2)-matrix corresponding to the Euler angles in angles.
    """
    phi, theta, psi = angles
    uz_phi = np.array([[np.exp(-(1/2)*phi*1j), 0],
                       [0, np.exp((1/2)*phi*1j)]], dtype=complex)
    uy_theta = np.array([[math.cos(theta/2), math.sin(theta/2)],
                         [-math.sin(theta/2), math.cos(theta/2)]], dtype=complex)
    ux_psi = np.array([[math.cos(psi/2), math.sin(psi/2)*1j],
                       [math.sin(psi/2)*1j, math.cos(psi/2)]], dtype=complex)
    return np.dot(uz_phi, np.dot(uy_theta, ux_psi))


def convert_su2_to_so3(matrix: np.ndarray) -> np.ndarray:
    """Computes SO(3)-matrix from input SU(2)-matrix.

    Args:
        matrix: The SU(2)-matrix for which a corresponding SO(3)-matrix needs to be computed.

    Returns:
        The SO(3)-matrix corresponding to ``matrix``.

    Raises:
        ValueError: if ``matrix`` is not an SU(2)-matrix.
    """
    if matrix.shape != (2, 2):
        raise ValueError(
            'Conversion from SU2 called on matrix of shape', matrix.shape)

    if abs(np.linalg.det(matrix) - 1) > 1e-4:
        raise ValueError(
            'Conversion from SU2 called on determinant of', np.linalg.det(matrix))

    matrix = matrix.astype(np.complex)
    a = np.real(matrix[0][0])
    b = np.imag(matrix[0][0])
    c = -np.real(matrix[0][1])
    d = -np.imag(matrix[0][1])
    rotation = np.array([[a**2-b**2-c**2+d**2, 2*a*b+2*c*d, -2*a*c+2*b*d],
                         [-2*a*b+2*c*d, a**2-b**2+c**2-d**2, 2*a*d+2*b*c],
                         [2*a*c+2*b*d, 2*b*c-2*a*d, a**2+b**2-c**2-d**2]], dtype=float)
    return rotation


def solve_decomposition_angle(matrix: np.ndarray) -> float:
    """Computes angle for balanced commutator of SO(3)-matrix ``matrix``.

    Computes angle a so that the SO(3)-matrix ``matrix`` can be decomposed
    as commutator [v,w] where v and w are both rotations of a about some axis.
    The computation is done by solving a trigonometric equation using scipy.optimize.fsolve.

    Args:
        matrix: The SO(3)-matrix for which the decomposition angle needs to be computed.

    Returns:
        Angle a so that matrix = [v,w] with v and w rotations of a about some axis.

    Raises:
        ValueError: if ``matrix`` is not an SO(3)-matrix.
    """
    descr_method = 'Computation of decomposition angle'
    if matrix.shape != (3, 3):
        raise ValueError(
            descr_method + 'called on matrix of shape', matrix.shape)

    if abs(np.linalg.det(matrix) - 1) > 1e-4:
        raise ValueError(
            descr_method + 'called on determinant of', np.linalg.det(matrix))

    trace = _compute_trace_so3(matrix)
    angle = math.acos((1/2)*(trace-1))

    def objective(phi):
        rhs = 2 * math.sin(phi / 2) ** 2
        rhs *= math.sqrt(1 - math.sin(phi / 2) ** 4)
        lhs = math.sin(angle / 2)
        return rhs - lhs

    decomposition_angle = scipy.optimize.fsolve(objective, angle)[0]
    return decomposition_angle


def compute_euler_angles_from_s03(matrix: np.ndarray) -> Tuple[float, float, float]:
    """Computes the Euler angles from the input SO(3)-matrix.

    Uses the algorithm from Gregory Slabaugh,
    see `here <https://www.gregslabaugh.net/publications/euler.pdf>`_.

    Args:
        matrix: The SO(3)-matrix for which the Euler angles need to be computed.

    Returns:
        A tuple ``(phi, theta, psi)``, which indicate rotations about the Z, Y and X axis,
        respectively.
    """
    if matrix[2][0] != 1 and matrix[2][1] != -1:
        theta = -math.asin(matrix[2][0])
        psi = math.atan2(matrix[2][1]/math.cos(theta),
                         matrix[2][2]/math.cos(theta))
        phi = math.atan2(matrix[1][0]/math.cos(theta),
                         matrix[0][0]/math.cos(theta))
        return phi, theta, psi
    else:
        phi = 0
        if matrix[2][0] == 1:
            theta = math.pi/2
            psi = phi + math.atan2(matrix[0][1], matrix[0][2])
        else:
            theta = -math.pi/2
            psi = -phi + math.atan2(-matrix[0][1], -matrix[0][2])
        return phi, theta, psi


def _compute_trace_so3(matrix: np.ndarray) -> float:
    """Computes trace of an SO(3)-matrix.

    Args:
        matrix: an SO(3)-matrix

    Returns:
        Trace of ``matrix``.

    Raises:
        ValueError: if ``matrix`` is not an SO(3)-matrix.
    """
    if matrix.shape != (3, 3):
        raise ValueError(
            'Computation of trace SO(3) called on matrix of shape', matrix.shape)

    if abs(np.linalg.det(matrix) - 1) > 1e-4:
        raise ValueError(
            'Computation of trace SO(3) called on determinant of', np.linalg.det(matrix))

    trace = np.matrix.trace(matrix)
    trace_rounded = min(trace, 3)
    return trace_rounded


def compute_rotation_between(from_vector: np.ndarray, to_vector: np.ndarray) -> np.ndarray:
    """Computes the SO(3)-matrix for rotating ``from_vector`` to ``to_vector``.

    Args:
        from_vector: unit vector of size 3
        to_vector: unit vector of size 3

    Returns:
        SO(3)-matrix that brings ``from_vector`` to ``to_vector``.

    Raises:
        ValueError: if at least one of ``from_vector`` of ``to_vector`` is not a 3-dim unit vector.
    """
    # descr_method = 'Computation rotation between vectors'
    # if from_vector.shape != (3,):
    #     raise ValueError(
    #         descr_method + 'called on matrix of shape', from_vector.shape)

    # if to_vector.shape != (3,):
    #     raise ValueError(
    #         descr_method + 'called on matrix of shape', to_vector.shape)

    # if abs(np.linalg.norm(from_vector)-1.0) > 1e-4:
    #     raise ValueError(
    #         descr_method + 'called on vector with norm', np.linalg.norm(from_vector))

    # if abs(np.linalg.norm(to_vector)-1.0) > 1e-4:
    #     raise ValueError(
    #         descr_method + 'called on vector with norm', np.linalg.norm(to_vector))
    from_vector = from_vector / np.linalg.norm(from_vector)
    to_vector = to_vector / np.linalg.norm(to_vector)

    v = np.cross(from_vector, to_vector)
    c = np.dot(from_vector, to_vector)
    cross_product_matrix = _cross_product_matrix(v)
    rotation_matrix = np.identity(3) + cross_product_matrix + 1 / \
        (1+c)*np.dot(cross_product_matrix, cross_product_matrix)
    return rotation_matrix


def _cross_product_matrix(v: np.ndarray) -> np.ndarray:
    """Computes cross product matrix from vector.

    Args:
        v: Vector for which cross product matrix needs to be computed.

    Returns:
        The cross product matrix corresponding to vector ``v``.
    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _compute_commutator_so3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes the commutator of the SO(3)-matrices ``a`` and ``b``.

    The computation uses the fact that the inverse of an SO(3)-matrix is equal to its transpose.

    Args:
        a: SO(3)-matrix
        b: SO(3)-matrix

    Returns:
        The commutator [a,b] of ``a`` and ``b`` w

    Raises:
        ValueError: if at least one of ``a`` or ``b`` is not an SO(3)-matrix.
    """
    if a.shape != (3, 3):
        raise ValueError(
            'Computation of trace SO(3) called on matrix of shape', a.shape)

    if abs(np.linalg.det(a) - 1) > 1e-4:
        raise ValueError(
            'Computation of trace SO(3) called on determinant of', np.linalg.det(a))

    if b.shape != (3, 3):
        raise ValueError(
            'Computation of trace SO(3) called on matrix of shape', b.shape)

    if abs(np.linalg.det(b) - 1) > 1e-4:
        raise ValueError(
            'Computation of trace SO(3) called on determinant of', np.linalg.det(b))

    # pylint:disable=assignment-from-no-return
    a_dagger = np.matrix.getH(a)
    b_dagger = np.matrix.getH(b)

    return np.dot(np.dot(np.dot(a, b), a_dagger), b_dagger)


# pylint: disable=invalid-name
def compute_rotation_from_angle_and_axis(angle: float, axis: np.ndarray) -> np.ndarray:
    """Computes the SO(3)-matrix corresponding to the rotation of ``angle`` about ``axis``.

    Args:
        angle: The angle of the rotation.
        axis: The axis of the rotation.

    Returns:
        SO(3)-matrix that represents a rotation of ``angle`` about ``axis``.

    Raises:
        ValueError: if ``axis`` is not a 3-dim unit vector.
    """
    descr_method = 'Computation rotation from angle and axis'
    if axis.shape != (3,):
        raise ValueError(
            descr_method + 'called on matrix of shape', axis.shape)

    if abs(np.linalg.norm(axis)-1.0) > 1e-4:
        raise ValueError(
            descr_method + 'called on vector with norm', np.linalg.norm(axis))

    res = math.cos(angle) * np.identity(3) + \
        math.sin(angle) * _cross_product_matrix(axis)
    res += (1 - math.cos(angle)) * np.outer(axis, axis)
    return res


def compute_rotation_axis(matrix: np.ndarray) -> np.ndarray:
    """Computes rotation axis of SO(3)-matrix.

    Args:
        matrix: The SO(3)-matrix for which rotation angle needs to be computed.

    Returns:
        The rotation axis of the SO(3)-matrix ``matrix``.

    Raises:
        ValueError: if ``matrix`` is not an SO(3)-matrix.
    """
    if matrix.shape != (3, 3):
        raise ValueError(
            'Computation of trace SO(3) called on matrix of shape', matrix.shape)

    if abs(np.linalg.det(matrix) - 1) > 1e-4:
        raise ValueError(
            'Computation of trace SO(3) called on determinant of', np.linalg.det(matrix))

    trace = _compute_trace_so3(matrix)
    theta = math.acos((1/2)*(trace-1))
    if math.sin(theta) > 1e-10:
        x = (1/(2*math.sin(theta)))*(matrix[2][1]-matrix[1][2])
        y = (1/(2*math.sin(theta)))*(matrix[0][2]-matrix[2][0])
        z = (1/(2*math.sin(theta)))*(matrix[1][0]-matrix[0][1])
    else:
        x = 1.0
        y = 0.0
        z = 0.0
    return np.array([x, y, z])

# TODO: unittesten


def convert_so3_to_su2(matrix: np.ndarray) -> np.ndarray:
    """Converts an SO(3)-matrix to a corresponding SU(2)-matrix.

    Args:
        matrix: SO(3)-matrix to convert.

    Returns:
        SU(2)-matrix corresponding to SO(3)-matrix ``matrix``.

    Raises:
        ValueError: if ``matrix`` is not an SO(3)-matrix.
    """
    if matrix.shape != (3, 3):
        raise ValueError(
            'Computation of trace SO(3) called on matrix of shape', matrix.shape)

    if abs(np.linalg.det(matrix) - 1) > 1e-4:
        raise ValueError(
            'Computation of trace SO(3) called on determinant of', np.linalg.det(matrix))

    return compute_su2_from_euler_angles(compute_euler_angles_from_so3(matrix))
