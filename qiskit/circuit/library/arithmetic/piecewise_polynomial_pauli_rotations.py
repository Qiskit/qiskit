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

"""Piecewise-polynomially-controlled Pauli rotations."""

from typing import List, Optional
import numpy as np

from qiskit.circuit import QuantumRegister, AncillaRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError

from .functional_pauli_rotations import FunctionalPauliRotations
from .polynomial_pauli_rotations import PolynomialPauliRotations
from .integer_comparator import IntegerComparator


class PiecewisePolynomialPauliRotations(FunctionalPauliRotations):
    r"""Piecewise-polynomially-controlled Pauli rotations.

    This class implements a piecewise polynomial (not necessarily continuous) function,
    :math:`f(x)`, on qubit amplitudes, which is defined through breakpoints and coefficients as
    follows.
    Suppose the breakpoints :math:`(x_0, ..., x_J)` are a subset of :math:`[0, 2^n-1]`, where
    :math:`n` is the number of state qubits. Further on, denote the corresponding coefficients by
    :math:`[a_{j,1},...,a_{j,d}]`, where :math:`d` is the highest degree among all polynomials.

    Then :math:`f(x)` is defined as:

    .. math::

        f(x) = \begin{cases}
            0, x < x_0 \\
            \sum_{i=0}^{i=d}a_{j,i}/2 x^i, x_j \leq x < x_{j+1}
            \end{cases}

    where if given the same number of breakpoints as polynomials, we implicitly assume
    :math:`x_{J+1} = 2^n`.

    .. note::

        Note the :math:`1/2` factor in the coefficients of :math:`f(x)`, this is consistent with
        Qiskit's Pauli rotations.

    Examples:
        >>> from qiskit import QuantumCircuit
        >>> from qiskit.circuit.library.arithmetic.piecewise_polynomial_pauli_rotations import\
        ... PiecewisePolynomialPauliRotations
        >>> qubits, breakpoints, coeffs = (2, [0, 2], [[0, -1.2],[-1, 1, 3]])
        >>> poly_r = PiecewisePolynomialPauliRotations(num_state_qubits=qubits,
        ...breakpoints=breakpoints, coeffs=coeffs)
        >>>
        >>> qc = QuantumCircuit(poly_r.num_qubits)
        >>> qc.h(list(range(qubits)));
        >>> qc.append(poly_r.to_instruction(), list(range(qc.num_qubits)));
        >>> qc.draw()
             ┌───┐┌──────────┐
        q_0: ┤ H ├┤0         ├
             ├───┤│          │
        q_1: ┤ H ├┤1         ├
             └───┘│          │
        q_2: ─────┤2         ├
                  │  pw_poly │
        q_3: ─────┤3         ├
                  │          │
        q_4: ─────┤4         ├
                  │          │
        q_5: ─────┤5         ├
                  └──────────┘

    References:
        [1]: Haener, T., Roetteler, M., & Svore, K. M. (2018).
             Optimizing Quantum Circuits for Arithmetic.
             `arXiv:1805.12445 <http://arxiv.org/abs/1805.12445>`_

        [2]: Carrera Vazquez, A., Hiptmair, R., & Woerner, S. (2022).
             Enhancing the Quantum Linear Systems Algorithm using Richardson Extrapolation.
             `ACM Transactions on Quantum Computing 3, 1, Article 2 <https://doi.org/10.1145/3490631>`_
    """

    def __init__(
        self,
        num_state_qubits: Optional[int] = None,
        breakpoints: Optional[List[int]] = None,
        coeffs: Optional[List[List[float]]] = None,
        basis: str = "Y",
        name: str = "pw_poly",
    ) -> None:
        """
        Args:
            num_state_qubits: The number of qubits representing the state.
            breakpoints: The breakpoints to define the piecewise-linear function.
                Defaults to ``[0]``.
            coeffs: The coefficients of the polynomials for different segments of the
            piecewise-linear function. ``coeffs[j][i]`` is the coefficient of the i-th power of x
            for the j-th polynomial.
                Defaults to linear: ``[[1]]``.
            basis: The type of Pauli rotation (``'X'``, ``'Y'``, ``'Z'``).
            name: The name of the circuit.
        """
        # store parameters
        self._breakpoints = breakpoints if breakpoints is not None else [0]
        self._coeffs = coeffs if coeffs is not None else [[1]]

        # store a list of coefficients as homogeneous polynomials adding 0's where necessary
        self._hom_coeffs = []
        self._degree = len(max(self._coeffs, key=len)) - 1
        for poly in self._coeffs:
            self._hom_coeffs.append(poly + [0] * (self._degree + 1 - len(poly)))

        super().__init__(num_state_qubits=num_state_qubits, basis=basis, name=name)

    @property
    def breakpoints(self) -> List[int]:
        """The breakpoints of the piecewise polynomial function.

        The function is polynomial in the intervals ``[point_i, point_{i+1}]`` where the last
        point implicitly is ``2**(num_state_qubits + 1)``.

        Returns:
            The list of breakpoints.
        """
        if (
            self.num_state_qubits is not None
            and len(self._breakpoints) == len(self.coeffs)
            and self._breakpoints[-1] < 2**self.num_state_qubits
        ):
            return self._breakpoints + [2**self.num_state_qubits]

        return self._breakpoints

    @breakpoints.setter
    def breakpoints(self, breakpoints: List[int]) -> None:
        """Set the breakpoints.

        Args:
            breakpoints: The new breakpoints.
        """
        self._invalidate()
        self._breakpoints = breakpoints

        if self.num_state_qubits and breakpoints:
            self._reset_registers(self.num_state_qubits)

    @property
    def coeffs(self) -> List[List[float]]:
        """The coefficients of the polynomials.

        Returns:
            The polynomial coefficients per interval as nested lists.
        """
        return self._coeffs

    @coeffs.setter
    def coeffs(self, coeffs: List[List[float]]) -> None:
        """Set the polynomials.

        Args:
            coeffs: The new polynomials.
        """
        self._invalidate()
        self._coeffs = coeffs

        # update the homogeneous polynomials and degree
        self._hom_coeffs = []
        self._degree = len(max(self._coeffs, key=len)) - 1
        for poly in self._coeffs:
            self._hom_coeffs.append(poly + [0] * (self._degree + 1 - len(poly)))

        if self.num_state_qubits and coeffs:
            self._reset_registers(self.num_state_qubits)

    @property
    def mapped_coeffs(self) -> List[List[float]]:
        """The coefficients mapped to the internal representation, since we only compare
        x>=breakpoint.

        Returns:
            The mapped coefficients.
        """
        mapped_coeffs = []

        # First polynomial
        mapped_coeffs.append(self._hom_coeffs[0])
        for i in range(1, len(self._hom_coeffs)):
            mapped_coeffs.append([])
            for j in range(0, self._degree + 1):
                mapped_coeffs[i].append(self._hom_coeffs[i][j] - self._hom_coeffs[i - 1][j])

        return mapped_coeffs

    @property
    def contains_zero_breakpoint(self) -> bool:
        """Whether 0 is the first breakpoint.

        Returns:
            True, if 0 is the first breakpoint, otherwise False.
        """
        return np.isclose(0, self.breakpoints[0])

    def evaluate(self, x: float) -> float:
        """Classically evaluate the piecewise polynomial rotation.

        Args:
            x: Value to be evaluated at.

        Returns:
            Value of piecewise polynomial function at x.
        """

        y = 0
        for i in range(0, len(self.breakpoints)):
            y = y + (x >= self.breakpoints[i]) * (np.poly1d(self.mapped_coeffs[i][::-1])(x))

        return y

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid."""
        valid = True

        if self.num_state_qubits is None:
            valid = False
            if raise_on_failure:
                raise AttributeError("The number of qubits has not been set.")

        if self.num_qubits < self.num_state_qubits + 1:
            valid = False
            if raise_on_failure:
                raise CircuitError(
                    "Not enough qubits in the circuit, need at least "
                    "{}.".format(self.num_state_qubits + 1)
                )

        if len(self.breakpoints) != len(self.coeffs) + 1:
            valid = False
            if raise_on_failure:
                raise ValueError("Mismatching number of breakpoints and polynomials.")

        return valid

    def _reset_registers(self, num_state_qubits: Optional[int]) -> None:
        """Reset the registers."""
        self.qregs = []

        if num_state_qubits:
            qr_state = QuantumRegister(num_state_qubits)
            qr_target = QuantumRegister(1)
            self.qregs = [qr_state, qr_target]

            # Calculate number of ancilla qubits required
            num_ancillas = num_state_qubits + 1
            if self.contains_zero_breakpoint:
                num_ancillas -= 1
            if num_ancillas > 0:
                qr_ancilla = AncillaRegister(num_ancillas)
                self.add_register(qr_ancilla)

    def _build(self):
        """If not already built, build the circuit."""
        if self._is_built:
            return

        super()._build()

        circuit = QuantumCircuit(*self.qregs, name=self.name)
        qr_state = circuit.qubits[: self.num_state_qubits]
        qr_target = [circuit.qubits[self.num_state_qubits]]
        # Ancilla for the comparator circuit
        qr_ancilla = circuit.qubits[self.num_state_qubits + 1 :]

        # apply comparators and controlled linear rotations
        for i, point in enumerate(self.breakpoints[:-1]):
            if i == 0 and self.contains_zero_breakpoint:
                # apply rotation
                poly_r = PolynomialPauliRotations(
                    num_state_qubits=self.num_state_qubits,
                    coeffs=self.mapped_coeffs[i],
                    basis=self.basis,
                )
                circuit.append(poly_r.to_gate(), qr_state[:] + qr_target)

            else:
                # apply Comparator
                comp = IntegerComparator(num_state_qubits=self.num_state_qubits, value=point)
                qr_state_full = qr_state[:] + [qr_ancilla[0]]  # add compare qubit
                qr_remaining_ancilla = qr_ancilla[1:]  # take remaining ancillas

                circuit.append(
                    comp.to_gate(), qr_state_full[:] + qr_remaining_ancilla[: comp.num_ancillas]
                )

                # apply controlled rotation
                poly_r = PolynomialPauliRotations(
                    num_state_qubits=self.num_state_qubits,
                    coeffs=self.mapped_coeffs[i],
                    basis=self.basis,
                )
                circuit.append(
                    poly_r.to_gate().control(), [qr_ancilla[0]] + qr_state[:] + qr_target
                )

                # uncompute comparator
                circuit.append(
                    comp.to_gate().inverse(),
                    qr_state_full[:] + qr_remaining_ancilla[: comp.num_ancillas],
                )

        self.append(circuit.to_gate(), self.qubits)
