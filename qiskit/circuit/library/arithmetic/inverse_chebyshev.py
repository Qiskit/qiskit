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

"""Piecewise polynomial approximation to arcsin(1/x)."""

from typing import Optional
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

from qiskit.circuit import QuantumRegister, AncillaRegister
from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit
from qiskit.circuit.exceptions import CircuitError

from .piecewise_polynomial_pauli_rotations import PiecewisePolynomialPauliRotations
from .integer_comparator import IntegerComparator

# pylint: disable=invalid-name


class InverseChebyshev(BlueprintCircuit):
    r"""Piecewise Chebyshev approximation of the inverse function.

    Building block of HHL. For a given constant :math:`C` and error tolerance :math:`\epsilon`,
    this class implements a piecewise polynomial approximation on :math:`n` qubits to
    :math:`arcsin(C/x)`, where :math:`x \in [a,2^n - 1]` and :math:`a` depends on :math:`\epsilon`.

    The values of the parameters are calculated according to [1].

    Examples:
        >>> from qiskit import QuantumCircuit
        >>> from qiskit.circuit.library.arithmetic.inverse_chebyshev import InverseChebyshev
        >>> num_qubits = 4
        >>> err_tolerance = 0.1
        >>> inv_cheb = InverseChebyshev(num_qubits, err_tolerance)
        >>> inv_cheb._build()
        >>> qc = QuantumCircuit(num_qubits + 1 + inv_cheb.num_ancillas)
        >>> qc.h(list(range(num_qubits)))
        <qiskit.circuit.instructionset.InstructionSet object at 0x00000297722900B8>
        >>> qc.append(inv_cheb.to_instruction(), list(range(qc.num_qubits)))
        >>> qc.draw()
              ┌───┐┌────────────┐
         q_0: ┤ H ├┤0           ├
              ├───┤│            │
         q_1: ┤ H ├┤1           ├
              ├───┤│            │
         q_2: ┤ H ├┤2           ├
              ├───┤│            │
         q_3: ┤ H ├┤3           ├
              └───┘│            │
         q_4: ─────┤4           ├
                   │            │
         q_5: ─────┤5           ├
                   │            │
         q_6: ─────┤6           ├
                   │   inv_cheb │
         q_7: ─────┤7           ├
                   │            │
         q_8: ─────┤8           ├
                   │            │
         q_9: ─────┤9           ├
                   │            │
        q_10: ─────┤10          ├
                   │            │
        q_11: ─────┤11          ├
                   │            │
        q_12: ─────┤12          ├
                   │            │
        q_13: ─────┤13          ├
                   └────────────┘

    References:

        [1]: Carrera Vazquez, A., Hiptmair, R., & Woerner, S. (2020).
             Enhancing the Quantum Linear Systems Algorithm using Richardson Extrapolation.
             `arXiv:2009.04484 <http://arxiv.org/abs/2009.04484>`_

        [2]: Haener, T., Roetteler, M., & Svore, K. M. (2018).
             Optimizing Quantum Circuits for Arithmetic.
             `arXiv:1805.12445 <http://arxiv.org/abs/1805.12445>`_
    """

    def __init__(self,
                 num_state_qubits: Optional[int] = None,
                 epsilon: float = 1e-2,
                 constant: float = 1,
                 kappa: float = 1,
                 name: str = 'inv_cheb') -> None:
        """
        Args:
            num_state_qubits: number of qubits representing the state.
            epsilon: accuracy of the approximation.
            constant: :math:`C` in :math:`arcsin(C/x)`.
            kappa: condition number of the system.
            name: The name of the circuit object.
        """
        super().__init__(name=name)

        # define internal parameters
        self._num_state_qubits = None

        # Store parameters
        self._epsilon = epsilon
        self._kappa = kappa
        self._constant = constant

        self._breakpoints = None
        self._polynomials = None
        self._poly_r = None

        self.num_state_qubits = num_state_qubits

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        valid = True

        if self.num_state_qubits is None:
            valid = False
            if raise_on_failure:
                raise AttributeError('The number of qubits has not been set.')

        if self.num_qubits < self.num_state_qubits + 1:
            valid = False
            if raise_on_failure:
                raise CircuitError('Not enough qubits in the circuit, need at least '
                                   '{}.'.format(self.num_state_qubits + 1))

        if len(self._breakpoints) != len(self._polynomials):
            valid = False
            if raise_on_failure:
                raise ValueError('Mismatching number of breakpoints and polynomials.')

        return valid

    @property
    def epsilon(self) -> float:
        """The error tolerance.

                Returns:
                    The error tolerance.
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon: Optional[float]) -> None:
        """Set the error tolerance.

        Note that this may change the underlying quantum register, if the number of state qubits
        changes.

        Args:
            epsilon: The new error tolerance.
        """
        if self._epsilon is None or epsilon != self._epsilon:
            self._invalidate()
            self._epsilon = epsilon

            self._reset_registers(self.num_state_qubits)

    @property
    def constant(self) -> float:
        r"""The constant :math:`C` in :math:`arcsin(C/x)`.

                Returns:
                    The constant :math:`C` in :math:`arcsin(C/x)`.
        """
        return self._constant

    @constant.setter
    def constant(self, constant: Optional[float]) -> None:
        r"""Set the constant :math:`C` in :math:`arcsin(C/x)`.

        Note that this may change the underlying quantum register, if the number of state qubits
        changes.

        Args:
            constant: The new constant :math:`C` in :math:`arcsin(C/x)`.
        """
        if self._constant is None or constant != self._constant:
            self._invalidate()
            self._constant = constant

            self._reset_registers(self.num_state_qubits)

    @property
    def num_state_qubits(self) -> int:
        r"""The number of state qubits representing the state :math:`|x\rangle`.

        Returns:
            The number of state qubits.
        """
        return self._num_state_qubits

    @num_state_qubits.setter
    def num_state_qubits(self, num_state_qubits: Optional[int]) -> None:
        """Set the number of state qubits.

        Note that this may change the underlying quantum register, if the number of state qubits
        changes.

        Args:
            num_state_qubits: The new number of qubits.
        """
        if self._num_state_qubits is None or num_state_qubits != self._num_state_qubits:
            self._invalidate()
            self._num_state_qubits = num_state_qubits

            self._reset_registers(num_state_qubits)

    def _reset_registers(self, num_state_qubits: Optional[int]) -> None:
        if num_state_qubits:
            qr_state = QuantumRegister(num_state_qubits)
            qr_target = QuantumRegister(1)
            self.qregs = [qr_state, qr_target]

            if self._poly_r is not None:
                num_ancillas = self._poly_r.num_ancillas
                if num_ancillas > 0:
                    self._ancillas = []
                    qr_ancilla = AncillaRegister(num_ancillas)
                    self.add_register(qr_ancilla)
        else:
            self.qregs = []

    def _build(self):
        """Build the circuit. The operation is considered successful when q_objective is
        :math:`|1>`."""
        # We perform the identity operation on [1,a].
        # int(round()) necessary to compensate for computer precision.
        if self.num_state_qubits is not None:
            N_l = 2 ** self.num_state_qubits
            a = int(round(N_l ** (2 / 3)))

            # Calculate the degree of the polynomial and the number of intervals
            r = 2 * self._constant / a + np.sqrt(np.abs(1 - (2 * self._constant / a) ** 2))
            degree = int(np.log(1 + (16.23 * np.sqrt(np.log(r) ** 2 + (np.pi / 2) ** 2) *
                                     self._kappa * (2 * self._kappa - self._epsilon)) /
                                self._epsilon))
            num_intervals = int(np.ceil(np.log((N_l - 1) / a) / np.log(5)))

            # Calculate breakpoints and polynomials
            self._breakpoints = []
            self._polynomials = []
            for i in range(0, num_intervals):
                # Add the breakpoint to the list
                self._breakpoints.append(a * (5 ** i))
                # Define the right breakpoint of the interval
                if i == num_intervals - 1:
                    r_breakpoint = N_l - 1
                else:
                    r_breakpoint = a * (5 ** (i + 1))
                # Calculate the polynomial approximating the function on the current interval
                poly = Chebyshev.interpolate(lambda x: np.arcsin(self._constant / x), degree,
                                             domain=[self._breakpoints[i], r_breakpoint])
                # Convert polynomial to the standard basis and rescale it for the rotation gates
                poly = 2 * poly.convert(kind=np.polynomial.Polynomial).coef
                # Convert to list and append
                self._polynomials.append(poly.tolist())

            self._poly_r = PiecewisePolynomialPauliRotations(self.num_state_qubits,
                                                             self._breakpoints, self._polynomials)
            # poly_r has been updated, so we need to update the ancilla register
            self._reset_registers(self.num_state_qubits)

        super()._build()

        qr_state = self.qubits[:self.num_state_qubits]
        qr_target = [self.qubits[self.num_state_qubits]]
        qr_ancillas = self.qubits[self.num_state_qubits + 1:]

        # For x<a we apply the identity, so q_objective has to be set manually to :math:`|1>`.
        comp = IntegerComparator(num_state_qubits=self.num_state_qubits, value=a, geq=False)

        qr = qr_state[:] + [qr_ancillas[0]]  # add ancilla as compare qubit
        qr_remaining_ancilla = qr_ancillas[1:]  # take remaining ancillas

        self.append(comp.to_gate(),
                    qr[:] + qr_remaining_ancilla[:comp.num_ancillas])

        # Apply a CNOT gate to the objective.
        self.cx(qr_ancillas[0], qr_target[0])

        # Uncompute comparator.
        self.append(comp.to_gate().inverse(),
                    qr[:] + qr_remaining_ancilla[:comp.num_ancillas])

        # Apply polynomial approximation for x>=a.
        self._poly_r._build()
        self.append(self._poly_r.to_instruction(), qr_state[:] + qr_target + qr_ancillas[:])
