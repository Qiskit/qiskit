# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
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

from qiskit.circuit import QuantumRegister
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

    The values of the parameters are calculated according to https://arxiv.org/abs/2009.04484.
    """

    def __init__(self,
                 num_state_qubits: Optional[int] = None,
                 epsilon: Optional[float] = None,
                 constant: Optional[float] = None,
                 kappa: Optional[float] = None,
                 name: str = 'inv_cheb') -> None:
        """Construct the piecewise Chebyshev approximation of the inverse function.

        Args:
            num_state_qubits: number of qubits representing the state.
            epsilon: accuracy of the approximation.
                Defaults to ``1e-2``.
            constant: :math:`C` in :math:`arcsin(C/x)`.
                Defaults to ``1``.
            kappa: condition number of the system.
                Defaults to ``1``.
            name: The name of the circuit object.
        """
        super().__init__(name=name)

        # define internal parameters
        self._num_state_qubits = None

        # Store parameters
        self._epsilon = epsilon if epsilon is not None else 1e-2
        self._kappa = kappa if kappa is not None else 1
        self._constant = constant if constant is not None else 1

        # We perform the identity operation on [1,a].
        # int(round()) necessary to compensate for computer precision.
        self._N_l = 2 ** num_state_qubits
        self._a = int(round(self._N_l ** (2/3)))

        # Calculate the degree of the polynomial and the number of intervals
        r = 2 * self._constant / self._a + np.sqrt(np.abs(1 - (2 * self._constant / self._a) ** 2))
        self._degree = int(np.log(1 + (16.23 * np.sqrt(np.log(r) ** 2 + (np.pi/2) ** 2) *
                                       self._kappa * (2 * self._kappa - self._epsilon)) /
                                  self._epsilon))
        self._num_intervals = int(np.ceil(np.log((self._N_l - 1) / self._a) / np.log(5)))

        # Calculate breakpoints and polynomials
        self._breakpoints = []
        self._polynomials = []
        for i in range(0, self._num_intervals):
            # Add the breakpoint to the list
            self._breakpoints.append(self._a * (5 ** i))
            # Define the right breakpoint of the interval
            if i == self._num_intervals - 1:
                r_breakpoint = self._N_l - 1
            else:
                r_breakpoint = self._a * (5 ** (i + 1))
            # Calculate the polynomial approximating the function on the current interval
            poly = Chebyshev.interpolate(lambda x: np.arcsin(self._constant / x), self._degree,
                                         domain=[self._breakpoints[i], r_breakpoint])
            # Convert polynomial to the standard basis and rescale it for the rotation gates
            poly = 2 * poly.convert(kind=np.polynomial.Polynomial).coef
            # Convert to list and append
            self._polynomials.append(poly.tolist())

        self._poly_r = PiecewisePolynomialPauliRotations(num_state_qubits, self._breakpoints,
                                                         self._polynomials)

        self.num_state_qubits = num_state_qubits

    @property
    def num_ancilla_qubits(self):
        """Return the number of required ancillas."""
        return self._poly_r.num_ancilla_qubits

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

            if self.num_ancilla_qubits > 0:
                qr_ancilla = QuantumRegister(self.num_ancilla_qubits)
                self.qregs += [qr_ancilla]
        else:
            self.qregs = []

    def _build(self):
        """Build the circuit. The operation is considered successful when q_objective is
        :math:`|1>`."""
        super()._build()

        qr_state = self.qubits[:self.num_state_qubits]
        qr_target = [self.qubits[self.num_state_qubits]]
        qr_ancillas = self.qubits[self.num_state_qubits + 1:]

        # For x<a we apply the identity, so q_objective has to be set manually to :math:`|1>`.
        comp = IntegerComparator(num_state_qubits=self.num_state_qubits, value=self._a, geq=False)

        qr = qr_state[:] + [qr_ancillas[0]]  # add ancilla as compare qubit
        qr_remaining_ancilla = qr_ancillas[1:]  # take remaining ancillas

        self.append(comp.to_gate(),
                    qr[:] + qr_remaining_ancilla[:comp.num_ancilla_qubits])

        # Apply a CNOT gate to the objective.
        self.cx(qr_ancillas[0], qr_target[0])

        # Uncompute comparator.
        self.append(comp.to_gate().inverse(),
                    qr[:] + qr_remaining_ancilla[:comp.num_ancilla_qubits])

        # Apply polynomial approximation for x>=a.
        self.append(self._poly_r.to_instruction(), qr_state[:] + qr_target + qr_ancillas[:])
