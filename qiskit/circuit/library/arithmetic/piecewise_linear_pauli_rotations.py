# -*- coding: utf-8 -*-

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

# pylint: disable=no-member

"""Piecewise-linearly-controlled rotation."""

from typing import List, Optional
import numpy as np

from qiskit.circuit import QuantumRegister
from qiskit.circuit.exceptions import CircuitError

from .functional_pauli_rotations import FunctionalPauliRotations
from .linear_pauli_rotations import LinearPauliRotations
from .integer_comparator import IntegerComparator


class PiecewiseLinearPauliRotations(FunctionalPauliRotations):
    r"""Piecewise-linearly-controlled Pauli rotations.

    For a piecewise linear (not necessarily continuous) function :math:`f(x)`, which is defined
    through breakpoints, slopes and offsets as follows.
    Suppose the breakpoints :math:`(x_0, ..., x_J)` are a subset of :math:`[0, 2^n-1]`, where
    :math:`n` is the number of state qubits. Further on, denote the corresponding slopes and
    offsets by :math:`a_j` and :math:`b_j` respectively.
    Then f(x) is defined as:

    .. math::

        f(x) = \begin{cases}
            0, x < x_0 \\
            a_j (x - x_j) + b_j, x_j \leq x < x_{j+1}
            \end{cases}

    where we implicitly assume :math:`x_{J+1} = 2^n`.
    """

    def __init__(self,
                 num_state_qubits: Optional[int] = None,
                 breakpoints: Optional[List[int]] = None,
                 slopes: Optional[List[float]] = None,
                 offsets: Optional[List[float]] = None,
                 basis: str = 'Y',
                 name: str = 'pw_lin') -> None:
        """Construct piecewise-linearly-controlled Pauli rotations.

        Args:
            num_state_qubits: The number of qubits representing the state.
            breakpoints: The breakpoints to define the piecewise-linear function.
                Defaults to ``[0]``.
            slopes: The slopes for different segments of the piecewise-linear function.
                Defaults to ``[1]``.
            offsets: The offsets for different segments of the piecewise-linear function.
                Defaults to ``[0]``.
            basis: The type of Pauli rotation (``'X'``, ``'Y'``, ``'Z'``).
            name: The name of the circuit.
        """
        # store parameters
        self._breakpoints = breakpoints if breakpoints is not None else [0]
        self._slopes = slopes if slopes is not None else [1]
        self._offsets = offsets if offsets is not None else [0]

        super().__init__(num_state_qubits=num_state_qubits, basis=basis, name=name)

    @property
    def breakpoints(self) -> List[int]:
        """The breakpoints of the piecewise linear function.

        The function is linear in the intervals ``[point_i, point_{i+1}]`` where the last
        point implicitely is ``2**(num_state_qubits + 1)``.
        """
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
    def slopes(self) -> List[int]:
        """The breakpoints of the piecewise linear function.

        The function is linear in the intervals ``[point_i, point_{i+1}]`` where the last
        point implicitely is ``2**(num_state_qubits + 1)``.
        """
        return self._slopes

    @slopes.setter
    def slopes(self, slopes: List[float]) -> None:
        """Set the slopes.

        Args:
            slopes: The new slopes.
        """
        self._invalidate()
        self._slopes = slopes

    @property
    def offsets(self) -> List[float]:
        """The breakpoints of the piecewise linear function.

        The function is linear in the intervals ``[point_i, point_{i+1}]`` where the last
        point implicitely is ``2**(num_state_qubits + 1)``.
        """
        return self._offsets

    @offsets.setter
    def offsets(self, offsets: List[float]) -> None:
        """Set the offsets.

        Args:
            offsets: The new offsets.
        """
        self._invalidate()
        self._offsets = offsets

    @property
    def mapped_slopes(self) -> List[float]:
        """The slopes mapped to the internal representation.

        Returns:
            The mapped slopes.
        """
        mapped_slopes = np.zeros_like(self.slopes)
        for i, slope in enumerate(self.slopes):
            mapped_slopes[i] = slope - sum(mapped_slopes[:i])

        return mapped_slopes

    @property
    def mapped_offsets(self) -> List[float]:
        """The offsets mapped to the internal representation.

        Returns:
            The mapped offsets.
        """
        mapped_offsets = np.zeros_like(self.offsets)
        for i, (offset, slope, point) in enumerate(zip(self.offsets,
                                                       self.slopes,
                                                       self.breakpoints)):
            mapped_offsets[i] = offset - slope * point - sum(mapped_offsets[:i])

        return mapped_offsets

    @property
    def contains_zero_breakpoint(self) -> bool:
        """Whether 0 is the first breakpoint.

        Returns:
            True, if 0 is the first breakpoint, otherwise False.
        """
        return np.isclose(0, self.breakpoints[0])

    def evaluate(self, x: float) -> float:
        """Classically evaluate the piecewise linear rotation.

        Args:
            x: Value to be evaluated at.

        Returns:
            Value of piecewise linear function at x.
        """

        y = (x >= self.breakpoints[0]) * (x * self.mapped_slopes[0] + self.mapped_offsets[0])
        for i in range(1, len(self.breakpoints)):
            y = y + (x >= self.breakpoints[i]) * (x * self.mapped_slopes[i] +
                                                  self.mapped_offsets[i])

        return y

    @property
    def num_ancilla_qubits(self) -> int:
        """The number of ancilla qubits.

        Returns:
            The number of ancilla qubits in the circuit.
        """
        num_ancilla_qubits = self.num_state_qubits - 1 + len(self.breakpoints)
        if self.contains_zero_breakpoint:
            num_ancilla_qubits -= 1
        return num_ancilla_qubits

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

        if len(self.breakpoints) != len(self.slopes) or len(self.breakpoints) != len(self.offsets):
            valid = False
            if raise_on_failure:
                raise ValueError('Mismatching sizes of breakpoints, slopes and offsets.')

        return valid

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
        super()._build()

        qr_state = self.qubits[:self.num_state_qubits]
        qr_target = [self.qubits[self.num_state_qubits]]
        qr_ancilla = self.qubits[self.num_state_qubits + 1:]

        # apply comparators and controlled linear rotations
        for i, point in enumerate(self.breakpoints):
            if i == 0 and self.contains_zero_breakpoint:
                # apply rotation
                lin_r = LinearPauliRotations(num_state_qubits=self.num_state_qubits,
                                             slope=self.mapped_slopes[i],
                                             offset=self.mapped_offsets[i],
                                             basis=self.basis)
                self.append(lin_r.to_gate(), qr_state[:] + qr_target)

            else:
                if self.contains_zero_breakpoint:
                    i_compare = i - 1
                else:
                    i_compare = i

                # apply Comparator
                comp = IntegerComparator(num_state_qubits=self.num_state_qubits, value=point)
                qr = qr_state[:] + [qr_ancilla[i_compare]]  # add ancilla as compare qubit
                qr_remaining_ancilla = qr_ancilla[i_compare + 1:]  # take remaining ancillas

                self.append(comp.to_gate(),
                            qr[:] + qr_remaining_ancilla[:comp.num_ancilla_qubits])

                # apply controlled rotation
                lin_r = LinearPauliRotations(num_state_qubits=self.num_state_qubits,
                                             slope=self.mapped_slopes[i],
                                             offset=self.mapped_offsets[i],
                                             basis=self.basis)
                self.append(lin_r.to_gate().control(),
                            [qr_ancilla[i_compare]] + qr_state[:] + qr_target)

                # uncompute comparator
                self.append(comp.to_gate().inverse(),
                            qr[:] + qr_remaining_ancilla[:comp.num_ancilla_qubits])
