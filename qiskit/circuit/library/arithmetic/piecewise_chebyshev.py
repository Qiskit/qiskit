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

"""Piecewise polynomial Chebyshev approximation to a given f(x)."""

from typing import Callable, List, Optional
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

from qiskit.circuit import QuantumRegister, AncillaRegister
from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit
from qiskit.circuit.exceptions import CircuitError

from .piecewise_polynomial_pauli_rotations import PiecewisePolynomialPauliRotations


class PiecewiseChebyshev(BlueprintCircuit):
    r"""Piecewise Chebyshev approximation to an input function.

    For a given function :math:`f(x)` and degree :math:`d`, this class implements a piecewise
    polynomial Chebyshev approximation on :math:`n` qubits to :math:`f(x)` on the given intervals.
    All the polynomials in the approximation are of degree :math:`d`.

    The values of the parameters are calculated according to [1].

    Examples:
        >>> import numpy as np
>>> from qiskit import QuantumCircuit
>>> from qiskit.circuit.library.arithmetic.piecewise_chebyshev import PiecewiseChebyshev
        >>> f_x, degree, breakpoints, num_state_qubits = lambda x: np.arcsin(1 / x), 2, [2, 4], 2
        >>> pw_approximation = PiecewiseChebyshev(f_x, degree, breakpoints, num_state_qubits)
        >>> pw_approximation._build()
        >>> qc = QuantumCircuit(pw_approximation.num_qubits)
        >>> qc.h(list(range(num_state_qubits)))
        <qiskit.circuit.instructionset.InstructionSet object at 0x000002136F7ECFD0>
        >>> qc.append(pw_approximation.to_instruction(), list(range(qc.num_qubits)))
        <qiskit.circuit.instructionset.InstructionSet object at 0x000002136F861240>
        >>> qc.draw()
             ┌───┐┌──────────┐
        q_0: ┤ H ├┤0         ├
             ├───┤│          │
        q_1: ┤ H ├┤1         ├
             └───┘│          │
        q_2: ─────┤2         ├
                  │  pw_cheb │
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
    """

    def __init__(self,
                 f_x: Callable[[int], float],
                 degree: Optional[int] = None,
                 breakpoints: Optional[List[int]] = None,
                 num_state_qubits: Optional[int] = None,
                 name: str = 'pw_cheb') -> None:
        r"""
        Args:
            f_x: the function to be approximated.
            degree: the degree of the polynomials.
                Defaults to ``1``.
            breakpoints: the breakpoints to define the piecewise-linear function.
                Defaults to the full interval.
            num_state_qubits: number of qubits representing the state.
            name: The name of the circuit object.
        """
        super().__init__(name=name)

        # define internal parameters
        self._num_state_qubits = None

        # Store parameters
        self._f_x = f_x
        self._degree = degree if degree is not None else 1
        self._breakpoints = breakpoints if breakpoints is not None else [0]

        self._polynomials = None
        self._poly_r = None

        self.num_state_qubits = num_state_qubits

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        valid = True

        if self._f_x is None:
            valid = False
            if raise_on_failure:
                raise AttributeError('The function to be approximated has not been set.')

        if self._degree is None:
            valid = False
            if raise_on_failure:
                raise AttributeError('The degree of the polynomials has not been set.')

        if self._breakpoints is None:
            valid = False
            if raise_on_failure:
                raise AttributeError('The breakpoints have not been set.')

        if self.num_state_qubits is None:
            valid = False
            if raise_on_failure:
                raise AttributeError('The number of qubits has not been set.')

        if self.num_qubits < self.num_state_qubits + 1:
            valid = False
            if raise_on_failure:
                raise CircuitError('Not enough qubits in the circuit, need at least '
                                   '{}.'.format(self.num_state_qubits + 1))

        return valid

    @property
    def f_x(self) -> Callable[[int], float]:
        """The function to be approximated.

                Returns:
                    The function to be approximated.
        """
        return self._f_x

    @f_x.setter
    def f_x(self, f_x: Optional[Callable[[int], float]]) -> None:
        """Set the function to be approximated.

        Note that this may change the underlying quantum register, if the number of state qubits
        changes.

        Args:
            f_x: The new function to be approximated.
        """
        if self._f_x is None or f_x != self._f_x:
            self._invalidate()
            self._f_x = f_x

            self._reset_registers(self.num_state_qubits)

    @property
    def degree(self) -> int:
        """The degree of the polynomials.

                Returns:
                    The degree of the polynomials.
        """
        return self._degree

    @degree.setter
    def degree(self, degree: Optional[int]) -> None:
        """Set the error tolerance.

        Note that this may change the underlying quantum register, if the number of state qubits
        changes.

        Args:
            degree: The new degree.
        """
        if self._degree is None or degree != self._degree:
            self._invalidate()
            self._degree = degree

            self._reset_registers(self.num_state_qubits)

    @property
    def breakpoints(self) -> List[int]:
        """The breakpoints for the piecewise approximation.

                Returns:
                    The breakpoints for the piecewise approximation.
        """
        return self._breakpoints

    @breakpoints.setter
    def breakpoints(self, breakpoints: Optional[List[int]]) -> None:
        """Set the breakpoints for the piecewise approximation.

        Note that this may change the underlying quantum register, if the number of state qubits
        changes.

        Args:
            breakpoints: The new breakpoints for the piecewise approximation.
        """
        if self._breakpoints is None or breakpoints != self._breakpoints:
            self._invalidate()
            self._breakpoints = breakpoints

            self._reset_registers(self.num_state_qubits)

    @property
    def polynomials(self) -> List[List[float]]:
        """The polynomials for the piecewise approximation.

                Returns:
                    The polynomials for the piecewise approximation.
        """
        return self._polynomials

    @polynomials.setter
    def polynomials(self, polynomials: Optional[List[List[float]]]) -> None:
        """Set the polynomials for the piecewise approximation.

        Note that this may change the underlying quantum register, if the number of state qubits
        changes.

        Args:
            polynomials: The new breakpoints for the piecewise approximation.
        """
        if self._polynomials is None or polynomials != self._polynomials:
            self._invalidate()
            self._polynomials = polynomials

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

            # Set breakpoints if they haven't been set
            if num_state_qubits is not None and self._breakpoints is None:
                self.breakpoints = [0, 2 ** num_state_qubits]

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
        if self.num_state_qubits:
            n_l = 2 ** self.num_state_qubits

            # Add n_l as a
            num_intervals = len(self._breakpoints)

            # Calculate the polynomials
            self._polynomials = []
            for i in range(0, num_intervals - 1):
                # Calculate the polynomial approximating the function on the current interval
                poly = Chebyshev.interpolate(self._f_x, self._degree,
                                             domain=[self._breakpoints[i],
                                                     self._breakpoints[i + 1]])
                # Convert polynomial to the standard basis and rescale it for the rotation gates
                poly = 2 * poly.convert(kind=np.polynomial.Polynomial).coef
                # Convert to list and append
                self._polynomials.append(poly.tolist())

            # If the last breakpoint is < n_l, add the identity polynomial
            if self._breakpoints[-1] < n_l:
                self.polynomials = self._polynomials + [[2 * np.arcsin(1)]]
                # Add n_l as the last breakpoint since that's what the algorithm expects
                self.breakpoints = self._breakpoints + [n_l]

            # If the first breakpoint is > 1, add the identity polynomial
            if self._breakpoints[0] > 0:
                self.polynomials = [[2 * np.arcsin(1)]] + self._polynomials
                self.breakpoints = [0] + self._breakpoints

            self._poly_r = PiecewisePolynomialPauliRotations(self.num_state_qubits,
                                                             self._breakpoints, self._polynomials)
            self._reset_registers(self.num_state_qubits)

        super()._build()

        qr_state = self.qubits[:self.num_state_qubits]
        qr_target = [self.qubits[self.num_state_qubits]]
        qr_ancillas = self.qubits[self.num_state_qubits + 1:]

        # Apply polynomial approximation
        self._poly_r._build()
        self.append(self._poly_r.to_instruction(), qr_state[:] + qr_target + qr_ancillas[:])
