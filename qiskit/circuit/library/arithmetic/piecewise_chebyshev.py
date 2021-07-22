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

        .. jupyter-execute::

            import numpy as np
            from qiskit import QuantumCircuit
            from qiskit.circuit.library.arithmetic.piecewise_chebyshev import PiecewiseChebyshev
            f_x, degree, breakpoints, num_state_qubits = lambda x: np.arcsin(1 / x), 2, [2, 4], 2
            pw_approximation = PiecewiseChebyshev(f_x, degree, breakpoints, num_state_qubits)
            pw_approximation._build()
            qc = QuantumCircuit(pw_approximation.num_qubits)
            qc.h(list(range(num_state_qubits)))
            qc.append(pw_approximation.to_instruction(), qc.qubits)
            qc.draw(output='mpl')

    References:

        [1]: Haener, T., Roetteler, M., & Svore, K. M. (2018).
             Optimizing Quantum Circuits for Arithmetic.
             `arXiv:1805.12445 <http://arxiv.org/abs/1805.12445>`_
    """

    def __init__(
        self,
        f_x: Callable[[int], float],
        degree: Optional[int] = None,
        breakpoints: Optional[List[int]] = None,
        num_state_qubits: Optional[int] = None,
        name: str = "pw_cheb",
    ) -> None:
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

        self.num_state_qubits = num_state_qubits

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        valid = True

        if self._f_x is None:
            valid = False
            if raise_on_failure:
                raise AttributeError("The function to be approximated has not been set.")

        if self._degree is None:
            valid = False
            if raise_on_failure:
                raise AttributeError("The degree of the polynomials has not been set.")

        if self._breakpoints is None:
            valid = False
            if raise_on_failure:
                raise AttributeError("The breakpoints have not been set.")

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
        breakpoints = self._breakpoints

        # it the state qubits are set ensure that the breakpoints match beginning and end
        if self.num_state_qubits is not None:
            num_states = 2 ** self.num_state_qubits

            # If the last breakpoint is < num_states, add the identity polynomial
            if breakpoints[-1] < num_states:
                breakpoints = breakpoints + [num_states]

            # If the first breakpoint is > 0, add the identity polynomial
            if breakpoints[0] > 0:
                breakpoints = [0] + breakpoints

        return breakpoints

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
            self._breakpoints = breakpoints if breakpoints is not None else [0]

            self._reset_registers(self.num_state_qubits)

    @property
    def polynomials(self) -> List[List[float]]:
        """The polynomials for the piecewise approximation.

        Returns:
            The polynomials for the piecewise approximation.
        """
        if self.num_state_qubits is None:
            return [[]]

        # note this must be the private attribute since we handle missing breakpoints at
        # 0 and 2 ^ num_qubits here (e.g. if the function we approximate is not defined at 0
        # and the user takes that into account we just add an identity)
        breakpoints = self._breakpoints
        # Need to take into account the case in which no breakpoints were provided in first place
        if breakpoints == [0]:
            breakpoints = [0, 2 ** self.num_state_qubits]

        num_intervals = len(breakpoints)

        # Calculate the polynomials
        polynomials = []
        for i in range(0, num_intervals - 1):
            # Calculate the polynomial approximating the function on the current interval
            poly = Chebyshev.interpolate(
                self.f_x, self.degree, domain=[breakpoints[i], breakpoints[i + 1]]
            )
            # Convert polynomial to the standard basis and rescale it for the rotation gates
            poly = 2 * poly.convert(kind=np.polynomial.Polynomial).coef
            # Convert to list and append
            polynomials.append(poly.tolist())

        # If the last breakpoint is < 2 ** num_qubits, add the identity polynomial
        if breakpoints[-1] < 2 ** self.num_state_qubits:
            polynomials = polynomials + [[2 * np.arcsin(1)]]

        # If the first breakpoint is > 0, add the identity polynomial
        if breakpoints[0] > 0:
            polynomials = [[2 * np.arcsin(1)]] + polynomials

        return polynomials

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
        if num_state_qubits is not None:
            qr_state = QuantumRegister(num_state_qubits, "state")
            qr_target = QuantumRegister(1, "target")
            self.qregs = [qr_state, qr_target]
            self._ancillas = []
            self._qubits = qr_state[:] + qr_target[:]
            self._qubit_set = set(self._qubits)

            num_ancillas = num_state_qubits
            if num_ancillas > 0:
                qr_ancilla = AncillaRegister(num_ancillas)
                self.add_register(qr_ancilla)

        else:
            self.qregs = []
            self._qubits = []
            self._qubit_set = set()
            self._ancillas = []

    def _build(self):
        """Build the circuit. The operation is considered successful when q_objective is
        :math:`|1>`"""
        # do not build the circuit if _data is already populated
        if self._data is not None:
            return

        self._data = []

        # check whether the configuration is valid
        self._check_configuration()

        poly_r = PiecewisePolynomialPauliRotations(
            self.num_state_qubits, self.breakpoints, self.polynomials, name=self.name
        )

        # qr_state = self.qubits[: self.num_state_qubits]
        # qr_target = [self.qubits[self.num_state_qubits]]
        # qr_ancillas = self.qubits[self.num_state_qubits + 1 :]

        # Apply polynomial approximation
        self.append(poly_r.to_gate(), self.qubits)
