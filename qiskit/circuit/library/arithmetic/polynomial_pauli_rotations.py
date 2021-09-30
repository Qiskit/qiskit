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

"""Polynomially controlled Pauli-rotations."""

import warnings
from typing import List, Optional, Dict, Sequence

from itertools import product

from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError

from .functional_pauli_rotations import FunctionalPauliRotations


def _binomial_coefficients(n):
    """ "Return a dictionary of binomial coefficients

    Based-on/forked from sympy's binomial_coefficients() function [#]

    .. [#] https://github.com/sympy/sympy/blob/sympy-1.5.1/sympy/ntheory/multinomial.py
    """

    data = {(0, n): 1, (n, 0): 1}
    temp = 1
    for k in range(1, n // 2 + 1):
        temp = (temp * (n - k + 1)) // k
        data[k, n - k] = data[n - k, k] = temp
    return data


def _large_coefficients_iter(m, n):
    """ "Return an iterator of multinomial coefficients

    Based-on/forked from sympy's multinomial_coefficients_iterator() function [#]

    .. [#] https://github.com/sympy/sympy/blob/sympy-1.5.1/sympy/ntheory/multinomial.py
    """
    if m < 2 * n or n == 1:
        coefficients = _multinomial_coefficients(m, n)
        for key, value in coefficients.items():
            yield (key, value)
    else:
        coefficients = _multinomial_coefficients(n, n)
        coefficients_dict = {}
        for key, value in coefficients.items():
            coefficients_dict[tuple(filter(None, key))] = value
        coefficients = coefficients_dict

        temp = [n] + [0] * (m - 1)
        temp_a = tuple(temp)
        b = tuple(filter(None, temp_a))
        yield (temp_a, coefficients[b])
        if n:
            j = 0  # j will be the leftmost nonzero position
        else:
            j = m
        # enumerate tuples in co-lex order
        while j < m - 1:
            # compute next tuple
            temp_j = temp[j]
            if j:
                temp[j] = 0
                temp[0] = temp_j
            if temp_j > 1:
                temp[j + 1] += 1
                j = 0
            else:
                j += 1
                temp[j] += 1

            temp[0] -= 1
            temp_a = tuple(temp)
            b = tuple(filter(None, temp_a))
            yield (temp_a, coefficients[b])


def _multinomial_coefficients(m, n):
    """ "Return an iterator of multinomial coefficients

    Based-on/forked from sympy's multinomial_coefficients() function [#]

    .. [#] https://github.com/sympy/sympy/blob/sympy-1.5.1/sympy/ntheory/multinomial.py
    """
    if not m:
        if n:
            return {}
        return {(): 1}
    if m == 2:
        return _binomial_coefficients(n)
    if m >= 2 * n and n > 1:
        return dict(_large_coefficients_iter(m, n))
    if n:
        j = 0
    else:
        j = m
    temp = [n] + [0] * (m - 1)
    res = {tuple(temp): 1}
    while j < m - 1:
        temp_j = temp[j]
        if j:
            temp[j] = 0
            temp[0] = temp_j
        if temp_j > 1:
            temp[j + 1] += 1
            j = 0
            start = 1
            v = 0
        else:
            j += 1
            start = j + 1
            v = res[tuple(temp)]
            temp[j] += 1
        for k in range(start, m):
            if temp[k]:
                temp[k] -= 1
                v += res[tuple(temp)]
                temp[k] += 1
        temp[0] -= 1
        res[tuple(temp)] = (v * temp_j) // (n - temp[0])
    return res


class PolynomialPauliRotations(FunctionalPauliRotations):
    r"""A circuit implementing polynomial Pauli rotations.

    For a polynomial :math`p(x)`, a basis state :math:`|i\rangle` and a target qubit
    :math:`|0\rangle` this operator acts as:

    .. math::

        |i\rangle |0\rangle \mapsto \cos(p(i)) |i\rangle |0\rangle + \sin(p(i)) |i\rangle |1\rangle

    Let n be the number of qubits representing the state, d the degree of p(x) and q_i the qubits,
    where q_0 is the least significant qubit. Then for

    .. math::

        x = \sum_{i=0}^{n-1} 2^i q_i,

    we can write

    .. math::

        p(x) = \sum_{j=0}^{j=d} c_j x_j

    where :math:`c` are the input coefficients, ``coeffs``.
    """

    def __init__(
        self,
        num_state_qubits: Optional[int] = None,
        coeffs: Optional[List[float]] = None,
        basis: str = "Y",
        reverse: bool = False,
        name: str = "poly",
    ) -> None:
        """Prepare an approximation to a state with amplitudes specified by a polynomial.

        Args:
            num_state_qubits: The number of qubits representing the state.
            coeffs: The coefficients of the polynomial. ``coeffs[i]`` is the coefficient of the
                i-th power of x. Defaults to linear: [0, 1].
            basis: The type of Pauli rotation ('X', 'Y', 'Z').
            reverse: If True, apply the polynomial with the reversed list of qubits
                (i.e. q_n as q_0, q_n-1 as q_1, etc).
            name: The name of the circuit.
        """
        # set default internal parameters
        self._coeffs = coeffs or [0, 1]
        self._reverse = reverse
        if self._reverse is True:
            warnings.warn(
                "The reverse flag has been deprecated. "
                "Use circuit.reverse_bits() to reverse order of qubits.",
                DeprecationWarning,
            )

        # initialize super (after setting coeffs)
        super().__init__(num_state_qubits=num_state_qubits, basis=basis, name=name)

    @property
    def coeffs(self) -> List[float]:
        """The multiplicative factor in the rotation angle of the controlled rotations.

        The rotation angles are ``slope * 2^0``, ``slope * 2^1``, ... , ``slope * 2^(n-1)`` where
        ``n`` is the number of state qubits.

        Returns:
            The rotation angle common in all controlled rotations.
        """
        return self._coeffs

    @coeffs.setter
    def coeffs(self, coeffs: List[float]) -> None:
        """Set the multiplicative factor of the rotation angles.

        Args:
            The slope of the rotation angles.
        """
        self._invalidate()
        self._coeffs = coeffs

    @property
    def degree(self) -> int:
        """Return the degree of the polynomial, equals to the number of coefficients minus 1.

        Returns:
            The degree of the polynomial. If the coefficients have not been set, return 0.
        """
        if self.coeffs:
            return len(self.coeffs) - 1
        return 0

    @property
    def reverse(self) -> bool:
        """Whether to apply the rotations on the reversed list of qubits.

        Returns:
            True, if the rotations are applied on the reversed list, False otherwise.
        """
        return self._reverse

    @property
    def num_ancilla_qubits(self):
        """Deprecated. Use num_ancillas instead."""
        warnings.warn(
            "The PolynomialPauliRotations.num_ancilla_qubits property is deprecated "
            "as of 0.16.0. It will be removed no earlier than 3 months after the release "
            "date. You should use the num_ancillas property instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.num_ancillas

    def _reset_registers(self, num_state_qubits):
        if num_state_qubits is not None:
            # set new register of appropriate size
            qr_state = QuantumRegister(num_state_qubits, name="state")
            qr_target = QuantumRegister(1, name="target")

            self.qregs = [qr_state, qr_target]
        else:
            self.qregs = []

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
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

        return valid

    def _get_rotation_coefficients(self) -> Dict[Sequence[int], float]:
        """Compute the coefficient of each monomial.

        Returns:
            A dictionary with pairs ``{control_state: rotation angle}`` where ``control_state``
            is a tuple of ``0`` or ``1`` bits.
        """
        # determine the control states
        all_combinations = list(product([0, 1], repeat=self.num_state_qubits))
        valid_combinations = []
        for combination in all_combinations:
            if 0 < sum(combination) <= self.degree:
                valid_combinations += [combination]

        rotation_coeffs = {control_state: 0 for control_state in valid_combinations}

        # compute the coefficients for the control states
        for i, coeff in enumerate(self.coeffs[1:]):
            i += 1  # since we skip the first element we need to increase i by one

            # iterate over the multinomial coefficients
            for comb, num_combs in _multinomial_coefficients(self.num_state_qubits, i).items():
                control_state = ()
                power = 1
                for j, qubit in enumerate(comb):
                    if qubit > 0:  # means we control on qubit i
                        control_state += (1,)
                        power *= 2 ** (j * qubit)
                    else:
                        control_state += (0,)

                # Add angle
                rotation_coeffs[control_state] += coeff * num_combs * power

        return rotation_coeffs

    def _build(self):
        # do not build the circuit if _data is already populated
        if self._data is not None:
            return

        self._data = []

        # check whether the configuration is valid
        self._check_configuration()

        circuit = QuantumCircuit(*self.qregs, name=self.name)
        qr_state = circuit.qubits[: self.num_state_qubits]
        qr_target = circuit.qubits[self.num_state_qubits]

        rotation_coeffs = self._get_rotation_coefficients()

        if self.basis == "x":
            circuit.rx(self.coeffs[0], qr_target)
        elif self.basis == "y":
            circuit.ry(self.coeffs[0], qr_target)
        else:
            circuit.rz(self.coeffs[0], qr_target)

        for c in rotation_coeffs:
            qr_control = []
            if self.reverse:
                for i, _ in enumerate(c):
                    if c[i] > 0:
                        qr_control.append(qr_state[qr_state.size - i - 1])
            else:
                for i, _ in enumerate(c):
                    if c[i] > 0:
                        qr_control.append(qr_state[i])

            # apply controlled rotations
            if len(qr_control) > 1:
                if self.basis == "x":
                    circuit.mcrx(rotation_coeffs[c], qr_control, qr_target)
                elif self.basis == "y":
                    circuit.mcry(rotation_coeffs[c], qr_control, qr_target)
                else:
                    circuit.mcrz(rotation_coeffs[c], qr_control, qr_target)

            elif len(qr_control) == 1:
                if self.basis == "x":
                    circuit.crx(rotation_coeffs[c], qr_control[0], qr_target)
                elif self.basis == "y":
                    circuit.cry(rotation_coeffs[c], qr_control[0], qr_target)
                else:
                    circuit.crz(rotation_coeffs[c], qr_control[0], qr_target)

        self.append(circuit.to_gate(), self.qubits)
