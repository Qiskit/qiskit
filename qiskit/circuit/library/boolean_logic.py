# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=no-member

"""Implementations of boolean logic quantum circuits."""

from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library.standard_gates import MCXGate


class AND(QuantumCircuit):
    r"""A circuit implementing the logical AND operation on a number of qubits.

    For the AND operation the state :math:`|1\rangle` is interpreted as ``True``. The result
    qubit is flipped, if the state of all variable qubits is ``True``. In this format, the AND
    operation equals a multi-controlled X gate, which is controlled on all variable qubits.
    Using a list of flags however, qubits can be skipped or negated. Practically, the flags
    allow to skip controls or to apply pre- and post-X gates to the negated qubits.

    The AND gate without special flags equals the multi-controlled-X gate:

    .. jupyter-execute::
        :hide-code:

        from qiskit.circuit.library import AND
        import qiskit.tools.jupyter
        circuit = AND(5)
        %circuit_library_info circuit

    Using flags we can negate qubits or skip them. For instance, if we have 5 qubits and want to
    return ``True`` if the first qubit is ``False`` and the last two are ``True`` we use the flags
    ``[-1, 0, 0, 1, 1]``.

    .. jupyter-execute::
        :hide-code:

        from qiskit.circuit.library import AND
        import qiskit.tools.jupyter
        circuit = AND(5, flags=[-1, 0, 0, 1, 1])
        %circuit_library_info circuit

    """

    def __init__(self, num_variable_qubits: int, flags: Optional[List[int]] = None,
                 mcx_mode: str = 'noancilla') -> None:
        """Create a new logical AND circuit.

        Args:
            num_variable_qubits: The qubits of which the OR is computed. The result will be written
                into an additional result qubit.
            flags: A list of +1/0/-1 marking negations or omisiions of qubits.
            mcx_mode: The mode to be used to implement the multi-controlled X gate.
        """
        # store num_variables_qubits and flags
        self.num_variable_qubits = num_variable_qubits
        self.flags = flags

        # add registers
        qr_variable = QuantumRegister(num_variable_qubits, name='variable')
        qr_result = QuantumRegister(1, name='result')

        super().__init__(qr_variable, qr_result, name='and')

        # determine the control qubits: all that have a nonzero flag
        flags = flags or [1] * num_variable_qubits
        control_qubits = [q for q, flag in zip(qr_variable, flags) if flag != 0]

        # determine the qubits that need to be flipped (if a flag is < 0)
        flip_qubits = [q for q, flag in zip(qr_variable, flags) if flag < 0]

        # determine the number of ancillas
        self.num_ancilla_qubits = MCXGate.get_num_ancilla_qubits(len(control_qubits), mode=mcx_mode)
        if self.num_ancilla_qubits > 0:
            qr_ancilla = QuantumRegister(self.num_ancilla_qubits, 'ancilla')
            self.add_register(qr_ancilla)
        else:
            qr_ancilla = []

        if len(flip_qubits) > 0:
            self.x(flip_qubits)
        self.mcx(control_qubits, qr_result[:], qr_ancilla[:], mode=mcx_mode)
        if len(flip_qubits) > 0:
            self.x(flip_qubits)


class OR(QuantumCircuit):
    r"""A circuit implementing the logical OR operation on a number of qubits.

    For the OR operation the state :math:`|1\rangle` is interpreted as ``True``. The result
    qubit is flipped, if the state of any variable qubit is ``True``. The OR is implemented using
    a multi-open-controlled X gate (i.e. flips if the state is :math:`|0\rangle`) and
    applying an X gate on the result qubit.
    Using a list of flags, qubits can be skipped or negated.

    The OR gate without special flags:

    .. jupyter-execute::
        :hide-code:

        from qiskit.circuit.library import OR
        import qiskit.tools.jupyter
        circuit = OR(5)
        %circuit_library_info circuit

    Using flags we can negate qubits or skip them. For instance, if we have 5 qubits and want to
    return ``True`` if the first qubit is ``False`` or one of the last two are ``True`` we use the
    flags ``[-1, 0, 0, 1, 1]``.

    .. jupyter-execute::
        :hide-code:

        from qiskit.circuit.library import OR
        import qiskit.tools.jupyter
        circuit = OR(5, flags=[-1, 0, 0, 1, 1])
        %circuit_library_info circuit

    """

    def __init__(self, num_variable_qubits: int, flags: Optional[List[int]] = None,
                 mcx_mode: str = 'noancilla') -> None:
        """Create a new logical OR circuit.

        Args:
            num_variable_qubits: The qubits of which the OR is computed. The result will be written
                into an additional result qubit.
            flags: A list of +1/0/-1 marking negations or omisiions of qubits.
            mcx_mode: The mode to be used to implement the multi-controlled X gate.
        """
        # store num_variables_qubits and flags
        self.num_variable_qubits = num_variable_qubits
        self.flags = flags

        # add registers
        qr_variable = QuantumRegister(num_variable_qubits, name='variable')
        qr_result = QuantumRegister(1, name='result')

        super().__init__(qr_variable, qr_result, name='or')

        # determine the control qubits: all that have a nonzero flag
        flags = flags or [1] * num_variable_qubits
        control_qubits = [q for q, flag in zip(qr_variable, flags) if flag != 0]

        # determine the qubits that need to be flipped (if a flag is > 0)
        flip_qubits = [q for q, flag in zip(qr_variable, flags) if flag > 0]

        # determine the number of ancillas
        self.num_ancilla_qubits = MCXGate.get_num_ancilla_qubits(len(control_qubits), mode=mcx_mode)
        if self.num_ancilla_qubits > 0:
            qr_ancilla = QuantumRegister(self.num_ancilla_qubits, 'ancilla')
            self.add_register(qr_ancilla)
        else:
            qr_ancilla = []

        self.x(qr_result)
        if len(flip_qubits) > 0:
            self.x(flip_qubits)
        self.mcx(control_qubits, qr_result[:], qr_ancilla[:], mode=mcx_mode)
        if len(flip_qubits) > 0:
            self.x(flip_qubits)


class XOR(QuantumCircuit):
    """An n_qubit circuit for bitwise xor-ing the input with some integer ``amount``.

    The ``amount`` is xor-ed in bitstring form with the input.

    This circuit can also represent addition by ``amount`` over the finite field GF(2).
    """

    def __init__(self,
                 num_qubits: int,
                 amount: Optional[int] = None,
                 seed: Optional[int] = None,
                 ) -> None:
        """Return a circuit implementing bitwise xor.

        Args:
            num_qubits: the width of circuit.
            amount: the xor amount in decimal form.
            seed: random seed in case a random xor is requested.

        Raises:
            CircuitError: if the xor bitstring exceeds available qubits.

        Reference Circuit:
            .. jupyter-execute::
                :hide-code:

                from qiskit.circuit.library import XOR
                import qiskit.tools.jupyter
                circuit = XOR(5, seed=42)
                %circuit_library_info circuit
        """
        super().__init__(num_qubits, name="xor")

        if amount is not None:
            if len(bin(amount)[2:]) > num_qubits:
                raise CircuitError("Bits in 'amount' exceed circuit width")
        else:
            rng = np.random.default_rng(seed)
            amount = rng.integers(0, 2**num_qubits)

        for i in range(num_qubits):
            bit = amount & 1
            amount = amount >> 1
            if bit == 1:
                self.x(i)


class InnerProduct(QuantumCircuit):
    """An n_qubit circuit that computes the inner product of two registers."""

    def __init__(self, num_qubits: int) -> None:
        """Return a circuit to compute the inner product of 2 n-qubit registers.

        This implementation uses CZ gates.

        Args:
            num_qubits: width of top and bottom registers (half total circuit width)

        Reference Circuit:
            .. jupyter-execute::
                :hide-code:

                from qiskit.circuit.library import InnerProduct
                import qiskit.tools.jupyter
                circuit = InnerProduct(5)
                %circuit_library_info circuit
        """
        qr_a = QuantumRegister(num_qubits)
        qr_b = QuantumRegister(num_qubits)
        super().__init__(qr_a, qr_b, name="inner_product")

        for i in range(num_qubits):
            self.cz(qr_a[i], qr_b[i])
