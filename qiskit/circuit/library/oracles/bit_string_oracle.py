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

"""The bit string oracle."""

from typing import List, Optional
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import MCXGate


class BitStringOracle(QuantumCircuit):
    r"""The bit string oracle.

    Adds a -1 phase if all the objective qubits are in the specified state.
    E.g. the circuit  ``BitStringOracle(3, [0, 1], '10')`` multiplies the
    phase of all states with -1 where qubit 0 is in state :math:`|1\rangle`
    and qubit 1 is in state :math:`|0\rangle`.

    This circuit can for instance be used to implement a reflection about the
    :math:`|0\rangle^{\otimes n}` state, as

        BitStringOracle(n, list(range(n)), '0' * n)

    It is also used in Quantum Amplitude Estimation applications where sometimes
    'bad' states are specified as states where the certain qubits are :math:`|0\rangle`
    and we want to reflect about these states.

    Examples:
        >>> BitStringOracle.from_label(4, [0, 2, 3], '001').draw()
                 ┌───┐     ┌───┐
        state_0: ┤ X ├──■──┤ X ├
                 └───┘  │  └───┘
        state_1: ───────┼───────
                 ┌───┐  │  ┌───┐
        state_2: ┤ X ├──■──┤ X ├
                 ├───┤┌─┴─┐├───┤
        state_3: ┤ H ├┤ X ├┤ H ├
                 └───┘└───┘└───┘

        >>> BitStringOracle.from_label('xx1').draw()
        state_0: ─────

        state_1: ─────
                 ┌───┐
        state_2: ┤ Z ├
                 └───┘
    """

    def __init__(self,
                 num_state_qubits: int,
                 objective_qubits: List[int],
                 states: Optional[str] = None,
                 mcx: str = 'noancilla',
                 name: str = 'S_f') -> None:
        """
        Args:
            num_state_qubits: The number of qubits.
            objective_qubits: The objective qubits.
            states: The states the objective qubits should be in. Per default all are
                assumed to be 1.
            mcx: The mode for the multi-controlled X gate.
            name: The name of the circuit.

        Raises:
            ValueError: If ``objective_qubits`` contains an invalid index.
            ValueError: If ``states`` contains an invalid character.
        """
        qr_state = QuantumRegister(num_state_qubits, 'state')
        super().__init__(qr_state, name=name)

        # if we have no objective qubits we are done
        if len(objective_qubits) == 0:
            return

        num_ancillas = MCXGate.get_num_ancilla_qubits(len(objective_qubits) - 1, mcx)
        if num_ancillas > 0:
            qr_ancilla = AncillaRegister(num_ancillas, 'ancilla')
            self.add_register(qr_ancilla)
        else:
            qr_ancilla = []

        if any(qubit >= num_state_qubits for qubit in objective_qubits):
            raise ValueError('Qubit index out of range, max {}, provided {}'.format(
                num_state_qubits, objective_qubits))

        # get a list of which qubits to flip (those that are marked as '0')
        flip_qubits = []
        if states is not None:  # no qubits need to be flipped if states are all '1'
            if len(states) != len(objective_qubits):  # check for matching length
                raise ValueError('Length of states mismatches number of objective qubits, '
                                 '{} != {}'.format(len(states), len(objective_qubits)))

            for qubit, state in zip(objective_qubits, states):
                if state == '0':
                    flip_qubits += [qubit]
                elif state != '1':  # if not 1, then it was an invalid value
                    raise ValueError('Qubit states must consist of only 0 and 1: {}'.format(states))

        if len(flip_qubits) > 0:
            self.x(flip_qubits)
        if len(objective_qubits) == 1:
            self.z(objective_qubits[0])
        else:
            self.h(objective_qubits[-1])
            self.mcx(objective_qubits[:-1], objective_qubits[-1], qr_ancilla[:], mode=mcx)
            self.h(objective_qubits[-1])
        if len(flip_qubits) > 0:
            self.x(flip_qubits)

    @staticmethod
    def from_label(bitstr: str, mcx: str = 'noancilla') -> QuantumCircuit:
        """Construct the oracle from a bitstring directly.

        The bitstring can mark objective qubits with '0' or '1' for the respective state
        they should be in and use 'x' for idle qubits.

        Examples:
            >>> BitStringOracle.from_label('0x01').draw()
                     ┌───┐     ┌───┐
            state_0: ┤ X ├──■──┤ X ├
                     └───┘  │  └───┘
            state_1: ───────┼───────
                     ┌───┐  │  ┌───┐
            state_2: ┤ X ├──■──┤ X ├
                     ├───┤┌─┴─┐├───┤
            state_3: ┤ H ├┤ X ├┤ H ├
                     └───┘└───┘└───┘

        Args:
            bitstr: The bitstring.
            mcx: The method to implement the multi-controlled X gate.

        Raises:
            ValueError: If ``bitstr`` contains an unsupported character.

        Returns:
            The bitstring oracle.
        """
        num_qubits = len(bitstr)

        states, objective_qubits = [], []
        for i, character in enumerate(bitstr):
            if character == 'x':  # x indicates an idle qubit
                pass
            elif character in ['0', '1']:
                states.append(character)
                objective_qubits.append(i)
            else:
                raise ValueError('Qubit states must consist of only x, 0, 1: {}'.format(states))

        return BitStringOracle(num_qubits, objective_qubits, states, mcx=mcx)
