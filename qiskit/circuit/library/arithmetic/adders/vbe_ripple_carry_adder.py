# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compute the sum of two qubit registers using Classical Addition."""

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister

from .adder import Adder


class VBERippleCarryAdder(Adder):
    r"""A circuit that uses Classical Addition/Plain Adder to perform in-place addition on two qubit registers.

    As an example, a classical adder circuit that performs addition on two 3-qubit sized
    registers is as follows:

    .. parsed-literal::
      
                ┌────────┐                                                      ┌───────────┐┌──────┐
           a_0: ┤1       ├──────────────────────────────────────────────────────┤1          ├┤1     ├
                │        │┌────────┐                       ┌───────────┐┌──────┐│           ││      │
           a_1: ┤        ├┤1       ├───────────────────────┤1          ├┤1     ├┤           ├┤      ├
                │        ││        │┌────────┐     ┌──────┐│           ││      ││           ││      │
           a_2: ┤        ├┤        ├┤1       ├──■──┤1     ├┤           ├┤      ├┤           ├┤      ├
                │        ││        ││        │  │  │      ││           ││      ││           ││      │
           b_0: ┤2       ├┤        ├┤        ├──┼──┤      ├┤           ├┤      ├┤2          ├┤2     ├
                │        ││        ││        │  │  │      ││           ││      ││           ││  Sum │
           b_1: ┤  Carry ├┤2       ├┤        ├──┼──┤      ├┤2          ├┤2     ├┤  Carry_dg ├┤      ├
                │        ││        ││        │┌─┴─┐│      ││           ││  Sum ││           ││      │
           b_2: ┤        ├┤  Carry ├┤2       ├┤ X ├┤2     ├┤  Carry_dg ├┤      ├┤           ├┤      ├
                │        ││        ││  Carry │└───┘│  Sum ││           ││      ││           ││      │
        cout_0: ┤        ├┤        ├┤3       ├─────┤      ├┤           ├┤      ├┤           ├┤      ├
                │        ││        ││        │     │      ││           ││      ││           ││      │
         cin_0: ┤0       ├┤        ├┤        ├─────┤      ├┤           ├┤      ├┤0          ├┤0     ├
                │        ││        ││        │     │      ││           ││      ││           │└──────┘
         cin_1: ┤3       ├┤0       ├┤        ├─────┤      ├┤0          ├┤0     ├┤3          ├────────
                └────────┘│        ││        │     │      ││           │└──────┘└───────────┘
         cin_2: ──────────┤3       ├┤0       ├─────┤0     ├┤3          ├─────────────────────────────
                          └────────┘└────────┘     └──────┘└───────────┘
        

    Here *Carry* and *Sum* gates correspond to the gates introduced in [1].
    *Carry_dg* correspond to the inverse of the *Carry* gate. Note that
    in this implementation the input register qubits are ordered as all qubits from
    the first input register, followed by all qubits from the second input register.
    This is different ordering as compared to Figure 2 in [1], which leads to a different
    drawing of the circuit.

    **References:**

    [1] Vedral et al., Quantum Networks for Elementary Arithmetic Operations, 1995.
    `arXiv:quant-ph/9511018 <https://arxiv.org/pdf/quant-ph/9511018.pdf>`_

    """

    def __init__(self,
                 num_state_qubits: int,
                 modular: bool = False,
                 name: str = 'VBERippleCarryAdder'
                 ) -> None:
        """
        Args:
            num_state_qubits: The size of the register.
            modular: If True, perform addition modulo ``2 ** num_state_qubits``. This needs
                one less qubit (namely no ``cout`` qubit).
            name: The name of the circuit.

        Raises:
            ValueError: If ``num_state_qubits`` is lower than 1.
        """
        if num_state_qubits < 1:
            raise ValueError('The number of qubits must be at least 1.')

        super().__init__(num_state_qubits, name=name)

        # define the input registers
        qr_a = QuantumRegister(num_state_qubits, name='a')
        qr_b = QuantumRegister(num_state_qubits, name='b')
        self.add_register(qr_a, qr_b)

        if not modular:
            qr_cout = QuantumRegister(1, name='cout')
            self.add_register(qr_cout)

        qr_cin = AncillaRegister(num_state_qubits, name='cin')
        self.add_register(qr_cin)

        # corresponds to Carry gate in [1]
        qc_carry = QuantumCircuit(4, name='Carry')
        qc_carry.ccx(1, 2, 3)
        qc_carry.cx(1, 2)
        qc_carry.ccx(0, 2, 3)
        qc_instruction_carry = qc_carry.to_gate()

        # corresponds to Sum gate in [1]
        qc_sum = QuantumCircuit(3, name='Sum')
        qc_sum.cx(1, 2)
        qc_sum.cx(0, 2)
        qc_instruction_sum = qc_sum.to_gate()

        # Build a temporary subcircuit that adds a to b,
        # storing the result in b

        for j in range(num_state_qubits - 1):
            self.append(qc_instruction_carry, [qr_cin[j], qr_a[j], qr_b[j], qr_cin[j+1]])

        if not modular:
            self.append(qc_instruction_carry, [qr_cin[num_state_qubits - 1],
                                               qr_a[num_state_qubits - 1],
                                               qr_b[num_state_qubits - 1],
                                               qr_cout])
            self.cx(qr_a[num_state_qubits - 1], qr_b[num_state_qubits - 1])

        self.append(qc_instruction_sum, [qr_cin[num_state_qubits - 1],
                                         qr_a[num_state_qubits - 1], qr_b[num_state_qubits - 1]])

        for j in reversed(range(num_state_qubits - 1)):
            self.append(qc_instruction_carry.inverse(), [qr_cin[j], qr_a[j], qr_b[j], qr_cin[j+1]])
            self.append(qc_instruction_sum, [qr_cin[j], qr_a[j], qr_b[j]])
