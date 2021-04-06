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
# that they have been altered from the originals

"""Compute the sum of two qubit registers using Classical Addition."""

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister


class ClassicalAdder(QuantumCircuit):
    r"""A circuit that uses Classical Addition to perform in-place addition on two qubit registers.

     Circuit to compute the sum of two qubit registers using the Classical Addition Part from [1].
     Given two equally sized input registers that store quantum states
    :math:`|a\rangle` and :math:`|b\rangle`, performs addition of numbers that
    can be represented by the states, storing the resulting state in-place in the second register:

    .. math::

        |a\rangle |b\rangle \mapsto |a\rangle |a+b\rangle

    Here :math:`|a\rangle` (and correspondingly :math:`|b\rangle`) stands for the direct product
    :math:`|a_n\rangle \otimes |a_{n-1}\rangle \ldots |a_{1}\rangle \otimes |a_{0}\rangle`
    which denotes a quantum register prepared with the value :math:`a = 2^{0}a_{0} + 2^{1}a_{1} +
    \ldots 2^{n}a_{n}`[2].
    As an example, a classical adder circuit that performs addition on two 3-qubit sized
    registers is as follows:

    .. parsed-literal::
        
                   ┌────────┐                                                   ┌────────┐┌──────┐
        input_a_0: ┤1       ├───────────────────────────────────────────────────┤1       ├┤1     ├
                   │        │┌────────┐                       ┌────────┐┌──────┐│        ││      │
        input_a_1: ┤        ├┤1       ├───────────────────────┤1       ├┤1     ├┤        ├┤      ├
                   │        ││        │┌────────┐     ┌──────┐│        ││      ││        ││      │
        input_a_2: ┤        ├┤        ├┤1       ├──■──┤1     ├┤        ├┤      ├┤        ├┤      ├
                   │        ││        ││        │  │  │      ││        ││      ││        ││      │
        input_b_0: ┤2       ├┤        ├┤        ├──┼──┤      ├┤        ├┤      ├┤2       ├┤2     ├
                   │        ││        ││        │  │  │      ││        ││      ││        ││  Sum │
        input_b_1: ┤  Carry ├┤2       ├┤        ├──┼──┤      ├┤2       ├┤2     ├┤  Carry ├┤      ├
                   │        ││        ││        │┌─┴─┐│      ││        ││  Sum ││        ││      │
        input_b_2: ┤        ├┤  Carry ├┤2       ├┤ X ├┤2     ├┤  Carry ├┤      ├┤        ├┤      ├
                   │        ││        ││  Carry │└───┘│  Sum ││        ││      ││        ││      │
           cout_0: ┤        ├┤        ├┤3       ├─────┤      ├┤        ├┤      ├┤        ├┤      ├
                   │        ││        ││        │     │      ││        ││      ││        ││      │
            cin_0: ┤0       ├┤        ├┤        ├─────┤      ├┤        ├┤      ├┤0       ├┤0     ├
                   │        ││        ││        │     │      ││        ││      ││        │└──────┘
            cin_1: ┤3       ├┤0       ├┤        ├─────┤      ├┤0       ├┤0     ├┤3       ├────────
                   └────────┘│        ││        │     │      ││        │└──────┘└────────┘
            cin_2: ──────────┤3       ├┤0       ├─────┤0     ├┤3       ├──────────────────────────
                             └────────┘└────────┘     └──────┘└────────┘  
      
    
    Here *Carry* and *Sum* gates correspond to the gates introduced in [1]. Note that
    in this implementation the input register qubits are ordered as all qubits from
    the first input register, followed by all qubits from the second input register.
    This is different ordering as compared to Figure 2 in [1], which leads to a different
    drawing of the circuit.

    **References**

    [1] Thomas G.Draper, 2000. "Addition on a Quantum Computer"
    `Journal https://arxiv.org/pdf/quant-ph/0008033.pdf`_

    [2] Vedral et al., Quantum Networks for Elementary Arithmetic Operations, 1995.
    `arXiv:quant-ph/9511018 <https://arxiv.org/pdf/quant-ph/9511018.pdf>`_
    """

    def __init__(self, 
                 num_state_qubits: int, 
                 name: str = 'ClassicalAdder'
                 ) -> None:
        """
        Args:
            num_state_qubits: The size of the register.
            name: The name of the circuit.
        Raises:
            ValueError: If ``num_state_qubits`` is lower than 1.
        """
        if num_state_qubits < 1:
            raise ValueError('The number of qubits must be at least 1.')
        # define the registers
        qr_a = QuantumRegister(num_state_qubits, name='input_a')
        qr_b = QuantumRegister(num_state_qubits, name='input_b')
        qr_cin = AncillaRegister(num_state_qubits, name='cin')
        qr_cout = QuantumRegister(1, name='cout')

        # initialize the circuit
        super().__init__(qr_a, qr_b, qr_cout, qr_cin, name=name)

        #corresponds to Carry gate in [1]
        qc_carry = QuantumCircuit(4, name='Carry')
        qc_carry.ccx(1, 2, 3)
        qc_carry.cx(1, 2)
        qc_carry.ccx(0, 2, 3)
        qc_instruction_carry = qc_carry.to_instruction()

        #corresponds to Sum gate in [1]
        qc_sum = QuantumCircuit(3, name='Sum')
        qc_sum.cx(1, 2)
        qc_sum.cx(0, 2)
        qc_instruction_sum = qc_sum.to_instruction()

        # Build a temporary subcircuit that adds a to b,
        # storing the result in b

        for j in range(num_state_qubits - 1):
            self.append(qc_instruction_carry, [qr_cin[j], qr_a[j], qr_b[j], qr_cin[j+1]])

        self.append(qc_instruction_carry, [qr_cin[num_state_qubits - 1],
                                           qr_a[num_state_qubits - 1], qr_b[num_state_qubits - 1],
                                           qr_cout])
        self.cx(qr_a[num_state_qubits - 1], qr_b[num_state_qubits - 1])
        self.append(qc_instruction_sum, [qr_cin[num_state_qubits - 1],
                                         qr_a[num_state_qubits - 1], qr_b[num_state_qubits - 1]])

        for j in reversed(range(num_state_qubits - 1)):
            self.append(qc_instruction_carry, [qr_cin[j], qr_a[j], qr_b[j], qr_cin[j+1]])
            self.append(qc_instruction_sum, [qr_cin[j], qr_a[j], qr_b[j]])

