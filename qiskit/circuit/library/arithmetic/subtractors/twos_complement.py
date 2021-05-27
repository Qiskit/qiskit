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

"""Compute Two's Complement of a given qubit."""

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library.arithmetic import DraperQFTAdder

class TwosComplement(QuantumCircuit):
    r"""A circuit that obtains Two's Complement on one qubit register.
    Circuit to compute the two's complement of one qubit register Part from [1].
    
    As an example, a Two's Complement circuit that performs two's complement on a 3-qubit sized
    register is as follows:
    .. parsed-literal::
                   ┌───┐ ░       ░ ┌───┐┌───────────┐
        input_b_0: ┤ X ├─░───────░─┤ X ├┤3          ├─────
                   ├───┤ ░       ░ ├───┤│           │
        input_b_1: ┤ X ├─░───────░─┤ X ├┤4          ├─────
                   └───┘ ░       ░ ├───┤│           │
        input_b_2: ──────░───────░─┤ X ├┤5          ├─────
                         ░ ┌───┐ ░ └───┘│  QFTAdder │┌───┐
            cin_0: ──────░─┤ X ├─░──────┤0          ├┤ X ├
                         ░ └───┘ ░      │           │└───┘
            cin_1: ──────░───────░──────┤1          ├─────
                         ░       ░      │           │
            cin_2: ──────░───────░──────┤2          ├─────
                         ░       ░      └───────────┘
  
   
    **Reference**
    [1] Thomas G.Draper, 2000. "Addition on a Quantum Computer"
    `Journal https://arxiv.org/pdf/quant-ph/0008033.pdf`_
    """

    def __init__(self, 
                 num_state_qubits: int, 
                 adder=None,
                 name: str = 'TwosComplement'
                 ) -> None:
        """
        Args:
            num_state_qubits: The size of the register.
            adder: The adder used to add 1 to the input state. This must be a modular adder.
            name: The name of the circuit.
        Raises:
            ValueError: If ``num_state_qubits`` is lower than 1.
        """
        
        if num_state_qubits < 1:
            raise ValueError('The number of qubits must be at least 1.')
        if adder is None:
            adder = DraperQFTAdder(num_state_qubits,modular=False)
        # get the number of qubits needed
        num_qubits = adder.num_qubits
        num_helper_qubits = adder.num_ancillas
        
        # define the registers
        b_qr = QuantumRegister(num_state_qubits, name='input_b')
        carry_qr=QuantumRegister(1, name='carry')
        one_qr = AncillaRegister(num_state_qubits, name='cin')
        if num_helper_qubits != 0:
            qr_h = AncillaRegister(num_helper_qubits)

        
        # initialize the circuit
        if num_helper_qubits != 0 :
            super().__init__(b_qr, carry_qr, one_qr,qr_h, name=name)
        else:
            super().__init__(b_qr, carry_qr, one_qr, name=name)
        #if num_carry_qubits > 0:
        #    qr_c = QuantumRegister(num_carry_qubits)
        #    self.add_register(qr_c)
        # adder helper qubits if required
        #if num_helper_qubits > 0:
        #    qr_h = AncillaRegister(num_helper_qubits)  # helper/ancilla qubits
        #    self.add_register(qr_h)
        
        # Build a temporary subcircuit that obtains two's complement of b,


    #flippling circuit and adding 1
        self.barrier()
        self.x(one_qr[0])
        self.barrier()
        for j in range(num_state_qubits):
            self.x(b_qr[j])
        self.append(adder,one_qr[:]+b_qr[:]+carry_qr[:])
        self.x(one_qr[0])