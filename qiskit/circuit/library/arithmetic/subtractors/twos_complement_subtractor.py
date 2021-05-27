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

"""Compute the difference of two qubit registers using Two's Complement Subtraction."""
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
#from qiskit.circuit.library.arithmetic.adders.adder import Adder
from qiskit.circuit.library.arithmetic.adders import CDKMRippleCarryAdder,DraperQFTAdder,VBERippleCarryAdder
from qiskit.circuit.library.arithmetic.subtractor import TwosComplement
class Subtractor(QuantumCircuit):
    r"""A circuit that uses Two's Complement Subtraction to perform in-place subtraction on two qubit registers.
     Circuit to compute the difference of two qubit registers.
     Given two equally sized input registers that store quantum states
    :math:`|a\rangle` and :math:`|b\rangle`, performs subtraction of numbers that
    can be represented by the states, storing the resulting state in-place in the second register:
    .. math::
        |a\rangle |b\rangle \mapsto |a\rangle |a-b\rangle
    Here :math:`|a\rangle` (and correspondingly :math:`|b\rangle`) stands for the direct product
    :math:`|a_n\rangle \otimes |a_{n-1}\rangle \ldots |a_{1}\rangle \otimes |a_{0}\rangle`
    which denotes a quantum register prepared with the value :math:`a = 2^{0}a_{0} + 2^{1}a_{1} +
    \ldots 2^{n}a_{n}`[1].
    As an example, a subtractor circuit that performs two's complement on :math:`|b\rangle`and 
    performs addition on two 3-qubit sized registers is as follows:
    .. parsed-literal::
         a_0: ──────────────────────────────────────■──────■────────■──────────────────────────────────────
                                                    │      │        │
         a_1: ──────────────────────────────────────┼──────┼────────┼────────■──────■──────────────────────
               ░       ░ ┌───┐┌───────────┐┌──────┐ │P(π)  │        │        │      │              ┌──────┐
         b_0: ─░───────░─┤ X ├┤2          ├┤0     ├─■──────┼────────┼────────┼──────┼──────────────┤0     ├
               ░       ░ ├───┤│           ││      │        │P(π/2)  │        │P(π)  │              │      │
         b_1: ─░───────░─┤ X ├┤3          ├┤1 qft ├────────■────────┼────────■──────┼──────────────┤1 qft ├
               ░       ░ └───┘│           ││      │                 │P(π/4)         │P(π/2)        │      │
   carry_b_0: ─░───────░──────┤4          ├┤2     ├─────────────────■───────────────■────────■─────┤2     ├
               ░       ░      │  QFTAdder │└──────┘                                          │P(π) └──────┘
   carry_a_0: ────────────────┤           ├──────────────────────────────────────────────────■─────────────
               ░ ┌───┐ ░      │           │ ┌───┐
        a0_0: ─░─┤ X ├─░──────┤0          ├─┤ X ├──────────────────────────────────────────────────────────
               ░ └───┘ ░      │           │ └───┘
        a0_1: ─░───────░──────┤1          ├────────────────────────────────────────────────────────────────
               ░       ░      └───────────┘
           
    **References**
    
    [1] Vedral et al., Quantum Networks for Elementary Arithmetic Operations, 1995.
    `arXiv:quant-ph/9511018 <https://arxiv.org/pdf/quant-ph/9511018.pdf>`_
    """

    #def __init__(self, num_state_qubits: int, adder: Optional[adder] = None):
    def __init__(self, num_state_qubits: int, adder=None, name: str = 'Subtractor'):
        if adder is None:
            adder = DraperQFTAdder(num_state_qubits+1,modular=True)
            ##adder = RippleCarryAdder(num_state_qubits+1,modular=True)
        twos_complement = TwosComplement(num_state_qubits)
        # get the number of qubits needed
        num_qubits = adder.num_qubits
        num_helper_qubits = max(adder.num_ancillas,twos_complement.num_ancillas)
        
        # construct the registers
        qr_a = QuantumRegister(num_state_qubits, 'a')  # input a
        qr_b = QuantumRegister(num_state_qubits, 'b')  # input b
        carry_b=QuantumRegister(1,'carry_b')
        carry_a=AncillaRegister(1,'carry_a')
        qr_h=AncillaRegister(num_helper_qubits)
        
        super().__init__(qr_a,qr_b,carry_b,carry_a,qr_h, name=name)
        
        
        self.compose(twos_complement, qubits=qr_b[:]+carry_b[:]+qr_h[:twos_complement.num_ancillas],inplace=True)
        
        # adder
        self.compose(adder,qubits=qr_a[:]+carry_a[:]+qr_b[:]+carry_b[:]+qr_h[:adder.num_ancillas],inplace=True)