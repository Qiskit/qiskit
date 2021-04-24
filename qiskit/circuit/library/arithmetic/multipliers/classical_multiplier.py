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

"""Compute the product of two qubit registers using classical multiplication approach."""

from typing import Optional
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library.arithmetic.adders.adder import Adder


class ClassicalMultiplier(QuantumCircuit):
    r"""A multiplication circuit to store product of two input registers out-of-place.

    Circuit to compute the product of two qubit registers using the approach from [1].
    Given two equally sized input registers that store quantum states
    :math:`|a\rangle` and :math:`|b\rangle`, performs multiplication of numbers that
    can be represented by the states, storing the resulting state out-of-place
    in a third output register:

    .. math::

        |a\rangle |b\rangle |0\rangle \mapsto |a\rangle |b\rangle |a \cdot b\rangle

    Here :math:`|a\rangle` (and correspondingly :math:`|b\rangle`) stands for the direct product
    :math:`|a_n\rangle \otimes |a_{n-1}\rangle \ldots |a_{1}\rangle \otimes |a_{0}\rangle`
    which denotes a quantum register prepared with the value :math:`a = 2^{0}a_{0} + 2^{1}a_{1} +
    \ldots 2^{n}a_{n}` [2].

    As an example, a multiplier circuit that performs multiplication on two 2-qubit sized
    registers is as follows:

    .. parsed-literal::

          a_0: ──────────■───────────────────────────────
                         │
          a_1: ──────────┼────────────────────■──────────
               ┌─────────┴─────────┐┌─────────┴─────────┐
          b_0: ┤0                  ├┤0                  ├
               │                   ││                   │
          b_1: ┤1                  ├┤1                  ├
               │                   ││                   │
        out_0: ┤2                  ├┤                   ├
               │                   ││                   │
        out_1: ┤3 RippleCarryAdder ├┤2 RippleCarryAdder ├
               │                   ││                   │
        out_2: ┤4                  ├┤3                  ├
               │                   ││                   │
        out_3: ┤                   ├┤4                  ├
               │                   ││                   │
        aux_0: ┤5                  ├┤5                  ├
               └───────────────────┘└───────────────────┘

    Multiplication in this circuit is implemented in a classical approach by performing
    a series of shifted additions using one of the input registers while the qubits
    from the other input register act as control qubits for the adders.

    **References:**

    [1] Häner et al., Optimizing Quantum Circuits for Arithmetic, 2018.
    `arXiv:1805.12445 <https://arxiv.org/pdf/1805.12445.pdf>`_

    [2] Vedral et al., Quantum Networks for Elementary Arithmetic Operations, 1995.
    `arXiv:quant-ph/9511018 <https://arxiv.org/pdf/quant-ph/9511018.pdf>`_

    """

    def __init__(self,
                 num_state_qubits: int,
                 adder: Optional[Adder] = None,
                 name: str = 'ClassicalMultiplier') -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits in either input register for
                state :math:`|a\rangle` or :math:`|b\rangle`. The two input
                registers must have the same number of qubits.
            adder: adder circuit to be used for performing multiplication.
            name: The name of the circuit object.
        Raises:
            ValueError: If ``num_state_qubits`` is lower than 1.
        """
        if num_state_qubits < 1:
            raise ValueError('The number of qubits must be at least 1.')

        qr_a = QuantumRegister(num_state_qubits, name='a')
        qr_b = QuantumRegister(num_state_qubits, name='b')
        qr_out = QuantumRegister(2 * num_state_qubits, name='out')

        # initialize quantum circuit with register list
        super().__init__(qr_a, qr_b, qr_out, name=name)

        # prepare adder as controlled gate
        if not adder:
            from qiskit.circuit.library import RippleCarryAdder
            adder = RippleCarryAdder(num_state_qubits)
        controlled_adder = adder.to_gate().control(1)

        # get the number of helper qubits needed
        num_helper_qubits = adder.num_ancillas

        # add helper qubits if required
        if num_helper_qubits > 0:
            qr_h = AncillaRegister(num_helper_qubits)  # helper/ancilla qubits
            self.add_register(qr_h)

        # build multiplication circuit
        for i in range(num_state_qubits):
            qr_list = [qr_a[i]] + qr_b[:] + qr_out[i:num_state_qubits+i+1]
            if num_helper_qubits > 0:
                qr_list.extend(qr_h[:])
            self.append(controlled_adder, qr_list)
