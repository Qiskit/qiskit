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
from qiskit.circuit import QuantumRegister, AncillaRegister, QuantumCircuit

from .multiplier import Multiplier


class HRSCumulativeMultiplier(Multiplier):
    r"""A multiplication circuit to store product of two input registers out-of-place.

    Circuit uses the approach from [1]. As an example, a multiplier circuit that
    performs multiplication on two 3-qubit sized registers with the default adder
    is as follows:

    .. parsed-literal::

          a_0: ────■─────────────────────────
                   │
          a_1: ────┼─────────■───────────────
                   │         │
          a_2: ────┼─────────┼─────────■─────
               ┌───┴────┐┌───┴────┐┌───┴────┐
          b_0: ┤0       ├┤0       ├┤0       ├
               │        ││        ││        │
          b_1: ┤1       ├┤1       ├┤1       ├
               │        ││        ││        │
          b_2: ┤2       ├┤2       ├┤2       ├
               │        ││        ││        │
        out_0: ┤3       ├┤        ├┤        ├
               │        ││        ││        │
        out_1: ┤4       ├┤3       ├┤        ├
               │  Adder ││  Adder ││  Adder │
        out_2: ┤5       ├┤4       ├┤3       ├
               │        ││        ││        │
        out_3: ┤6       ├┤5       ├┤4       ├
               │        ││        ││        │
        out_4: ┤        ├┤6       ├┤5       ├
               │        ││        ││        │
        out_5: ┤        ├┤        ├┤6       ├
               │        ││        ││        │
        aux_0: ┤7       ├┤7       ├┤7       ├
               └────────┘└────────┘└────────┘

    Multiplication in this circuit is implemented in a classical approach by performing
    a series of shifted additions using one of the input registers while the qubits
    from the other input register act as control qubits for the adders.

    **References:**

    [1] Häner et al., Optimizing Quantum Circuits for Arithmetic, 2018.
    `arXiv:1805.12445 <https://arxiv.org/pdf/1805.12445.pdf>`_

    """

    def __init__(
        self,
        num_state_qubits: int,
        adder: Optional[QuantumCircuit] = None,
        name: str = "HRSCumulativeMultiplier",
    ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits in either input register for
                state :math:`|a\rangle` or :math:`|b\rangle`. The two input
                registers must have the same number of qubits.
            adder: adder circuit to be used for performing multiplication. The
                CDKMRippleCarryAdder is used as default if no adder is provided.
            name: The name of the circuit object.
        """
        super().__init__(num_state_qubits, name=name)

        # define the registers
        qr_a = QuantumRegister(num_state_qubits, name="a")
        qr_b = QuantumRegister(num_state_qubits, name="b")
        qr_out = QuantumRegister(2 * num_state_qubits, name="out")
        self.add_register(qr_a, qr_b, qr_out)

        # prepare adder as controlled gate
        if adder is None:
            from qiskit.circuit.library import CDKMRippleCarryAdder

            adder = CDKMRippleCarryAdder(num_state_qubits)
        controlled_adder = adder.to_gate().control(1)

        # get the number of helper qubits needed
        num_helper_qubits = adder.num_ancillas

        # add helper qubits if required
        if num_helper_qubits > 0:
            qr_h = AncillaRegister(num_helper_qubits, name='aux')  # helper/ancilla qubits
            self.add_register(qr_h)

        # build multiplication circuit
        for i in range(num_state_qubits):
            qr_list = [qr_a[i]] + qr_b[:] + qr_out[i : num_state_qubits + i + 1]
            if num_helper_qubits > 0:
                qr_list.extend(qr_h[:])
            self.append(controlled_adder, qr_list)
