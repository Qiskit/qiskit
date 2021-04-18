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

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister


class ClassicalMultiplier(QuantumCircuit):
    r"""A circuit that uses classical multiplication approach to store product of two input registers out-of-place.

    **References:**

    [1] Häner et al., Optimizing Quantum Circuits for Arithmetic, 2018.
    `arXiv:1805.12445 <https://arxiv.org/pdf/1805.12445.pdf>`_

    """

    def __init__(self,
                 num_state_qubits: int,
                 name: str = 'ClassicalMultiplier'
                 ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits in either input register for
                state :math:`|a\rangle` or :math:`|b\rangle`. The two input
                registers must have the same number of qubits.
            name: The name of the circuit object.
        Raises:
            ValueError: If ``num_state_qubits`` is lower than 1.
        """
        if num_state_qubits < 1:
            raise ValueError('The number of qubits must be at least 1.')

        qr_a = QuantumRegister(num_state_qubits, name='a')
        qr_b = QuantumRegister(num_state_qubits, name='b')
        qr_out = QuantumRegister(2 * num_state_qubits, name='out')
        qr_aux = AncillaRegister(1, name='aux')

        # initialize quantum circuit with register list
        super().__init__(qr_a, qr_b, qr_out, qr_aux, name=name)

        from qiskit.circuit.library import RippleCarryAdder
        # build multiplication circuit
        for i in range(num_state_qubits):
            self.append(
                RippleCarryAdder(num_state_qubits).to_gate().control(1),
                [qr_a[i]] + qr_b[:] + qr_out[i:num_state_qubits+i+1] + qr_aux[:]
            )
