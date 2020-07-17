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

"""InnerProduct circuit."""


from qiskit.circuit import QuantumRegister, QuantumCircuit


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
