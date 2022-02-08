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

"""Test library of phase estimation circuits."""

import unittest

import numpy as np

from qiskit.test.base import QiskitTestCase
from qiskit import BasicAer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PhaseEstimation, QFT
from qiskit.quantum_info import Statevector


class TestPhaseEstimation(QiskitTestCase):
    """Test the phase estimation circuit."""

    def assertPhaseEstimationIsCorrect(
        self, pec: QuantumCircuit, eigenstate: QuantumCircuit, phase_as_binary: str
    ):
        r"""Assert that the phase estimation circuit implements the correct transformation.

        Applying the phase estimation circuit on a target register which holds the eigenstate
        :math:`|u\rangle` (say the last register), the final state should be

        .. math::

            |\phi_1\rangle \cdots |\phi_t\rangle |u\rangle

        where the eigenvalue is written as :math:`e^{2\pi i \phi}` and the angle is represented
        in binary fraction, i.e. :math:`\phi = 0.\phi_1 \ldots \phi_t`.

        Args:
            pec: The circuit implementing the phase estimation circuit.
            eigenstate: The eigenstate as circuit.
            phase_as_binary: The phase of the eigenvalue in a binary fraction. E.g. if the
                phase is 0.25, the binary fraction is '01' as 0.01 = 0 * 0.5 + 1 * 0.25 = 0.25.
        """

        # the target state
        eigenstate_as_vector = Statevector.from_instruction(eigenstate).data
        reference = eigenstate_as_vector

        zero, one = [1, 0], [0, 1]
        for qubit in phase_as_binary[::-1]:
            reference = np.kron(reference, zero if qubit == "0" else one)

        # the simulated state
        circuit = QuantumCircuit(pec.num_qubits)
        circuit.compose(
            eigenstate,
            list(range(pec.num_qubits - eigenstate.num_qubits, pec.num_qubits)),
            inplace=True,
        )
        circuit.compose(pec, inplace=True)
        # TODO use Statevector for simulation once Qiskit/qiskit-terra#4681 is resolved
        # actual = Statevector.from_instruction(circuit).data
        backend = BasicAer.get_backend("statevector_simulator")
        actual = execute(circuit, backend).result().get_statevector()

        np.testing.assert_almost_equal(reference, actual)

    def test_phase_estimation(self):
        """Test the standard phase estimation circuit."""
        with self.subTest("U=S, psi=|1>"):
            unitary = QuantumCircuit(1)
            unitary.s(0)

            eigenstate = QuantumCircuit(1)
            eigenstate.x(0)

            # eigenvalue is 1j = exp(2j pi 0.25) thus phi = 0.25 = 0.010 = '010'
            # using three digits as 3 evaluation qubits are used
            phase_as_binary = "0100"

            pec = PhaseEstimation(4, unitary)

            self.assertPhaseEstimationIsCorrect(pec, eigenstate, phase_as_binary)

        with self.subTest("U=SZ, psi=|11>"):
            unitary = QuantumCircuit(2)
            unitary.z(0)
            unitary.s(1)

            eigenstate = QuantumCircuit(2)
            eigenstate.x([0, 1])

            # eigenvalue is -1j = exp(2j pi 0.75) thus phi = 0.75 = 0.110 = '110'
            # using three digits as 3 evaluation qubits are used
            phase_as_binary = "110"

            pec = PhaseEstimation(3, unitary)

            self.assertPhaseEstimationIsCorrect(pec, eigenstate, phase_as_binary)

        with self.subTest("a 3-q unitary"):
            #      ┌───┐
            # q_0: ┤ X ├──■────■───────
            #      ├───┤  │    │
            # q_1: ┤ X ├──■────■───────
            #      ├───┤┌───┐┌─┴─┐┌───┐
            # q_2: ┤ X ├┤ H ├┤ X ├┤ H ├
            #      └───┘└───┘└───┘└───┘
            unitary = QuantumCircuit(3)
            unitary.x([0, 1, 2])
            unitary.cz(0, 1)
            unitary.h(2)
            unitary.ccx(0, 1, 2)
            unitary.h(2)

            #      ┌───┐
            # q_0: ┤ H ├──■────■──
            #      └───┘┌─┴─┐  │
            # q_1: ─────┤ X ├──┼──
            #           └───┘┌─┴─┐
            # q_2: ──────────┤ X ├
            #                └───┘
            eigenstate = QuantumCircuit(3)
            eigenstate.h(0)
            eigenstate.cx(0, 1)
            eigenstate.cx(0, 2)

            # the unitary acts as identity on the eigenstate, thus the phase is 0
            phase_as_binary = "00"

            pec = PhaseEstimation(2, unitary)

            self.assertPhaseEstimationIsCorrect(pec, eigenstate, phase_as_binary)

    def test_phase_estimation_iqft_setting(self):
        """Test default and custom setting of the QFT circuit."""
        unitary = QuantumCircuit(1)
        unitary.s(0)

        with self.subTest("default QFT"):
            pec = PhaseEstimation(3, unitary)
            expected_qft = QFT(3, inverse=True, do_swaps=False)
            self.assertEqual(pec.decompose().data[-1][0].definition, expected_qft.decompose())

        with self.subTest("custom QFT"):
            iqft = QFT(3, approximation_degree=2).inverse()
            pec = PhaseEstimation(3, unitary, iqft=iqft)
            self.assertEqual(pec.decompose().data[-1][0].definition, iqft.decompose())


if __name__ == "__main__":
    unittest.main()
