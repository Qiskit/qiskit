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

"""Test library of multi-controlled multi-target circuits."""

import unittest
from ddt import ddt, data, unpack
import numpy as np

from qiskit.test.base import QiskitTestCase
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCMT, MCMTVChain, CHGate, XGate, ZGate, CXGate, CZGate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.states import state_fidelity


@ddt
class TestMCMT(QiskitTestCase):
    """Test the multi-controlled multi-target circuit."""

    @data(MCMT, MCMTVChain)
    def test_mcmt_as_normal_control(self, mcmt_class):
        """Test that the MCMT can act as normal control gate."""
        qc = QuantumCircuit(2)
        mcmt = mcmt_class(gate=CHGate(), num_ctrl_qubits=1, num_target_qubits=1)
        qc = qc.compose(mcmt, [0, 1])

        ref = QuantumCircuit(2)
        ref.ch(0, 1)

        self.assertEqual(qc, ref)

    def test_missing_qubits(self):
        """Test that an error is raised if qubits are missing."""
        with self.subTest(msg="no control qubits"):
            with self.assertRaises(AttributeError):
                _ = MCMT(XGate(), num_ctrl_qubits=0, num_target_qubits=1)

        with self.subTest(msg="no target qubits"):
            with self.assertRaises(AttributeError):
                _ = MCMT(ZGate(), num_ctrl_qubits=4, num_target_qubits=0)

    def test_different_gate_types(self):
        """Test the different supported input types for the target gate."""
        x_circ = QuantumCircuit(1)
        x_circ.x(0)
        for input_gate in [x_circ, QuantumCircuit.cx, QuantumCircuit.x, "cx", "x", CXGate()]:
            with self.subTest(input_gate=input_gate):
                mcmt = MCMT(input_gate, 2, 2)
                if isinstance(input_gate, QuantumCircuit):
                    self.assertEqual(mcmt.gate.definition[0][0], XGate())
                    self.assertEqual(len(mcmt.gate.definition), 1)
                else:
                    self.assertEqual(mcmt.gate, XGate())

    def test_mcmt_v_chain_ancilla_test(self):
        """Test too few and too many ancillas for the MCMT V-chain mode."""
        with self.subTest(msg="insufficient number of auxiliary qubits on gate"):
            qc = QuantumCircuit(5)
            mcmt = MCMTVChain(ZGate(), 3, 1)
            with self.assertRaises(QiskitError):
                qc.append(mcmt, range(5))

        with self.subTest(msg="too many auxiliary qubits on gate"):
            qc = QuantumCircuit(9)
            mcmt = MCMTVChain(ZGate(), 3, 1)
            with self.assertRaises(QiskitError):
                qc.append(mcmt, range(9))

    @data(
        [CZGate(), 1, 1],
        [CHGate(), 1, 1],
        [CZGate(), 3, 3],
        [CHGate(), 3, 3],
        [CZGate(), 1, 5],
        [CHGate(), 1, 5],
        [CZGate(), 5, 1],
        [CHGate(), 5, 1],
    )
    @unpack
    def test_mcmt_v_chain_simulation(self, cgate, num_controls, num_targets):
        """Test the MCMT V-chain implementation test on a simulation."""
        controls = QuantumRegister(num_controls)
        targets = QuantumRegister(num_targets)

        subsets = [tuple(range(i)) for i in range(num_controls + 1)]
        for subset in subsets:
            qc = QuantumCircuit(targets, controls)
            # Initialize all targets to 1, just to be sure that
            # the generic gate has some effect (f.e. Z gate has no effect
            # on a 0 state)
            qc.x(targets)

            num_ancillas = max(0, num_controls - 1)

            if num_ancillas > 0:
                ancillas = QuantumRegister(num_ancillas)
                qc.add_register(ancillas)
                qubits = controls[:] + targets[:] + ancillas[:]
            else:
                qubits = controls[:] + targets[:]

            for i in subset:
                qc.x(controls[i])

            mcmt = MCMTVChain(cgate, num_controls, num_targets)
            qc.compose(mcmt, qubits, inplace=True)

            for i in subset:
                qc.x(controls[i])

            vec = Statevector.from_label("0" * qc.num_qubits).evolve(qc)

            # target register is initially |11...1>, with length equal to 2**(n_targets)
            vec_exp = np.array([0] * (2 ** (num_targets) - 1) + [1])

            if isinstance(cgate, CZGate):
                # Z gate flips the last qubit only if it's applied an odd number of times
                if len(subset) == num_controls and (num_controls % 2) == 1:
                    vec_exp[-1] = -1
            elif isinstance(cgate, CHGate):
                # if all the control qubits have been activated,
                # we repeatedly apply the kronecker product of the Hadamard
                # with itself and then multiply the results for the original
                # state of the target qubits
                if len(subset) == num_controls:
                    h_i = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
                    h_tot = np.array([1])
                    for _ in range(num_targets):
                        h_tot = np.kron(h_tot, h_i)
                    vec_exp = np.dot(h_tot, vec_exp)
            else:
                raise ValueError(f"Test not implement for gate: {cgate}")

            # append the remaining part of the state
            vec_exp = np.concatenate(
                (vec_exp, [0] * (2 ** (num_controls + num_ancillas + num_targets) - vec_exp.size))
            )
            f_i = state_fidelity(vec, vec_exp)
            self.assertAlmostEqual(f_i, 1)


if __name__ == "__main__":
    unittest.main()
