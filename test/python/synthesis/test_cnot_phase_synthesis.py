# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test cnot circuit and cnot-phase circuit synthesis algorithms"""

import unittest
import ddt
from numpy import pi

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis.linear import synth_cnot_count_full_pmh
from qiskit.synthesis.linear_phase import synth_cnot_phase_aam
from test import QiskitTestCase


@ddt.ddt
class TestGraySynth(QiskitTestCase):
    """Test the Gray-Synth algorithm."""

    @ddt.data(
        (["s", "t", "z", "s", "t", "t"],),
        # Angles applied on PhaseGate are 'angles%numpy.pi',
        # So, to get PhaseGate(numpy.pi) we subtract a tiny value from pi.
        ([pi / 2, pi / 4, pi - 1e-09, pi / 2, pi / 4, pi / 4],),
        (["s", "t", "z", "s", "t", pi / 4],),
    )
    @ddt.unpack
    def test_gray_synth(self, angles):
        """Test synthesis of a small parity network via gray_synth.

        The algorithm should take the following matrix as an input:
        S =
        [[0, 1, 1, 0, 1, 1],
         [0, 1, 1, 0, 1, 0],
         [0, 0, 0, 1, 1, 0],
         [1, 0, 0, 1, 1, 1],
         [0, 1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0]]

        Along with some rotation angles:
        ['s', 't', 'z', 's', 't', 't'])

        which together specify the Fourier expansion in the sum-over-paths representation
        of a quantum circuit.

        And should return the following circuit (or an equivalent one):
                          ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐
        q_0: |0>──────────┤ X ├┤ X ├┤ T ├┤ X ├┤ X ├┤ X ├┤ X ├┤ T ├┤ X ├┤ T ├┤ X ├┤ X ├┤ Z ├┤ X ├
                          └─┬─┘└─┬─┘└───┘└─┬─┘└─┬─┘└─┬─┘└─┬─┘└───┘└─┬─┘└───┘└─┬─┘└─┬─┘└───┘└─┬─┘
        q_1: |0>────────────┼────┼─────────■────┼────┼────┼─────────┼─────────┼────┼─────────■──
                            │    │              │    │    │         │         │    │
        q_2: |0>───────■────■────┼──────────────■────┼────┼─────────┼────■────┼────┼────────────
                ┌───┐┌─┴─┐┌───┐  │                   │    │         │  ┌─┴─┐  │    │
        q_3: |0>┤ S ├┤ X ├┤ S ├──■───────────────────┼────┼─────────■──┤ X ├──┼────┼────────────
                └───┘└───┘└───┘                      │    │            └───┘  │    │
        q_4: |0>─────────────────────────────────────■────┼───────────────────■────┼────────────
                                                          │                        │
        q_5: |0>──────────────────────────────────────────■────────────────────────■────────────

        """

        cnots = [
            [0, 1, 1, 0, 1, 1],
            [0, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [1, 0, 0, 1, 1, 1],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0],
        ]
        c_gray = synth_cnot_phase_aam(cnots, angles)
        unitary_gray = UnitaryGate(Operator(c_gray))

        # Create the circuit displayed above:
        q = QuantumRegister(6, "q")
        c_compare = QuantumCircuit(q)
        c_compare.s(q[3])
        c_compare.cx(q[2], q[3])
        c_compare.s(q[3])
        c_compare.cx(q[2], q[0])
        c_compare.cx(q[3], q[0])
        c_compare.t(q[0])
        c_compare.cx(q[1], q[0])
        c_compare.cx(q[2], q[0])
        c_compare.cx(q[4], q[0])
        c_compare.cx(q[5], q[0])
        c_compare.t(q[0])
        c_compare.cx(q[3], q[0])
        c_compare.t(q[0])
        c_compare.cx(q[2], q[3])
        c_compare.cx(q[4], q[0])
        c_compare.cx(q[5], q[0])
        c_compare.z(q[0])
        c_compare.cx(q[1], q[0])
        unitary_compare = UnitaryGate(Operator(c_compare))

        # Check if the two circuits are equivalent
        self.assertEqual(unitary_gray, unitary_compare)

    def test_paper_example(self):
        """Test synthesis of a diagonal operator from the paper.

        The diagonal operator in Example 4.2
            U|x> = e^(2.pi.i.f(x))|x>,
        where
            f(x) = 1/8*(x1^x2 + x0 + x0^x3 + x0^x1^x2 + x0^x1^x3 + x0^x1)

        The algorithm should take the following matrix as an input:
        S = [[0, 1, 1, 1, 1, 1],
             [1, 0, 0, 1, 1, 1],
             [1, 0, 0, 1, 0, 0],
             [0, 0, 1, 0, 1, 0]]

        and only T gates as phase rotations,

        And should return the following circuit (or an equivalent one):
                ┌───┐┌───┐     ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐     ┌───┐┌───┐
        q_0: |0>┤ T ├┤ X ├─────┤ T ├┤ X ├┤ X ├┤ T ├┤ X ├┤ T ├┤ X ├┤ T ├─────┤ X ├┤ X ├
                ├───┤└─┬─┘┌───┐└───┘└─┬─┘└─┬─┘└───┘└─┬─┘└───┘└─┬─┘└───┘┌───┐└─┬─┘└─┬─┘
        q_1: |0>┤ X ├──┼──┤ T ├───────■────┼─────────┼─────────┼───────┤ X ├──■────┼──
                └─┬─┘  │  └───┘            │         │         │       └─┬─┘       │
        q_2: |0>──■────┼───────────────────┼─────────■─────────┼─────────■─────────┼──
                       │                   │                   │                   │
        q_3: |0>───────■───────────────────■───────────────────■───────────────────■──
        """
        cnots = [[0, 1, 1, 1, 1, 1], [1, 0, 0, 1, 1, 1], [1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0]]
        angles = ["t"] * 6
        c_gray = synth_cnot_phase_aam(cnots, angles)
        unitary_gray = UnitaryGate(Operator(c_gray))

        # Create the circuit displayed above:
        q = QuantumRegister(4, "q")
        c_compare = QuantumCircuit(q)
        c_compare.t(q[0])
        c_compare.cx(q[2], q[1])
        c_compare.cx(q[3], q[0])
        c_compare.t(q[0])
        c_compare.t(q[1])
        c_compare.cx(q[1], q[0])
        c_compare.cx(q[3], q[0])
        c_compare.t(q[0])
        c_compare.cx(q[2], q[0])
        c_compare.t(q[0])
        c_compare.cx(q[3], q[0])
        c_compare.t(q[0])
        c_compare.cx(q[2], q[1])
        c_compare.cx(q[1], q[0])
        c_compare.cx(q[3], q[0])
        unitary_compare = UnitaryGate(Operator(c_compare))

        # Check if the two circuits are equivalent
        self.assertEqual(unitary_gray, unitary_compare)

    def test_ccz(self):
        """Test synthesis of the doubly-controlled Z gate.

        The diagonal operator in Example 4.3
            U|x> = e^(2.pi.i.f(x))|x>,
        where
            f(x) = 1/8*(x0 + x1 + x2 - x0^x1 - x0^x2 - x1^x2 + x0^x1^x2)

        The algorithm should take the following matrix as an input:
        S = [[1, 0, 0, 1, 1, 0, 1],
             [0, 1, 0, 1, 0, 1, 1],
             [0, 0, 1, 0, 1, 1, 1]]

        and only T and T* gates as phase rotations,

        And should return the following circuit (or an equivalent one):
                ┌───┐
        q_0: |0>┤ T ├───────■──────────────■───────────────────■──────────────■──
                └───┘┌───┐┌─┴─┐┌───┐       │                   │            ┌─┴─┐
        q_1: |0>─────┤ T ├┤ X ├┤ T*├───────┼─────────■─────────┼─────────■──┤ X ├
                     └───┘└───┘└───┘┌───┐┌─┴─┐┌───┐┌─┴─┐┌───┐┌─┴─┐┌───┐┌─┴─┐└───┘
        q_2: |0>────────────────────┤ T ├┤ X ├┤ T*├┤ X ├┤ T*├┤ X ├┤ T ├┤ X ├─────
                                    └───┘└───┘└───┘└───┘└───┘└───┘└───┘└───┘
        """
        cnots = [[1, 0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 1, 1]]
        angles = ["t", "t", "t", "tdg", "tdg", "tdg", "t"]
        c_gray = synth_cnot_phase_aam(cnots, angles)
        unitary_gray = UnitaryGate(Operator(c_gray))

        # Create the circuit displayed above:
        q = QuantumRegister(3, "q")
        c_compare = QuantumCircuit(q)
        c_compare.t(q[0])
        c_compare.t(q[1])
        c_compare.cx(q[0], q[1])
        c_compare.tdg(q[1])
        c_compare.t(q[2])
        c_compare.cx(q[0], q[2])
        c_compare.tdg(q[2])
        c_compare.cx(q[1], q[2])
        c_compare.tdg(q[2])
        c_compare.cx(q[0], q[2])
        c_compare.t(q[2])
        c_compare.cx(q[1], q[2])
        c_compare.cx(q[0], q[1])
        unitary_compare = UnitaryGate(Operator(c_compare))

        # Check if the two circuits are equivalent
        self.assertEqual(unitary_gray, unitary_compare)

    def test_zero_parity_columns_are_dropped(self):
        """Test if columns whose parity vector is all-zero, are dropped
        along with their angles.
        """
        cnots = [
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        angles = [0.1, 0.2, 0.3, 0.4]  # one per column
        ckt_gray = synth_cnot_phase_aam(cnots, angles)
        oper_list = [ckt_instr.operation for ckt_instr in ckt_gray]
        oper_name_list = [oper.name for oper in oper_list]
        oper_params_list = [oper.params for oper in oper_list]

        # Only one parity is present on qubit:0, so, no CNOTs applied
        self.assertNotIn("cx", oper_name_list)

        # Angle 0.1, 0.4 were attached to a zero-parity column 0, 3 and should
        # have been dropped.
        self.assertFalse([0.1] in oper_params_list)
        self.assertFalse([0.4] in oper_params_list)

    def test_unit_parities_need_no_cnots(self):
        """
        Test that an input consisting only of unit parities e_0, e_1, ..., e_{n-1}
        produces a circuit with zero CNOTs and only Rz gates.
        """
        n = 5
        # Identity matrix: column k is the unit parity e_k.
        cnots = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        angles = [0.1, 0.2, 0.3, 0.4, 0.5]  # one angle per column

        ckt_gray = synth_cnot_phase_aam(cnots, angles)
        oper_list = [ckt_instr.operation for ckt_instr in ckt_gray]
        oper_name_list = [oper.name for oper in oper_list]
        oper_params_list = [oper.params for oper in oper_list]

        # Unit partiy should have no CNOTs applied
        self.assertNotIn("cx", oper_name_list)

        # All angles should be present
        for ang in angles:
            self.assertIn([ang], oper_params_list)


@ddt.ddt
class TestPatelMarkovHayes(QiskitTestCase):
    """Test the Patel-Markov-Hayes algorithm for synthesizing linear
    CNOT-only circuits."""

    def test_patel_markov_hayes(self):
        """Test synthesis of a small linear circuit
        (example from paper, Figure 3).

        The algorithm should take the following matrix as an input:
        S = [[1, 1, 0, 0, 0, 0],
             [1, 0, 0, 1, 1, 0],
             [0, 1, 0, 0, 1, 0],
             [1, 1, 1, 1, 1, 1],
             [1, 1, 0, 1, 1, 1],
             [0, 0, 1, 1, 1, 0]]

        And should return the following circuit (or an equivalent one):
                          ┌───┐
        q_0: |0>──────────┤ X ├──────────────────────────────────────────■────■────■──
                          └─┬─┘┌───┐                                   ┌─┴─┐  │    │
        q_1: |0>────────────■──┤ X ├────────────────────────────────■──┤ X ├──┼────┼──
                     ┌───┐     └─┬─┘┌───┐          ┌───┐          ┌─┴─┐└───┘  │    │
        q_2: |0>─────┤ X ├───────┼──┤ X ├───────■──┤ X ├───────■──┤ X ├───────┼────┼──
                ┌───┐└─┬─┘       │  └─┬─┘┌───┐┌─┴─┐└─┬─┘       │  └───┘       │  ┌─┴─┐
        q_3: |0>┤ X ├──┼─────────■────┼──┤ X ├┤ X ├──■────■────┼──────────────┼──┤ X ├
                └─┬─┘  │              │  └─┬─┘├───┤       │  ┌─┴─┐          ┌─┴─┐└───┘
        q_4: |0>──■────┼──────────────■────■──┤ X ├───────┼──┤ X ├──────────┤ X ├─────
                       │                      └─┬─┘     ┌─┴─┐└───┘          └───┘
        q_5: |0>───────■────────────────────────■───────┤ X ├─────────────────────────
                                                        └───┘
        """
        state = [
            [1, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 0],
        ]
        c_patel = synth_cnot_count_full_pmh(state)
        unitary_patel = UnitaryGate(Operator(c_patel))

        # Create the circuit displayed above:
        q = QuantumRegister(6, "q")
        c_compare = QuantumCircuit(q)
        c_compare.cx(q[4], q[3])
        c_compare.cx(q[5], q[2])
        c_compare.cx(q[1], q[0])
        c_compare.cx(q[3], q[1])
        c_compare.cx(q[4], q[2])
        c_compare.cx(q[4], q[3])
        c_compare.cx(q[5], q[4])
        c_compare.cx(q[2], q[3])
        c_compare.cx(q[3], q[2])
        c_compare.cx(q[3], q[5])
        c_compare.cx(q[2], q[4])
        c_compare.cx(q[1], q[2])
        c_compare.cx(q[0], q[1])
        c_compare.cx(q[0], q[4])
        c_compare.cx(q[0], q[3])
        unitary_compare = UnitaryGate(Operator(c_compare))

        # Check if the two circuits are equivalent
        self.assertEqual(unitary_patel, unitary_compare)


if __name__ == "__main__":
    unittest.main()
