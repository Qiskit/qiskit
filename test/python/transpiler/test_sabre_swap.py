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

"""Test the Sabre Swap pass"""

import unittest

import ddt

from qiskit.circuit.library import CCXGate, HGate, Measure, SwapGate
from qiskit.transpiler.passes import SabreSwap, TrivialLayout
from qiskit.transpiler import CouplingMap, PassManager
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.utils import optionals


def looping_circuit(uphill_swaps=1, additional_local_minimum_gates=0):
    """A circuit that causes SabreSwap to loop infinitely.

    This looks like (using cz gates to show the symmetry, though we actually output cx for testing
    purposes):

    .. parsed-literal::

         q_0: ─■────────────────
               │
         q_1: ─┼──■─────────────
               │  │
         q_2: ─┼──┼──■──────────
               │  │  │
         q_3: ─┼──┼──┼──■───────
               │  │  │  │
         q_4: ─┼──┼──┼──┼─────■─
               │  │  │  │     │
         q_5: ─┼──┼──┼──┼──■──■─
               │  │  │  │  │
         q_6: ─┼──┼──┼──┼──┼────
               │  │  │  │  │
         q_7: ─┼──┼──┼──┼──■──■─
               │  │  │  │     │
         q_8: ─┼──┼──┼──┼─────■─
               │  │  │  │
         q_9: ─┼──┼──┼──■───────
               │  │  │
        q_10: ─┼──┼──■──────────
               │  │
        q_11: ─┼──■─────────────
               │
        q_12: ─■────────────────

    where `uphill_swaps` is the number of qubits separating the inner-most gate (representing how
    many swaps need to be made that all increase the heuristics), and
    `additional_local_minimum_gates` is how many extra gates to add on the outside (these increase
    the size of the region of stability).
    """
    outers = 4 + additional_local_minimum_gates
    n_qubits = 2 * outers + 4 + uphill_swaps
    # This is (most of) the front layer, which is a bunch of outer qubits in the
    # coupling map.
    outer_pairs = [(i, n_qubits - i - 1) for i in range(outers)]
    inner_heuristic_peak = [
        # This gate is completely "inside" all the others in the front layer in
        # terms of the coupling map, so it's the only one that we can in theory
        # make progress towards without making the others worse.
        (outers + 1, outers + 2 + uphill_swaps),
        # These are the only two gates in the extended set, and they both get
        # further apart if you make a swap to bring the above gate closer
        # together, which is the trick that creates the "heuristic hill".
        (outers, outers + 1),
        (outers + 2 + uphill_swaps, outers + 3 + uphill_swaps),
    ]
    qc = QuantumCircuit(n_qubits)
    for pair in outer_pairs + inner_heuristic_peak:
        qc.cx(*pair)
    return qc


@ddt.ddt
class TestSabreSwap(QiskitTestCase):
    """Tests the SabreSwap pass."""

    def test_trivial_case(self):
        """Test that an already mapped circuit is unchanged.
                  ┌───┐┌───┐
        q_0: ──■──┤ H ├┤ X ├──■──
             ┌─┴─┐└───┘└─┬─┘  │
        q_1: ┤ X ├──■────■────┼──
             └───┘┌─┴─┐       │
        q_2: ──■──┤ X ├───────┼──
             ┌─┴─┐├───┤       │
        q_3: ┤ X ├┤ X ├───────┼──
             └───┘└─┬─┘     ┌─┴─┐
        q_4: ───────■───────┤ X ├
                            └───┘
        """
        coupling = CouplingMap.from_ring(5)

        qr = QuantumRegister(5, "q")
        qc = QuantumCircuit(qr)
        qc.cx(0, 1)  # free
        qc.cx(2, 3)  # free
        qc.h(0)  # free
        qc.cx(1, 2)  # F
        qc.cx(1, 0)
        qc.cx(4, 3)  # F
        qc.cx(0, 4)

        passmanager = PassManager(SabreSwap(coupling, "basic"))
        new_qc = passmanager.run(qc)

        self.assertEqual(new_qc, qc)

    def test_lookahead_mode(self):
        """Test lookahead mode's lookahead finds single SWAP gate.
                  ┌───┐
        q_0: ──■──┤ H ├───────────────
             ┌─┴─┐└───┘
        q_1: ┤ X ├──■────■─────────■──
             └───┘┌─┴─┐  │         │
        q_2: ──■──┤ X ├──┼────■────┼──
             ┌─┴─┐└───┘┌─┴─┐┌─┴─┐┌─┴─┐
        q_3: ┤ X ├─────┤ X ├┤ X ├┤ X ├
             └───┘     └───┘└───┘└───┘
        q_4: ─────────────────────────

        """
        coupling = CouplingMap.from_line(5)

        qr = QuantumRegister(5, "q")
        qc = QuantumCircuit(qr)
        qc.cx(0, 1)  # free
        qc.cx(2, 3)  # free
        qc.h(0)  # free
        qc.cx(1, 2)  # free
        qc.cx(1, 3)  # F
        qc.cx(2, 3)  # E
        qc.cx(1, 3)  # E

        pm = PassManager(SabreSwap(coupling, "lookahead"))
        new_qc = pm.run(qc)

        self.assertEqual(new_qc.num_nonlocal_gates(), 7)

    def test_do_not_change_cm(self):
        """Coupling map should not change.
        See https://github.com/Qiskit/qiskit-terra/issues/5675"""
        cm_edges = [(1, 0), (2, 0), (2, 1), (3, 2), (3, 4), (4, 2)]
        coupling = CouplingMap(cm_edges)

        passmanager = PassManager(SabreSwap(coupling))
        _ = passmanager.run(QuantumCircuit(1))

        self.assertEqual(set(cm_edges), set(coupling.get_edges()))

    def test_do_not_reorder_measurements(self):
        """Test that SabreSwap doesn't reorder measurements to the same classical bit.

        With the particular coupling map used in this test and the 3q ccx gate, the routing would
        invariably the measurements if the classical successors are not accurately tracked.
        Regression test of gh-7950."""
        coupling = CouplingMap([(0, 2), (2, 0), (1, 2), (2, 1)])
        qc = QuantumCircuit(3, 1)
        qc.compose(CCXGate().definition, [0, 1, 2], [])  # Unroll CCX to 2q operations.
        qc.h(0)
        qc.barrier()
        qc.measure(0, 0)  # This measure is 50/50 between the Z states.
        qc.measure(1, 0)  # This measure always overwrites with 0.
        passmanager = PassManager(SabreSwap(coupling))
        transpiled = passmanager.run(qc)

        last_h, h_qubits, _ = transpiled.data[-4]
        self.assertIsInstance(last_h, HGate)
        first_measure, first_measure_qubits, _ = transpiled.data[-2]
        second_measure, second_measure_qubits, _ = transpiled.data[-1]
        self.assertIsInstance(first_measure, Measure)
        self.assertIsInstance(second_measure, Measure)
        # Assert that the first measure is on the same qubit that the HGate was applied to, and the
        # second measurement is on a different qubit (though we don't care which exactly - that
        # depends a little on the randomisation of the pass).
        self.assertEqual(h_qubits, first_measure_qubits)
        self.assertNotEqual(h_qubits, second_measure_qubits)

    # The 'basic' method can't get stuck in the same way.
    @ddt.data("lookahead", "decay")
    def test_no_infinite_loop(self, method):
        """Test that the 'release value' mechanisms allow SabreSwap to make progress even on
        circuits that get stuck in a stable local minimum of the lookahead parameters."""
        qc = looping_circuit(3, 1)
        qc.measure_all()
        coupling_map = CouplingMap.from_line(qc.num_qubits)
        routing_pass = PassManager(SabreSwap(coupling_map, method))

        n_swap_gates = 0

        def leak_number_of_swaps(cls, *args, **kwargs):
            nonlocal n_swap_gates
            n_swap_gates += 1
            if n_swap_gates > 1_000:
                raise Exception("SabreSwap seems to be stuck in a loop")
            # pylint: disable=bad-super-call
            return super(SwapGate, cls).__new__(cls, *args, **kwargs)

        with unittest.mock.patch.object(SwapGate, "__new__", leak_number_of_swaps):
            routed = routing_pass.run(qc)

        routed_ops = routed.count_ops()
        del routed_ops["swap"]
        self.assertEqual(routed_ops, qc.count_ops())
        couplings = {
            tuple(routed.find_bit(bit).index for bit in qargs)
            for _, qargs, _ in routed.data
            if len(qargs) == 2
        }
        # Asserting equality to the empty set gives better errors on failure than asserting that
        # `couplings <= coupling_map`.
        self.assertEqual(couplings - set(coupling_map.get_edges()), set())

        # Assert that the same keys are produced by a simulation - this is a test that the inserted
        # swaps route the qubits correctly.
        if not optionals.HAS_AER:
            return

        from qiskit import Aer

        sim = Aer.get_backend("aer_simulator")
        in_results = sim.run(qc, shots=4096).result().get_counts()
        out_results = sim.run(routed, shots=4096).result().get_counts()
        self.assertEqual(set(in_results), set(out_results))

    def test_classical_condition(self):
        """Test that :class:`.SabreSwap` correctly accounts for classical conditions in its
        reckoning on whether a node is resolved or not.  If it is not handled correctly, the second
        gate might not appear in the output.

        Regression test of gh-8040."""
        with self.subTest("1 bit in register"):
            qc = QuantumCircuit(2, 1)
            qc.z(0)
            qc.z(0).c_if(qc.cregs[0], 0)
            cm = CouplingMap([(0, 1), (1, 0)])
            expected = PassManager([TrivialLayout(cm)]).run(qc)
            actual = PassManager([TrivialLayout(cm), SabreSwap(cm)]).run(qc)
            self.assertEqual(expected, actual)
        with self.subTest("multiple registers"):
            cregs = [ClassicalRegister(3), ClassicalRegister(4)]
            qc = QuantumCircuit(QuantumRegister(2, name="q"), *cregs)
            qc.z(0)
            qc.z(0).c_if(cregs[0], 0)
            qc.z(0).c_if(cregs[1], 0)
            cm = CouplingMap([(0, 1), (1, 0)])
            expected = PassManager([TrivialLayout(cm)]).run(qc)
            actual = PassManager([TrivialLayout(cm), SabreSwap(cm)]).run(qc)
            self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
