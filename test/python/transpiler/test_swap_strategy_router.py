# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for swap strategy routers."""

from qiskit.circuit import QuantumCircuit, Qubit, QuantumRegister
from qiskit.transpiler import PassManager, CouplingMap, Layout, TranspilerError

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit.library.n_local import QAOAAnsatz
from qiskit.converters import circuit_to_dag
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import Pauli
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes import SetLayout

from qiskit.test import QiskitTestCase

from qiskit.transpiler.passes.routing.swap_strategies import (
    SwapStrategy,
    FindCommutingPauliEvolutions,
    Commuting2qGateRouter,
)


class TestPauliEvolutionSwapStrategies(QiskitTestCase):
    """A class to test the swap strategies transpiler passes."""

    def setUp(self):
        """Assume a linear coupling map."""
        super().setUp()
        cmap = CouplingMap(couplinglist=[(0, 1), (1, 2), (2, 3)])
        swap_strat = SwapStrategy(cmap, swap_layers=[[(0, 1), (2, 3)], [(1, 2)]])

        self.pm_ = PassManager(
            [
                FindCommutingPauliEvolutions(),
                Commuting2qGateRouter(swap_strat),
            ]
        )

    def test_basic_zz(self):
        """Test to decompose a ZZ-based evolution op.

        The expected circuit is:

        ..parsed-literal::
                                                           ┌────────────────┐
            q_0: ───────────────────X──────────────────────┤0               ├
                 ┌────────────────┐ │ ┌────────────────┐   │  exp(-i ZZ)(3) │
            q_1: ┤0               ├─X─┤0               ├─X─┤1               ├
                 │  exp(-i ZZ)(1) │   │  exp(-i ZZ)(2) │ │ └────────────────┘
            q_2: ┤1               ├─X─┤1               ├─X───────────────────
                 └────────────────┘ │ └────────────────┘
            q_3: ───────────────────X────────────────────────────────────────


        """

        op = PauliSumOp.from_list([("IZZI", 1), ("ZIIZ", 2), ("ZIZI", 3)])
        circ = QuantumCircuit(4)
        circ.append(PauliEvolutionGate(op, 1), range(4))

        swapped = self.pm_.run(circ)

        expected = QuantumCircuit(4)
        expected.append(PauliEvolutionGate(Pauli("ZZ"), 1), (1, 2))
        expected.swap(0, 1)
        expected.swap(2, 3)
        expected.append(PauliEvolutionGate(Pauli("ZZ"), 2), (1, 2))
        expected.swap(1, 2)
        expected.append(PauliEvolutionGate(Pauli("ZZ"), 3), (0, 1))

        self.assertEqual(swapped, expected)

    def test_basic_xx(self):
        """Test to route an XX-based evolution op.

        The op is :code:`[("XXII", -1), ("IIXX", 1), ("XIIX", -2), ("IXIX", 2)]`.

        The expected circuit is:

        ..parsed-literal::

                  ┌────────────────┐
            q_0: ─┤0               ├─X─────────────────────────────────────────
                  │  exp(-i XX)(3) │ │ ┌─────────────────┐
            q_1: ─┤1               ├─X─┤0                ├─X───────────────────
                 ┌┴────────────────┤   │  exp(-i XX)(-6) │ │ ┌────────────────┐
            q_2: ┤0                ├─X─┤1                ├─X─┤0               ├
                 │  exp(-i XX)(-3) │ │ └─────────────────┘   │  exp(-i XX)(6) │
            q_3: ┤1                ├─X───────────────────────┤1               ├
                 └─────────────────┘                         └────────────────┘
        """

        op = PauliSumOp.from_list([("XXII", -1), ("IIXX", 1), ("XIIX", -2), ("IXIX", 2)])

        circ = QuantumCircuit(4)
        circ.append(PauliEvolutionGate(op, 3), range(4))

        swapped = self.pm_.run(circ)

        expected = QuantumCircuit(4)
        expected.append(PauliEvolutionGate(Pauli("XX"), 3), (0, 1))
        expected.append(PauliEvolutionGate(Pauli("XX"), -3), (2, 3))
        expected.swap(0, 1)
        expected.swap(2, 3)
        expected.append(PauliEvolutionGate(Pauli("XX"), -6), (1, 2))
        expected.swap(1, 2)
        expected.append(PauliEvolutionGate(Pauli("XX"), 6), (2, 3))

        self.assertEqual(swapped, expected)

    def test_basic_xx_with_measure(self):
        """Test to route an XX-based evolution op with measures.

        The op is :code:`[("XXII", -1), ("IIXX", 1), ("XIIX", -2), ("IXIX", 2)]`.

        The expected circuit is:

        ..parsed-literal::

                     ┌────────────────┐                                            ░    ┌─┐
               q_0: ─┤0               ├─X──────────────────────────────────────────░────┤M├──────
                     │  exp(-i XX)(3) │ │ ┌─────────────────┐                      ░    └╥┘   ┌─┐
               q_1: ─┤1               ├─X─┤0                ├─X────────────────────░─────╫────┤M├
                    ┌┴────────────────┤   │  exp(-i XX)(-6) │ │ ┌────────────────┐ ░ ┌─┐ ║    └╥┘
               q_2: ┤0                ├─X─┤1                ├─X─┤0               ├─░─┤M├─╫─────╫─
                    │  exp(-i XX)(-3) │ │ └─────────────────┘   │  exp(-i XX)(6) │ ░ └╥┘ ║ ┌─┐ ║
               q_3: ┤1                ├─X───────────────────────┤1               ├─░──╫──╫─┤M├─╫─
                    └─────────────────┘                         └────────────────┘ ░  ║  ║ └╥┘ ║
            meas: 4/══════════════════════════════════════════════════════════════════╩══╩══╩══╩═
                                                                                      0  1  2  3
        """

        op = PauliSumOp.from_list([("XXII", -1), ("IIXX", 1), ("XIIX", -2), ("IXIX", 2)])

        circ = QuantumCircuit(4, 4)
        circ.append(PauliEvolutionGate(op, 3), range(4))
        circ.barrier()
        for idx in range(4):
            circ.measure(idx, idx)

        swapped = self.pm_.run(circ)

        expected = QuantumCircuit(4, 4)
        expected.append(PauliEvolutionGate(Pauli("XX"), 3), (0, 1))
        expected.append(PauliEvolutionGate(Pauli("XX"), -3), (2, 3))
        expected.swap(0, 1)
        expected.swap(2, 3)
        expected.append(PauliEvolutionGate(Pauli("XX"), -6), (1, 2))
        expected.swap(1, 2)
        expected.append(PauliEvolutionGate(Pauli("XX"), 6), (2, 3))
        expected.barrier()
        expected.measure(2, 0)
        expected.measure(0, 1)
        expected.measure(3, 2)
        expected.measure(1, 3)

        self.assertEqual(swapped, expected)

    def test_qaoa(self):
        """Test the QAOA with a custom mixer.

        This test ensures that single-qubit gates end up on the correct qubits. The mixer
        uses Ry gates and the operator is :code:`[("IZZI", 1), ("ZIIZ", 2), ("ZIZI", 3)]`.

        ..parsed-literal:

                 ┌───┐                                              ┌──────────────────┐»
            q_0: ┤ H ├─────────────────────X────────────────────────┤0                 ├»
                 ├───┤┌──────────────────┐ │ ┌──────────────────┐   │  exp(-i ZZ)(3.0) │»
            q_1: ┤ H ├┤0                 ├─X─┤0                 ├─X─┤1                 ├»
                 ├───┤│  exp(-i ZZ)(1.0) │   │  exp(-i ZZ)(2.0) │ │ └──────────────────┘»
            q_2: ┤ H ├┤1                 ├─X─┤1                 ├─X─────────────────────»
                 ├───┤└──────────────────┘ │ └──────────────────┘                       »
            q_3: ┤ H ├─────────────────────X────────────────────────────────────────────»
                 └───┘                                                                  »
            «     ┌────────┐                    ┌──────────────────┐         »
            «q_0: ┤ Ry(-1) ├────────────────────┤0                 ├────X────»
            «     ├────────┤┌──────────────────┐│  exp(-i ZZ)(6.0) │    │    »
            «q_1: ┤ Ry(-3) ├┤0                 ├┤1                 ├────X────»
            «     ├────────┤│  exp(-i ZZ)(4.0) │└──────────────────┘         »
            «q_2: ┤ Ry(0)  ├┤1                 ├─────────X───────────────────»
            «     ├────────┤└──────────────────┘         │                   »
            «q_3: ┤ Ry(-2) ├─────────────────────────────X───────────────────»
            «     └────────┘                                                 »
            «                         ┌────────┐
            «q_0: ────────────────────┤ Ry(-3) ├
            «     ┌──────────────────┐├────────┤
            «q_1: ┤0                 ├┤ Ry(-1) ├
            «     │  exp(-i ZZ)(2.0) │├────────┤
            «q_2: ┤1                 ├┤ Ry(-2) ├
            «     └──────────────────┘├────────┤
            «q_3: ────────────────────┤ Ry(0)  ├
            «                         └────────┘
        """

        mixer = QuantumCircuit(4)
        for idx in range(4):
            mixer.ry(-idx, idx)

        op = PauliSumOp.from_list([("IZZI", 1), ("ZIIZ", 2), ("ZIZI", 3)])
        circ = QAOAAnsatz(op, reps=2, mixer_operator=mixer)

        swapped = self.pm_.run(circ.decompose())
        param_dict = {p: idx + 1 for idx, p in enumerate(swapped.parameters)}
        swapped.assign_parameters(param_dict, inplace=True)

        # There is some optionality in the order of two instructions. Both are valid.
        valid_expected = []
        for order in [0, 1]:
            expected = QuantumCircuit(4)
            expected.h(range(4))
            expected.append(PauliEvolutionGate(Pauli("ZZ"), 1), (1, 2))
            expected.swap(0, 1)
            expected.swap(2, 3)
            expected.append(PauliEvolutionGate(Pauli("ZZ"), 2), (1, 2))
            expected.swap(1, 2)
            expected.append(PauliEvolutionGate(Pauli("ZZ"), 3), (0, 1))
            expected.ry(-1, 0)
            expected.ry(-3, 1)
            expected.ry(0, 2)
            expected.ry(-2, 3)
            if order == 0:
                expected.append(PauliEvolutionGate(Pauli("ZZ"), 6), (0, 1))
                expected.append(PauliEvolutionGate(Pauli("ZZ"), 4), (2, 1))
            else:
                expected.append(PauliEvolutionGate(Pauli("ZZ"), 4), (2, 1))
                expected.append(PauliEvolutionGate(Pauli("ZZ"), 6), (0, 1))

            expected.swap(0, 1)
            expected.swap(2, 3)
            expected.append(PauliEvolutionGate(Pauli("ZZ"), 2), (1, 2))
            expected.ry(-3, 0)
            expected.ry(-1, 1)
            expected.ry(-2, 2)
            expected.ry(0, 3)

            valid_expected.append(expected == swapped)

        self.assertEqual(set(valid_expected), {True, False})

    def test_enlarge_with_ancilla(self):
        """This pass tests that idle qubits after an embedding are left idle."""

        # Create a four qubit problem.
        op = PauliSumOp.from_list([("IZZI", 1), ("ZIIZ", 2), ("ZIZI", 3)])

        circ = QuantumCircuit(4)
        circ.append(PauliEvolutionGate(op, 1), range(4))

        # Create a four qubit quantum circuit.
        backend_cmap = CouplingMap(couplinglist=[(0, 1), (1, 2), (1, 3), (3, 4)])

        swap_cmap = CouplingMap(couplinglist=[(0, 1), (1, 2), (2, 3)])
        swap_strat = SwapStrategy(swap_cmap, swap_layers=(((0, 1), (2, 3)), ((1, 2),)))

        initial_layout = Layout.from_intlist([0, 1, 3, 4], *circ.qregs)

        pm_pre = PassManager(
            [
                FindCommutingPauliEvolutions(),
                Commuting2qGateRouter(swap_strat),
                SetLayout(initial_layout),
                FullAncillaAllocation(backend_cmap),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ]
        )

        embedded = pm_pre.run(circ)

        expected = QuantumCircuit(5)
        expected.append(PauliEvolutionGate(Pauli("ZZ"), 1), (1, 3))
        expected.swap(0, 1)
        expected.swap(3, 4)
        expected.append(PauliEvolutionGate(Pauli("ZZ"), 2), (1, 3))
        expected.swap(1, 3)
        expected.append(PauliEvolutionGate(Pauli("ZZ"), 3), (0, 1))

        self.assertEqual(embedded, expected)

    def test_ccx(self):
        """Test that extra multi-qubit operations are properly adjusted.

        Here, we test that the circuit

        .. parsed-literal::

                 ┌──────────────────────────┐
            q_0: ┤0                         ├──■──
                 │                          │┌─┴─┐
            q_1: ┤1 exp(-it (IZZ + ZIZ))(1) ├┤ X ├
                 │                          │└─┬─┘
            q_2: ┤2                         ├──■──
                 └──────────────────────────┘
            q_3: ─────────────────────────────────

        becomes

        .. parsed-literal::

                 ┌─────────────────┐                      ┌───┐
            q_0: ┤0                ├─X────────────────────┤ X ├
                 │  exp(-it ZZ)(1) │ │ ┌─────────────────┐└─┬─┘
            q_1: ┤1                ├─X─┤0                ├──■──
                 └─────────────────┘   │  exp(-it ZZ)(2) │  │
            q_2: ──────────────────────┤1                ├──■──
                                       └─────────────────┘
            q_3: ──────────────────────────────────────────────


        as expected. I.e. the Toffoli is properly adjusted at the end.
        """
        cmap = CouplingMap(couplinglist=[(0, 1), (1, 2)])
        swap_strat = SwapStrategy(cmap, swap_layers=(((0, 1),),))

        pm_ = PassManager(
            [
                FindCommutingPauliEvolutions(),
                Commuting2qGateRouter(swap_strat),
            ]
        )
        op = PauliSumOp.from_list([("IZZ", 1), ("ZIZ", 2)])
        circ = QuantumCircuit(4)
        circ.append(PauliEvolutionGate(op, 1), range(3))
        circ.ccx(0, 2, 1)

        swapped = pm_.run(circ)

        expected = QuantumCircuit(4)
        expected.append(PauliEvolutionGate(Pauli("ZZ"), 1), (0, 1))
        expected.swap(0, 1)
        expected.append(PauliEvolutionGate(Pauli("ZZ"), 2), (1, 2))
        expected.ccx(1, 2, 0)

        self.assertEqual(swapped, expected)


class TestSwapRouterExceptions(QiskitTestCase):
    """Test that exceptions are properly raises."""

    def setUp(self):
        """Setup useful variables."""
        super().setUp()

        # A fully connected problem.
        op = PauliSumOp.from_list(
            [("IIZZ", 1), ("IZIZ", 1), ("ZIIZ", 1), ("IZZI", 1), ("ZIZI", 1), ("ZZII", 1)]
        )
        self.circ = QuantumCircuit(4)
        self.circ.append(PauliEvolutionGate(op, 1), range(4))

        self.swap_cmap = CouplingMap(couplinglist=[(0, 1), (1, 2), (2, 3)])

        # This swap strategy does not reach full connectivity.
        self.swap_strat = SwapStrategy(self.swap_cmap, swap_layers=(((0, 1), (2, 3)),))

    def test_no_swap_strategy(self):
        """Test raise on no swap strategy."""

        pm_ = PassManager([FindCommutingPauliEvolutions(), Commuting2qGateRouter()])

        with self.assertRaises(TranspilerError):
            pm_.run(self.circ)

    def test_dangling_qubits(self):
        """Test that dangling qubits are not allowed."""

        loose = [Qubit() for _ in [None] * 5]
        reg = QuantumRegister(5)
        qc = QuantumCircuit(loose, reg)

        message = "Circuit has qubits not contained in the qubit register."
        with self.assertRaisesRegex(TranspilerError, message):
            Commuting2qGateRouter(self.swap_strat).run(circuit_to_dag(qc))

    def test_too_many_registers(self):
        """Check that we raise if there are too many registers."""
        qc = QuantumCircuit(QuantumRegister(5), QuantumRegister(4))

        message = "Commuting2qGateRouter runs on circuits with one quantum register."
        with self.assertRaisesRegex(TranspilerError, message):
            Commuting2qGateRouter(self.swap_strat).run(circuit_to_dag(qc))

    def test_deficient_swap_strategy(self):
        """Test to raise when all edges cannot be implemented."""

        pm_ = PassManager([FindCommutingPauliEvolutions(), Commuting2qGateRouter()])

        with self.assertRaises(TranspilerError):
            pm_.run(self.circ)
