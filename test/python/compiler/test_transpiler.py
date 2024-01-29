# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests basic functionality of the transpile function"""

import copy
import io
import math
import os
import sys
import unittest
from logging import StreamHandler, getLogger

from test import combine  # pylint: disable=wrong-import-order
from unittest.mock import patch

import numpy as np
import rustworkx as rx
from ddt import data, ddt, unpack

from qiskit import BasicAer, ClassicalRegister, QuantumCircuit, QuantumRegister, pulse, qasm3, qpy
from qiskit.circuit import (
    Clbit,
    ControlFlowOp,
    ForLoopOp,
    Gate,
    IfElseOp,
    Parameter,
    Qubit,
    Reset,
    SwitchCaseOp,
    WhileLoopOp,
)
from qiskit.circuit.annotated_operation import (
    AnnotatedOperation,
    InverseModifier,
    ControlModifier,
    PowerModifier,
)
from qiskit.circuit.classical import expr
from qiskit.circuit.delay import Delay
from qiskit.circuit.library import (
    CXGate,
    CZGate,
    ECRGate,
    HGate,
    RXGate,
    RYGate,
    RZGate,
    SGate,
    SXGate,
    SXdgGate,
    SdgGate,
    U1Gate,
    U2Gate,
    UGate,
    XGate,
    ZGate,
)
from qiskit.circuit.measure import Measure
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode, DAGOutNode
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import BackendV2
from qiskit.providers.fake_provider import (
    FakeBoeblingen,
    FakeMelbourne,
    FakeMumbaiV2,
    FakeNairobiV2,
    FakeRueschlikon,
    FakeSherbrooke,
    FakeVigo,
)
from qiskit.providers.options import Options
from qiskit.pulse import InstructionScheduleMap
from qiskit.quantum_info import Operator, random_unitary
from qiskit.test import QiskitTestCase, slow_test
from qiskit.utils import parallel
from qiskit.transpiler import CouplingMap, Layout, PassManager, TransformationPass
from qiskit.transpiler.exceptions import TranspilerError, CircuitTooWideForTarget
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements, GateDirection, VF2PostLayout
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager, level_0_pass_manager
from qiskit.transpiler.target import InstructionProperties, Target


class CustomCX(Gate):
    """Custom CX gate representation."""

    def __init__(self):
        super().__init__("custom_cx", 2, [])

    def _define(self):
        self._definition = QuantumCircuit(2)
        self._definition.cx(0, 1)


def connected_qubits(physical: int, coupling_map: CouplingMap) -> set:
    """Get the physical qubits that have a connection to this one in the coupling map."""
    for component in coupling_map.connected_components():
        if physical in (qubits := set(component.graph.nodes())):
            return qubits
    raise ValueError(f"physical qubit {physical} is not in the coupling map")


@ddt
class TestTranspile(QiskitTestCase):
    """Test transpile function."""

    def test_empty_transpilation(self):
        """Test that transpiling an empty list is a no-op.  Regression test of gh-7287."""
        self.assertEqual(transpile([]), [])

    def test_pass_manager_none(self):
        """Test passing the default (None) pass manager to the transpiler.

        It should perform the default qiskit flow:
        unroll, swap_mapper, cx_direction, cx_cancellation, optimize_1q_gates
        and should be equivalent to using tools.compile
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])

        coupling_map = [[1, 0]]
        basis_gates = ["u1", "u2", "u3", "cx", "id"]

        backend = BasicAer.get_backend("qasm_simulator")
        circuit2 = transpile(
            circuit,
            backend=backend,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
        )

        circuit3 = transpile(
            circuit, backend=backend, coupling_map=coupling_map, basis_gates=basis_gates
        )
        self.assertEqual(circuit2, circuit3)

    @data(0, 1, 2, 3)
    def test_num_processes_kwarg_concurrent_default(self, num_processes):
        """Test that num_processes kwarg works when the system default parallel is false"""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        target = FakeMumbaiV2().target
        res = transpile([qc] * 3, target=target, num_processes=num_processes)
        self.assertIsInstance(res, list)
        for circ in res:
            self.assertIsInstance(circ, QuantumCircuit)

    def test_transpile_basis_gates_no_backend_no_coupling_map(self):
        """Verify transpile() works with no coupling_map or backend."""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])

        basis_gates = ["u1", "u2", "u3", "cx", "id"]
        circuit2 = transpile(circuit, basis_gates=basis_gates, optimization_level=0)
        resources_after = circuit2.count_ops()
        self.assertEqual({"u2": 2, "cx": 4}, resources_after)

    def test_transpile_non_adjacent_layout(self):
        """Transpile pipeline can handle manual layout on non-adjacent qubits.

        circuit:

        .. parsed-literal::

                  ┌───┐
            qr_0: ┤ H ├──■──────────── -> 1
                  └───┘┌─┴─┐
            qr_1: ─────┤ X ├──■─────── -> 2
                       └───┘┌─┴─┐
            qr_2: ──────────┤ X ├──■── -> 3
                            └───┘┌─┴─┐
            qr_3: ───────────────┤ X ├ -> 5
                                 └───┘

        device:
        0  -  1  -  2  -  3  -  4  -  5  -  6

              |     |     |     |     |     |

              13 -  12  - 11 -  10 -  9  -  8  -   7
        """
        qr = QuantumRegister(4, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[2], qr[3])

        coupling_map = FakeMelbourne().configuration().coupling_map
        basis_gates = FakeMelbourne().configuration().basis_gates
        initial_layout = [None, qr[0], qr[1], qr[2], None, qr[3]]

        new_circuit = transpile(
            circuit,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            initial_layout=initial_layout,
        )

        qubit_indices = {bit: idx for idx, bit in enumerate(new_circuit.qubits)}

        for instruction in new_circuit.data:
            if isinstance(instruction.operation, CXGate):
                self.assertIn([qubit_indices[x] for x in instruction.qubits], coupling_map)

    def test_transpile_qft_grid(self):
        """Transpile pipeline can handle 8-qubit QFT on 14-qubit grid."""
        qr = QuantumRegister(8)
        circuit = QuantumCircuit(qr)
        for i, _ in enumerate(qr):
            for j in range(i):
                circuit.cp(math.pi / float(2 ** (i - j)), qr[i], qr[j])
            circuit.h(qr[i])

        coupling_map = FakeMelbourne().configuration().coupling_map
        basis_gates = FakeMelbourne().configuration().basis_gates
        new_circuit = transpile(circuit, basis_gates=basis_gates, coupling_map=coupling_map)

        qubit_indices = {bit: idx for idx, bit in enumerate(new_circuit.qubits)}

        for instruction in new_circuit.data:
            if isinstance(instruction.operation, CXGate):
                self.assertIn([qubit_indices[x] for x in instruction.qubits], coupling_map)

    def test_already_mapped_1(self):
        """Circuit not remapped if matches topology.

        See: https://github.com/Qiskit/qiskit-terra/issues/342
        """
        backend = FakeRueschlikon()
        coupling_map = backend.configuration().coupling_map
        basis_gates = backend.configuration().basis_gates

        qr = QuantumRegister(16, "qr")
        cr = ClassicalRegister(16, "cr")
        qc = QuantumCircuit(qr, cr)
        qc.cx(qr[3], qr[14])
        qc.cx(qr[5], qr[4])
        qc.h(qr[9])
        qc.cx(qr[9], qr[8])
        qc.x(qr[11])
        qc.cx(qr[3], qr[4])
        qc.cx(qr[12], qr[11])
        qc.cx(qr[13], qr[4])
        qc.measure(qr, cr)

        new_qc = transpile(
            qc,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            initial_layout=Layout.generate_trivial_layout(qr),
        )
        qubit_indices = {bit: idx for idx, bit in enumerate(new_qc.qubits)}
        cx_qubits = [instr.qubits for instr in new_qc.data if instr.operation.name == "cx"]
        cx_qubits_physical = [
            [qubit_indices[ctrl], qubit_indices[tgt]] for [ctrl, tgt] in cx_qubits
        ]
        self.assertEqual(
            sorted(cx_qubits_physical), [[3, 4], [3, 14], [5, 4], [9, 8], [12, 11], [13, 4]]
        )

    def test_already_mapped_via_layout(self):
        """Test that a manual layout that satisfies a coupling map does not get altered.

        See: https://github.com/Qiskit/qiskit-terra/issues/2036

        circuit:

        .. parsed-literal::

                  ┌───┐                  ┌───┐ ░ ┌─┐
            qn_0: ┤ H ├──■────────────■──┤ H ├─░─┤M├─── -> 9
                  └───┘  │            │  └───┘ ░ └╥┘
            qn_1: ───────┼────────────┼────────░──╫──── -> 6
                         │            │        ░  ║
            qn_2: ───────┼────────────┼────────░──╫──── -> 5
                         │            │        ░  ║
            qn_3: ───────┼────────────┼────────░──╫──── -> 0
                         │            │        ░  ║
            qn_4: ───────┼────────────┼────────░──╫──── -> 1
                  ┌───┐┌─┴─┐┌──────┐┌─┴─┐┌───┐ ░  ║ ┌─┐
            qn_5: ┤ H ├┤ X ├┤ P(2) ├┤ X ├┤ H ├─░──╫─┤M├ -> 4
                  └───┘└───┘└──────┘└───┘└───┘ ░  ║ └╥┘
            cn: 2/════════════════════════════════╩══╩═
                                                  0  1

        device:
        0 -- 1 -- 2 -- 3 -- 4
        |                   |
        5 -- 6 -- 7 -- 8 -- 9
        |                   |
        10 - 11 - 12 - 13 - 14
        |                   |
        15 - 16 - 17 - 18 - 19
        """
        basis_gates = ["u1", "u2", "u3", "cx", "id"]
        coupling_map = [
            [0, 1],
            [0, 5],
            [1, 0],
            [1, 2],
            [2, 1],
            [2, 3],
            [3, 2],
            [3, 4],
            [4, 3],
            [4, 9],
            [5, 0],
            [5, 6],
            [5, 10],
            [6, 5],
            [6, 7],
            [7, 6],
            [7, 8],
            [7, 12],
            [8, 7],
            [8, 9],
            [9, 4],
            [9, 8],
            [9, 14],
            [10, 5],
            [10, 11],
            [10, 15],
            [11, 10],
            [11, 12],
            [12, 7],
            [12, 11],
            [12, 13],
            [13, 12],
            [13, 14],
            [14, 9],
            [14, 13],
            [14, 19],
            [15, 10],
            [15, 16],
            [16, 15],
            [16, 17],
            [17, 16],
            [17, 18],
            [18, 17],
            [18, 19],
            [19, 14],
            [19, 18],
        ]

        q = QuantumRegister(6, name="qn")
        c = ClassicalRegister(2, name="cn")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[5])
        qc.cx(q[0], q[5])
        qc.p(2, q[5])
        qc.cx(q[0], q[5])
        qc.h(q[0])
        qc.h(q[5])
        qc.barrier(q)
        qc.measure(q[0], c[0])
        qc.measure(q[5], c[1])

        initial_layout = [
            q[3],
            q[4],
            None,
            None,
            q[5],
            q[2],
            q[1],
            None,
            None,
            q[0],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]

        new_qc = transpile(
            qc, coupling_map=coupling_map, basis_gates=basis_gates, initial_layout=initial_layout
        )
        qubit_indices = {bit: idx for idx, bit in enumerate(new_qc.qubits)}
        cx_qubits = [instr.qubits for instr in new_qc.data if instr.operation.name == "cx"]
        cx_qubits_physical = [
            [qubit_indices[ctrl], qubit_indices[tgt]] for [ctrl, tgt] in cx_qubits
        ]
        self.assertEqual(sorted(cx_qubits_physical), [[9, 4], [9, 4]])

    def test_transpile_bell(self):
        """Test Transpile Bell.

        If all correct some should exists.
        """
        backend = BasicAer.get_backend("qasm_simulator")

        qubit_reg = QuantumRegister(2, name="q")
        clbit_reg = ClassicalRegister(2, name="c")
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        circuits = transpile(qc, backend)
        self.assertIsInstance(circuits, QuantumCircuit)

    def test_transpile_bell_discrete_basis(self):
        """Test that it's possible to transpile a very simple circuit to a discrete stabiliser-like
        basis.  In general, we do not make any guarantees about the possibility or quality of
        transpilation in these situations, but this is at least useful as a check that stuff that
        _could_ be possible remains so."""

        target = Target(num_qubits=2)
        for one_q in [XGate(), SXGate(), SXdgGate(), SGate(), SdgGate(), ZGate()]:
            target.add_instruction(one_q, {(0,): None, (1,): None})
        # This is only in one direction, and not the direction we're going to attempt to lay it out
        # onto, so we can test the basis translation.
        target.add_instruction(ECRGate(), {(1, 0): None})

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        # Try with the initial layout in both directions to ensure we're dealing with the basis
        # having only a single direction.
        self.assertIsInstance(transpile(qc, target=target, initial_layout=[0, 1]), QuantumCircuit)
        self.assertIsInstance(transpile(qc, target=target, initial_layout=[1, 0]), QuantumCircuit)

    def test_transpile_one(self):
        """Test transpile a single circuit.

        Check that the top-level `transpile` function returns
        a single circuit."""
        backend = BasicAer.get_backend("qasm_simulator")

        qubit_reg = QuantumRegister(2)
        clbit_reg = ClassicalRegister(2)
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        circuit = transpile(qc, backend)
        self.assertIsInstance(circuit, QuantumCircuit)

    def test_transpile_two(self):
        """Test transpile two circuits.

        Check that the transpiler returns a list of two circuits.
        """
        backend = BasicAer.get_backend("qasm_simulator")

        qubit_reg = QuantumRegister(2)
        clbit_reg = ClassicalRegister(2)
        qubit_reg2 = QuantumRegister(2)
        clbit_reg2 = ClassicalRegister(2)
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = QuantumCircuit(qubit_reg, qubit_reg2, clbit_reg, clbit_reg2, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        circuits = transpile([qc, qc_extra], backend)
        self.assertIsInstance(circuits, list)
        self.assertEqual(len(circuits), 2)

        for circuit in circuits:
            self.assertIsInstance(circuit, QuantumCircuit)

    def test_transpile_singleton(self):
        """Test transpile a single-element list with a circuit.

        Check that `transpile` returns a single-element list.

        See https://github.com/Qiskit/qiskit-terra/issues/5260
        """
        backend = BasicAer.get_backend("qasm_simulator")

        qubit_reg = QuantumRegister(2)
        clbit_reg = ClassicalRegister(2)
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        circuits = transpile([qc], backend)
        self.assertIsInstance(circuits, list)
        self.assertEqual(len(circuits), 1)
        self.assertIsInstance(circuits[0], QuantumCircuit)

    def test_mapping_correction(self):
        """Test mapping works in previous failed case."""
        backend = FakeRueschlikon()
        qr = QuantumRegister(name="qr", size=11)
        cr = ClassicalRegister(name="qc", size=11)
        circuit = QuantumCircuit(qr, cr)
        circuit.u(1.564784764685993, -1.2378965763410095, 2.9746763177861713, qr[3])
        circuit.u(1.2269835563676523, 1.1932982847014162, -1.5597357740824318, qr[5])
        circuit.cx(qr[5], qr[3])
        circuit.p(0.856768317675967, qr[3])
        circuit.u(-3.3911273825190915, 0.0, 0.0, qr[5])
        circuit.cx(qr[3], qr[5])
        circuit.u(2.159209321625547, 0.0, 0.0, qr[5])
        circuit.cx(qr[5], qr[3])
        circuit.u(0.30949966910232335, 1.1706201763833217, 1.738408691990081, qr[3])
        circuit.u(1.9630571407274755, -0.6818742967975088, 1.8336534616728195, qr[5])
        circuit.u(1.330181833806101, 0.6003162754946363, -3.181264980452862, qr[7])
        circuit.u(0.4885914820775024, 3.133297443244865, -2.794457469189904, qr[8])
        circuit.cx(qr[8], qr[7])
        circuit.p(2.2196187596178616, qr[7])
        circuit.u(-3.152367609631023, 0.0, 0.0, qr[8])
        circuit.cx(qr[7], qr[8])
        circuit.u(1.2646005789809263, 0.0, 0.0, qr[8])
        circuit.cx(qr[8], qr[7])
        circuit.u(0.7517780502091939, 1.2828514296564781, 1.6781179605443775, qr[7])
        circuit.u(0.9267400575390405, 2.0526277839695153, 2.034202361069533, qr[8])
        circuit.u(2.550304293455634, 3.8250017126569698, -2.1351609599720054, qr[1])
        circuit.u(0.9566260876600556, -1.1147561503064538, 2.0571590492298797, qr[4])
        circuit.cx(qr[4], qr[1])
        circuit.p(2.1899329069137394, qr[1])
        circuit.u(-1.8371715243173294, 0.0, 0.0, qr[4])
        circuit.cx(qr[1], qr[4])
        circuit.u(0.4717053496327104, 0.0, 0.0, qr[4])
        circuit.cx(qr[4], qr[1])
        circuit.u(2.3167620677708145, -1.2337330260253256, -0.5671322899563955, qr[1])
        circuit.u(1.0468499525240678, 0.8680750644809365, -1.4083720073192485, qr[4])
        circuit.u(2.4204244021892807, -2.211701932616922, 3.8297006565735883, qr[10])
        circuit.u(0.36660280497727255, 3.273119149343493, -1.8003362351299388, qr[6])
        circuit.cx(qr[6], qr[10])
        circuit.p(1.067395863586385, qr[10])
        circuit.u(-0.7044917541291232, 0.0, 0.0, qr[6])
        circuit.cx(qr[10], qr[6])
        circuit.u(2.1830003849921527, 0.0, 0.0, qr[6])
        circuit.cx(qr[6], qr[10])
        circuit.u(2.1538343756723917, 2.2653381826084606, -3.550087952059485, qr[10])
        circuit.u(1.307627685019188, -0.44686656993522567, -2.3238098554327418, qr[6])
        circuit.u(2.2046797998462906, 0.9732961754855436, 1.8527865921467421, qr[9])
        circuit.u(2.1665254613904126, -1.281337664694577, -1.2424905413631209, qr[0])
        circuit.cx(qr[0], qr[9])
        circuit.p(2.6209599970201007, qr[9])
        circuit.u(0.04680566321901303, 0.0, 0.0, qr[0])
        circuit.cx(qr[9], qr[0])
        circuit.u(1.7728411151289603, 0.0, 0.0, qr[0])
        circuit.cx(qr[0], qr[9])
        circuit.u(2.4866395967434443, 0.48684511243566697, -3.0069186877854728, qr[9])
        circuit.u(1.7369112924273789, -4.239660866163805, 1.0623389015296005, qr[0])
        circuit.barrier(qr)
        circuit.measure(qr, cr)

        circuits = transpile(circuit, backend)

        self.assertIsInstance(circuits, QuantumCircuit)

    def test_transpiler_layout_from_intlist(self):
        """A list of ints gives layout to correctly map circuit.
        virtual  physical
         q1_0  -  4   ---[H]---
         q2_0  -  5
         q2_1  -  6   ---[H]---
         q3_0  -  8
         q3_1  -  9
         q3_2  -  10  ---[H]---

        """
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        qr3 = QuantumRegister(3, "qr3")
        qc = QuantumCircuit(qr1, qr2, qr3)
        qc.h(qr1[0])
        qc.h(qr2[1])
        qc.h(qr3[2])
        layout = [4, 5, 6, 8, 9, 10]

        cmap = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [7, 8],
            [9, 8],
            [9, 10],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]

        new_circ = transpile(
            qc, backend=None, coupling_map=cmap, basis_gates=["u2"], initial_layout=layout
        )
        qubit_indices = {bit: idx for idx, bit in enumerate(new_circ.qubits)}
        mapped_qubits = []

        for instruction in new_circ.data:
            mapped_qubits.append(qubit_indices[instruction.qubits[0]])

        self.assertEqual(mapped_qubits, [4, 6, 10])

    def test_mapping_multi_qreg(self):
        """Test mapping works for multiple qregs."""
        backend = FakeRueschlikon()
        qr = QuantumRegister(3, name="qr")
        qr2 = QuantumRegister(1, name="qr2")
        qr3 = QuantumRegister(4, name="qr3")
        cr = ClassicalRegister(3, name="cr")
        qc = QuantumCircuit(qr, qr2, qr3, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr2[0])
        qc.cx(qr[1], qr3[2])
        qc.measure(qr, cr)

        circuits = transpile(qc, backend)

        self.assertIsInstance(circuits, QuantumCircuit)

    def test_transpile_circuits_diff_registers(self):
        """Transpile list of circuits with different qreg names."""
        backend = FakeRueschlikon()
        circuits = []
        for _ in range(2):
            qr = QuantumRegister(2)
            cr = ClassicalRegister(2)
            circuit = QuantumCircuit(qr, cr)
            circuit.h(qr[0])
            circuit.cx(qr[0], qr[1])
            circuit.measure(qr, cr)
            circuits.append(circuit)

        circuits = transpile(circuits, backend)
        self.assertIsInstance(circuits[0], QuantumCircuit)

    def test_wrong_initial_layout(self):
        """Test transpile with a bad initial layout."""
        backend = FakeMelbourne()

        qubit_reg = QuantumRegister(2, name="q")
        clbit_reg = ClassicalRegister(2, name="c")
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        bad_initial_layout = [
            QuantumRegister(3, "q")[0],
            QuantumRegister(3, "q")[1],
            QuantumRegister(3, "q")[2],
        ]

        with self.assertRaises(TranspilerError):
            transpile(qc, backend, initial_layout=bad_initial_layout)

    def test_parameterized_circuit_for_simulator(self):
        """Verify that a parameterized circuit can be transpiled for a simulator backend."""
        qr = QuantumRegister(2, name="qr")
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")
        qc.rz(theta, qr[0])

        transpiled_qc = transpile(qc, backend=BasicAer.get_backend("qasm_simulator"))

        expected_qc = QuantumCircuit(qr)
        expected_qc.append(RZGate(theta), [qr[0]])
        self.assertEqual(expected_qc, transpiled_qc)

    def test_parameterized_circuit_for_device(self):
        """Verify that a parameterized circuit can be transpiled for a device backend."""
        qr = QuantumRegister(2, name="qr")
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")
        qc.rz(theta, qr[0])

        transpiled_qc = transpile(
            qc, backend=FakeMelbourne(), initial_layout=Layout.generate_trivial_layout(qr)
        )

        qr = QuantumRegister(14, "q")
        expected_qc = QuantumCircuit(qr, global_phase=-1 * theta / 2.0)
        expected_qc.append(U1Gate(theta), [qr[0]])

        self.assertEqual(expected_qc, transpiled_qc)

    def test_parameter_expression_circuit_for_simulator(self):
        """Verify that a circuit including expressions of parameters can be
        transpiled for a simulator backend."""
        qr = QuantumRegister(2, name="qr")
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")
        square = theta * theta
        qc.rz(square, qr[0])

        transpiled_qc = transpile(qc, backend=BasicAer.get_backend("qasm_simulator"))

        expected_qc = QuantumCircuit(qr)
        expected_qc.append(RZGate(square), [qr[0]])
        self.assertEqual(expected_qc, transpiled_qc)

    def test_parameter_expression_circuit_for_device(self):
        """Verify that a circuit including expressions of parameters can be
        transpiled for a device backend."""
        qr = QuantumRegister(2, name="qr")
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")
        square = theta * theta
        qc.rz(square, qr[0])

        transpiled_qc = transpile(
            qc, backend=FakeMelbourne(), initial_layout=Layout.generate_trivial_layout(qr)
        )

        qr = QuantumRegister(14, "q")
        expected_qc = QuantumCircuit(qr, global_phase=-1 * square / 2.0)
        expected_qc.append(U1Gate(square), [qr[0]])
        self.assertEqual(expected_qc, transpiled_qc)

    def test_final_measurement_barrier_for_devices(self):
        """Verify BarrierBeforeFinalMeasurements pass is called in default pipeline for devices."""
        qasm_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qasm")
        circ = QuantumCircuit.from_qasm_file(os.path.join(qasm_dir, "example.qasm"))
        layout = Layout.generate_trivial_layout(*circ.qregs)
        orig_pass = BarrierBeforeFinalMeasurements()
        with patch.object(BarrierBeforeFinalMeasurements, "run", wraps=orig_pass.run) as mock_pass:
            transpile(
                circ,
                coupling_map=FakeRueschlikon().configuration().coupling_map,
                initial_layout=layout,
            )
            self.assertTrue(mock_pass.called)

    def test_do_not_run_gatedirection_with_symmetric_cm(self):
        """When the coupling map is symmetric, do not run GateDirection."""
        qasm_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qasm")
        circ = QuantumCircuit.from_qasm_file(os.path.join(qasm_dir, "example.qasm"))
        layout = Layout.generate_trivial_layout(*circ.qregs)
        coupling_map = []
        for node1, node2 in FakeRueschlikon().configuration().coupling_map:
            coupling_map.append([node1, node2])
            coupling_map.append([node2, node1])

        orig_pass = GateDirection(CouplingMap(coupling_map))
        with patch.object(GateDirection, "run", wraps=orig_pass.run) as mock_pass:
            transpile(circ, coupling_map=coupling_map, initial_layout=layout)
            self.assertFalse(mock_pass.called)

    def test_optimize_to_nothing(self):
        """Optimize gates up to fixed point in the default pipeline
        See https://github.com/Qiskit/qiskit-terra/issues/2035
        """
        #       ┌───┐     ┌───┐┌───┐┌───┐     ┌───┐
        # q0_0: ┤ H ├──■──┤ X ├┤ Y ├┤ Z ├──■──┤ H ├──■────■──
        #       └───┘┌─┴─┐└───┘└───┘└───┘┌─┴─┐└───┘┌─┴─┐┌─┴─┐
        # q0_1: ─────┤ X ├───────────────┤ X ├─────┤ X ├┤ X ├
        #            └───┘               └───┘     └───┘└───┘
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.x(qr[0])
        circ.y(qr[0])
        circ.z(qr[0])
        circ.cx(qr[0], qr[1])
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.cx(qr[0], qr[1])

        after = transpile(circ, coupling_map=[[0, 1], [1, 0]], basis_gates=["u3", "u2", "u1", "cx"])

        expected = QuantumCircuit(QuantumRegister(2, "q"), global_phase=-np.pi / 2)
        msg = f"after:\n{after}\nexpected:\n{expected}"
        self.assertEqual(after, expected, msg=msg)

    def test_pass_manager_empty(self):
        """Test passing an empty PassManager() to the transpiler.

        It should perform no transformations on the circuit.
        """
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        resources_before = circuit.count_ops()

        pass_manager = PassManager()
        out_circuit = pass_manager.run(circuit)
        resources_after = out_circuit.count_ops()

        self.assertDictEqual(resources_before, resources_after)

    def test_move_measurements(self):
        """Measurements applied AFTER swap mapping."""
        backend = FakeRueschlikon()
        cmap = backend.configuration().coupling_map
        qasm_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qasm")
        circ = QuantumCircuit.from_qasm_file(os.path.join(qasm_dir, "move_measurements.qasm"))

        lay = [0, 1, 15, 2, 14, 3, 13, 4, 12, 5, 11, 6]
        out = transpile(circ, initial_layout=lay, coupling_map=cmap, routing_method="stochastic")
        out_dag = circuit_to_dag(out)
        meas_nodes = out_dag.named_nodes("measure")
        for meas_node in meas_nodes:
            is_last_measure = all(
                isinstance(after_measure, DAGOutNode)
                for after_measure in out_dag.quantum_successors(meas_node)
            )
            self.assertTrue(is_last_measure)

    @data(0, 1, 2, 3)
    def test_init_resets_kept_preset_passmanagers(self, optimization_level):
        """Test initial resets kept at all preset transpilation levels"""
        num_qubits = 5
        qc = QuantumCircuit(num_qubits)
        qc.reset(range(num_qubits))

        num_resets = transpile(qc, optimization_level=optimization_level).count_ops()["reset"]
        self.assertEqual(num_resets, num_qubits)

    @data(0, 1, 2, 3)
    def test_initialize_reset_is_not_removed(self, optimization_level):
        """The reset in front of initializer should NOT be removed at beginning"""
        qr = QuantumRegister(1, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], [qr[0]])
        qc.initialize([1.0 / math.sqrt(2), -1.0 / math.sqrt(2)], [qr[0]])

        after = transpile(qc, basis_gates=["reset", "u3"], optimization_level=optimization_level)
        self.assertEqual(after.count_ops()["reset"], 2, msg=f"{after}\n does not have 2 resets.")

    def test_initialize_FakeMelbourne(self):
        """Test that the zero-state resets are remove in a device not supporting them."""
        desired_vector = [1 / math.sqrt(2), 0, 0, 0, 0, 0, 0, 1 / math.sqrt(2)]
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2]])

        out = transpile(qc, backend=FakeMelbourne())
        out_dag = circuit_to_dag(out)
        reset_nodes = out_dag.named_nodes("reset")

        self.assertEqual(len(reset_nodes), 3)

    def test_non_standard_basis(self):
        """Test a transpilation with a non-standard basis"""
        qr1 = QuantumRegister(1, "q1")
        qr2 = QuantumRegister(2, "q2")
        qr3 = QuantumRegister(3, "q3")
        qc = QuantumCircuit(qr1, qr2, qr3)
        qc.h(qr1[0])
        qc.h(qr2[1])
        qc.h(qr3[2])
        layout = [4, 5, 6, 8, 9, 10]

        cmap = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [7, 8],
            [9, 8],
            [9, 10],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]

        circuit = transpile(
            qc, backend=None, coupling_map=cmap, basis_gates=["h"], initial_layout=layout
        )

        dag_circuit = circuit_to_dag(circuit)
        resources_after = dag_circuit.count_ops()
        self.assertEqual({"h": 3}, resources_after)

    def test_hadamard_to_rot_gates(self):
        """Test a transpilation from H to Rx, Ry gates"""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.h(0)

        expected = QuantumCircuit(qr, global_phase=np.pi / 2)
        expected.append(RYGate(theta=np.pi / 2), [0])
        expected.append(RXGate(theta=np.pi), [0])

        circuit = transpile(qc, basis_gates=["rx", "ry"], optimization_level=0)
        self.assertEqual(circuit, expected)

    def test_basis_subset(self):
        """Test a transpilation with a basis subset of the standard basis"""
        qr = QuantumRegister(1, "q1")
        qc = QuantumCircuit(qr)
        qc.h(qr[0])
        qc.x(qr[0])
        qc.t(qr[0])

        layout = [4]

        cmap = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [7, 8],
            [9, 8],
            [9, 10],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]

        circuit = transpile(
            qc, backend=None, coupling_map=cmap, basis_gates=["u3"], initial_layout=layout
        )

        dag_circuit = circuit_to_dag(circuit)
        resources_after = dag_circuit.count_ops()
        self.assertEqual({"u3": 1}, resources_after)

    def test_check_circuit_width(self):
        """Verify transpilation of circuit with virtual qubits greater than
        physical qubits raises error"""
        cmap = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [7, 8],
            [9, 8],
            [9, 10],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]

        qc = QuantumCircuit(15, 15)

        with self.assertRaises(CircuitTooWideForTarget):
            transpile(qc, coupling_map=cmap)

    @data(0, 1, 2, 3)
    def test_ccx_routing_method_none(self, optimization_level):
        """CCX without routing method."""

        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)

        out = transpile(
            qc,
            routing_method="none",
            basis_gates=["u", "cx"],
            initial_layout=[0, 1, 2],
            seed_transpiler=0,
            coupling_map=[[0, 1], [1, 2]],
            optimization_level=optimization_level,
        )

        self.assertTrue(Operator(qc).equiv(out))

    @data(0, 1, 2, 3)
    def test_ccx_routing_method_none_failed(self, optimization_level):
        """CCX without routing method cannot be routed."""

        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)

        with self.assertRaises(TranspilerError):
            transpile(
                qc,
                routing_method="none",
                basis_gates=["u", "cx"],
                initial_layout=[0, 1, 2],
                seed_transpiler=0,
                coupling_map=[[0, 1], [1, 2]],
                optimization_level=optimization_level,
            )

    @data(0, 1, 2, 3)
    def test_ms_unrolls_to_cx(self, optimization_level):
        """Verify a Rx,Ry,Rxx circuit transpile to a U3,CX target."""

        qc = QuantumCircuit(2)
        qc.rx(math.pi / 2, 0)
        qc.ry(math.pi / 4, 1)
        qc.rxx(math.pi / 4, 0, 1)

        out = transpile(qc, basis_gates=["u3", "cx"], optimization_level=optimization_level)

        self.assertTrue(Operator(qc).equiv(out))

    @data(0, 1, 2, 3)
    def test_ms_can_target_ms(self, optimization_level):
        """Verify a Rx,Ry,Rxx circuit can transpile to an Rx,Ry,Rxx target."""

        qc = QuantumCircuit(2)
        qc.rx(math.pi / 2, 0)
        qc.ry(math.pi / 4, 1)
        qc.rxx(math.pi / 4, 0, 1)

        out = transpile(qc, basis_gates=["rx", "ry", "rxx"], optimization_level=optimization_level)

        self.assertTrue(Operator(qc).equiv(out))

    @data(0, 1, 2, 3)
    def test_cx_can_target_ms(self, optimization_level):
        """Verify a U3,CX circuit can transpiler to a Rx,Ry,Rxx target."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(math.pi / 4, [0, 1])

        out = transpile(qc, basis_gates=["rx", "ry", "rxx"], optimization_level=optimization_level)

        self.assertTrue(Operator(qc).equiv(out))

    @data(0, 1, 2, 3)
    def test_measure_doesnt_unroll_ms(self, optimization_level):
        """Verify a measure doesn't cause an Rx,Ry,Rxx circuit to unroll to U3,CX."""

        qc = QuantumCircuit(2, 2)
        qc.rx(math.pi / 2, 0)
        qc.ry(math.pi / 4, 1)
        qc.rxx(math.pi / 4, 0, 1)
        qc.measure([0, 1], [0, 1])
        out = transpile(qc, basis_gates=["rx", "ry", "rxx"], optimization_level=optimization_level)

        self.assertEqual(qc, out)

    @data(
        ["cx", "u3"],
        ["cz", "u3"],
        ["cz", "rx", "rz"],
        ["rxx", "rx", "ry"],
        ["iswap", "rx", "rz"],
    )
    def test_block_collection_runs_for_non_cx_bases(self, basis_gates):
        """Verify block collection is run when a single two qubit gate is in the basis."""
        twoq_gate, *_ = basis_gates

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.cx(0, 1)
        qc.cx(0, 1)

        out = transpile(qc, basis_gates=basis_gates, optimization_level=3)

        self.assertLessEqual(out.count_ops()[twoq_gate], 2)

    @unpack
    @data(
        (["u3", "cx"], {"u3": 1, "cx": 1}),
        (["rx", "rz", "iswap"], {"rx": 6, "rz": 12, "iswap": 2}),
        (["rx", "ry", "rxx"], {"rx": 6, "ry": 5, "rxx": 1}),
    )
    def test_block_collection_reduces_1q_gate(self, basis_gates, gate_counts):
        """For synthesis to non-U3 bases, verify we minimize 1q gates."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        out = transpile(qc, basis_gates=basis_gates, optimization_level=3)

        self.assertTrue(Operator(out).equiv(qc))
        self.assertTrue(set(out.count_ops()).issubset(basis_gates))
        for basis_gate in basis_gates:
            self.assertLessEqual(out.count_ops()[basis_gate], gate_counts[basis_gate])

    @combine(
        optimization_level=[0, 1, 2, 3],
        basis_gates=[
            ["u3", "cx"],
            ["rx", "rz", "iswap"],
            ["rx", "ry", "rxx"],
        ],
    )
    def test_translation_method_synthesis(self, optimization_level, basis_gates):
        """Verify translation_method='synthesis' gets to the basis."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        out = transpile(
            qc,
            translation_method="synthesis",
            basis_gates=basis_gates,
            optimization_level=optimization_level,
        )

        self.assertTrue(Operator(out).equiv(qc))
        self.assertTrue(set(out.count_ops()).issubset(basis_gates))

    def test_transpiled_custom_gates_calibration(self):
        """Test if transpiled calibrations is equal to custom gates circuit calibrations."""
        custom_180 = Gate("mycustom", 1, [3.14])
        custom_90 = Gate("mycustom", 1, [1.57])

        circ = QuantumCircuit(2)
        circ.append(custom_180, [0])
        circ.append(custom_90, [1])

        with pulse.build() as q0_x180:
            pulse.play(pulse.library.Gaussian(20, 1.0, 3.0), pulse.DriveChannel(0))
        with pulse.build() as q1_y90:
            pulse.play(pulse.library.Gaussian(20, -1.0, 3.0), pulse.DriveChannel(1))

        # Add calibration
        circ.add_calibration(custom_180, [0], q0_x180)
        circ.add_calibration(custom_90, [1], q1_y90)

        backend = FakeBoeblingen()
        transpiled_circuit = transpile(
            circ,
            backend=backend,
            layout_method="trivial",
        )
        self.assertEqual(transpiled_circuit.calibrations, circ.calibrations)
        self.assertEqual(list(transpiled_circuit.count_ops().keys()), ["mycustom"])
        self.assertEqual(list(transpiled_circuit.count_ops().values()), [2])

    def test_transpiled_basis_gates_calibrations(self):
        """Test if the transpiled calibrations is equal to basis gates circuit calibrations."""
        circ = QuantumCircuit(2)
        circ.h(0)

        with pulse.build() as q0_x180:
            pulse.play(pulse.library.Gaussian(20, 1.0, 3.0), pulse.DriveChannel(0))

        # Add calibration
        circ.add_calibration("h", [0], q0_x180)

        backend = FakeBoeblingen()
        transpiled_circuit = transpile(
            circ,
            backend=backend,
        )
        self.assertEqual(transpiled_circuit.calibrations, circ.calibrations)

    def test_transpile_calibrated_custom_gate_on_diff_qubit(self):
        """Test if the custom, non calibrated gate raises QiskitError."""
        custom_180 = Gate("mycustom", 1, [3.14])

        circ = QuantumCircuit(2)
        circ.append(custom_180, [0])

        with pulse.build() as q0_x180:
            pulse.play(pulse.library.Gaussian(20, 1.0, 3.0), pulse.DriveChannel(0))

        # Add calibration
        circ.add_calibration(custom_180, [1], q0_x180)

        backend = FakeBoeblingen()
        with self.assertRaises(QiskitError):
            transpile(circ, backend=backend, layout_method="trivial")

    def test_transpile_calibrated_nonbasis_gate_on_diff_qubit(self):
        """Test if the non-basis gates are transpiled if they are on different qubit that
        is not calibrated."""
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.h(1)

        with pulse.build() as q0_x180:
            pulse.play(pulse.library.Gaussian(20, 1.0, 3.0), pulse.DriveChannel(0))

        # Add calibration
        circ.add_calibration("h", [1], q0_x180)

        backend = FakeBoeblingen()
        transpiled_circuit = transpile(
            circ,
            backend=backend,
        )
        self.assertEqual(transpiled_circuit.calibrations, circ.calibrations)
        self.assertEqual(set(transpiled_circuit.count_ops().keys()), {"u2", "h"})

    def test_transpile_subset_of_calibrated_gates(self):
        """Test transpiling a circuit with both basis gate (not-calibrated) and
        a calibrated gate on different qubits."""
        x_180 = Gate("mycustom", 1, [3.14])

        circ = QuantumCircuit(2)
        circ.h(0)
        circ.append(x_180, [0])
        circ.h(1)

        with pulse.build() as q0_x180:
            pulse.play(pulse.library.Gaussian(20, 1.0, 3.0), pulse.DriveChannel(0))

        circ.add_calibration(x_180, [0], q0_x180)
        circ.add_calibration("h", [1], q0_x180)  # 'h' is calibrated on qubit 1

        transpiled_circ = transpile(circ, FakeBoeblingen(), layout_method="trivial")
        self.assertEqual(set(transpiled_circ.count_ops().keys()), {"u2", "mycustom", "h"})

    def test_parameterized_calibrations_transpile(self):
        """Check that gates can be matched to their calibrations before and after parameter
        assignment."""
        tau = Parameter("tau")
        circ = QuantumCircuit(3, 3)
        circ.append(Gate("rxt", 1, [2 * 3.14 * tau]), [0])

        def q0_rxt(tau):
            with pulse.build() as q0_rxt:
                pulse.play(pulse.library.Gaussian(20, 0.4 * tau, 3.0), pulse.DriveChannel(0))
            return q0_rxt

        circ.add_calibration("rxt", [0], q0_rxt(tau), [2 * 3.14 * tau])

        transpiled_circ = transpile(circ, FakeBoeblingen(), layout_method="trivial")
        self.assertEqual(set(transpiled_circ.count_ops().keys()), {"rxt"})
        circ = circ.assign_parameters({tau: 1})
        transpiled_circ = transpile(circ, FakeBoeblingen(), layout_method="trivial")
        self.assertEqual(set(transpiled_circ.count_ops().keys()), {"rxt"})

    def test_inst_durations_from_calibrations(self):
        """Test that circuit calibrations can be used instead of explicitly
        supplying inst_durations.
        """
        qc = QuantumCircuit(2)
        qc.append(Gate("custom", 1, []), [0])

        with pulse.build() as cal:
            pulse.play(pulse.library.Gaussian(20, 1.0, 3.0), pulse.DriveChannel(0))
        qc.add_calibration("custom", [0], cal)

        out = transpile(qc, scheduling_method="alap")
        self.assertEqual(out.duration, cal.duration)

    @data(0, 1, 2, 3)
    def test_multiqubit_gates_calibrations(self, opt_level):
        """Test multiqubit gate > 2q with calibrations works

        Adapted from issue description in https://github.com/Qiskit/qiskit-terra/issues/6572
        """
        circ = QuantumCircuit(5)
        custom_gate = Gate("my_custom_gate", 5, [])
        circ.append(custom_gate, [0, 1, 2, 3, 4])
        circ.measure_all()
        backend = FakeBoeblingen()
        with pulse.build(backend, name="custom") as my_schedule:
            pulse.play(
                pulse.library.Gaussian(duration=128, amp=0.1, sigma=16), pulse.drive_channel(0)
            )
            pulse.play(
                pulse.library.Gaussian(duration=128, amp=0.1, sigma=16), pulse.drive_channel(1)
            )
            pulse.play(
                pulse.library.Gaussian(duration=128, amp=0.1, sigma=16), pulse.drive_channel(2)
            )
            pulse.play(
                pulse.library.Gaussian(duration=128, amp=0.1, sigma=16), pulse.drive_channel(3)
            )
            pulse.play(
                pulse.library.Gaussian(duration=128, amp=0.1, sigma=16), pulse.drive_channel(4)
            )
            pulse.play(
                pulse.library.Gaussian(duration=128, amp=0.1, sigma=16), pulse.ControlChannel(1)
            )
            pulse.play(
                pulse.library.Gaussian(duration=128, amp=0.1, sigma=16), pulse.ControlChannel(2)
            )
            pulse.play(
                pulse.library.Gaussian(duration=128, amp=0.1, sigma=16), pulse.ControlChannel(3)
            )
            pulse.play(
                pulse.library.Gaussian(duration=128, amp=0.1, sigma=16), pulse.ControlChannel(4)
            )
        circ.add_calibration("my_custom_gate", [0, 1, 2, 3, 4], my_schedule, [])
        trans_circ = transpile(circ, backend, optimization_level=opt_level, layout_method="trivial")
        self.assertEqual({"measure": 5, "my_custom_gate": 1, "barrier": 1}, trans_circ.count_ops())

    @data(0, 1, 2, 3)
    def test_circuit_with_delay(self, optimization_level):
        """Verify a circuit with delay can transpile to a scheduled circuit."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)

        out = transpile(
            qc,
            scheduling_method="alap",
            basis_gates=["h", "cx"],
            instruction_durations=[("h", 0, 200), ("cx", [0, 1], 700)],
            optimization_level=optimization_level,
        )

        self.assertEqual(out.duration, 1200)

    def test_delay_converts_to_dt(self):
        """Test that a delay instruction is converted to units of dt given a backend."""
        qc = QuantumCircuit(2)
        qc.delay(1000, [0], unit="us")

        backend = FakeRueschlikon()
        backend.configuration().dt = 0.5e-6
        out = transpile([qc, qc], backend)
        self.assertEqual(out[0].data[0].operation.unit, "dt")
        self.assertEqual(out[1].data[0].operation.unit, "dt")

        out = transpile(qc, dt=1e-9)
        self.assertEqual(out.data[0].operation.unit, "dt")

    def test_scheduling_backend_v2(self):
        """Test that scheduling method works with Backendv2."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        backend = FakeMumbaiV2()
        out = transpile([qc, qc], backend, scheduling_method="alap")
        self.assertIn("delay", out[0].count_ops())
        self.assertIn("delay", out[1].count_ops())

    @data(1, 2, 3)
    def test_no_infinite_loop(self, optimization_level):
        """Verify circuit cost always descends and optimization does not flip flop indefinitely."""
        qc = QuantumCircuit(1)
        qc.ry(0.2, 0)

        out = transpile(
            qc, basis_gates=["id", "p", "sx", "cx"], optimization_level=optimization_level
        )

        # Expect a -pi/2 global phase for the U3 to RZ/SX conversion, and
        # a -0.5 * theta phase for RZ to P twice, once at theta, and once at 3 pi
        # for the second and third RZ gates in the U3 decomposition.
        expected = QuantumCircuit(
            1, global_phase=-np.pi / 2 - 0.5 * (-0.2 + np.pi) - 0.5 * 3 * np.pi
        )
        expected.p(-np.pi, 0)
        expected.sx(0)
        expected.p(np.pi - 0.2, 0)
        expected.sx(0)

        error_message = (
            f"\nOutput circuit:\n{out!s}\n{Operator(out).data}\n"
            f"Expected circuit:\n{expected!s}\n{Operator(expected).data}"
        )
        self.assertEqual(out, expected, error_message)

    @data(0, 1, 2, 3)
    def test_transpile_preserves_circuit_metadata(self, optimization_level):
        """Verify that transpile preserves circuit metadata in the output."""
        circuit = QuantumCircuit(2, metadata={"experiment_id": "1234", "execution_number": 4})
        circuit.h(0)
        circuit.cx(0, 1)

        cmap = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [7, 8],
            [9, 8],
            [9, 10],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]

        res = transpile(
            circuit,
            basis_gates=["id", "p", "sx", "cx"],
            coupling_map=cmap,
            optimization_level=optimization_level,
        )
        self.assertEqual(circuit.metadata, res.metadata)

    @data(0, 1, 2, 3)
    def test_transpile_optional_registers(self, optimization_level):
        """Verify transpile accepts circuits without registers end-to-end."""

        qubits = [Qubit() for _ in range(3)]
        clbits = [Clbit() for _ in range(3)]

        qc = QuantumCircuit(qubits, clbits)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        qc.measure(qubits, clbits)

        out = transpile(qc, FakeBoeblingen(), optimization_level=optimization_level)

        self.assertEqual(len(out.qubits), FakeBoeblingen().configuration().num_qubits)
        self.assertEqual(len(out.clbits), len(clbits))

    @data(0, 1, 2, 3)
    def test_translate_ecr_basis(self, optimization_level):
        """Verify that rewriting in ECR basis is efficient."""
        circuit = QuantumCircuit(2)
        circuit.append(random_unitary(4, seed=1), [0, 1])
        circuit.barrier()
        circuit.cx(0, 1)
        circuit.barrier()
        circuit.swap(0, 1)
        circuit.barrier()
        circuit.iswap(0, 1)

        res = transpile(circuit, basis_gates=["u", "ecr"], optimization_level=optimization_level)
        self.assertEqual(res.count_ops()["ecr"], 9)
        self.assertTrue(Operator(res).equiv(circuit))

    def test_optimize_ecr_basis(self):
        """Test highest optimization level can optimize over ECR."""
        circuit = QuantumCircuit(2)
        circuit.swap(1, 0)
        circuit.iswap(0, 1)

        res = transpile(circuit, basis_gates=["u", "ecr"], optimization_level=3)
        self.assertEqual(res.count_ops()["ecr"], 1)
        self.assertTrue(Operator(res).equiv(circuit))

    def test_approximation_degree_invalid(self):
        """Test invalid approximation degree raises."""
        circuit = QuantumCircuit(2)
        circuit.swap(0, 1)
        with self.assertRaises(QiskitError):
            transpile(circuit, basis_gates=["u", "cz"], approximation_degree=1.1)

    def test_approximation_degree(self):
        """Test more approximation gives lower-cost circuit."""
        circuit = QuantumCircuit(2)
        circuit.swap(0, 1)
        circuit.h(0)
        circ_10 = transpile(
            circuit,
            basis_gates=["u", "cx"],
            translation_method="synthesis",
            approximation_degree=0.1,
        )
        circ_90 = transpile(
            circuit,
            basis_gates=["u", "cx"],
            translation_method="synthesis",
            approximation_degree=0.9,
        )
        self.assertLess(circ_10.depth(), circ_90.depth())

    @data(0, 1, 2, 3)
    def test_synthesis_translation_method_with_single_qubit_gates(self, optimization_level):
        """Test that synthesis basis translation works for solely 1q circuit"""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        res = transpile(
            qc,
            basis_gates=["id", "rz", "x", "sx", "cx"],
            translation_method="synthesis",
            optimization_level=optimization_level,
        )
        expected = QuantumCircuit(3, global_phase=3 * np.pi / 4)
        expected.rz(np.pi / 2, 0)
        expected.rz(np.pi / 2, 1)
        expected.rz(np.pi / 2, 2)
        expected.sx(0)
        expected.sx(1)
        expected.sx(2)
        expected.rz(np.pi / 2, 0)
        expected.rz(np.pi / 2, 1)
        expected.rz(np.pi / 2, 2)
        self.assertEqual(res, expected)

    @data(0, 1, 2, 3)
    def test_synthesis_translation_method_with_gates_outside_basis(self, optimization_level):
        """Test that synthesis translation works for circuits with single gates outside bassis"""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        res = transpile(
            qc,
            basis_gates=["id", "rz", "x", "sx", "cx"],
            translation_method="synthesis",
            optimization_level=optimization_level,
        )
        if optimization_level != 3:
            self.assertTrue(Operator(qc).equiv(res))
            self.assertNotIn("swap", res.count_ops())
        else:
            # Optimization level 3 eliminates the pointless swap
            self.assertEqual(res, QuantumCircuit(2))

    @data(0, 1, 2, 3)
    def test_target_ideal_gates(self, opt_level):
        """Test that transpile() with a custom ideal sim target works."""
        theta = Parameter("θ")
        phi = Parameter("ϕ")
        lam = Parameter("λ")
        target = Target(num_qubits=2)
        target.add_instruction(UGate(theta, phi, lam), {(0,): None, (1,): None})
        target.add_instruction(CXGate(), {(0, 1): None})
        target.add_instruction(Measure(), {(0,): None, (1,): None})
        qubit_reg = QuantumRegister(2, name="q")
        clbit_reg = ClassicalRegister(2, name="c")
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])

        result = transpile(qc, target=target, optimization_level=opt_level)

        self.assertEqual(Operator.from_circuit(result), Operator.from_circuit(qc))

    @data(0, 1, 2, 3)
    def test_transpile_with_custom_control_flow_target(self, opt_level):
        """Test transpile() with a target and constrol flow ops."""
        target = FakeMumbaiV2().target
        target.add_instruction(ForLoopOp, name="for_loop")
        target.add_instruction(WhileLoopOp, name="while_loop")
        target.add_instruction(IfElseOp, name="if_else")
        target.add_instruction(SwitchCaseOp, name="switch_case")

        circuit = QuantumCircuit(6, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        circuit.cx(0, 1)
        circuit.cz(0, 2)
        circuit.append(CustomCX(), [1, 2], [])
        with circuit.for_loop((1,)):
            circuit.cx(0, 1)
            circuit.cz(0, 2)
            circuit.append(CustomCX(), [1, 2], [])
        with circuit.if_test((circuit.clbits[0], True)) as else_:
            circuit.cx(0, 1)
            circuit.cz(0, 2)
            circuit.append(CustomCX(), [1, 2], [])
        with else_:
            circuit.cx(3, 4)
            circuit.cz(3, 5)
            circuit.append(CustomCX(), [4, 5], [])
            with circuit.while_loop((circuit.clbits[0], True)):
                circuit.cx(3, 4)
                circuit.cz(3, 5)
                circuit.append(CustomCX(), [4, 5], [])
        with circuit.switch(circuit.cregs[0]) as case_:
            with case_(0):
                circuit.cx(0, 1)
                circuit.cz(0, 2)
                circuit.append(CustomCX(), [1, 2], [])
            with case_(1):
                circuit.cx(1, 2)
                circuit.cz(1, 3)
                circuit.append(CustomCX(), [2, 3], [])
        transpiled = transpile(
            circuit, optimization_level=opt_level, target=target, seed_transpiler=12434
        )
        # Tests of the complete validity of a circuit are mostly done at the indiviual pass level;
        # here we're just checking that various passes do appear to have run.
        self.assertIsInstance(transpiled, QuantumCircuit)
        # Assert layout ran.
        self.assertIsNot(getattr(transpiled, "_layout", None), None)

        def _visit_block(circuit, qubit_mapping=None):
            for instruction in circuit:
                qargs = tuple(qubit_mapping[x] for x in instruction.qubits)
                self.assertTrue(target.instruction_supported(instruction.operation.name, qargs))
                if isinstance(instruction.operation, ControlFlowOp):
                    for block in instruction.operation.blocks:
                        new_mapping = {
                            inner: qubit_mapping[outer]
                            for outer, inner in zip(instruction.qubits, block.qubits)
                        }
                        _visit_block(block, new_mapping)
                # Assert unrolling ran.
                self.assertNotIsInstance(instruction.operation, CustomCX)
                # Assert translation ran.
                self.assertNotIsInstance(instruction.operation, CZGate)

        # Assert routing ran.
        _visit_block(
            transpiled,
            qubit_mapping={qubit: index for index, qubit in enumerate(transpiled.qubits)},
        )

    @data(1, 2, 3)
    def test_transpile_identity_circuit_no_target(self, opt_level):
        """Test circuit equivalent to identity is optimized away for all optimization levels >0.

        Reproduce taken from https://github.com/Qiskit/qiskit-terra/issues/9217
        """
        qr1 = QuantumRegister(3, "state")
        qr2 = QuantumRegister(2, "ancilla")
        cr = ClassicalRegister(2, "c")
        qc = QuantumCircuit(qr1, qr2, cr)
        qc.h(qr1[0])
        qc.cx(qr1[0], qr1[1])
        qc.cx(qr1[1], qr1[2])
        qc.cx(qr1[1], qr1[2])
        qc.cx(qr1[0], qr1[1])
        qc.h(qr1[0])

        empty_qc = QuantumCircuit(qr1, qr2, cr)
        result = transpile(qc, optimization_level=opt_level)
        self.assertEqual(empty_qc, result)

    @data(0, 1, 2, 3)
    def test_initial_layout_with_loose_qubits(self, opt_level):
        """Regression test of gh-10125."""
        qc = QuantumCircuit([Qubit(), Qubit()])
        qc.cx(0, 1)
        transpiled = transpile(qc, initial_layout=[1, 0], optimization_level=opt_level)
        self.assertIsNotNone(transpiled.layout)
        self.assertEqual(
            transpiled.layout.initial_layout, Layout({0: qc.qubits[1], 1: qc.qubits[0]})
        )

    @data(0, 1, 2, 3)
    def test_initial_layout_with_overlapping_qubits(self, opt_level):
        """Regression test of gh-10125."""
        qr1 = QuantumRegister(2, "qr1")
        qr2 = QuantumRegister(bits=qr1[:])
        qc = QuantumCircuit(qr1, qr2)
        qc.cx(0, 1)
        transpiled = transpile(qc, initial_layout=[1, 0], optimization_level=opt_level)
        self.assertIsNotNone(transpiled.layout)
        self.assertEqual(
            transpiled.layout.initial_layout, Layout({0: qc.qubits[1], 1: qc.qubits[0]})
        )

    @combine(opt_level=[0, 1, 2, 3], basis=[["rz", "x"], ["rx", "z"], ["rz", "y"], ["ry", "x"]])
    def test_paulis_to_constrained_1q_basis(self, opt_level, basis):
        """Test that Pauli-gate circuits can be transpiled to constrained 1q bases that do not
        contain any root-Pauli gates."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.barrier()
        qc.y(0)
        qc.barrier()
        qc.z(0)
        transpiled = transpile(qc, basis_gates=basis, optimization_level=opt_level)
        self.assertGreaterEqual(set(basis) | {"barrier"}, transpiled.count_ops().keys())
        self.assertEqual(Operator(qc), Operator(transpiled))

    @data(0, 1, 2, 3)
    def test_barrier_not_output(self, opt_level):
        """Test that barriers added as part internal transpiler operations do not leak out."""
        qc = QuantumCircuit(2, 2)
        qc.cx(0, 1)
        qc.measure(range(2), range(2))
        tqc = transpile(
            qc,
            initial_layout=[1, 4],
            coupling_map=[[1, 2], [2, 3], [3, 4]],
            optimization_level=opt_level,
        )
        self.assertNotIn("barrier", tqc.count_ops())

    @data(0, 1, 2, 3)
    def test_barrier_not_output_input_preservered(self, opt_level):
        """Test that barriers added as part internal transpiler operations do not leak out."""
        qc = QuantumCircuit(2, 2)
        qc.cx(0, 1)
        qc.measure_all()
        tqc = transpile(
            qc,
            initial_layout=[1, 4],
            coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]],
            optimization_level=opt_level,
        )
        op_counts = tqc.count_ops()
        self.assertEqual(op_counts["barrier"], 1)
        for inst in tqc.data:
            if inst.operation.name == "barrier":
                self.assertEqual(len(inst.qubits), 2)

    @combine(opt_level=[0, 1, 2, 3])
    def test_transpile_annotated_ops(self, opt_level):
        """Test transpilation of circuits with annotated operations."""
        qc = QuantumCircuit(3)
        qc.append(AnnotatedOperation(SGate(), InverseModifier()), [0])
        qc.append(AnnotatedOperation(XGate(), ControlModifier(1)), [1, 2])
        qc.append(AnnotatedOperation(HGate(), PowerModifier(3)), [2])
        expected = QuantumCircuit(3)
        expected.sdg(0)
        expected.cx(1, 2)
        expected.h(2)
        transpiled = transpile(qc, optimization_level=opt_level, seed_transpiler=42)
        self.assertNotIn("annotated", transpiled.count_ops().keys())
        self.assertEqual(Operator(qc), Operator(transpiled))
        self.assertEqual(Operator(qc), Operator(expected))

    @combine(opt_level=[0, 1, 2, 3])
    def test_transpile_annotated_ops_with_backend(self, opt_level):
        """Test transpilation of circuits with annotated operations given a backend."""
        qc = QuantumCircuit(3)
        qc.append(AnnotatedOperation(SGate(), InverseModifier()), [0])
        qc.append(AnnotatedOperation(XGate(), ControlModifier(1)), [1, 2])
        qc.append(AnnotatedOperation(HGate(), PowerModifier(3)), [2])
        backend = FakeMelbourne()
        transpiled = transpile(
            qc, optimization_level=opt_level, backend=backend, seed_transpiler=42
        )
        self.assertLessEqual(set(transpiled.count_ops().keys()), {"u1", "u2", "u3", "cx"})


@ddt
class TestPostTranspileIntegration(QiskitTestCase):
    """Test that the output of `transpile` is usable in various other integration contexts."""

    def _regular_circuit(self):
        a = Parameter("a")
        regs = [
            QuantumRegister(2, name="q0"),
            QuantumRegister(3, name="q1"),
            ClassicalRegister(2, name="c0"),
        ]
        bits = [Qubit(), Qubit(), Clbit()]
        base = QuantumCircuit(*regs, bits)
        base.h(0)
        base.measure(0, 0)
        base.cx(0, 1)
        base.cz(0, 2)
        base.cz(0, 3)
        base.cz(1, 4)
        base.cx(1, 5)
        base.measure(1, 1)
        base.append(CustomCX(), [3, 6])
        base.append(CustomCX(), [5, 4])
        base.append(CustomCX(), [5, 3])
        base.append(CustomCX(), [2, 4]).c_if(base.cregs[0], 3)
        base.ry(a, 4)
        base.measure(4, 2)
        return base

    def _control_flow_circuit(self):
        a = Parameter("a")
        regs = [
            QuantumRegister(2, name="q0"),
            QuantumRegister(3, name="q1"),
            ClassicalRegister(2, name="c0"),
        ]
        bits = [Qubit(), Qubit(), Clbit()]
        base = QuantumCircuit(*regs, bits)
        base.h(0)
        base.measure(0, 0)
        with base.if_test((base.cregs[0], 1)) as else_:
            base.cx(0, 1)
            base.cz(0, 2)
            base.cz(0, 3)
        with else_:
            base.cz(1, 4)
            with base.for_loop((1, 2)):
                base.cx(1, 5)
        base.measure(2, 2)
        with base.while_loop((2, False)):
            base.append(CustomCX(), [3, 6])
            base.append(CustomCX(), [5, 4])
            base.append(CustomCX(), [5, 3])
            base.append(CustomCX(), [2, 4])
            base.ry(a, 4)
            base.measure(4, 2)
        with base.switch(base.cregs[0]) as case_:
            with case_(0, 1):
                base.cz(3, 5)
            with case_(case_.DEFAULT):
                base.cz(1, 4)
                base.append(CustomCX(), [2, 4])
                base.append(CustomCX(), [3, 4])
        return base

    def _control_flow_expr_circuit(self):
        a = Parameter("a")
        regs = [
            QuantumRegister(2, name="q0"),
            QuantumRegister(3, name="q1"),
            ClassicalRegister(2, name="c0"),
        ]
        bits = [Qubit(), Qubit(), Clbit()]
        base = QuantumCircuit(*regs, bits)
        base.h(0)
        base.measure(0, 0)
        with base.if_test(expr.equal(base.cregs[0], 1)) as else_:
            base.cx(0, 1)
            base.cz(0, 2)
            base.cz(0, 3)
        with else_:
            base.cz(1, 4)
            with base.for_loop((1, 2)):
                base.cx(1, 5)
        base.measure(2, 2)
        with base.while_loop(expr.logic_not(bits[2])):
            base.append(CustomCX(), [3, 6])
            base.append(CustomCX(), [5, 4])
            base.append(CustomCX(), [5, 3])
            base.append(CustomCX(), [2, 4])
            base.ry(a, 4)
            base.measure(4, 2)
        with base.switch(expr.bit_and(base.cregs[0], 2)) as case_:
            with case_(0, 1):
                base.cz(3, 5)
            with case_(case_.DEFAULT):
                base.cz(1, 4)
                base.append(CustomCX(), [2, 4])
                base.append(CustomCX(), [3, 4])
        return base

    @data(0, 1, 2, 3)
    def test_qpy_roundtrip(self, optimization_level):
        """Test that the output of a transpiled circuit can be round-tripped through QPY."""
        transpiled = transpile(
            self._regular_circuit(),
            backend=FakeMelbourne(),
            optimization_level=optimization_level,
            seed_transpiler=2022_10_17,
        )
        # Round-tripping the layout is out-of-scope for QPY while it's a private attribute.
        transpiled._layout = None
        buffer = io.BytesIO()
        qpy.dump(transpiled, buffer)
        buffer.seek(0)
        round_tripped = qpy.load(buffer)[0]
        self.assertEqual(round_tripped, transpiled)

    @data(0, 1, 2, 3)
    def test_qpy_roundtrip_backendv2(self, optimization_level):
        """Test that the output of a transpiled circuit can be round-tripped through QPY."""
        transpiled = transpile(
            self._regular_circuit(),
            backend=FakeMumbaiV2(),
            optimization_level=optimization_level,
            seed_transpiler=2022_10_17,
        )

        # Round-tripping the layout is out-of-scope for QPY while it's a private attribute.
        transpiled._layout = None
        buffer = io.BytesIO()
        qpy.dump(transpiled, buffer)
        buffer.seek(0)
        round_tripped = qpy.load(buffer)[0]

        self.assertEqual(round_tripped, transpiled)

    @data(0, 1, 2, 3)
    def test_qpy_roundtrip_control_flow(self, optimization_level):
        """Test that the output of a transpiled circuit with control flow can be round-tripped
        through QPY."""
        if optimization_level == 3 and sys.platform == "win32":
            self.skipTest(
                "This test case triggers a bug in the eigensolver routine on windows. "
                "See #10345 for more details."
            )

        backend = FakeMelbourne()
        transpiled = transpile(
            self._control_flow_circuit(),
            backend=backend,
            basis_gates=backend.configuration().basis_gates
            + ["if_else", "for_loop", "while_loop", "switch_case"],
            optimization_level=optimization_level,
            seed_transpiler=2022_10_17,
        )
        # Round-tripping the layout is out-of-scope for QPY while it's a private attribute.
        transpiled._layout = None
        buffer = io.BytesIO()
        qpy.dump(transpiled, buffer)
        buffer.seek(0)
        round_tripped = qpy.load(buffer)[0]
        self.assertEqual(round_tripped, transpiled)

    @data(0, 1, 2, 3)
    def test_qpy_roundtrip_control_flow_backendv2(self, optimization_level):
        """Test that the output of a transpiled circuit with control flow can be round-tripped
        through QPY."""
        backend = FakeMumbaiV2()
        backend.target.add_instruction(IfElseOp, name="if_else")
        backend.target.add_instruction(ForLoopOp, name="for_loop")
        backend.target.add_instruction(WhileLoopOp, name="while_loop")
        backend.target.add_instruction(SwitchCaseOp, name="switch_case")
        transpiled = transpile(
            self._control_flow_circuit(),
            backend=backend,
            optimization_level=optimization_level,
            seed_transpiler=2022_10_17,
        )
        # Round-tripping the layout is out-of-scope for QPY while it's a private attribute.
        transpiled._layout = None
        buffer = io.BytesIO()
        qpy.dump(transpiled, buffer)
        buffer.seek(0)
        round_tripped = qpy.load(buffer)[0]
        self.assertEqual(round_tripped, transpiled)

    @data(0, 1, 2, 3)
    def test_qpy_roundtrip_control_flow_expr(self, optimization_level):
        """Test that the output of a transpiled circuit with control flow including `Expr` nodes can
        be round-tripped through QPY."""
        if optimization_level == 3 and sys.platform == "win32":
            self.skipTest(
                "This test case triggers a bug in the eigensolver routine on windows. "
                "See #10345 for more details."
            )
        backend = FakeMelbourne()
        transpiled = transpile(
            self._control_flow_expr_circuit(),
            backend=backend,
            basis_gates=backend.configuration().basis_gates
            + ["if_else", "for_loop", "while_loop", "switch_case"],
            optimization_level=optimization_level,
            seed_transpiler=2023_07_26,
        )
        buffer = io.BytesIO()
        qpy.dump(transpiled, buffer)
        buffer.seek(0)
        round_tripped = qpy.load(buffer)[0]
        self.assertEqual(round_tripped, transpiled)

    @data(0, 1, 2, 3)
    def test_qpy_roundtrip_control_flow_expr_backendv2(self, optimization_level):
        """Test that the output of a transpiled circuit with control flow including `Expr` nodes can
        be round-tripped through QPY."""
        backend = FakeMumbaiV2()
        backend.target.add_instruction(IfElseOp, name="if_else")
        backend.target.add_instruction(ForLoopOp, name="for_loop")
        backend.target.add_instruction(WhileLoopOp, name="while_loop")
        backend.target.add_instruction(SwitchCaseOp, name="switch_case")
        transpiled = transpile(
            self._control_flow_circuit(),
            backend=backend,
            optimization_level=optimization_level,
            seed_transpiler=2023_07_26,
        )
        buffer = io.BytesIO()
        qpy.dump(transpiled, buffer)
        buffer.seek(0)
        round_tripped = qpy.load(buffer)[0]
        self.assertEqual(round_tripped, transpiled)

    @data(0, 1, 2, 3)
    def test_qasm3_output(self, optimization_level):
        """Test that the output of a transpiled circuit can be dumped into OpenQASM 3."""
        transpiled = transpile(
            self._regular_circuit(),
            backend=FakeMelbourne(),
            optimization_level=optimization_level,
            seed_transpiler=2022_10_17,
        )
        # TODO: There's not a huge amount we can sensibly test for the output here until we can
        # round-trip the OpenQASM 3 back into a Terra circuit.  Mostly we're concerned that the dump
        # itself doesn't throw an error, though.
        self.assertIsInstance(qasm3.dumps(transpiled).strip(), str)

    @data(0, 1, 2, 3)
    def test_qasm3_output_control_flow(self, optimization_level):
        """Test that the output of a transpiled circuit with control flow can be dumped into
        OpenQASM 3."""
        backend = FakeMumbaiV2()
        backend.target.add_instruction(IfElseOp, name="if_else")
        backend.target.add_instruction(ForLoopOp, name="for_loop")
        backend.target.add_instruction(WhileLoopOp, name="while_loop")
        backend.target.add_instruction(SwitchCaseOp, name="switch_case")
        transpiled = transpile(
            self._control_flow_circuit(),
            backend=backend,
            optimization_level=optimization_level,
            seed_transpiler=2022_10_17,
        )
        # TODO: There's not a huge amount we can sensibly test for the output here until we can
        # round-trip the OpenQASM 3 back into a Terra circuit.  Mostly we're concerned that the dump
        # itself doesn't throw an error, though.
        self.assertIsInstance(
            qasm3.dumps(transpiled, experimental=qasm3.ExperimentalFeatures.SWITCH_CASE_V1).strip(),
            str,
        )

    @data(0, 1, 2, 3)
    def test_qasm3_output_control_flow_expr(self, optimization_level):
        """Test that the output of a transpiled circuit with control flow and `Expr` nodes can be
        dumped into OpenQASM 3."""
        backend = FakeMumbaiV2()
        backend.target.add_instruction(IfElseOp, name="if_else")
        backend.target.add_instruction(ForLoopOp, name="for_loop")
        backend.target.add_instruction(WhileLoopOp, name="while_loop")
        backend.target.add_instruction(SwitchCaseOp, name="switch_case")
        transpiled = transpile(
            self._control_flow_circuit(),
            backend=backend,
            optimization_level=optimization_level,
            seed_transpiler=2023_07_26,
        )
        # TODO: There's not a huge amount we can sensibly test for the output here until we can
        # round-trip the OpenQASM 3 back into a Terra circuit.  Mostly we're concerned that the dump
        # itself doesn't throw an error, though.
        self.assertIsInstance(
            qasm3.dumps(transpiled, experimental=qasm3.ExperimentalFeatures.SWITCH_CASE_V1).strip(),
            str,
        )

    @data(0, 1, 2, 3)
    def test_transpile_target_no_measurement_error(self, opt_level):
        """Test that transpile with a target which contains ideal measurement works

        Reproduce from https://github.com/Qiskit/qiskit-terra/issues/8969
        """
        target = Target()
        target.add_instruction(Measure(), {(0,): None})
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        res = transpile(qc, target=target, optimization_level=opt_level)
        self.assertEqual(qc, res)

    def test_transpile_final_layout_updated_with_post_layout(self):
        """Test that the final layout is correctly set when vf2postlayout runs.

        Reproduce from #10457
        """

        def _get_index_layout(transpiled_circuit: QuantumCircuit, num_source_qubits: int):
            """Return the index layout of a transpiled circuit"""
            layout = transpiled_circuit.layout
            if layout is None:
                return list(range(num_source_qubits))

            pos_to_virt = {v: k for k, v in layout.input_qubit_mapping.items()}
            qubit_indices = []
            for index in range(num_source_qubits):
                qubit_idx = layout.initial_layout[pos_to_virt[index]]
                if layout.final_layout is not None:
                    qubit_idx = layout.final_layout[transpiled_circuit.qubits[qubit_idx]]
                qubit_indices.append(qubit_idx)
            return qubit_indices

        vf2_post_layout_called = False

        def callback(**kwargs):
            nonlocal vf2_post_layout_called
            if isinstance(kwargs["pass_"], VF2PostLayout):
                vf2_post_layout_called = True
                self.assertIsNotNone(kwargs["property_set"]["post_layout"])

        backend = FakeVigo()
        qubits = 3
        qc = QuantumCircuit(qubits)
        for i in range(5):
            qc.cx(i % qubits, int(i + qubits / 2) % qubits)

        tqc = transpile(qc, backend=backend, seed_transpiler=4242, callback=callback)
        self.assertTrue(vf2_post_layout_called)
        self.assertEqual([3, 2, 1], _get_index_layout(tqc, qubits))


class StreamHandlerRaiseException(StreamHandler):
    """Handler class that will raise an exception on formatting errors."""

    def handleError(self, record):
        raise sys.exc_info()


class TestLogTranspile(QiskitTestCase):
    """Testing the log_transpile option."""

    def setUp(self):
        super().setUp()
        logger = getLogger()
        self.addCleanup(logger.setLevel, logger.level)
        logger.setLevel("DEBUG")
        self.output = io.StringIO()
        logger.addHandler(StreamHandlerRaiseException(self.output))
        self.circuit = QuantumCircuit(QuantumRegister(1))

    def assertTranspileLog(self, log_msg):
        """Runs the transpiler and check for logs containing specified message"""
        transpile(self.circuit)
        self.output.seek(0)
        # Filter unrelated log lines
        output_lines = self.output.readlines()
        transpile_log_lines = [x for x in output_lines if log_msg in x]
        self.assertTrue(len(transpile_log_lines) > 0)

    def test_transpile_log_time(self):
        """Check Total Transpile Time is logged"""
        self.assertTranspileLog("Total Transpile Time")


class TestTranspileCustomPM(QiskitTestCase):
    """Test transpile function with custom pass manager"""

    def test_custom_multiple_circuits(self):
        """Test transpiling with custom pass manager and multiple circuits.
        This tests created a deadlock, so it needs to be monitored for timeout.
        See: https://github.com/Qiskit/qiskit-terra/issues/3925
        """
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        pm_conf = PassManagerConfig(
            initial_layout=None,
            basis_gates=["u1", "u2", "u3", "cx"],
            coupling_map=CouplingMap([[0, 1]]),
            backend_properties=None,
            seed_transpiler=1,
        )
        passmanager = level_0_pass_manager(pm_conf)

        transpiled = passmanager.run([qc, qc])

        expected = QuantumCircuit(QuantumRegister(2, "q"))
        expected.append(U2Gate(0, 3.141592653589793), [0])
        expected.cx(0, 1)

        self.assertEqual(len(transpiled), 2)
        self.assertEqual(transpiled[0], expected)
        self.assertEqual(transpiled[1], expected)


@ddt
class TestTranspileParallel(QiskitTestCase):
    """Test transpile() in parallel."""

    def setUp(self):
        super().setUp()

        # Force parallel execution to True to test multiprocessing for this class
        original_val = parallel.PARALLEL_DEFAULT

        def restore_default():
            parallel.PARALLEL_DEFAULT = original_val

        self.addCleanup(restore_default)
        parallel.PARALLEL_DEFAULT = True

    @data(0, 1, 2, 3)
    def test_parallel_multiprocessing(self, opt_level):
        """Test parallel dispatch works with multiprocessing."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        backend = FakeMumbaiV2()
        pm = generate_preset_pass_manager(opt_level, backend)
        res = pm.run([qc, qc])
        for circ in res:
            self.assertIsInstance(circ, QuantumCircuit)

    @data(0, 1, 2, 3)
    def test_parallel_with_target(self, opt_level):
        """Test that parallel dispatch works with a manual target."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        target = FakeMumbaiV2().target
        res = transpile([qc] * 3, target=target, optimization_level=opt_level)
        self.assertIsInstance(res, list)
        for circ in res:
            self.assertIsInstance(circ, QuantumCircuit)

    @data(0, 1, 2, 3)
    def test_parallel_num_processes_kwarg(self, num_processes):
        """Test that num_processes kwarg works when the system default parallel is true"""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        target = FakeMumbaiV2().target
        res = transpile([qc] * 3, target=target, num_processes=num_processes)
        self.assertIsInstance(res, list)
        for circ in res:
            self.assertIsInstance(circ, QuantumCircuit)

    @data(0, 1, 2, 3)
    def test_parallel_dispatch(self, opt_level):
        """Test that transpile in parallel works for all optimization levels."""
        backend = FakeRueschlikon()
        qr = QuantumRegister(16)
        cr = ClassicalRegister(16)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        for k in range(1, 15):
            qc.cx(qr[0], qr[k])
        qc.measure(qr, cr)
        qlist = [qc for k in range(15)]
        tqc = transpile(
            qlist, backend=backend, optimization_level=opt_level, seed_transpiler=424242
        )
        result = backend.run(tqc, seed_simulator=4242424242, shots=1000).result()
        counts = result.get_counts()
        for count in counts:
            self.assertTrue(math.isclose(count["0000000000000000"], 500, rel_tol=0.1))
            self.assertTrue(math.isclose(count["0111111111111111"], 500, rel_tol=0.1))

    def test_parallel_dispatch_lazy_cal_loading(self):
        """Test adding calibration by lazy loading in parallel environment."""

        class TestAddCalibration(TransformationPass):
            """A fake pass to test lazy pulse qobj loading in parallel environment."""

            def __init__(self, target):
                """Instantiate with target."""
                super().__init__()
                self.target = target

            def run(self, dag):
                """Run test pass that adds calibration of SX gate of qubit 0."""
                dag.add_calibration(
                    "sx",
                    qubits=(0,),
                    schedule=self.target["sx"][(0,)].calibration,  # PulseQobj is parsed here
                )
                return dag

        backend = FakeMumbaiV2()

        # This target has PulseQobj entries that provides a serialized schedule data
        pass_ = TestAddCalibration(backend.target)
        pm = PassManager(passes=[pass_])
        self.assertIsNone(backend.target["sx"][(0,)]._calibration._definition)

        qc = QuantumCircuit(1)
        qc.sx(0)
        qc_copied = [qc for _ in range(10)]

        qcs_cal_added = pm.run(qc_copied)
        ref_cal = backend.target["sx"][(0,)].calibration
        for qc_test in qcs_cal_added:
            added_cal = qc_test.calibrations["sx"][((0,), ())]
            self.assertEqual(added_cal, ref_cal)

    @data(0, 1, 2, 3)
    def test_parallel_singleton_conditional_gate(self, opt_level):
        """Test that singleton mutable instance doesn't lose state in parallel."""
        backend = FakeNairobiV2()
        circ = QuantumCircuit(2, 1)
        circ.h(0)
        circ.measure(0, circ.clbits[0])
        circ.z(1).c_if(circ.clbits[0], 1)
        res = transpile(
            [circ, circ], backend, optimization_level=opt_level, seed_transpiler=123456769
        )
        self.assertTrue(res[0].data[-1].operation.mutable)
        self.assertEqual(res[0].data[-1].operation.condition, (res[0].clbits[0], 1))

    @data(0, 1, 2, 3)
    def test_backendv2_and_basis_gates(self, opt_level):
        """Test transpile() with BackendV2 and basis_gates set."""
        backend = FakeNairobiV2()
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.cz(0, 3)
        qc.cz(0, 4)
        qc.measure_all()
        tqc = transpile(
            qc,
            backend=backend,
            basis_gates=["u", "cz"],
            optimization_level=opt_level,
            seed_transpiler=12345678942,
        )
        op_count = set(tqc.count_ops())
        self.assertEqual({"u", "cz", "measure", "barrier"}, op_count)
        for inst in tqc.data:
            if inst.operation.name not in {"u", "cz"}:
                continue
            qubits = tuple(tqc.find_bit(x).index for x in inst.qubits)
            self.assertIn(qubits, backend.target.qargs)

    @data(0, 1, 2, 3)
    def test_backendv2_and_coupling_map(self, opt_level):
        """Test transpile() with custom coupling map."""
        backend = FakeNairobiV2()
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.cz(0, 3)
        qc.cz(0, 4)
        qc.measure_all()
        cmap = CouplingMap.from_line(5, bidirectional=False)
        tqc = transpile(
            qc,
            backend=backend,
            coupling_map=cmap,
            optimization_level=opt_level,
            seed_transpiler=12345678942,
        )
        op_count = set(tqc.count_ops())
        self.assertTrue({"rz", "sx", "x", "cx", "measure", "barrier"}.issuperset(op_count))
        for inst in tqc.data:
            if len(inst.qubits) == 2:
                qubit_0 = tqc.find_bit(inst.qubits[0]).index
                qubit_1 = tqc.find_bit(inst.qubits[1]).index
                self.assertEqual(qubit_1, qubit_0 + 1)

    def test_transpile_with_multiple_coupling_maps(self):
        """Test passing a different coupling map for every circuit"""
        backend = FakeNairobiV2()

        qc = QuantumCircuit(3)
        qc.cx(0, 2)

        # Add a connection between 0 and 2 so that transpile does not change
        # the gates
        cmap = CouplingMap.from_line(7)
        cmap.add_edge(0, 2)

        with self.assertRaisesRegex(TranspilerError, "Only a single input coupling"):
            # Initial layout needed to prevent transpiler from relabeling
            # qubits to avoid doing the swap
            transpile(
                [qc] * 2,
                backend,
                coupling_map=[backend.coupling_map, cmap],
                initial_layout=(0, 1, 2),
            )

    @data(0, 1, 2, 3)
    def test_backend_and_custom_gate(self, opt_level):
        """Test transpile() with BackendV2, custom basis pulse gate."""
        backend = FakeNairobiV2()
        inst_map = InstructionScheduleMap()
        inst_map.add("newgate", [0, 1], pulse.ScheduleBlock())
        newgate = Gate("newgate", 2, [])
        circ = QuantumCircuit(2)
        circ.append(newgate, [0, 1])
        tqc = transpile(
            circ, backend, inst_map=inst_map, basis_gates=["newgate"], optimization_level=opt_level
        )
        self.assertEqual(len(tqc.data), 1)
        self.assertEqual(tqc.data[0].operation, newgate)
        qubits = tuple(tqc.find_bit(x).index for x in tqc.data[0].qubits)
        self.assertIn(qubits, backend.target.qargs)


@ddt
class TestTranspileMultiChipTarget(QiskitTestCase):
    """Test transpile() with a disjoint coupling map."""

    def setUp(self):
        super().setUp()

        class FakeMultiChip(BackendV2):
            """Fake multi chip backend."""

            def __init__(self):
                super().__init__()
                graph = rx.generators.directed_heavy_hex_graph(3)
                num_qubits = len(graph) * 3
                rng = np.random.default_rng(seed=12345678942)
                rz_props = {}
                x_props = {}
                sx_props = {}
                measure_props = {}
                delay_props = {}
                self._target = Target("Fake multi-chip backend", num_qubits=num_qubits)
                for i in range(num_qubits):
                    qarg = (i,)
                    rz_props[qarg] = InstructionProperties(error=0.0, duration=0.0)
                    x_props[qarg] = InstructionProperties(
                        error=rng.uniform(1e-6, 1e-4), duration=rng.uniform(1e-8, 9e-7)
                    )
                    sx_props[qarg] = InstructionProperties(
                        error=rng.uniform(1e-6, 1e-4), duration=rng.uniform(1e-8, 9e-7)
                    )
                    measure_props[qarg] = InstructionProperties(
                        error=rng.uniform(1e-3, 1e-1), duration=rng.uniform(1e-8, 9e-7)
                    )
                    delay_props[qarg] = None
                self._target.add_instruction(XGate(), x_props)
                self._target.add_instruction(SXGate(), sx_props)
                self._target.add_instruction(RZGate(Parameter("theta")), rz_props)
                self._target.add_instruction(Measure(), measure_props)
                self._target.add_instruction(Delay(Parameter("t")), delay_props)
                cz_props = {}
                for i in range(3):
                    for root_edge in graph.edge_list():
                        offset = i * len(graph)
                        edge = (root_edge[0] + offset, root_edge[1] + offset)
                        cz_props[edge] = InstructionProperties(
                            error=rng.uniform(1e-5, 5e-3), duration=rng.uniform(1e-8, 9e-7)
                        )
                self._target.add_instruction(CZGate(), cz_props)

            @property
            def target(self):
                return self._target

            @property
            def max_circuits(self):
                return None

            @classmethod
            def _default_options(cls):
                return Options(shots=1024)

            def run(self, circuit, **kwargs):
                raise NotImplementedError

        self.backend = FakeMultiChip()

    @data(0, 1, 2, 3)
    def test_basic_connected_circuit(self, opt_level):
        """Test basic connected circuit on disjoint backend"""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.measure_all()
        tqc = transpile(qc, self.backend, optimization_level=opt_level)
        for inst in tqc.data:
            qubits = tuple(tqc.find_bit(x).index for x in inst.qubits)
            op_name = inst.operation.name
            if op_name == "barrier":
                continue
            self.assertIn(qubits, self.backend.target[op_name])

    @data(0, 1, 2, 3)
    def test_triple_circuit(self, opt_level):
        """Test a split circuit with one circuit component per chip."""
        qc = QuantumCircuit(30)
        qc.h(0)
        qc.h(10)
        qc.h(20)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        qc.cx(0, 6)
        qc.cx(0, 7)
        qc.cx(0, 8)
        qc.cx(0, 9)
        qc.ecr(10, 11)
        qc.ecr(10, 12)
        qc.ecr(10, 13)
        qc.ecr(10, 14)
        qc.ecr(10, 15)
        qc.ecr(10, 16)
        qc.ecr(10, 17)
        qc.ecr(10, 18)
        qc.ecr(10, 19)
        qc.cy(20, 21)
        qc.cy(20, 22)
        qc.cy(20, 23)
        qc.cy(20, 24)
        qc.cy(20, 25)
        qc.cy(20, 26)
        qc.cy(20, 27)
        qc.cy(20, 28)
        qc.cy(20, 29)
        qc.measure_all()

        if opt_level == 0:
            with self.assertRaises(TranspilerError):
                tqc = transpile(qc, self.backend, optimization_level=opt_level, seed_transpiler=42)
            return

        tqc = transpile(qc, self.backend, optimization_level=opt_level, seed_transpiler=42)
        for inst in tqc.data:
            qubits = tuple(tqc.find_bit(x).index for x in inst.qubits)
            op_name = inst.operation.name
            if op_name == "barrier":
                continue
            self.assertIn(qubits, self.backend.target[op_name])

    def test_disjoint_control_flow(self):
        """Test control flow circuit on disjoint coupling map."""
        qc = QuantumCircuit(6, 1)
        qc.h(0)
        qc.ecr(0, 1)
        qc.cx(0, 2)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], True)):
            qc.reset(0)
            qc.cz(1, 0)
        qc.h(3)
        qc.cz(3, 4)
        qc.cz(3, 5)
        target = self.backend.target
        target.add_instruction(Reset(), {(i,): None for i in range(target.num_qubits)})
        target.add_instruction(IfElseOp, name="if_else")
        tqc = transpile(qc, target=target)
        edges = set(target.build_coupling_map().graph.edge_list())

        def _visit_block(circuit, qubit_mapping=None):
            for instruction in circuit:
                if instruction.operation.name == "barrier":
                    continue
                qargs = tuple(qubit_mapping[x] for x in instruction.qubits)
                self.assertTrue(target.instruction_supported(instruction.operation.name, qargs))
                if isinstance(instruction.operation, ControlFlowOp):
                    for block in instruction.operation.blocks:
                        new_mapping = {
                            inner: qubit_mapping[outer]
                            for outer, inner in zip(instruction.qubits, block.qubits)
                        }
                        _visit_block(block, new_mapping)
                elif len(qargs) == 2:
                    self.assertIn(qargs, edges)
                self.assertIn(instruction.operation.name, target)

        _visit_block(
            tqc,
            qubit_mapping={qubit: index for index, qubit in enumerate(tqc.qubits)},
        )

    def test_disjoint_control_flow_shared_classical(self):
        """Test circuit with classical data dependency between connected components."""
        creg = ClassicalRegister(19)
        qc = QuantumCircuit(25)
        qc.add_register(creg)
        qc.h(0)
        for i in range(18):
            qc.cx(0, i + 1)
        for i in range(18):
            qc.measure(i, creg[i])
        with qc.if_test((creg, 0)):
            qc.h(20)
            qc.ecr(20, 21)
            qc.ecr(20, 22)
            qc.ecr(20, 23)
            qc.ecr(20, 24)
        target = self.backend.target
        target.add_instruction(Reset(), {(i,): None for i in range(target.num_qubits)})
        target.add_instruction(IfElseOp, name="if_else")
        tqc = transpile(qc, target=target)

        def _visit_block(circuit, qubit_mapping=None):
            for instruction in circuit:
                if instruction.operation.name == "barrier":
                    continue
                qargs = tuple(qubit_mapping[x] for x in instruction.qubits)
                self.assertTrue(target.instruction_supported(instruction.operation.name, qargs))
                if isinstance(instruction.operation, ControlFlowOp):
                    for block in instruction.operation.blocks:
                        new_mapping = {
                            inner: qubit_mapping[outer]
                            for outer, inner in zip(instruction.qubits, block.qubits)
                        }
                        _visit_block(block, new_mapping)

        _visit_block(
            tqc,
            qubit_mapping={qubit: index for index, qubit in enumerate(tqc.qubits)},
        )

    @slow_test
    @data(2, 3)
    def test_six_component_circuit(self, opt_level):
        """Test input circuit with more than 1 component per backend component."""
        qc = QuantumCircuit(42)
        qc.h(0)
        qc.h(10)
        qc.h(20)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        qc.cx(0, 6)
        qc.cx(0, 7)
        qc.cx(0, 8)
        qc.cx(0, 9)
        qc.ecr(10, 11)
        qc.ecr(10, 12)
        qc.ecr(10, 13)
        qc.ecr(10, 14)
        qc.ecr(10, 15)
        qc.ecr(10, 16)
        qc.ecr(10, 17)
        qc.ecr(10, 18)
        qc.ecr(10, 19)
        qc.cy(20, 21)
        qc.cy(20, 22)
        qc.cy(20, 23)
        qc.cy(20, 24)
        qc.cy(20, 25)
        qc.cy(20, 26)
        qc.cy(20, 27)
        qc.cy(20, 28)
        qc.cy(20, 29)
        qc.h(30)
        qc.cx(30, 31)
        qc.cx(30, 32)
        qc.cx(30, 33)
        qc.h(34)
        qc.cx(34, 35)
        qc.cx(34, 36)
        qc.cx(34, 37)
        qc.h(38)
        qc.cx(38, 39)
        qc.cx(39, 40)
        qc.cx(39, 41)
        qc.measure_all()
        tqc = transpile(qc, self.backend, optimization_level=opt_level, seed_transpiler=42)
        for inst in tqc.data:
            qubits = tuple(tqc.find_bit(x).index for x in inst.qubits)
            op_name = inst.operation.name
            if op_name == "barrier":
                continue
            self.assertIn(qubits, self.backend.target[op_name])

    def test_six_component_circuit_level_1(self):
        """Test input circuit with more than 1 component per backend component."""
        opt_level = 1
        qc = QuantumCircuit(42)
        qc.h(0)
        qc.h(10)
        qc.h(20)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        qc.cx(0, 6)
        qc.cx(0, 7)
        qc.cx(0, 8)
        qc.cx(0, 9)
        qc.ecr(10, 11)
        qc.ecr(10, 12)
        qc.ecr(10, 13)
        qc.ecr(10, 14)
        qc.ecr(10, 15)
        qc.ecr(10, 16)
        qc.ecr(10, 17)
        qc.ecr(10, 18)
        qc.ecr(10, 19)
        qc.cy(20, 21)
        qc.cy(20, 22)
        qc.cy(20, 23)
        qc.cy(20, 24)
        qc.cy(20, 25)
        qc.cy(20, 26)
        qc.cy(20, 27)
        qc.cy(20, 28)
        qc.cy(20, 29)
        qc.h(30)
        qc.cx(30, 31)
        qc.cx(30, 32)
        qc.cx(30, 33)
        qc.h(34)
        qc.cx(34, 35)
        qc.cx(34, 36)
        qc.cx(34, 37)
        qc.h(38)
        qc.cx(38, 39)
        qc.cx(39, 40)
        qc.cx(39, 41)
        qc.measure_all()
        tqc = transpile(qc, self.backend, optimization_level=opt_level, seed_transpiler=42)
        for inst in tqc.data:
            qubits = tuple(tqc.find_bit(x).index for x in inst.qubits)
            op_name = inst.operation.name
            if op_name == "barrier":
                continue
            self.assertIn(qubits, self.backend.target[op_name])

    @data(0, 1, 2, 3)
    def test_shared_classical_between_components_condition(self, opt_level):
        """Test a condition sharing classical bits between components."""
        creg = ClassicalRegister(19)
        qc = QuantumCircuit(25)
        qc.add_register(creg)
        qc.h(0)
        for i in range(18):
            qc.cx(0, i + 1)
        for i in range(18):
            qc.measure(i, creg[i])

        qc.ecr(20, 21).c_if(creg, 0)
        tqc = transpile(qc, self.backend, optimization_level=opt_level)

        def _visit_block(circuit, qubit_mapping=None):
            for instruction in circuit:
                if instruction.operation.name == "barrier":
                    continue
                qargs = tuple(qubit_mapping[x] for x in instruction.qubits)
                self.assertTrue(
                    self.backend.target.instruction_supported(instruction.operation.name, qargs)
                )
                if isinstance(instruction.operation, ControlFlowOp):
                    for block in instruction.operation.blocks:
                        new_mapping = {
                            inner: qubit_mapping[outer]
                            for outer, inner in zip(instruction.qubits, block.qubits)
                        }
                        _visit_block(block, new_mapping)

        _visit_block(
            tqc,
            qubit_mapping={qubit: index for index, qubit in enumerate(tqc.qubits)},
        )

    @data(0, 1, 2, 3)
    def test_shared_classical_between_components_condition_large_to_small(self, opt_level):
        """Test a condition sharing classical bits between components."""
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(25)
        qc.add_register(creg)
        # Component 0
        qc.h(24)
        qc.cx(24, 23)
        qc.measure(24, creg[0])
        qc.measure(23, creg[1])
        # Component 1
        qc.h(0).c_if(creg, 0)
        for i in range(18):
            qc.ecr(0, i + 1).c_if(creg, 0)
        tqc = transpile(qc, self.backend, optimization_level=opt_level, seed_transpiler=123456789)

        def _visit_block(circuit, qubit_mapping=None):
            for instruction in circuit:
                if instruction.operation.name == "barrier":
                    continue
                qargs = tuple(qubit_mapping[x] for x in instruction.qubits)
                self.assertTrue(
                    self.backend.target.instruction_supported(instruction.operation.name, qargs)
                )
                if isinstance(instruction.operation, ControlFlowOp):
                    for block in instruction.operation.blocks:
                        new_mapping = {
                            inner: qubit_mapping[outer]
                            for outer, inner in zip(instruction.qubits, block.qubits)
                        }
                        _visit_block(block, new_mapping)

        _visit_block(
            tqc,
            qubit_mapping={qubit: index for index, qubit in enumerate(tqc.qubits)},
        )
        # Check that virtual qubits that interact with each other via quantum links are placed into
        # the same component of the coupling map.
        initial_layout = tqc.layout.initial_layout
        coupling_map = self.backend.target.build_coupling_map()
        components = [
            connected_qubits(initial_layout[qc.qubits[23]], coupling_map),
            connected_qubits(initial_layout[qc.qubits[0]], coupling_map),
        ]
        self.assertLessEqual({initial_layout[qc.qubits[i]] for i in [23, 24]}, components[0])
        self.assertLessEqual({initial_layout[qc.qubits[i]] for i in range(19)}, components[1])

        # Check clbits are in order.
        # Traverse the output dag over the sole clbit, checking that the qubits of the ops
        # go in order between the components. This is a sanity check to ensure that routing
        # doesn't reorder a classical data dependency between components. Inside a component
        # we have the dag ordering so nothing should be out of order within a component.
        tqc_dag = circuit_to_dag(tqc)
        qubit_map = {qubit: index for index, qubit in enumerate(tqc_dag.qubits)}
        input_node = tqc_dag.input_map[tqc_dag.clbits[0]]
        first_meas_node = tqc_dag._multi_graph.find_successors_by_edge(
            input_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
        )[0]
        # The first node should be a measurement
        self.assertIsInstance(first_meas_node.op, Measure)
        # This should be in the first component
        self.assertIn(qubit_map[first_meas_node.qargs[0]], components[0])
        op_node = tqc_dag._multi_graph.find_successors_by_edge(
            first_meas_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
        )[0]
        while isinstance(op_node, DAGOpNode):
            self.assertIn(qubit_map[op_node.qargs[0]], components[1])
            op_node = tqc_dag._multi_graph.find_successors_by_edge(
                op_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
            )[0]

    @data(1, 2, 3)
    def test_shared_classical_between_components_condition_large_to_small_reverse_index(
        self, opt_level
    ):
        """Test a condition sharing classical bits between components."""
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(25)
        qc.add_register(creg)
        # Component 0
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(0, creg[0])
        qc.measure(1, creg[1])
        # Component 1
        qc.h(24).c_if(creg, 0)
        for i in range(23, 5, -1):
            qc.ecr(24, i).c_if(creg, 0)
        tqc = transpile(qc, self.backend, optimization_level=opt_level, seed_transpiler=2023)

        def _visit_block(circuit, qubit_mapping=None):
            for instruction in circuit:
                if instruction.operation.name == "barrier":
                    continue
                qargs = tuple(qubit_mapping[x] for x in instruction.qubits)
                self.assertTrue(
                    self.backend.target.instruction_supported(instruction.operation.name, qargs)
                )
                if isinstance(instruction.operation, ControlFlowOp):
                    for block in instruction.operation.blocks:
                        new_mapping = {
                            inner: qubit_mapping[outer]
                            for outer, inner in zip(instruction.qubits, block.qubits)
                        }
                        _visit_block(block, new_mapping)

        _visit_block(
            tqc,
            qubit_mapping={qubit: index for index, qubit in enumerate(tqc.qubits)},
        )
        # Check that virtual qubits that interact with each other via quantum links are placed into
        # the same component of the coupling map.
        initial_layout = tqc.layout.initial_layout
        coupling_map = self.backend.target.build_coupling_map()
        components = [
            connected_qubits(initial_layout[qc.qubits[0]], coupling_map),
            connected_qubits(initial_layout[qc.qubits[6]], coupling_map),
        ]
        self.assertLessEqual({initial_layout[qc.qubits[i]] for i in range(2)}, components[0])
        self.assertLessEqual({initial_layout[qc.qubits[i]] for i in range(6, 25)}, components[1])

        # Check clbits are in order.
        # Traverse the output dag over the sole clbit, checking that the qubits of the ops
        # go in order between the components. This is a sanity check to ensure that routing
        # doesn't reorder a classical data dependency between components. Inside a component
        # we have the dag ordering so nothing should be out of order within a component.
        tqc_dag = circuit_to_dag(tqc)
        qubit_map = {qubit: index for index, qubit in enumerate(tqc_dag.qubits)}
        input_node = tqc_dag.input_map[tqc_dag.clbits[0]]
        first_meas_node = tqc_dag._multi_graph.find_successors_by_edge(
            input_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
        )[0]
        # The first node should be a measurement
        self.assertIsInstance(first_meas_node.op, Measure)
        # This shoulde be in the first ocmponent
        self.assertIn(qubit_map[first_meas_node.qargs[0]], components[0])
        op_node = tqc_dag._multi_graph.find_successors_by_edge(
            first_meas_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
        )[0]
        while isinstance(op_node, DAGOpNode):
            self.assertIn(qubit_map[op_node.qargs[0]], components[1])
            op_node = tqc_dag._multi_graph.find_successors_by_edge(
                op_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
            )[0]

    @data(1, 2, 3)
    def test_chained_data_dependency(self, opt_level):
        """Test 3 component circuit with shared clbits between each component."""
        creg = ClassicalRegister(1)
        qc = QuantumCircuit(30)
        qc.add_register(creg)
        # Component 0
        qc.h(0)
        for i in range(9):
            qc.cx(0, i + 1)
        measure_op = Measure()
        qc.append(measure_op, [9], [creg[0]])
        # Component 1
        qc.h(10).c_if(creg, 0)
        for i in range(11, 20):
            qc.ecr(10, i).c_if(creg, 0)
        measure_op = Measure()
        qc.append(measure_op, [19], [creg[0]])
        # Component 2
        qc.h(20).c_if(creg, 0)
        for i in range(21, 30):
            qc.cz(20, i).c_if(creg, 0)
        measure_op = Measure()
        qc.append(measure_op, [29], [creg[0]])
        tqc = transpile(qc, self.backend, optimization_level=opt_level, seed_transpiler=2023)

        def _visit_block(circuit, qubit_mapping=None):
            for instruction in circuit:
                if instruction.operation.name == "barrier":
                    continue
                qargs = tuple(qubit_mapping[x] for x in instruction.qubits)
                self.assertTrue(
                    self.backend.target.instruction_supported(instruction.operation.name, qargs)
                )
                if isinstance(instruction.operation, ControlFlowOp):
                    for block in instruction.operation.blocks:
                        new_mapping = {
                            inner: qubit_mapping[outer]
                            for outer, inner in zip(instruction.qubits, block.qubits)
                        }
                        _visit_block(block, new_mapping)

        _visit_block(
            tqc,
            qubit_mapping={qubit: index for index, qubit in enumerate(tqc.qubits)},
        )
        # Check that virtual qubits that interact with each other via quantum links are placed into
        # the same component of the coupling map.
        initial_layout = tqc.layout.initial_layout
        coupling_map = self.backend.target.build_coupling_map()
        components = [
            connected_qubits(initial_layout[qc.qubits[0]], coupling_map),
            connected_qubits(initial_layout[qc.qubits[10]], coupling_map),
            connected_qubits(initial_layout[qc.qubits[20]], coupling_map),
        ]
        self.assertLessEqual({initial_layout[qc.qubits[i]] for i in range(10)}, components[0])
        self.assertLessEqual({initial_layout[qc.qubits[i]] for i in range(10, 20)}, components[1])
        self.assertLessEqual({initial_layout[qc.qubits[i]] for i in range(20, 30)}, components[2])

        # Check clbits are in order.
        # Traverse the output dag over the sole clbit, checking that the qubits of the ops
        # go in order between the components. This is a sanity check to ensure that routing
        # doesn't reorder a classical data dependency between components. Inside a component
        # we have the dag ordering so nothing should be out of order within a component.
        tqc_dag = circuit_to_dag(tqc)
        qubit_map = {qubit: index for index, qubit in enumerate(tqc_dag.qubits)}
        input_node = tqc_dag.input_map[tqc_dag.clbits[0]]
        first_meas_node = tqc_dag._multi_graph.find_successors_by_edge(
            input_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
        )[0]
        self.assertIsInstance(first_meas_node.op, Measure)
        self.assertIn(qubit_map[first_meas_node.qargs[0]], components[0])
        op_node = tqc_dag._multi_graph.find_successors_by_edge(
            first_meas_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
        )[0]
        while not isinstance(op_node.op, Measure):
            self.assertIn(qubit_map[op_node.qargs[0]], components[1])
            op_node = tqc_dag._multi_graph.find_successors_by_edge(
                op_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
            )[0]
        self.assertIn(qubit_map[op_node.qargs[0]], components[1])
        op_node = tqc_dag._multi_graph.find_successors_by_edge(
            op_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
        )[0]
        while not isinstance(op_node.op, Measure):
            self.assertIn(qubit_map[op_node.qargs[0]], components[2])
            op_node = tqc_dag._multi_graph.find_successors_by_edge(
                op_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
            )[0]
        self.assertIn(qubit_map[op_node.qargs[0]], components[2])

    @data("sabre", "stochastic", "basic", "lookahead")
    def test_basic_connected_circuit_dense_layout(self, routing_method):
        """Test basic connected circuit on disjoint backend"""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.measure_all()
        tqc = transpile(
            qc,
            self.backend,
            layout_method="dense",
            routing_method=routing_method,
            seed_transpiler=42,
        )
        for inst in tqc.data:
            qubits = tuple(tqc.find_bit(x).index for x in inst.qubits)
            op_name = inst.operation.name
            if op_name == "barrier":
                continue
            self.assertIn(qubits, self.backend.target[op_name])

    # Lookahead swap skipped for performance
    @data("sabre", "stochastic", "basic")
    def test_triple_circuit_dense_layout(self, routing_method):
        """Test a split circuit with one circuit component per chip."""
        qc = QuantumCircuit(30)
        qc.h(0)
        qc.h(10)
        qc.h(20)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        qc.cx(0, 6)
        qc.cx(0, 7)
        qc.cx(0, 8)
        qc.cx(0, 9)
        qc.ecr(10, 11)
        qc.ecr(10, 12)
        qc.ecr(10, 13)
        qc.ecr(10, 14)
        qc.ecr(10, 15)
        qc.ecr(10, 16)
        qc.ecr(10, 17)
        qc.ecr(10, 18)
        qc.ecr(10, 19)
        qc.cy(20, 21)
        qc.cy(20, 22)
        qc.cy(20, 23)
        qc.cy(20, 24)
        qc.cy(20, 25)
        qc.cy(20, 26)
        qc.cy(20, 27)
        qc.cy(20, 28)
        qc.cy(20, 29)
        qc.measure_all()
        tqc = transpile(
            qc,
            self.backend,
            layout_method="dense",
            routing_method=routing_method,
            seed_transpiler=42,
        )
        for inst in tqc.data:
            qubits = tuple(tqc.find_bit(x).index for x in inst.qubits)
            op_name = inst.operation.name
            if op_name == "barrier":
                continue
            self.assertIn(qubits, self.backend.target[op_name])

    @data("sabre", "stochastic", "basic", "lookahead")
    def test_triple_circuit_invalid_layout(self, routing_method):
        """Test a split circuit with one circuit component per chip."""
        qc = QuantumCircuit(30)
        qc.h(0)
        qc.h(10)
        qc.h(20)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        qc.cx(0, 6)
        qc.cx(0, 7)
        qc.cx(0, 8)
        qc.cx(0, 9)
        qc.ecr(10, 11)
        qc.ecr(10, 12)
        qc.ecr(10, 13)
        qc.ecr(10, 14)
        qc.ecr(10, 15)
        qc.ecr(10, 16)
        qc.ecr(10, 17)
        qc.ecr(10, 18)
        qc.ecr(10, 19)
        qc.cy(20, 21)
        qc.cy(20, 22)
        qc.cy(20, 23)
        qc.cy(20, 24)
        qc.cy(20, 25)
        qc.cy(20, 26)
        qc.cy(20, 27)
        qc.cy(20, 28)
        qc.cy(20, 29)
        qc.measure_all()
        with self.assertRaises(TranspilerError):
            transpile(
                qc,
                self.backend,
                layout_method="trivial",
                routing_method=routing_method,
                seed_transpiler=42,
            )

    # Lookahead swap skipped for performance reasons
    @data("sabre", "stochastic", "basic")
    def test_six_component_circuit_dense_layout(self, routing_method):
        """Test input circuit with more than 1 component per backend component."""
        qc = QuantumCircuit(42)
        qc.h(0)
        qc.h(10)
        qc.h(20)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        qc.cx(0, 6)
        qc.cx(0, 7)
        qc.cx(0, 8)
        qc.cx(0, 9)
        qc.ecr(10, 11)
        qc.ecr(10, 12)
        qc.ecr(10, 13)
        qc.ecr(10, 14)
        qc.ecr(10, 15)
        qc.ecr(10, 16)
        qc.ecr(10, 17)
        qc.ecr(10, 18)
        qc.ecr(10, 19)
        qc.cy(20, 21)
        qc.cy(20, 22)
        qc.cy(20, 23)
        qc.cy(20, 24)
        qc.cy(20, 25)
        qc.cy(20, 26)
        qc.cy(20, 27)
        qc.cy(20, 28)
        qc.cy(20, 29)
        qc.h(30)
        qc.cx(30, 31)
        qc.cx(30, 32)
        qc.cx(30, 33)
        qc.h(34)
        qc.cx(34, 35)
        qc.cx(34, 36)
        qc.cx(34, 37)
        qc.h(38)
        qc.cx(38, 39)
        qc.cx(39, 40)
        qc.cx(39, 41)
        qc.measure_all()
        tqc = transpile(
            qc,
            self.backend,
            layout_method="dense",
            routing_method=routing_method,
            seed_transpiler=42,
        )
        for inst in tqc.data:
            qubits = tuple(tqc.find_bit(x).index for x in inst.qubits)
            op_name = inst.operation.name
            if op_name == "barrier":
                continue
            self.assertIn(qubits, self.backend.target[op_name])

    @data(0, 1, 2, 3)
    def test_transpile_target_with_qubits_without_ops(self, opt_level):
        """Test qubits without operations aren't ever used."""
        target = Target(num_qubits=5)
        target.add_instruction(XGate(), {(i,): InstructionProperties(error=0.5) for i in range(3)})
        target.add_instruction(HGate(), {(i,): InstructionProperties(error=0.5) for i in range(3)})
        target.add_instruction(
            CXGate(), {edge: InstructionProperties(error=0.5) for edge in [(0, 1), (1, 2), (2, 0)]}
        )
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        tqc = transpile(qc, target=target, optimization_level=opt_level)
        invalid_qubits = {3, 4}
        self.assertEqual(tqc.num_qubits, 5)
        for inst in tqc.data:
            for bit in inst.qubits:
                self.assertNotIn(tqc.find_bit(bit).index, invalid_qubits)

    @data(0, 1, 2, 3)
    def test_transpile_target_with_qubits_without_ops_with_routing(self, opt_level):
        """Test qubits without operations aren't ever used."""
        target = Target(num_qubits=5)
        target.add_instruction(XGate(), {(i,): InstructionProperties(error=0.5) for i in range(4)})
        target.add_instruction(HGate(), {(i,): InstructionProperties(error=0.5) for i in range(4)})
        target.add_instruction(
            CXGate(),
            {edge: InstructionProperties(error=0.5) for edge in [(0, 1), (1, 2), (2, 0), (2, 3)]},
        )
        qc = QuantumCircuit(4)
        qc.x(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 3)
        qc.cx(0, 3)
        tqc = transpile(qc, target=target, optimization_level=opt_level)
        invalid_qubits = {
            4,
        }
        self.assertEqual(tqc.num_qubits, 5)
        for inst in tqc.data:
            for bit in inst.qubits:
                self.assertNotIn(tqc.find_bit(bit).index, invalid_qubits)

    @data(0, 1, 2, 3)
    def test_transpile_target_with_qubits_without_ops_circuit_too_large(self, opt_level):
        """Test qubits without operations aren't ever used and error if circuit needs them."""
        target = Target(num_qubits=5)
        target.add_instruction(XGate(), {(i,): InstructionProperties(error=0.5) for i in range(3)})
        target.add_instruction(HGate(), {(i,): InstructionProperties(error=0.5) for i in range(3)})
        target.add_instruction(
            CXGate(), {edge: InstructionProperties(error=0.5) for edge in [(0, 1), (1, 2), (2, 0)]}
        )
        qc = QuantumCircuit(4)
        qc.x(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        with self.assertRaises(TranspilerError):
            transpile(qc, target=target, optimization_level=opt_level)

    @data(0, 1, 2, 3)
    def test_transpile_target_with_qubits_without_ops_circuit_too_large_disconnected(
        self, opt_level
    ):
        """Test qubits without operations aren't ever used if a disconnected circuit needs them."""
        target = Target(num_qubits=5)
        target.add_instruction(XGate(), {(i,): InstructionProperties(error=0.5) for i in range(3)})
        target.add_instruction(HGate(), {(i,): InstructionProperties(error=0.5) for i in range(3)})
        target.add_instruction(
            CXGate(), {edge: InstructionProperties(error=0.5) for edge in [(0, 1), (1, 2), (2, 0)]}
        )
        qc = QuantumCircuit(5)
        qc.x(0)
        qc.x(1)
        qc.x(3)
        qc.x(4)
        with self.assertRaises(TranspilerError):
            transpile(qc, target=target, optimization_level=opt_level)

    @data(0, 1, 2, 3)
    def test_transpile_does_not_affect_backend_coupling(self, opt_level):
        """Test that transpiliation of a circuit does not mutate the `CouplingMap` stored by a V2
        backend.  Regression test of gh-9997."""
        if opt_level == 3:
            raise unittest.SkipTest("unitary resynthesis fails due to gh-10004")
        qc = QuantumCircuit(127)
        for i in range(1, 127):
            qc.ecr(0, i)
        backend = FakeSherbrooke()
        original_map = copy.deepcopy(backend.coupling_map)
        transpile(qc, backend, optimization_level=opt_level)
        self.assertEqual(original_map, backend.coupling_map)

    @combine(
        optimization_level=[0, 1, 2, 3],
        scheduling_method=["asap", "alap"],
    )
    def test_transpile_target_with_qubits_without_delays_with_scheduling(
        self, optimization_level, scheduling_method
    ):
        """Test qubits without operations aren't ever used."""
        no_delay_qubits = [1, 3, 4]
        target = Target(num_qubits=5, dt=1)
        target.add_instruction(
            XGate(), {(i,): InstructionProperties(duration=160) for i in range(4)}
        )
        target.add_instruction(
            HGate(), {(i,): InstructionProperties(duration=160) for i in range(4)}
        )
        target.add_instruction(
            CXGate(),
            {
                edge: InstructionProperties(duration=800)
                for edge in [(0, 1), (1, 2), (2, 0), (2, 3)]
            },
        )
        target.add_instruction(
            Delay(Parameter("t")), {(i,): None for i in range(4) if i not in no_delay_qubits}
        )
        qc = QuantumCircuit(4)
        qc.x(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 3)
        qc.cx(0, 3)
        tqc = transpile(
            qc,
            target=target,
            optimization_level=optimization_level,
            scheduling_method=scheduling_method,
        )
        invalid_qubits = {
            4,
        }
        self.assertEqual(tqc.num_qubits, 5)
        for inst in tqc.data:
            for bit in inst.qubits:
                self.assertNotIn(tqc.find_bit(bit).index, invalid_qubits)
                if isinstance(inst.operation, Delay):
                    self.assertNotIn(tqc.find_bit(bit).index, no_delay_qubits)
