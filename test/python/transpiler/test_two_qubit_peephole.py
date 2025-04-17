# This code is part of Qiskit.
#
# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-function-docstring

"""
Tests for the default UnitarySynthesis transpiler pass.
"""

import math
import numpy as np
import ddt

from qiskit import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.random import random_unitary
from qiskit.transpiler import PassManager, CouplingMap, Target, InstructionProperties
from qiskit.transpiler.passes import TwoQubitPeepholeOptimization, TrivialLayout
from qiskit.circuit.library import (
    IGate,
    CXGate,
    RZGate,
    RXGate,
    SXGate,
    XGate,
    iSwapGate,
    ECRGate,
    UGate,
    ZGate,
    RYYGate,
    RZZGate,
    RXXGate,
)
from qiskit.circuit import Measure
from qiskit.circuit.controlflow import IfElseOp
from qiskit.circuit import Parameter, Gate

from test import combine  # pylint: disable=wrong-import-order
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from test.python.providers.fake_mumbai_v2 import (  # pylint: disable=wrong-import-order
    FakeMumbaiFractionalCX,
)
from ..legacy_cmaps import YORKTOWN_CMAP


class FakeBackend2QV2(GenericBackendV2):
    """A 2-qubit fake backend"""

    def __init__(self):
        super().__init__(num_qubits=2, basis_gates=["rx", "u"], seed=42)
        cx_props = {
            (0, 1): InstructionProperties(duration=5.23e-7, error=0.00098115),
        }
        self._target.add_instruction(CXGate(), cx_props)
        ecr_props = {
            (1, 0): InstructionProperties(duration=4.52e-9, error=0.0000132115),
        }
        self._target.add_instruction(ECRGate(), ecr_props)


class FakeBackend5QV2(GenericBackendV2):
    """A 5-qubit fake backend"""

    def __init__(self, bidirectional=True):
        super().__init__(num_qubits=5, basis_gates=["u"], seed=42)
        cx_props = {
            (0, 1): InstructionProperties(duration=5.23e-7, error=0.00098115),
            (3, 4): InstructionProperties(duration=5.23e-7, error=0.00098115),
        }
        if bidirectional:
            cx_props[(1, 0)] = InstructionProperties(duration=6.23e-7, error=0.00099115)
            cx_props[(4, 3)] = InstructionProperties(duration=7.23e-7, error=0.00099115)
        self._target.add_instruction(CXGate(), cx_props)
        ecr_props = {
            (1, 2): InstructionProperties(duration=4.52e-9, error=0.0000132115),
            (2, 3): InstructionProperties(duration=4.52e-9, error=0.0000132115),
        }
        if bidirectional:
            ecr_props[(2, 1)] = InstructionProperties(duration=5.52e-9, error=0.0000232115)
            ecr_props[(3, 2)] = InstructionProperties(duration=5.52e-9, error=0.0000232115)


@ddt
class TestTwoQubitPeepholeOptimization(QiskitTestCase):
    """Test TwoQubitPeepholeOptimization."""

    @combine(
        bidirectional=[True, False],
        dsc=(
            "test natural_direction works with transpile using a"
            "target with multiple 2q gates with bidirectional={bidirectional}"
        ),
        name="bidirectional_{bidirectional}",
    )
    def test_coupling_map_transpile_with_backendv2(self, bidirectional):
        backend = FakeBackend5QV2(bidirectional)
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.append(random_unitary(4, seed=1), [0, 1])
        circ_01 = TwoQubitPeepholeOptimization(backend.target)(circ)
        circ_01_index = {qubit: index for index, qubit in enumerate(circ_01.qubits)}
        self.assertGreaterEqual(len(circ_01.get_instructions("cx")), 1)
        for instr in circ_01.get_instructions("cx"):
            self.assertEqual(
                (0, 1), (circ_01_index[instr.qubits[0]], circ_01_index[instr.qubits[1]])
            )

    @combine(
        bidirectional=[True, False],
        dsc=(
            "Test direction with transpile using a "
            "target with multiple 2q gates with bidirectional={bidirectional}"
            "direction [0, 1] is lower error and should be picked."
        ),
        name="bidirectional_{bidirectional}",
    )
    def test_coupling_unequal_duration_with_backendv2(self, bidirectional):
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.append(random_unitary(4, seed=1), [1, 0])
        backend = FakeBackend5QV2(bidirectional)
        tqc = TwoQubitPeepholeOptimization(backend.target)(circ)
        tqc_index = {qubit: index for index, qubit in enumerate(tqc.qubits)}
        self.assertGreaterEqual(len(tqc.get_instructions("cx")), 1)
        for instr in tqc.get_instructions("cx"):
            self.assertEqual((0, 1), (tqc_index[instr.qubits[0]], tqc_index[instr.qubits[1]]))

    def test_non_overlapping_kak_gates_with_backendv2(self):
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.append(random_unitary(4, seed=1), [1, 0])
        backend = FakeBackend2QV2()
        tqc = TwoQubitPeepholeOptimization(backend.target)(circ)

        tqc_index = {qubit: index for index, qubit in enumerate(tqc.qubits)}
        self.assertGreaterEqual(len(tqc.get_instructions("ecr")), 1)
        for instr in tqc.get_instructions("ecr"):
            self.assertEqual((1, 0), (tqc_index[instr.qubits[0]], tqc_index[instr.qubits[1]]))

    def test_fractional_cx_with_backendv2(self):
        """Test fractional CX gets used if present in target."""
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.append(random_unitary(4, seed=1), [0, 1])
        backend = FakeMumbaiFractionalCX()
        synth_pass = TwoQubitPeepholeOptimization(target=backend.target)
        tqc = synth_pass(circ)
        tqc_index = {qubit: index for index, qubit in enumerate(tqc.qubits)}
        self.assertGreaterEqual(len(tqc.get_instructions("rzx")), 1)
        for instr in tqc.get_instructions("rzx"):
            self.assertEqual((0, 1), (tqc_index[instr.qubits[0]], tqc_index[instr.qubits[1]]))

    def test_reverse_direction(self):
        target = Target(2)
        target.add_instruction(CXGate(), {(0, 1): InstructionProperties(error=1.2e-6)})
        target.add_instruction(ECRGate(), {(0, 1): InstructionProperties(error=1.2e-7)})
        target.add_instruction(
            UGate(Parameter("theta"), Parameter("phi"), Parameter("lam")), {(0,): None, (1,): None}
        )
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.append(random_unitary(4, seed=1), [1, 0])
        tqc = TwoQubitPeepholeOptimization(target)(circ)
        tqc_index = {qubit: index for index, qubit in enumerate(tqc.qubits)}
        self.assertGreaterEqual(len(tqc.get_instructions("ecr")), 1)
        for instr in tqc.get_instructions("ecr"):
            self.assertEqual((0, 1), (tqc_index[instr.qubits[0]], tqc_index[instr.qubits[1]]))

    def test_controlled_basis(self):
        target = Target(2)
        target.add_instruction(RYYGate(np.pi / 8), {(0, 1): InstructionProperties(error=1.2e-6)})
        target.add_instruction(
            UGate(Parameter("theta"), Parameter("phi"), Parameter("lam")), {(0,): None, (1,): None}
        )
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.append(random_unitary(4, seed=1), [1, 0])
        tqc = TwoQubitPeepholeOptimization(target)(circ)
        self.assertGreaterEqual(len(tqc.get_instructions("ryy")), 1)
        self.assertEqual(Operator(tqc), Operator(circ))

    def test_approximation_controlled(self):
        target = Target(2)
        target.add_instruction(RZZGate(np.pi / 10), {(0, 1): InstructionProperties(error=0.006)})
        target.add_instruction(RXXGate(np.pi / 3), {(0, 1): InstructionProperties(error=0.01)})
        target.add_instruction(
            UGate(Parameter("theta"), Parameter("phi"), Parameter("lam")),
            {(0,): InstructionProperties(error=0.001), (1,): InstructionProperties(error=0.002)},
        )
        circ = QuantumCircuit(2)
        circ.append(random_unitary(4, seed=7), [1, 0])

        dag = circuit_to_dag(circ)
        dag_100 = TwoQubitPeepholeOptimization(target=target, approximation_degree=1.0).run(dag)
        dag_99 = TwoQubitPeepholeOptimization(target=target, approximation_degree=0.99).run(dag)
        self.assertGreaterEqual(dag_100.depth(), dag_99.depth())
        self.assertEqual(Operator(dag_to_circuit(dag_100)), Operator(circ))

    def test_mapping_control_flow(self):
        """Test that inner dags use proper qubit mapping."""
        qr = QuantumRegister(3, "q")
        qc = QuantumCircuit(qr)

        # Create target that supports CX only between 0 and 2.
        fake_target = Target()
        fake_target.add_instruction(CXGate(), {(0, 2): None})
        fake_target.add_instruction(
            UGate(Parameter("t"), Parameter("p"), Parameter("l")),
            {
                (0,): None,
                (1,): None,
                (2,): None,
            },
        )

        qc_uni1 = QuantumCircuit(2)
        qc_uni1.swap(0, 1)
        qc_uni1_mat = Operator(qc_uni1)

        loop_body = QuantumCircuit(2)
        loop_body.unitary(qc_uni1_mat, [0, 1])

        # Loop body uses qubits 0 and 2, mapped to 0 and 1 in the block.
        # If synthesis doesn't handle recursive mapping, it'll incorrectly
        # look for a CX on (0, 1) instead of on (0, 2).
        qc.for_loop((0,), None, loop_body, [0, 2], [])

        result = TwoQubitPeepholeOptimization(fake_target)(qc)
        self.assertIsInstance(result, QuantumCircuit)

    def test_single_qubit_with_target(self):
        """Test input circuit with only 1q works with target."""
        qc = QuantumCircuit(1)
        qc.append(ZGate(), [qc.qubits[0]])
        dag = circuit_to_dag(qc)
        backend = GenericBackendV2(num_qubits=5, seed=42)
        unitary_synth_pass = TwoQubitPeepholeOptimization(target=backend.target)
        result_dag = unitary_synth_pass.run(dag)
        result_qc = dag_to_circuit(result_dag)
        self.assertEqual(qc, result_qc)

    def test_two_qubit_identity_with_target(self):
        """Test input single qubit identity works with target."""
        qc = QuantumCircuit(2)
        qc.unitary(np.eye(4, dtype=complex), [0, 1])
        dag = circuit_to_dag(qc)
        backend = GenericBackendV2(num_qubits=5)
        unitary_synth_pass = TwoQubitPeepholeOptimization(target=backend.target)
        result_dag = unitary_synth_pass.run(dag)
        result_qc = dag_to_circuit(result_dag)
        self.assertEqual(result_qc, QuantumCircuit(2))

    def test_unitary_synthesis_with_ideal_and_variable_width_ops(self):
        """Test unitary synthesis works with a target that contains ideal and variadic ops."""
        qc = QuantumCircuit(2)
        qc.unitary(np.eye(4), [0, 1])
        dag = circuit_to_dag(qc)
        target = GenericBackendV2(num_qubits=5).target
        target.add_instruction(IfElseOp, name="if_else")
        target.add_instruction(ZGate())
        target.add_instruction(ECRGate())
        unitary_synth_pass = TwoQubitPeepholeOptimization(target=target)
        result_dag = unitary_synth_pass.run(dag)
        result_qc = dag_to_circuit(result_dag)
        self.assertEqual(result_qc, QuantumCircuit(2))

    def test_unitary_synthesis_custom_gate_target(self):
        qc = QuantumCircuit(2)
        qc.unitary(np.eye(4), [0, 1])
        dag = circuit_to_dag(qc)

        class CustomGate(Gate):
            """Custom Opaque Gate"""

            def __init__(self):
                super().__init__("custom", 2, [])

        target = Target(num_qubits=2)
        target.add_instruction(
            UGate(Parameter("t"), Parameter("p"), Parameter("l")), {(0,): None, (1,): None}
        )
        target.add_instruction(CustomGate(), {(0, 1): None, (1, 0): None})
        unitary_synth_pass = TwoQubitPeepholeOptimization(target=target)
        result_dag = unitary_synth_pass.run(dag)
        result_qc = dag_to_circuit(result_dag)
        self.assertEqual(result_qc, qc)

    def test_iswap_no_cx_synthesis_succeeds(self):
        """Test basis set with iswap but no cx can synthesize a circuit"""
        target = Target()
        theta = Parameter("theta")

        i_props = {
            (0,): InstructionProperties(duration=35.5e-9, error=0.000413),
            (1,): InstructionProperties(duration=35.5e-9, error=0.000502),
        }
        target.add_instruction(IGate(), i_props)
        rz_props = {
            (0,): InstructionProperties(duration=0, error=0),
            (1,): InstructionProperties(duration=0, error=0),
        }
        target.add_instruction(RZGate(theta), rz_props)
        sx_props = {
            (0,): InstructionProperties(duration=35.5e-9, error=0.000413),
            (1,): InstructionProperties(duration=35.5e-9, error=0.000502),
        }
        target.add_instruction(SXGate(), sx_props)
        x_props = {
            (0,): InstructionProperties(duration=35.5e-9, error=0.000413),
            (1,): InstructionProperties(duration=35.5e-9, error=0.000502),
        }
        target.add_instruction(XGate(), x_props)
        iswap_props = {
            (0, 1): InstructionProperties(duration=519.11e-9, error=0.01201),
            (1, 0): InstructionProperties(duration=554.66e-9, error=0.01201),
        }
        target.add_instruction(iSwapGate(), iswap_props)
        measure_props = {
            (0,): InstructionProperties(duration=5.813e-6, error=0.0751),
            (1,): InstructionProperties(duration=5.813e-6, error=0.0225),
        }
        target.add_instruction(Measure(), measure_props)

        qc = QuantumCircuit(2)
        cxmat = Operator(CXGate()).to_matrix()
        qc.unitary(cxmat, [0, 1])
        unitary_synth_pass = TwoQubitPeepholeOptimization(target=target)
        dag = circuit_to_dag(qc)
        result_dag = unitary_synth_pass.run(dag)
        result_qc = dag_to_circuit(result_dag)
        self.assertTrue(np.allclose(Operator(result_qc.to_gate()).to_matrix(), cxmat))

    def test_custom_rxx_gate_in_target(self):
        """Test synthesis with custom parameterized gate in target."""

        theta = Parameter("θ")
        lam = Parameter("λ")
        phi = Parameter("ϕ")

        target = Target(num_qubits=2)
        target.add_instruction(RZGate(lam))
        target.add_instruction(RXGate(phi))
        target.add_instruction(RXXGate(theta))

        qc = QuantumCircuit(2)
        qc.unitary(random_unitary(4, seed=1234), [0, 1])
        qc_transpiled = TwoQubitPeepholeOptimization(target=target)(qc)
        opcount = qc_transpiled.count_ops()
        self.assertTrue(set(opcount).issubset({"rz", "rx", "rxx"}))

        self.assertTrue(np.allclose(Operator(qc_transpiled), Operator(qc)))

    def test_custom_parameterized_gate_in_target_skips(self):
        """Test that synthesis is skipped with custom parameterized
        gate in target that is not RXX equivalent."""

        class CustomXYGate(Gate):
            """Custom Gate subclass that's not a standard gate and not RXX equivalent"""

            _standard_gate = None

            def __init__(self, theta: ParameterValueType, label=None):
                """Create new custom rotstion XY gate."""
                super().__init__("MyCustomXYGate", 2, [theta])

            def __array__(self, dtype=None):
                """Return a Numpy.array for the custom gate."""
                theta = self.params[0]
                cos = math.cos(theta)
                isin = 1j * math.sin(theta)
                return np.array(
                    [[1, 0, 0, 0], [0, cos, -isin, 0], [0, -isin, cos, 0], [0, 0, 0, 1]],
                    dtype=dtype,
                )

            def inverse(self, annotated: bool = False):
                return CustomXYGate(-self.params[0])

        theta = Parameter("θ")
        lam = Parameter("λ")
        phi = Parameter("ϕ")

        target = Target(num_qubits=2)
        target.add_instruction(RZGate(lam))
        target.add_instruction(RXGate(phi))
        target.add_instruction(CustomXYGate(theta))

        qc = QuantumCircuit(2)
        qc.unitary(random_unitary(4, seed=1234), [0, 1])
        qc_transpiled = TwoQubitPeepholeOptimization(target=target)(qc)
        opcount = qc_transpiled.count_ops()
        self.assertTrue(set(opcount).issubset({"unitary"}))
        self.assertTrue(np.allclose(Operator(qc_transpiled), Operator(qc)))

    def test_determinism(self):
        """Test that the decomposition is deterministic."""
        gate_counts = {"rx": 6, "rz": 12, "iswap": 2}
        basis_gates = ["rx", "rz", "iswap"]
        target = Target.from_configuration(basis_gates=basis_gates)
        pm = generate_preset_pass_manager(target=target, optimization_level=2, seed_transpiler=42)

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        for _ in range(10):
            out = pm.run(qc)
            self.assertTrue(Operator(out).equiv(qc))
            self.assertTrue(set(out.count_ops()).issubset(basis_gates))
            for basis_gate in basis_gates:
                self.assertLessEqual(out.count_ops()[basis_gate], gate_counts[basis_gate])

    @combine(gate=["unitary", "swap"])
    def test_two_qubit_synthesis_to_directional_cx_target(self, gate):
        """Verify two qubit unitaries are synthesized to match basis gates."""
        # TODO: should make check more explicit e.g. explicitly set gate
        # direction in test instead of using specific fake backend
        backend = GenericBackendV2(
            num_qubits=5,
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            coupling_map=YORKTOWN_CMAP,
            seed=1,
        )
        coupling_map = CouplingMap(backend.coupling_map)
        triv_layout_pass = TrivialLayout(coupling_map)

        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        if gate == "unitary":
            qc.unitary(random_unitary(4, seed=12), [0, 1])
        elif gate == "swap":
            qc.swap(qr[0], qr[1])

        unisynth_pass = TwoQubitPeepholeOptimization(
            target=backend.target,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        qc_out = pm.run(qc)
        self.assertEqual(Operator(qc), Operator(qc_out))
