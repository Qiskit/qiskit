# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test the multi-stage pass manager."""

from test import QiskitTestCase
from copy import copy

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.passmanager.optimization_pm import OptimizationPassManager
from qiskit.passmanager.lowering_pm import LoweringPassManager
from qiskit.passmanager.multi_stage_pm import MultiStagePassManager
from qiskit.transpiler import generate_preset_pass_manager, CouplingMap
from .tasks import (
    BaseTask,
    CircuitNoOp,
    CircuitAnalysis,
    CircuitToDAG,
    DAGNoOp,
    DAGRemoveIdentity,
    RequirePropertySet,
)


class PauliIR:
    """A bare minimum Pauli-string IR keeping a dense list of global Paulis to apply."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.instructions = []

    def apply(self, pauli: str):
        """Append a Pauli. Must have global size."""
        if len(pauli) != self.num_qubits:
            raise ValueError("Incompatible number of qubits")
        self.instructions.append(pauli)

    def to_circuit(self) -> QuantumCircuit:
        """Convert to a circuit."""
        circuit = QuantumCircuit(self.num_qubits)
        for pauli in self.instructions:
            circuit.pauli(pauli, circuit.qubits)
        return circuit


class RemovePauliIdentities(BaseTask[PauliIR, PauliIR]):
    """A pass removing Pauli identities."""

    def __init__(self):
        super().__init__(PauliIR)

    def run(self, passmanager_ir, property_set=None):
        to_remove = []
        for i, pauli in enumerate(passmanager_ir.instructions):
            if all(p == "I" for p in pauli):
                to_remove.append(i)

        # remove from the back to keep indices valid
        for i in reversed(to_remove):
            del passmanager_ir.instructions[i]

        return passmanager_ir


class PauliToCircuit(BaseTask[PauliIR, QuantumCircuit]):
    """Lower PauliIR to QuantumCircuit."""

    def __init__(self):
        super().__init__(PauliIR)

    def run(self, passmanager_ir, property_set=None):
        return passmanager_ir.to_circuit()


class TestMultiStagePM(QiskitTestCase):
    """Test the multi-stage pass manager."""

    def test_stage_names(self):
        """Test getting the stage names."""
        opt1 = OptimizationPassManager([CircuitNoOp()])
        lower = LoweringPassManager(CircuitToDAG())
        opt2 = OptimizationPassManager([DAGNoOp()])

        staged_pm = MultiStagePassManager(first=opt1, lower=lower, last=opt2)
        expected = ("first", "lower", "last")
        with self.subTest(msg="stage names"):
            self.assertEqual(expected, staged_pm.stages)

        with self.subTest(msg="getting stages"):
            self.assertEqual(opt1, getattr(staged_pm, expected[0]))
            self.assertEqual(lower, getattr(staged_pm, expected[1]))
            self.assertEqual(opt2, getattr(staged_pm, expected[2]))

    def test_multi_ir(self):
        """Test changing the IR in the pipeline."""
        pauli = OptimizationPassManager([RemovePauliIdentities()])
        pauli_to_circuit = LoweringPassManager(PauliToCircuit())
        circuit = OptimizationPassManager([CircuitNoOp()])
        qc_to_dag = LoweringPassManager(CircuitToDAG())  # "circuit_to_dag" is reserved
        dag = OptimizationPassManager([DAGNoOp()])

        staged_pm = MultiStagePassManager(
            pauli=pauli,
            pauli_to_circuit=pauli_to_circuit,
            circuit=circuit,
            circuit_to_dag=qc_to_dag,
            dag=dag,
        )
        input_program = PauliIR(3)
        input_program.apply("XYZ")
        input_program.apply("ZZI")
        input_program.apply("III")
        input_program.apply("IYI")

        output_program = staged_pm.run(input_program)

        expected = QuantumCircuit(3)
        expected.pauli("XYZ", [0, 1, 2])
        expected.pauli("ZZI", [0, 1, 2])
        expected.pauli("IYI", [0, 1, 2])

        self.assertEqual(circuit_to_dag(expected), output_program)

    def test_generate_preset_pm_compatibility(self):
        """Check the existing pipeline can be a stage."""

        pauli = OptimizationPassManager([RemovePauliIdentities()])
        pauli_to_circuit = LoweringPassManager(PauliToCircuit())
        circuit = OptimizationPassManager([CircuitNoOp()])
        qc_to_dag = LoweringPassManager(CircuitToDAG())  # "circuit_to_dag" is reserved
        dag = generate_preset_pass_manager(
            coupling_map=CouplingMap.from_line(10), basis_gates=["sx", "x", "rz", "cx"]
        )

        staged_pm = MultiStagePassManager(
            pauli=pauli,
            pauli_to_circuit=pauli_to_circuit,
            circuit=circuit,
            circuit_to_dag=qc_to_dag,
            dag=dag,
        )
        input_program = PauliIR(3)
        input_program.apply("ZZI")
        input_program.apply("ZZI")
        input_program.apply("III")
        input_program.apply("IXI")

        output_program = staged_pm.run(input_program)

        self.assertEqual(output_program.count_ops(), {"x": 1})
        self.assertEqual(output_program.num_qubits(), 10)

    def test_invalid_ir(self):
        """Test invalid IR pipeline.

        This can be set up by the user since we cannot generally validate the types
        are correct.
        """
        opt1 = OptimizationPassManager([CircuitNoOp()])
        opt2 = OptimizationPassManager([DAGNoOp()])

        invalid_staged_pm = MultiStagePassManager(opt1=opt1, opt2=opt2)
        input_program = QuantumCircuit(1)

        # This TypeError is only raised since the passes check the input type at runtime
        with self.assertRaises(TypeError):
            _ = invalid_staged_pm.run(input_program)

    def test_stage_replacement(self):
        """Test replacing stages."""
        circuit_noop = OptimizationPassManager([CircuitNoOp()])
        lower = LoweringPassManager(CircuitToDAG())
        dag_noop = OptimizationPassManager([DAGNoOp()])
        remove_iden = OptimizationPassManager([DAGRemoveIdentity()])

        staged_pm = MultiStagePassManager(
            circuit_opt=circuit_noop, circuit_to_dag=lower, dag_opt=dag_noop
        )

        circuit = QuantumCircuit(1)
        circuit.rz(1e-10, 0)

        with self.subTest(msg="no op"):
            out = staged_pm.run(circuit)
            self.assertEqual(circuit_to_dag(circuit), out)

        # change the DAG optimization stage to remove close-to-identity gates
        staged_pm.dag_opt = remove_iden

        with self.subTest(msg="remove identities"):
            out = staged_pm.run(circuit)
            self.assertEqual(0, sum(out.count_ops().values()))

    def test_callback(self):
        """Test the callback works."""
        circuit = OptimizationPassManager([CircuitNoOp(), CircuitAnalysis()])
        lower = LoweringPassManager(CircuitToDAG())
        dag = OptimizationPassManager([DAGRemoveIdentity()])

        staged_pm = MultiStagePassManager(circuit=circuit, circuit_to_dag=lower, dag=dag)

        # Store all data from the callback to verify it has been passed correctly. This should
        # contain 4 tasks: CircuitNoOp, CircuitAnalysis, CircuitToDag, and DAGRemoveIdentity
        properties = []

        def callback(task, passmanager_ir, property_set, running_time, count):
            properties.append(
                (
                    task.__class__.__name__,
                    copy(passmanager_ir),
                    copy(property_set),
                    running_time,
                    count,
                )
            )

        input_program = QuantumCircuit(3)
        input_program.h(0)
        input_program.rz(1e-10, 0)
        input_program.cswap(0, 1, 2)

        out = staged_pm.run(input_program, callback=callback)

        # Test after task 0
        self.assertEqual(properties[0][0], "CircuitNoOp")
        self.assertEqual(properties[0][1], input_program)
        self.assertEqual(properties[0][2], {})  # empty property set

        # Test after task 1
        self.assertEqual(properties[1][0], "CircuitAnalysis")
        self.assertEqual(properties[1][1], input_program)
        self.assertEqual(properties[1][2]["ops"], input_program.count_ops())

        # Test after task 2
        self.assertEqual(properties[2][0], "CircuitToDAG")
        self.assertEqual(properties[2][1], circuit_to_dag(input_program))
        self.assertEqual(properties[2][2]["ops"], properties[1][2]["ops"])  # property set remains

        # Test after task 3
        self.assertEqual(properties[3][0], "DAGRemoveIdentity")
        self.assertEqual(properties[3][1], out)
        self.assertEqual(properties[3][2]["ops"], properties[1][2]["ops"])

        # All recorded runtimes should be non-negative, and the tasks should be counted
        for i, prop in enumerate(properties):
            self.assertGreaterEqual(prop[3], 0.0)  # runtime
            self.assertEqual(prop[4], i + 1)  # task count

    def test_single_stage(self):
        """Test a pipeline with a single stage."""
        pm = MultiStagePassManager(circuit_opt=OptimizationPassManager([CircuitNoOp()]))

        circuit = QuantumCircuit(2)
        circuit.h(0)

        self.assertEqual(circuit, pm.run(circuit))

    def test_initial_property_set(self):
        """Test that a pre-populated property set is visible to tasks."""
        pm = MultiStagePassManager(
            circuit_opt=OptimizationPassManager([RequirePropertySet("seed")])
        )
        initial = {"seed": 42}
        circuit = QuantumCircuit(1)

        with self.subTest(msg="valid initial set"):
            out = pm.run(circuit, property_set=initial)
            self.assertEqual(out, QuantumCircuit(1))

        with self.subTest(msg="missing initial set"):
            with self.assertRaises(ValueError):
                _ = pm.run(QuantumCircuit(1))
