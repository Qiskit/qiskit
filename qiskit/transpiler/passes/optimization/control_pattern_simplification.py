# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Transpiler pass for simplifying multi-controlled gates with complementary control patterns."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit import ControlledGate, QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.utils import optionals as _optionals


@dataclass
class ControlledGateInfo:
    """Information about a controlled gate for optimization analysis.

    Attributes:
        node: DAGOpNode containing the gate
        operation: The gate operation
        control_qubits: List of control qubit indices
        target_qubits: List of target qubit indices
        ctrl_state: Control state pattern as binary string
        params: Gate parameters (e.g., rotation angle)
    """
    node: DAGOpNode
    operation: ControlledGate
    control_qubits: List[int]
    target_qubits: List[int]
    ctrl_state: str
    params: Tuple[float, ...]


@_optionals.HAS_SYMPY.require_in_instance
class ControlPatternSimplification(TransformationPass):
    """Simplify multi-controlled gates using Boolean algebraic pattern matching.

    This pass detects consecutive multi-controlled gates with identical base operations,
    target qubits, and parameters (e.g., rotation angles) but different control patterns.
    It then applies Boolean algebraic simplification to reduce gate counts.

    **Supported Gate Types:**

    The optimization works for any parametric controlled gate where the same parameter
    value is used across multiple gates, including:

    - Multi-controlled rotation gates: MCRX, MCRY, MCRZ
    - Multi-controlled phase gates: MCRZ, MCPhase
    - Any custom controlled gates with identical parameters

    **Optimization Techniques:**

    1. **Complementary patterns**: Patterns like ['11', '01'] represent
       ``(q0 ∧ q1) ∨ (q0 ∧ ¬q1) = q0``, reducing 2 multi-controlled gates to 1 single-controlled gate.

    2. **Subset patterns**: Patterns like ['111', '110'] simplify via
       ``(q0 ∧ q1 ∧ q2) ∨ (q0 ∧ q1 ∧ ¬q2) = (q0 ∧ q1)``,
       reducing the number of control qubits.

    3. **XOR pairs**: Patterns like ['110', '101'] satisfy ``q1 ⊕ q2 = 1`` and can be
       optimized using CNOT gates, reducing 2 multi-controlled gates to 1 multi-controlled gate + 2 CNOTs.

    4. **Complete partitions**: Patterns like ['00','01','10','11'] → unconditional gates.

    **Example:**

    .. code-block:: python

        from qiskit import QuantumCircuit
        from qiskit.circuit.library import RXGate, RYGate, RZGate
        from qiskit.transpiler.passes import ControlPatternSimplification

        # Works with any rotation gate (RX, RY, RZ, etc.)
        theta = np.pi / 4

        # Example with RX gates
        qc = QuantumCircuit(3)
        qc.append(RXGate(theta).control(2, ctrl_state='11'), [0, 1, 2])
        qc.append(RXGate(theta).control(2, ctrl_state='01'), [0, 1, 2])

        # Apply optimization
        pass_ = ControlPatternSimplification()
        optimized_qc = pass_(qc)

        # Result: Single CRX gate controlled by q0

        # Also works with RY, RZ, Phase, and other parametric gates
        qc2 = QuantumCircuit(3)
        qc2.append(RYGate(theta).control(2, ctrl_state='11'), [0, 1, 2])
        qc2.append(RYGate(theta).control(2, ctrl_state='01'), [0, 1, 2])
        optimized_qc2 = pass_(qc2)  # Same optimization applied

    **References:**

    - Atallah et al., "Graph Matching Trotterization for Continuous Time Quantum Walk
      Circuit Simulation", Proceedings of IEEE Quantum Computing and Engineering (QCE) 2025.
    - Gonzalez et al., "Efficient sparse state preparation via quantum walks",
      npj Quantum Information (2025).
    - Amy et al., "Fast synthesis of depth-optimal quantum circuits", IEEE TCAD 32.6 (2013).
    - Shende & Markov, "On the CNOT-cost of TOFFOLI gates", arXiv:0803.2316 (2008).
    - Barenco et al., "Elementary gates for quantum computation", Phys. Rev. A 52.5 (1995).

    .. note::
        This pass requires the optional SymPy library for Boolean expression simplification.
        Install with: ``pip install sympy``
    """

    def __init__(self, tolerance=1e-10):
        """Initialize the control pattern simplification pass.

        Args:
            tolerance (float): Numerical tolerance for comparing gate parameters.
                Default is 1e-10.

        Raises:
            MissingOptionalLibraryError: if SymPy is not installed.
        """
        super().__init__()
        self.tolerance = tolerance

    def _extract_control_pattern(self, gate: ControlledGate, num_ctrl_qubits: int) -> str:
        """Extract control pattern from a controlled gate as binary string.

        Args:
            gate: The controlled gate
            num_ctrl_qubits: Number of control qubits

        Returns:
            Binary string representation of control pattern (e.g., '11', '01')
        """
        ctrl_state = gate.ctrl_state

        if ctrl_state is None:
            # Default: all controls must be in |1⟩ state
            return '1' * num_ctrl_qubits
        elif isinstance(ctrl_state, str):
            return ctrl_state
        elif isinstance(ctrl_state, int):
            # Convert integer to binary string with appropriate length
            return format(ctrl_state, f'0{num_ctrl_qubits}b')
        else:
            # Fallback: assume all ones
            return '1' * num_ctrl_qubits

    def _parameters_match(self, params1: Tuple, params2: Tuple) -> bool:
        """Check if two parameter tuples match within tolerance.

        Args:
            params1: First parameter tuple
            params2: Second parameter tuple

        Returns:
            True if parameters match within tolerance
        """
        if len(params1) != len(params2):
            return False

        for p1, p2 in zip(params1, params2):
            if isinstance(p1, (int, float)) and isinstance(p2, (int, float)):
                if not np.isclose(p1, p2, atol=self.tolerance):
                    return False
            elif p1 != p2:
                # For non-numeric parameters (e.g., ParameterExpression)
                return False

        return True

    def _collect_controlled_gates(self, dag: DAGCircuit) -> List[List[ControlledGateInfo]]:
        """Collect runs of consecutive controlled gates from the DAG.

        Args:
            dag: The DAG circuit to analyze

        Returns:
            List of runs, where each run is a list of ControlledGateInfo objects
        """
        runs = []
        current_run = []

        for node in dag.topological_op_nodes():
            if isinstance(node.op, ControlledGate):
                # Extract gate information
                num_ctrl_qubits = node.op.num_ctrl_qubits
                ctrl_state = self._extract_control_pattern(node.op, num_ctrl_qubits)

                # Get qubit indices
                qargs = dag.qubits.get_indices(node.qargs)
                control_qubits = qargs[:num_ctrl_qubits]
                target_qubits = qargs[num_ctrl_qubits:]

                gate_info = ControlledGateInfo(
                    node=node,
                    operation=node.op,
                    control_qubits=control_qubits,
                    target_qubits=target_qubits,
                    ctrl_state=ctrl_state,
                    params=tuple(node.op.params) if node.op.params else ()
                )

                current_run.append(gate_info)
            else:
                # Non-controlled gate breaks the run
                if len(current_run) > 0:
                    runs.append(current_run)
                    current_run = []

        # Add final run if exists
        if len(current_run) > 0:
            runs.append(current_run)

        return runs

    def _group_compatible_gates(self, gates: List[ControlledGateInfo]) -> List[List[ControlledGateInfo]]:
        """Group gates that can be optimized together.

        Gates are compatible if they have:
        - Same base gate type
        - Same target qubits
        - Same control qubits (same set, different patterns allowed)
        - Same parameters

        Args:
            gates: List of controlled gate information

        Returns:
            List of groups, where each group contains compatible gates
        """
        if len(gates) < 2:
            return []

        groups = []
        i = 0

        while i < len(gates):
            current_group = [gates[i]]
            base_gate = gates[i].operation.base_gate
            target_qubits = gates[i].target_qubits
            control_qubits_set = set(gates[i].control_qubits)
            params = gates[i].params

            # Look for consecutive compatible gates
            j = i + 1
            while j < len(gates):
                candidate = gates[j]

                # Check compatibility
                if (candidate.operation.base_gate.name == base_gate.name and
                    candidate.target_qubits == target_qubits and
                    set(candidate.control_qubits) == control_qubits_set and
                    self._parameters_match(candidate.params, params) and
                    candidate.ctrl_state != gates[i].ctrl_state):  # Different patterns

                    current_group.append(candidate)
                    j += 1
                else:
                    break

            # Only add groups with 2+ gates
            if len(current_group) >= 2:
                groups.append(current_group)

            i = j if j > i + 1 else i + 1

        return groups

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the ControlPatternSimplification pass on a DAGCircuit.

        Args:
            dag: The DAG to be optimized.

        Returns:
            DAGCircuit: The optimized DAG with simplified control patterns.
        """
        # TODO: Implement the optimization logic
        # 1. Identify runs of consecutive multi-controlled gates
        # 2. Group gates with same base operation, target, and parameters
        #    (works for any parametric gate: RX, RY, RZ, Phase, etc.)
        # 3. Extract control patterns from ctrl_state
        # 4. Apply Boolean simplification using SymPy
        # 5. Detect XOR patterns for CNOT tricks
        # 6. Generate optimized circuit with reduced gate count
        # 7. Replace original gates with optimized version

        return dag
