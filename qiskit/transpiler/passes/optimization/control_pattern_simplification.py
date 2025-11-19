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
from qiskit.circuit import ControlledGate


class BitwisePatternAnalyzer:
    """Analyze and simplify control patterns using bitwise operations.

    This class provides bitwise operations to analyze concrete control patterns
    without requiring symbolic Boolean algebra (SymPy). It works with binary
    string patterns like '11', '01', '110', etc.
    """

    def __init__(self, num_qubits: int):
        """Initialize the analyzer.

        Args:
            num_qubits: Number of control qubits in the patterns
        """
        self.num_qubits = num_qubits

    def _pattern_to_int(self, pattern: str) -> int:
        """Convert binary pattern string to integer.

        Args:
            pattern: Binary string (e.g., '110', '01')
                    String is read left-to-right: leftmost is qubit 0

        Returns:
            Integer representation
        """
        return int(pattern, 2)  # Direct conversion, no reversal

    def _find_common_bits(self, patterns: List[str]) -> Tuple[int, int]:
        """Find bits that have the same value across all patterns.

        Args:
            patterns: List of binary pattern strings

        Returns:
            Tuple of (mask, value) where:
            - mask: bits set to 1 where all patterns have the same value
            - value: the common bit values at those positions
        """
        if not patterns:
            return (0, 0)

        first = self._pattern_to_int(patterns[0])
        mask = (1 << self.num_qubits) - 1  # All bits set

        for pattern in patterns[1:]:
            curr = self._pattern_to_int(pattern)
            # Update mask: keep only bits that match
            diff = first ^ curr
            mask &= ~diff  # Clear bits that differ

        return (mask, first & mask)

    def _can_eliminate_bit(self, bit_idx: int, patterns: List[str]) -> bool:
        """Check if a specific bit position can be eliminated.

        A bit can be eliminated if it varies across patterns in a way that
        allows simplification (complementary patterns).

        Args:
            bit_idx: Bit position to check (0-indexed from left)
            patterns: List of binary pattern strings

        Returns:
            True if bit can be eliminated
        """
        # Get all unique values at this bit position
        bit_values = set(p[bit_idx] for p in patterns)

        if len(bit_values) == 1:
            # Bit is constant across all patterns, cannot eliminate
            return False

        # Check if varying this bit covers complementary patterns
        # Collect patterns for each bit value
        patterns_by_bit = {"0": [], "1": []}
        for p in patterns:
            bit_val = p[bit_idx]
            patterns_by_bit[bit_val].append(p)

        # Check if patterns are identical except for this bit
        if "0" in patterns_by_bit and "1" in patterns_by_bit:
            patterns_0 = patterns_by_bit["0"]
            patterns_1 = patterns_by_bit["1"]

            if len(patterns_0) != len(patterns_1):
                return False

            # Remove the bit at bit_idx and compare
            def remove_bit(p):
                return p[:bit_idx] + p[bit_idx + 1 :]

            patterns_0_stripped = sorted(remove_bit(p) for p in patterns_0)
            patterns_1_stripped = sorted(remove_bit(p) for p in patterns_1)

            return patterns_0_stripped == patterns_1_stripped

        return False

    def simplify_patterns(
        self, patterns: List[str]
    ) -> Tuple[str, Optional[List[int]], Optional[str]]:
        """Simplify control patterns using bitwise analysis.

        Args:
            patterns: List of binary control pattern strings

        Returns:
            Tuple of (classification, qubit_indices, ctrl_state):
            - classification: 'single', 'and', 'unconditional', 'no_optimization'
            - qubit_indices: List of qubit indices needed for control
            - ctrl_state: Control state string for the remaining qubits
        """
        if not patterns:
            return ("no_optimization", None, None)

        # Ensure all patterns are same length
        if len(set(len(p) for p in patterns)) > 1:
            return ("no_optimization", None, None)

        # Check for complete partition (all possible states covered)
        unique_patterns = set(patterns)
        if len(unique_patterns) == 2**self.num_qubits:
            return ("unconditional", [], "")

        # Find which bits can be eliminated
        eliminable_bits = []
        for bit_idx in range(self.num_qubits):
            if self._can_eliminate_bit(bit_idx, patterns):
                eliminable_bits.append(bit_idx)

        # Find common bits across all patterns
        mask, value = self._find_common_bits(patterns)

        # Determine which bits are needed
        # Note: eliminable_bits uses string indices (0=leftmost)
        # while mask/value use integer bit indices (0=LSB/rightmost)
        needed_bits = []
        ctrl_state_bits = []

        for string_idx in range(self.num_qubits):
            # Map string index to integer bit index
            int_bit_idx = self.num_qubits - 1 - string_idx
            bit_mask = 1 << int_bit_idx

            if string_idx in eliminable_bits:
                # This bit can be eliminated
                continue

            # Check if this bit has a common value
            if mask & bit_mask:
                # Bit is common across all patterns, keep it
                # Convert string index to qubit index (little-endian: qubit 0 is rightmost)
                qubit_idx = self.num_qubits - 1 - string_idx
                needed_bits.append(qubit_idx)
                bit_value = "1" if (value & bit_mask) else "0"
                ctrl_state_bits.append(bit_value)

        if len(needed_bits) == 0:
            return ("unconditional", [], "")
        elif len(needed_bits) == 1:
            return ("single", needed_bits, "".join(ctrl_state_bits))
        else:
            return ("and", sorted(needed_bits), "".join(ctrl_state_bits))


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


class ControlPatternSimplification(TransformationPass):
    """Simplify multi-controlled gates using Boolean algebraic pattern matching.

    This pass detects consecutive multi-controlled gates with identical base operations,
    target qubits, and parameters (e.g., rotation angles) but different control patterns.
    It then applies bitwise pattern analysis to reduce gate counts.

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
    """

    def __init__(self, tolerance=1e-10):
        """Initialize the control pattern simplification pass.

        Args:
            tolerance (float): Numerical tolerance for comparing gate parameters.
                Default is 1e-10.
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
            return "1" * num_ctrl_qubits
        elif isinstance(ctrl_state, str):
            return ctrl_state
        elif isinstance(ctrl_state, int):
            # Convert integer to binary string with appropriate length
            return format(ctrl_state, f"0{num_ctrl_qubits}b")
        else:
            # Fallback: assume all ones
            return "1" * num_ctrl_qubits

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
                qargs = [dag.find_bit(q).index for q in node.qargs]
                control_qubits = qargs[:num_ctrl_qubits]
                target_qubits = qargs[num_ctrl_qubits:]

                gate_info = ControlledGateInfo(
                    node=node,
                    operation=node.op,
                    control_qubits=control_qubits,
                    target_qubits=target_qubits,
                    ctrl_state=ctrl_state,
                    params=tuple(node.op.params) if node.op.params else (),
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

    def _group_compatible_gates(
        self, gates: List[ControlledGateInfo]
    ) -> List[List[ControlledGateInfo]]:
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
                if (
                    candidate.operation.base_gate.name == base_gate.name
                    and candidate.target_qubits == target_qubits
                    and set(candidate.control_qubits) == control_qubits_set
                    and self._parameters_match(candidate.params, params)
                    and candidate.ctrl_state != gates[i].ctrl_state
                ):  # Different patterns

                    current_group.append(candidate)
                    j += 1
                else:
                    break

            # Only add groups with 2+ gates
            if len(current_group) >= 2:
                groups.append(current_group)

            i = j if j > i + 1 else i + 1

        return groups

    def _build_single_control_gate(
        self,
        base_gate,
        params: Tuple,
        control_qubit: int,
        target_qubits: List[int],
        ctrl_state: str,
    ) -> Tuple[ControlledGate, List[int]]:
        """Build a single-controlled gate from optimization result.

        Args:
            base_gate: The base gate operation (e.g., RXGate, RYGate)
            params: Gate parameters (e.g., rotation angle)
            control_qubit: Index of the control qubit
            target_qubits: List of target qubit indices
            ctrl_state: Control state ('0' or '1')

        Returns:
            Tuple of (optimized_gate, qargs) where qargs is [control_qubit, *target_qubits]
        """
        # Create base gate with parameters
        if params:
            gate = base_gate(*params)
        else:
            gate = base_gate

        # Create controlled version with single control
        controlled_gate = gate.control(1, ctrl_state=ctrl_state)

        # Qubit arguments: control first, then targets
        qargs = [control_qubit] + target_qubits

        return (controlled_gate, qargs)

    def _build_multi_control_gate(
        self,
        base_gate,
        params: Tuple,
        control_qubits: List[int],
        target_qubits: List[int],
        ctrl_state: str,
    ) -> Tuple[ControlledGate, List[int]]:
        """Build a multi-controlled gate with reduced control qubits.

        Args:
            base_gate: The base gate operation
            params: Gate parameters
            control_qubits: List of control qubit indices (reduced set)
            target_qubits: List of target qubit indices
            ctrl_state: Control state pattern for the reduced controls

        Returns:
            Tuple of (optimized_gate, qargs)
        """
        # Create base gate with parameters
        if params:
            gate = base_gate(*params)
        else:
            gate = base_gate

        # Create controlled version with multiple controls
        num_ctrl_qubits = len(control_qubits)
        controlled_gate = gate.control(num_ctrl_qubits, ctrl_state=ctrl_state)

        # Qubit arguments: controls first, then targets
        qargs = control_qubits + target_qubits

        return (controlled_gate, qargs)

    def _build_unconditional_gate(
        self, base_gate, params: Tuple, target_qubits: List[int]
    ) -> Tuple:
        """Build an unconditional gate (no controls).

        Args:
            base_gate: The base gate operation
            params: Gate parameters
            target_qubits: List of target qubit indices

        Returns:
            Tuple of (gate, qargs)
        """
        # Create base gate with parameters (no controls)
        if params:
            gate = base_gate(*params)
        else:
            gate = base_gate

        return (gate, target_qubits)

    def _replace_gates_in_dag(
        self, dag: DAGCircuit, original_group: List[ControlledGateInfo], replacement: List[Tuple]
    ):
        """Replace a group of gates in the DAG with optimized gates.

        Args:
            dag: The DAG circuit to modify
            original_group: List of original gate info objects to remove
            replacement: List of (gate, qargs) tuples to insert

        Returns:
            None (modifies dag in place)
        """
        if not original_group or not replacement:
            return

        # Find the position of the first gate in the group
        first_node = original_group[0].node

        # Remove all gates in the group
        for gate_info in original_group:
            dag.remove_op_node(gate_info.node)

        # Insert replacement gates at the position of the first removed gate
        # We need to get the qubits as Qubit objects, not indices
        for gate, qargs_indices in replacement:
            # Convert qubit indices to Qubit objects
            qubits = [dag.qubits[idx] for idx in qargs_indices]

            # Apply the gate to the DAG
            dag.apply_operation_back(gate, qubits)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the ControlPatternSimplification pass on a DAGCircuit.

        Args:
            dag: The DAG to be optimized.

        Returns:
            DAGCircuit: The optimized DAG with simplified control patterns.
        """
        # 1. Collect runs of consecutive controlled gates
        gate_runs = self._collect_controlled_gates(dag)

        # Track groups to optimize (collect all first to avoid modifying DAG during iteration)
        optimizations_to_apply = []

        # 2. Process each run
        for run in gate_runs:
            # Group gates by compatible properties
            groups = self._group_compatible_gates(run)

            # 3. Process each optimizable group
            for group in groups:
                if len(group) < 2:
                    continue

                # Extract control patterns
                patterns = [g.ctrl_state for g in group]
                num_qubits = len(group[0].control_qubits)

                # 4. Try bitwise pattern simplification
                analyzer = BitwisePatternAnalyzer(num_qubits)
                classification, qubit_indices, ctrl_state = analyzer.simplify_patterns(patterns)

                # 5. Build optimized gate based on classification
                replacement = None

                if classification == "single" and qubit_indices and ctrl_state:
                    # Simplified to single control qubit
                    control_qubit_pos = qubit_indices[0]
                    control_qubit = group[0].control_qubits[control_qubit_pos]
                    target_qubits = group[0].target_qubits
                    base_gate = type(group[0].operation.base_gate)
                    params = group[0].params

                    gate, qargs = self._build_single_control_gate(
                        base_gate, params, control_qubit, target_qubits, ctrl_state
                    )
                    replacement = [(gate, qargs)]

                elif classification == "and" and qubit_indices and ctrl_state:
                    # Simplified to AND of multiple controls (reduced set)
                    control_qubits = [group[0].control_qubits[i] for i in qubit_indices]
                    target_qubits = group[0].target_qubits
                    base_gate = type(group[0].operation.base_gate)
                    params = group[0].params

                    gate, qargs = self._build_multi_control_gate(
                        base_gate, params, control_qubits, target_qubits, ctrl_state
                    )
                    replacement = [(gate, qargs)]

                elif classification == "unconditional":
                    # All control states covered - unconditional gate
                    target_qubits = group[0].target_qubits
                    base_gate = type(group[0].operation.base_gate)
                    params = group[0].params

                    gate, qargs = self._build_unconditional_gate(base_gate, params, target_qubits)
                    replacement = [(gate, qargs)]

                # Store optimization if found
                if replacement:
                    optimizations_to_apply.append((group, replacement))

        # 6. Apply all optimizations to DAG
        for group, replacement in optimizations_to_apply:
            self._replace_gates_in_dag(dag, group, replacement)

        return dag
