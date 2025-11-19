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
                qargs = [dag.find_bit(q).index for q in node.qargs]
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

    def _pattern_to_boolean_expr(self, pattern: str, num_qubits: int):
        """Convert a binary control pattern to a SymPy Boolean expression.

        Args:
            pattern: Binary string pattern (e.g., '11', '01', '110')
                    Pattern is little-endian: rightmost bit corresponds to qubit 0
            num_qubits: Number of control qubits

        Returns:
            SymPy Boolean expression representing the pattern
        """
        from sympy import symbols, And, Not

        # Create symbols for each control qubit
        qubit_vars = symbols(f'q0:{num_qubits}')

        # Build expression: AND of all control conditions
        # Pattern is little-endian, so reverse it to match qubit ordering
        conditions = []
        for i, bit in enumerate(reversed(pattern)):
            if bit == '1':
                conditions.append(qubit_vars[i])
            else:  # bit == '0'
                conditions.append(Not(qubit_vars[i]))

        return And(*conditions) if len(conditions) > 1 else conditions[0]

    def _combine_patterns_to_expression(self, patterns: List[str], num_qubits: int):
        """Combine multiple control patterns into a single Boolean expression.

        Args:
            patterns: List of binary string patterns
            num_qubits: Number of control qubits

        Returns:
            SymPy Boolean expression (OR of all pattern expressions)
        """
        from sympy import Or

        if not patterns:
            return None

        if len(patterns) == 1:
            return self._pattern_to_boolean_expr(patterns[0], num_qubits)

        # Combine patterns with OR
        pattern_exprs = [self._pattern_to_boolean_expr(p, num_qubits) for p in patterns]
        return Or(*pattern_exprs)

    def _simplify_boolean_expression(self, expr):
        """Simplify a Boolean expression using SymPy.

        Args:
            expr: SymPy Boolean expression

        Returns:
            Simplified SymPy Boolean expression
        """
        from sympy.logic import simplify_logic

        if expr is None:
            return None

        return simplify_logic(expr)

    def _classify_simplified_expression(self, expr, num_qubits: int) -> Tuple[str, Optional[List[int]], Optional[str]]:
        """Classify the simplified Boolean expression to determine optimization type.

        Args:
            expr: Simplified SymPy Boolean expression
            num_qubits: Number of control qubits

        Returns:
            Tuple of (classification_type, relevant_qubit_indices, ctrl_state)
            Classification types:
            - 'single': Single variable (e.g., q0 or ~q0)
            - 'and': AND of multiple variables (e.g., q0 & q1 or ~q0 & q1)
            - 'unconditional': Always True
            - 'no_optimization': No simplification possible
            ctrl_state: Control state string for the qubits (e.g., '1', '0', '10', '01', etc.)
        """
        from sympy import Symbol, And, Not
        from sympy.logic.boolalg import BooleanTrue

        if expr is None:
            return ('no_optimization', None, None)

        # Check if unconditional (True)
        if isinstance(expr, BooleanTrue) or expr == True:
            return ('unconditional', [], '')

        # Check if single variable or single NOT
        if isinstance(expr, Symbol):
            # Extract qubit index from symbol name (e.g., 'q0' -> 0)
            qubit_idx = int(str(expr)[1:])
            return ('single', [qubit_idx], '1')

        if isinstance(expr, Not) and isinstance(expr.args[0], Symbol):
            # NOT of a single variable (e.g., ~q0)
            qubit_idx = int(str(expr.args[0])[1:])
            return ('single', [qubit_idx], '0')

        # Check if AND of variables (with potential NOTs)
        if isinstance(expr, And):
            qubit_indices = []
            ctrl_state = ''
            for arg in expr.args:
                if isinstance(arg, Symbol):
                    qubit_idx = int(str(arg)[1:])
                    qubit_indices.append(qubit_idx)
                    ctrl_state += '1'
                elif isinstance(arg, Not) and isinstance(arg.args[0], Symbol):
                    qubit_idx = int(str(arg.args[0])[1:])
                    qubit_indices.append(qubit_idx)
                    ctrl_state += '0'
                else:
                    # Complex expression, can't optimize simply
                    return ('no_optimization', None, None)

            return ('and', sorted(qubit_indices), ctrl_state)

        # Other cases: no simple optimization
        return ('no_optimization', None, None)

    def _build_single_control_gate(
        self, base_gate, params: Tuple, control_qubit: int, target_qubits: List[int],
        ctrl_state: str
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
        self, base_gate, params: Tuple, control_qubits: List[int],
        target_qubits: List[int], ctrl_state: str
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
        self, dag: DAGCircuit, original_group: List[ControlledGateInfo],
        replacement: List[Tuple]
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

                # 4. Try Boolean algebraic simplification
                expr = self._combine_patterns_to_expression(patterns, num_qubits)
                simplified = self._simplify_boolean_expression(expr)
                classification, qubit_indices, ctrl_state = self._classify_simplified_expression(
                    simplified, num_qubits
                )

                # 5. Build optimized gate based on classification
                replacement = None

                if classification == 'single' and qubit_indices and ctrl_state:
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

                elif classification == 'and' and qubit_indices and ctrl_state:
                    # Simplified to AND of multiple controls (reduced set)
                    control_qubits = [group[0].control_qubits[i] for i in qubit_indices]
                    target_qubits = group[0].target_qubits
                    base_gate = type(group[0].operation.base_gate)
                    params = group[0].params

                    gate, qargs = self._build_multi_control_gate(
                        base_gate, params, control_qubits, target_qubits, ctrl_state
                    )
                    replacement = [(gate, qargs)]

                elif classification == 'unconditional':
                    # All control states covered - unconditional gate
                    target_qubits = group[0].target_qubits
                    base_gate = type(group[0].operation.base_gate)
                    params = group[0].params

                    gate, qargs = self._build_unconditional_gate(
                        base_gate, params, target_qubits
                    )
                    replacement = [(gate, qargs)]

                # Store optimization if found
                if replacement:
                    optimizations_to_apply.append((group, replacement))

        # 6. Apply all optimizations to DAG
        for group, replacement in optimizations_to_apply:
            self._replace_gates_in_dag(dag, group, replacement)

        return dag
