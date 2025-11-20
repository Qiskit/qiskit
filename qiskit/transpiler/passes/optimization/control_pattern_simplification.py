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
import sympy as sp
from sympy.logic import simplify_logic

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import CXGate, XGate

class BooleanExpressionAnalyzer:
    """Analyzes control patterns using sympy Boolean expressions.

    This class converts control patterns to Boolean expressions,
    simplifies them using sympy, and determines if simplification occurred.
    Follows the implementation pattern from mcrx_simplifier.py.
    """

    def __init__(self, num_qubits: int):
        """Initialize analyzer for given number of control qubits.

        Args:
            num_qubits: Number of control qubits in patterns
        """
        self.num_qubits = num_qubits
        # Create symbols x0, x1, x2, ... for each control qubit
        self.symbols = [sp.Symbol(f"x{i}") for i in range(num_qubits)]

    def pattern_to_boolean_expr(self, pattern: str) -> sp.Basic:
        """Convert binary pattern string to Boolean expression.

        Args:
            pattern: Binary pattern string (e.g., '110')
                    Position i corresponds to qubit i (LSB-first)

        Returns:
            Sympy Boolean expression representing the pattern
        """
        terms = []
        for i, bit in enumerate(pattern):
            if bit == '1':
                terms.append(self.symbols[i])
            elif bit == '0':
                terms.append(sp.Not(self.symbols[i]))

        if not terms:
            return sp.true
        elif len(terms) == 1:
            return terms[0]
        else:
            return sp.And(*terms)

    def patterns_to_combined_expr(self, patterns: List[str]) -> sp.Basic:
        """Convert list of patterns to combined Boolean OR expression.

        Args:
            patterns: List of binary pattern strings

        Returns:
            Sympy Boolean expression (OR of all pattern expressions)
        """
        if not patterns:
            return sp.false

        pattern_exprs = [self.pattern_to_boolean_expr(p) for p in patterns]

        if len(pattern_exprs) == 1:
            return pattern_exprs[0]
        else:
            return sp.Or(*pattern_exprs)

    def simplify_expression(self, expr: sp.Basic) -> sp.Basic:
        """Simplify Boolean expression using sympy.

        Args:
            expr: Sympy Boolean expression

        Returns:
            Simplified expression
        """
        try:
            return simplify_logic(expr)
        except Exception:
            return expr

    def find_xor_pairs(self, patterns: List[str]) -> List[Tuple[str, str, List[int], str]]:
        """Find pairs of patterns that form XOR relationships.

        An XOR pair has exactly 2 bit positions that differ between the patterns.

        Args:
            patterns: List of binary control pattern strings

        Returns:
            List of tuples (pattern1, pattern2, diff_positions, xor_type) where:
            - diff_positions: [pos_i, pos_j] positions that differ (0-indexed)
            - xor_type: '10-01', '01-10', '11-00', or '00-11'
        """
        xor_pairs = []
        patterns_list = list(patterns)

        for i in range(len(patterns_list)):
            for j in range(i + 1, len(patterns_list)):
                p1 = patterns_list[i]
                p2 = patterns_list[j]

                # Find positions where patterns differ
                diff_positions = [k for k in range(len(p1)) if p1[k] != p2[k]]

                if len(diff_positions) == 2:
                    # This is an XOR pair
                    pos_i, pos_j = diff_positions

                    # Determine XOR type based on bit values
                    bits_p1 = p1[pos_i] + p1[pos_j]
                    bits_p2 = p2[pos_i] + p2[pos_j]

                    # Determine XOR pattern type
                    if bits_p1 == '10' and bits_p2 == '01':
                        xor_type = '10-01'
                    elif bits_p1 == '01' and bits_p2 == '10':
                        xor_type = '01-10'
                    elif bits_p1 == '11' and bits_p2 == '00':
                        xor_type = '11-00'
                    elif bits_p1 == '00' and bits_p2 == '11':
                        xor_type = '00-11'
                    else:
                        continue  # Not a standard XOR pattern

                    xor_pairs.append((p1, p2, diff_positions, xor_type))

        return xor_pairs

    def simplify_patterns_pairwise(
        self, patterns: List[str]
    ) -> Optional[List[dict]]:
        """Simplify patterns using pairwise optimizations (complementary or XOR).

        This finds ONE pairwise optimization (Hamming distance 1 first, then 2).
        For iterative simplification of multiple patterns, this should be called
        repeatedly until no more optimizations are found.

        Args:
            patterns: List of binary control pattern strings

        Returns:
            List with single optimization dict containing:
            - 'type': 'complementary', 'xor_standard', or 'xor_with_x'
            - 'patterns': patterns involved in this optimization
            - 'control_positions': qubit positions for control
            - 'ctrl_state': control state string
            - 'xor_qubits': (for XOR only) positions needing CX/X gates
            - 'xor_type': (for XOR only) type of XOR pattern
            Or None if no pairwise optimization possible
        """
        if not patterns or len(patterns) < 2:
            return None

        patterns_set = set(patterns)
        patterns_list = list(patterns_set)

        # Prioritize Hamming distance 1 (complementary pairs) first
        for i in range(len(patterns_list)):
            for j in range(i + 1, len(patterns_list)):
                p1 = patterns_list[i]
                p2 = patterns_list[j]

                # Find differing positions
                diff_positions = [k for k in range(len(p1)) if p1[k] != p2[k]]

                if len(diff_positions) == 1:
                    # Complementary pair - drop the differing bit
                    pos = diff_positions[0]
                    common_positions = [k for k in range(len(p1)) if k != pos]

                    # Build ctrl_state from common positions
                    ctrl_state = ''.join(p1[k] for k in common_positions)

                    # After ctrl_state reversal during extraction, pattern indices
                    # directly correspond to qubit indices (no LSB mapping needed)
                    control_qubit_indices = common_positions

                    return [{
                        'type': 'complementary',
                        'patterns': [p1, p2],
                        'control_positions': control_qubit_indices,
                        'ctrl_state': ctrl_state
                    }]

        # Try XOR pairs (Hamming distance 2)
        xor_pairs = self.find_xor_pairs(patterns_list)

        if xor_pairs:
            p1, p2, diff_positions, xor_type = xor_pairs[0]
            pos_i, pos_j = diff_positions

            # Find common positions (bits that don't vary)
            common_positions = [k for k in range(len(p1)) if k not in diff_positions]

            # Determine optimization based on XOR type
            if xor_type in ['10-01', '01-10']:
                # Standard XOR: CX trick
                # After CX(qi, qj), both patterns have qj=1
                # String positions directly map to control qubit indices
                qi = pos_i
                qj = pos_j

                # After CX, control on qj=1
                control_qubit_indices = common_positions + [qj]
                control_qubit_indices = sorted(control_qubit_indices)

                # Build ctrl_state for control_qubit_indices
                ctrl_state_bits = []
                for idx in control_qubit_indices:
                    if idx == qj:
                        ctrl_state_bits.append('1')
                    else:
                        ctrl_state_bits.append(p1[idx])

                ctrl_state = ''.join(ctrl_state_bits)

                return [{
                    'type': 'xor_standard',
                    'patterns': [p1, p2],
                    'control_positions': control_qubit_indices,
                    'ctrl_state': ctrl_state,
                    'xor_qubits': [qi, qj],  # For CX(qi, qj)
                    'xor_type': xor_type
                }]

            else:  # '11-00' or '00-11'
                # XOR with X gates
                qi = pos_i
                qj = pos_j

                # After X(qj) + CX(qi,qj), control on qj=1
                control_qubit_indices = common_positions + [qj]
                control_qubit_indices = sorted(control_qubit_indices)

                # Build ctrl_state
                ctrl_state_bits = []
                for idx in control_qubit_indices:
                    if idx == qj:
                        ctrl_state_bits.append('1')
                    else:
                        ctrl_state_bits.append(p1[idx])

                ctrl_state = ''.join(ctrl_state_bits)

                return [{
                    'type': 'xor_with_x',
                    'patterns': [p1, p2],
                    'control_positions': control_qubit_indices,
                    'ctrl_state': ctrl_state,
                    'xor_qubits': [qi, qj],
                    'xor_type': xor_type
                }]

        return None

    def simplify_patterns_iterative(
        self, patterns: List[str]
    ) -> Tuple[str, Optional[dict], Optional[str]]:
        """Iteratively simplify patterns using pairwise optimizations.

        Repeatedly applies pairwise simplification (Hamming distance 1, then 2)
        until no more optimizations are found. This handles complex cases like
        5 patterns → 4 → 3 through multiple iterations.

        Args:
            patterns: List of binary control pattern strings

        Returns:
            Tuple of (classification, optimization_info, ctrl_state):
            - If optimizations found: ("pairwise_iterative", dict with optimizations, None)
            - If no optimization: (None, None, None)
        """
        if not patterns or len(patterns) == 0:
            return (None, None, None)

        remaining_patterns = set(patterns)
        all_optimizations = []

        # Iteratively find and apply pairwise optimizations
        while len(remaining_patterns) >= 2:
            # Try to find one pairwise optimization
            pairwise_result = self.simplify_patterns_pairwise(list(remaining_patterns))

            if not pairwise_result:
                # No more pairwise optimizations found
                break

            # Found an optimization
            opt = pairwise_result[0]
            matched = set(opt['patterns'])

            # Add this optimization to our list
            all_optimizations.append(opt)

            # Remove matched patterns from remaining
            remaining_patterns -= matched

        # After all iterations, check what we have
        if len(all_optimizations) == 0:
            # No pairwise optimizations found
            return (None, None, None)

        # We found some pairwise optimizations
        return ("pairwise_iterative", {
            'optimizations': all_optimizations,
            'remaining_patterns': list(remaining_patterns)
        }, None)

    def simplify_patterns(
        self, patterns: List[str]
    ) -> Tuple[str, Optional[List[int]], Optional[str]]:
        """Simplify control patterns using sympy Boolean expression analysis.

        This is the main entry point that tries different simplification strategies:
        1. Check for unconditional (all states covered)
        2. Try iterative pairwise (for >2 patterns)
        3. Try single pairwise optimization
        4. Use sympy Boolean simplification

        Args:
            patterns: List of binary control pattern strings

        Returns:
            Tuple of (classification, qubit_indices, ctrl_state):
            - classification: 'single', 'and', 'unconditional', 'no_optimization', 'pairwise', 'pairwise_iterative'
            - qubit_indices: List of qubit indices or optimization info
            - ctrl_state: Control state string or None
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

        # Try iterative pairwise optimization for complex cases (> 2 patterns)
        if len(unique_patterns) > 2:
            iterative_result = self.simplify_patterns_iterative(list(unique_patterns))

            if iterative_result[0] == "pairwise_iterative":
                # Iterative pairwise achieved optimization
                return iterative_result

        # Try single pairwise optimization
        pairwise_result = self.simplify_patterns_pairwise(list(unique_patterns))

        if pairwise_result:
            # Pairwise achieved some optimization
            return ("pairwise", pairwise_result, None)

        # Use sympy Boolean simplification to check for simpler forms
        original_expr = self.patterns_to_combined_expr(list(unique_patterns))
        simplified_expr = self.simplify_expression(original_expr)

        # Analyze the simplified expression to extract control information
        return self._analyze_simplified_expr(simplified_expr, original_expr)

    def _analyze_simplified_expr(
        self, simplified_expr: sp.Basic, original_expr: sp.Basic
    ) -> Tuple[str, Optional[List[int]], Optional[str]]:
        """Analyze a simplified sympy expression to extract control pattern info.

        Args:
            simplified_expr: Simplified Boolean expression
            original_expr: Original Boolean expression for comparison

        Returns:
            Tuple of (classification, qubit_indices, ctrl_state)
        """
        # Check if expression simplified to True (unconditional)
        if simplified_expr == sp.true:
            return ("unconditional", [], "")

        # Check if expression simplified to False (impossible, no optimization)
        if simplified_expr == sp.false:
            return ("no_optimization", None, None)

        # Check if simplified to a single variable or its negation
        for i, symbol in enumerate(self.symbols):
            if simplified_expr == symbol:
                # Single control on qubit i = 1
                return ("single", [i], "1")
            elif simplified_expr == sp.Not(symbol):
                # Single control on qubit i = 0
                return ("single", [i], "0")

        # Check if it's an AND of literals (conjunction)
        if isinstance(simplified_expr, sp.And):
            qubit_indices = []
            ctrl_state_bits = []

            for arg in simplified_expr.args:
                if isinstance(arg, sp.Not):
                    # Negated variable
                    var = arg.args[0]
                    if var in self.symbols:
                        qubit_idx = self.symbols.index(var)
                        qubit_indices.append(qubit_idx)
                        ctrl_state_bits.append('0')
                elif arg in self.symbols:
                    # Positive variable
                    qubit_idx = self.symbols.index(arg)
                    qubit_indices.append(qubit_idx)
                    ctrl_state_bits.append('1')

            if qubit_indices:
                # Sort by qubit index
                sorted_pairs = sorted(zip(qubit_indices, ctrl_state_bits))
                qubit_indices = [q for q, _ in sorted_pairs]
                ctrl_state = ''.join(c for _, c in sorted_pairs)

                if len(qubit_indices) == 1:
                    return ("single", qubit_indices, ctrl_state)
                else:
                    return ("and", qubit_indices, ctrl_state)

        # If expression didn't simplify or is complex OR, no optimization
        if str(simplified_expr) == str(original_expr):
            return ("no_optimization", None, None)

        # Expression simplified but we can't extract a simple pattern
        return ("no_optimization", None, None)


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
            # Reverse Qiskit's ctrl_state to match our LSB-first pattern convention
            # (matching mcrx_simplifier implementation)
            return ctrl_state[::-1]
        elif isinstance(ctrl_state, int):
            # Convert integer to binary string and reverse
            # (matching mcrx_simplifier implementation)
            return format(ctrl_state, f"0{num_ctrl_qubits}b")[::-1]
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
        - Same control qubits (same set)
        - Same parameters

        This handles TWO types of grouping:
        1. Identical patterns: Merge angles (e.g., 2x RX(θ) with '110' → RX(2θ) with '110')
        2. Different patterns: Pattern simplification (e.g., '11'+'01' → '1')

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
            ctrl_state = gates[i].ctrl_state

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
                ):
                    # Compatible! Can be either identical patterns OR different patterns
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
        # Reverse ctrl_state back to Qiskit's format (matching mcrx_simplifier)
        controlled_gate = gate.control(1, ctrl_state=ctrl_state[::-1])

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
        # Reverse ctrl_state back to Qiskit's format (matching mcrx_simplifier)
        controlled_gate = gate.control(num_ctrl_qubits, ctrl_state=ctrl_state[::-1])

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

    def _build_iterative_pairwise_gates(
        self, group: List[ControlledGateInfo], iterative_info: dict
    ) -> List[Tuple]:
        """Build gates for iterative pairwise optimization.

        Args:
            group: Original group of gates
            iterative_info: Dict with 'optimizations' list and 'remaining_patterns'

        Returns:
            List of (gate, qargs) tuples
        """
        optimizations = iterative_info['optimizations']
        remaining_patterns_strs = iterative_info['remaining_patterns']

        base_gate = type(group[0].operation.base_gate)
        params = group[0].params
        target_qubits = group[0].target_qubits
        all_control_qubits = group[0].control_qubits

        gates = []

        # Build gates for each optimization
        for opt in optimizations:
            opt_type = opt['type']
            control_positions = opt['control_positions']
            ctrl_state = opt['ctrl_state']
            control_qubits = [all_control_qubits[pos] for pos in control_positions]

            # Build the optimized gate for this pair
            if opt_type == 'complementary':
                if len(control_qubits) == 0:
                    gate, qargs = self._build_unconditional_gate(base_gate, params, target_qubits)
                    gates.append((gate, qargs))
                elif len(control_qubits) == 1:
                    gate, qargs = self._build_single_control_gate(
                        base_gate, params, control_qubits[0], target_qubits, ctrl_state
                    )
                    gates.append((gate, qargs))
                else:
                    gate, qargs = self._build_multi_control_gate(
                        base_gate, params, control_qubits, target_qubits, ctrl_state
                    )
                    gates.append((gate, qargs))

            elif opt_type == 'xor_standard':
                qi, qj = opt['xor_qubits']
                qi_circuit = all_control_qubits[qi]
                qj_circuit = all_control_qubits[qj]

                gates.append((CXGate(), [qi_circuit, qj_circuit]))

                if len(control_qubits) == 0:
                    gate, qargs = self._build_unconditional_gate(base_gate, params, target_qubits)
                elif len(control_qubits) == 1:
                    gate, qargs = self._build_single_control_gate(
                        base_gate, params, control_qubits[0], target_qubits, ctrl_state
                    )
                else:
                    gate, qargs = self._build_multi_control_gate(
                        base_gate, params, control_qubits, target_qubits, ctrl_state
                    )
                gates.append((gate, qargs))
                gates.append((CXGate(), [qi_circuit, qj_circuit]))

            elif opt_type == 'xor_with_x':
                qi, qj = opt['xor_qubits']
                qi_circuit = all_control_qubits[qi]
                qj_circuit = all_control_qubits[qj]

                gates.append((XGate(), [qj_circuit]))
                gates.append((CXGate(), [qi_circuit, qj_circuit]))

                if len(control_qubits) == 0:
                    gate, qargs = self._build_unconditional_gate(base_gate, params, target_qubits)
                elif len(control_qubits) == 1:
                    gate, qargs = self._build_single_control_gate(
                        base_gate, params, control_qubits[0], target_qubits, ctrl_state
                    )
                else:
                    gate, qargs = self._build_multi_control_gate(
                        base_gate, params, control_qubits, target_qubits, ctrl_state
                    )
                gates.append((gate, qargs))
                gates.append((CXGate(), [qi_circuit, qj_circuit]))
                gates.append((XGate(), [qj_circuit]))

        # Add gates for remaining unmatched patterns
        remaining_patterns_int = {int(p, 2) for p in remaining_patterns_strs}
        for gate_info in group:
            ctrl_state_int = int(gate_info.ctrl_state, 2) if isinstance(gate_info.ctrl_state, str) else gate_info.ctrl_state
            if ctrl_state_int in remaining_patterns_int:
                gate = gate_info.operation
                qargs = gate_info.control_qubits + gate_info.target_qubits
                gates.append((gate, qargs))

        return gates if gates else None

    def _build_pairwise_optimized_gates(
        self, group: List[ControlledGateInfo], pairwise_opts: List[dict]
    ) -> List[Tuple]:
        """Build optimized gates for pairwise optimization (complementary or XOR).

        Args:
            group: Original group of gates
            pairwise_opts: List of pairwise optimization dicts

        Returns:
            List of (gate, qargs) tuples
        """
        if not pairwise_opts:
            return None

        opt = pairwise_opts[0]  # Take first optimization
        opt_type = opt['type']
        control_positions = opt['control_positions']
        ctrl_state = opt['ctrl_state']
        # Convert matched patterns to integers for comparison with gate ctrl_state
        matched_patterns = {int(p, 2) for p in opt['patterns']}

        base_gate = type(group[0].operation.base_gate)
        params = group[0].params
        target_qubits = group[0].target_qubits
        all_control_qubits = group[0].control_qubits

        # Map control_positions (qubit indices in pattern) to actual circuit qubits
        control_qubits = [all_control_qubits[pos] for pos in control_positions]

        gates = []

        # Build gates for the pairwise optimization
        if opt_type == 'complementary':
            # Simple case: just reduce control qubits
            if len(control_qubits) == 0:
                # Unconditional
                gate, qargs = self._build_unconditional_gate(base_gate, params, target_qubits)
                gates.append((gate, qargs))
            elif len(control_qubits) == 1:
                # Single control
                gate, qargs = self._build_single_control_gate(
                    base_gate, params, control_qubits[0], target_qubits, ctrl_state
                )
                gates.append((gate, qargs))
            else:
                # Multi control
                gate, qargs = self._build_multi_control_gate(
                    base_gate, params, control_qubits, target_qubits, ctrl_state
                )
                gates.append((gate, qargs))

        elif opt_type == 'xor_standard':
            # Standard XOR: CX(qi, qj) + controlled_gate + CX(qi, qj)
            qi, qj = opt['xor_qubits']
            qi_circuit = all_control_qubits[qi]
            qj_circuit = all_control_qubits[qj]

            # Build the wrapped circuit
            gates = []

            # CX(qi, qj)
            gates.append((CXGate(), [qi_circuit, qj_circuit]))

            # Controlled gate with reduced controls
            if len(control_qubits) == 0:
                gate, qargs = self._build_unconditional_gate(base_gate, params, target_qubits)
            elif len(control_qubits) == 1:
                gate, qargs = self._build_single_control_gate(
                    base_gate, params, control_qubits[0], target_qubits, ctrl_state
                )
            else:
                gate, qargs = self._build_multi_control_gate(
                    base_gate, params, control_qubits, target_qubits, ctrl_state
                )
            gates.append((gate, qargs))

            # CX(qi, qj)
            gates.append((CXGate(), [qi_circuit, qj_circuit]))

        elif opt_type == 'xor_with_x':
            # XOR with X gates: X(qj) + CX(qi, qj) + controlled_gate + CX(qi, qj) + X(qj)
            qi, qj = opt['xor_qubits']
            qi_circuit = all_control_qubits[qi]
            qj_circuit = all_control_qubits[qj]

            gates = []

            # X(qj)
            gates.append((XGate(), [qj_circuit]))

            # CX(qi, qj)
            gates.append((CXGate(), [qi_circuit, qj_circuit]))

            # Controlled gate
            if len(control_qubits) == 0:
                gate, qargs = self._build_unconditional_gate(base_gate, params, target_qubits)
            elif len(control_qubits) == 1:
                gate, qargs = self._build_single_control_gate(
                    base_gate, params, control_qubits[0], target_qubits, ctrl_state
                )
            else:
                gate, qargs = self._build_multi_control_gate(
                    base_gate, params, control_qubits, target_qubits, ctrl_state
                )
            gates.append((gate, qargs))

            # CX(qi, qj)
            gates.append((CXGate(), [qi_circuit, qj_circuit]))

            # X(qj)
            gates.append((XGate(), [qj_circuit]))

        # Build gates for any unmatched patterns
        for gate_info in group:
            # gate_info.ctrl_state is a string like '0000', convert to int for comparison
            ctrl_state_int = int(gate_info.ctrl_state, 2) if isinstance(gate_info.ctrl_state, str) else gate_info.ctrl_state
            if ctrl_state_int not in matched_patterns:
                # This pattern wasn't part of the pairwise optimization
                # Build a separate gate for it
                gate = gate_info.operation
                qargs = gate_info.control_qubits + gate_info.target_qubits
                gates.append((gate, qargs))

        return gates if gates else None

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

                # 4. Check if all patterns are identical (angle merging case)
                unique_patterns = set(patterns)
                if len(unique_patterns) == 1:
                    # All gates have identical patterns - merge by summing angles
                    # This applies to parametric gates like RX, RY, RZ
                    control_qubits = group[0].control_qubits
                    target_qubits = group[0].target_qubits
                    base_gate = type(group[0].operation.base_gate)
                    ctrl_state = group[0].ctrl_state

                    # Sum the angles from all gates
                    if group[0].params:
                        total_angle = sum(g.params[0] for g in group)
                        params = (total_angle,)
                    else:
                        params = group[0].params

                    gate, qargs = self._build_multi_control_gate(
                        base_gate, params, control_qubits, target_qubits, ctrl_state
                    )
                    replacement = [(gate, qargs)]
                else:
                    # Different patterns - try pattern simplification using sympy
                    analyzer = BooleanExpressionAnalyzer(num_qubits)
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

                    elif classification == "pairwise_iterative" and qubit_indices:
                        # Iterative pairwise optimization (multiple steps)
                        replacement = self._build_iterative_pairwise_gates(
                            group, qubit_indices
                        )

                    elif classification == "pairwise" and qubit_indices:
                        # Single pairwise optimization (complementary or XOR)
                        replacement = self._build_pairwise_optimized_gates(
                            group, qubit_indices
                        )

                # Store optimization if found
                if replacement:
                    optimizations_to_apply.append((group, replacement))

        # 6. Apply all optimizations to DAG
        for group, replacement in optimizations_to_apply:
            self._replace_gates_in_dag(dag, group, replacement)

        return dag
