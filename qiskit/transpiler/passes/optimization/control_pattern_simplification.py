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
from qiskit.circuit.library import CXGate, XGate


class BitwisePatternAnalyzer:
    """Analyzes control patterns using pure bitwise operations."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits

    def _find_common_bits(self, patterns: List[str]) -> Tuple[List[int], str]:
        """Find bit positions with same value across all patterns."""
        if not patterns:
            return [], ""
        common_positions = []
        common_values = []
        for pos in range(len(patterns[0])):
            bits_at_pos = set(p[pos] for p in patterns)
            if len(bits_at_pos) == 1:
                common_positions.append(pos)
                common_values.append(patterns[0][pos])
        return common_positions, "".join(common_values)

    def _check_single_variable_simplification(
        self, patterns: List[str]
    ) -> Optional[Tuple[int, str]]:
        """Check if patterns simplify to a single variable control."""
        if not patterns:
            return None
        n = self.num_qubits
        pattern_set = set(patterns)
        if len(pattern_set) != 2 ** (n - 1):
            return None

        for ctrl_pos in range(n):
            expected_with_1 = set()
            expected_with_0 = set()
            for combo in range(2 ** (n - 1)):
                pattern_1 = ""
                pattern_0 = ""
                bit_idx = 0
                for pos in range(n):
                    if pos == ctrl_pos:
                        pattern_1 += "1"
                        pattern_0 += "0"
                    else:
                        pattern_1 += str((combo >> bit_idx) & 1)
                        pattern_0 += str((combo >> bit_idx) & 1)
                        bit_idx += 1
                expected_with_1.add(pattern_1)
                expected_with_0.add(pattern_0)

            if pattern_set == expected_with_1:
                return (ctrl_pos, "1")
            if pattern_set == expected_with_0:
                return (ctrl_pos, "0")
        return None

    def simplify_patterns_pairwise(self, patterns: List[str]) -> Optional[List[dict]]:
        """Find ONE pairwise optimization (Hamming distance 1, 2, or n)."""
        if not patterns or len(patterns) < 2:
            return None

        patterns_list = sorted(set(patterns))

        # Try all pairs
        for i in range(len(patterns_list)):
            for j in range(i + 1, len(patterns_list)):
                p1, p2 = patterns_list[i], patterns_list[j]
                diff_positions = [k for k in range(len(p1)) if p1[k] != p2[k]]
                hamming = len(diff_positions)

                # Hamming distance 1: complementary pair
                if hamming == 1:
                    pos = diff_positions[0]
                    common_positions = [k for k in range(len(p1)) if k != pos]
                    ctrl_state = "".join(p1[k] for k in common_positions)
                    return [
                        {
                            "type": "complementary",
                            "patterns": [p1, p2],
                            "control_positions": common_positions,
                            "ctrl_state": ctrl_state,
                        }
                    ]

                # Hamming distance 2: XOR pair
                if hamming == 2:
                    pos_i, pos_j = diff_positions
                    bits_p1 = p1[pos_i] + p1[pos_j]
                    bits_p2 = p2[pos_i] + p2[pos_j]

                    # Determine XOR type
                    if (bits_p1, bits_p2) in [("10", "01"), ("01", "10")]:
                        xor_type = "xor_standard"
                    elif (bits_p1, bits_p2) in [("11", "00"), ("00", "11")]:
                        xor_type = "xor_with_x"
                    else:
                        continue

                    common_positions = [k for k in range(len(p1)) if k not in diff_positions]
                    control_positions = sorted(common_positions + [pos_j])
                    ctrl_state = "".join(
                        "1" if idx == pos_j else p1[idx] for idx in control_positions
                    )
                    return [
                        {
                            "type": xor_type,
                            "patterns": [p1, p2],
                            "control_positions": control_positions,
                            "ctrl_state": ctrl_state,
                            "xor_qubits": [pos_i, pos_j],
                        }
                    ]

                # Hamming distance n (all bits differ): XOR chain
                if hamming == len(p1) and len(p1) >= 2:
                    anchor = diff_positions[0]
                    targets = diff_positions[1:]
                    return [
                        {
                            "type": "xor_chain",
                            "patterns": [p1, p2],
                            "control_positions": targets,
                            "ctrl_state": p1[anchor] * len(targets),
                            "xor_anchor": anchor,
                            "xor_targets": targets,
                        }
                    ]

        return None

    def simplify_patterns_iterative(
        self, patterns: List[str]
    ) -> Tuple[str, Optional[dict], Optional[str]]:
        """Iteratively simplify patterns using pairwise optimizations."""
        if not patterns:
            return (None, None, None)

        remaining = sorted(set(patterns))
        all_optimizations = []

        while len(remaining) >= 2:
            result = self.simplify_patterns_pairwise(remaining)
            if not result:
                break
            opt = result[0]
            all_optimizations.append(opt)
            matched = set(opt["patterns"])
            remaining = [p for p in remaining if p not in matched]

        if not all_optimizations:
            return (None, None, None)

        return (
            "pairwise_iterative",
            {"optimizations": all_optimizations, "remaining_patterns": list(remaining)},
            None,
        )

    def simplify_patterns(
        self, patterns: List[str]
    ) -> Tuple[str, Optional[List[int]], Optional[str]]:
        """Main entry point for pattern simplification."""
        if not patterns or len(set(len(p) for p in patterns)) > 1:
            return ("no_optimization", None, None)

        unique = sorted(set(patterns))

        # Complete partition
        if len(unique) == 2**self.num_qubits:
            return ("unconditional", [], "")

        # Try iterative pairwise for > 2 patterns
        if len(unique) > 2:
            result = self.simplify_patterns_iterative(unique)
            if result[0] == "pairwise_iterative":
                return result

        # Try single pairwise
        pairwise = self.simplify_patterns_pairwise(unique)
        if pairwise:
            return ("pairwise", pairwise, None)

        # Bitwise analysis
        single_var = self._check_single_variable_simplification(unique)
        if single_var:
            return ("single", [single_var[0]], single_var[1])

        common_pos, common_vals = self._find_common_bits(unique)
        if common_pos and len(common_pos) < self.num_qubits:
            varying = [i for i in range(self.num_qubits) if i not in common_pos]
            if len(unique) == 2 ** len(varying):
                if len(common_pos) == 1:
                    return ("single", common_pos, common_vals)
                return ("and", common_pos, common_vals)

        return ("no_optimization", None, None)


@dataclass
class ControlledGateInfo:
    """Information about a controlled gate for optimization analysis."""

    node: DAGOpNode
    operation: ControlledGate
    control_qubits: List[int]
    target_qubits: List[int]
    ctrl_state: str
    params: Tuple[float, ...]


class ControlPatternSimplification(TransformationPass):
    """Simplify multi-controlled gates using Boolean algebraic pattern matching.

    This pass detects consecutive multi-controlled gates with identical base operations,
    target qubits, and parameters but different control patterns, then applies bitwise
    pattern analysis to reduce gate counts.

    **Optimization Techniques:**

    1. **Complementary patterns**: ['11', '01'] → single control on q0
    2. **Subset patterns**: ['111', '110'] → reduced control qubits
    3. **XOR pairs**: ['110', '101'] → CNOT + reduced multi-controlled gate
    4. **XOR chains**: ['000', '111'] → CX chain + reduced multi-controlled gate
    5. **Complete partitions**: ['00','01','10','11'] → unconditional gate

    **References:**

    - Atallah et al., "Graph Matching Trotterization for CTQW Circuit Simulation", IEEE QCE 2025
    - Gonzalez et al., "Efficient sparse state preparation via quantum walks", npj QI 2025
    """

    def __init__(self, tolerance=1e-10):
        super().__init__()
        self.tolerance = tolerance

    def _extract_control_pattern(self, gate: ControlledGate, num_ctrl: int) -> str:
        """Extract control pattern as binary string (LSB-first internally)."""
        ctrl_state = gate.ctrl_state
        if ctrl_state is None:
            return "1" * num_ctrl
        if isinstance(ctrl_state, str):
            return ctrl_state[::-1]
        if isinstance(ctrl_state, int):
            return format(ctrl_state, f"0{num_ctrl}b")[::-1]
        return "1" * num_ctrl

    def _parameters_match(self, params1: Tuple, params2: Tuple) -> bool:
        """Check if two parameter tuples match within tolerance."""
        if len(params1) != len(params2):
            return False
        for p1, p2 in zip(params1, params2):
            if isinstance(p1, (int, float)) and isinstance(p2, (int, float)):
                if not np.isclose(p1, p2, atol=self.tolerance):
                    return False
            elif p1 != p2:
                return False
        return True

    def _collect_controlled_gates(self, dag: DAGCircuit) -> List[List[ControlledGateInfo]]:
        """Collect runs of consecutive controlled gates from the DAG."""
        runs = []
        current_run = []

        for node in dag.topological_op_nodes():
            if isinstance(node.op, ControlledGate):
                num_ctrl = node.op.num_ctrl_qubits
                qargs = [dag.find_bit(q).index for q in node.qargs]
                gate_info = ControlledGateInfo(
                    node=node,
                    operation=node.op,
                    control_qubits=qargs[:num_ctrl],
                    target_qubits=qargs[num_ctrl:],
                    ctrl_state=self._extract_control_pattern(node.op, num_ctrl),
                    params=tuple(node.op.params) if node.op.params else (),
                )
                current_run.append(gate_info)
            else:
                if current_run:
                    runs.append(current_run)
                    current_run = []

        if current_run:
            runs.append(current_run)
        return runs

    def _group_compatible_gates(
        self, gates: List[ControlledGateInfo]
    ) -> List[List[ControlledGateInfo]]:
        """Group gates that can be optimized together."""
        if len(gates) < 2:
            return []

        groups = []
        i = 0
        while i < len(gates):
            group = [gates[i]]
            base = gates[i]
            j = i + 1
            while j < len(gates):
                cand = gates[j]
                if (
                    cand.operation.base_gate.name == base.operation.base_gate.name
                    and cand.target_qubits == base.target_qubits
                    and set(cand.control_qubits) == set(base.control_qubits)
                    and self._parameters_match(cand.params, base.params)
                ):
                    group.append(cand)
                    j += 1
                else:
                    break

            if len(group) >= 2:
                groups.append(group)
            i = j if j > i + 1 else i + 1

        return groups

    def _build_controlled_gate(
        self,
        base_gate,
        params: Tuple,
        control_qubits: List[int],
        target_qubits: List[int],
        ctrl_state: str,
    ) -> Tuple:
        """Build a controlled gate with given controls."""
        gate = base_gate(*params) if params else base_gate()
        if not control_qubits:
            return (gate, target_qubits)
        controlled = gate.control(len(control_qubits), ctrl_state=ctrl_state[::-1])
        return (controlled, control_qubits + target_qubits)

    def _build_optimized_gates(
        self,
        group: List[ControlledGateInfo],
        optimizations: List[dict],
        remaining_patterns: List[str],
    ) -> List[Tuple]:
        """Build optimized gates for a list of optimizations."""
        base_gate = type(group[0].operation.base_gate)
        params = group[0].params
        target_qubits = group[0].target_qubits
        all_ctrl_qubits = group[0].control_qubits

        gates = []

        for opt in optimizations:
            opt_type = opt["type"]
            ctrl_positions = opt["control_positions"]
            ctrl_state = opt["ctrl_state"]
            ctrl_qubits = [all_ctrl_qubits[p] for p in ctrl_positions]

            if opt_type == "complementary":
                gates.append(
                    self._build_controlled_gate(
                        base_gate, params, ctrl_qubits, target_qubits, ctrl_state
                    )
                )

            elif opt_type == "xor_standard":
                qi, qj = opt["xor_qubits"]
                qi_c, qj_c = all_ctrl_qubits[qi], all_ctrl_qubits[qj]
                gates.append((CXGate(), [qi_c, qj_c]))
                gates.append(
                    self._build_controlled_gate(
                        base_gate, params, ctrl_qubits, target_qubits, ctrl_state
                    )
                )
                gates.append((CXGate(), [qi_c, qj_c]))

            elif opt_type == "xor_with_x":
                qi, qj = opt["xor_qubits"]
                qi_c, qj_c = all_ctrl_qubits[qi], all_ctrl_qubits[qj]
                gates.append((XGate(), [qj_c]))
                gates.append((CXGate(), [qi_c, qj_c]))
                gates.append(
                    self._build_controlled_gate(
                        base_gate, params, ctrl_qubits, target_qubits, ctrl_state
                    )
                )
                gates.append((CXGate(), [qi_c, qj_c]))
                gates.append((XGate(), [qj_c]))

            elif opt_type == "xor_chain":
                anchor = opt["xor_anchor"]
                targets = opt["xor_targets"]
                anchor_c = all_ctrl_qubits[anchor]
                target_cs = [all_ctrl_qubits[t] for t in targets]

                for tc in target_cs:
                    gates.append((CXGate(), [anchor_c, tc]))
                gates.append(
                    self._build_controlled_gate(
                        base_gate, params, ctrl_qubits, target_qubits, ctrl_state
                    )
                )
                for tc in reversed(target_cs):
                    gates.append((CXGate(), [anchor_c, tc]))

        # Add remaining unmatched patterns
        remaining_int = {int(p, 2) for p in remaining_patterns}
        for g in group:
            ctrl_int = int(g.ctrl_state, 2) if isinstance(g.ctrl_state, str) else g.ctrl_state
            if ctrl_int in remaining_int:
                gates.append((g.operation, g.control_qubits + g.target_qubits))

        return gates if gates else None

    def _replace_gates_in_dag(
        self, dag: DAGCircuit, group: List[ControlledGateInfo], replacement: List[Tuple]
    ):
        """Replace a group of gates in the DAG with optimized gates."""
        if not group or not replacement:
            return

        for gate_info in group:
            dag.remove_op_node(gate_info.node)

        for gate, qargs_indices in replacement:
            qubits = [dag.qubits[idx] for idx in qargs_indices]
            dag.apply_operation_back(gate, qubits)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the ControlPatternSimplification pass on a DAGCircuit."""
        gate_runs = self._collect_controlled_gates(dag)
        optimizations_to_apply = []

        for run in gate_runs:
            for group in self._group_compatible_gates(run):
                if len(group) < 2:
                    continue

                patterns = [g.ctrl_state for g in group]
                num_qubits = len(group[0].control_qubits)
                unique = set(patterns)

                # Identical patterns: merge angles
                if len(unique) == 1:
                    base_gate = type(group[0].operation.base_gate)
                    params = (sum(g.params[0] for g in group),) if group[0].params else ()
                    replacement = [
                        self._build_controlled_gate(
                            base_gate,
                            params,
                            group[0].control_qubits,
                            group[0].target_qubits,
                            group[0].ctrl_state,
                        )
                    ]
                else:
                    analyzer = BitwisePatternAnalyzer(num_qubits)
                    classification, info, ctrl_state = analyzer.simplify_patterns(patterns)
                    replacement = None

                    if classification == "single" and info and ctrl_state:
                        ctrl_qubit = group[0].control_qubits[info[0]]
                        replacement = [
                            self._build_controlled_gate(
                                type(group[0].operation.base_gate),
                                group[0].params,
                                [ctrl_qubit],
                                group[0].target_qubits,
                                ctrl_state,
                            )
                        ]

                    elif classification == "and" and info and ctrl_state:
                        ctrl_qubits = [group[0].control_qubits[i] for i in info]
                        replacement = [
                            self._build_controlled_gate(
                                type(group[0].operation.base_gate),
                                group[0].params,
                                ctrl_qubits,
                                group[0].target_qubits,
                                ctrl_state,
                            )
                        ]

                    elif classification == "unconditional":
                        replacement = [
                            self._build_controlled_gate(
                                type(group[0].operation.base_gate),
                                group[0].params,
                                [],
                                group[0].target_qubits,
                                "",
                            )
                        ]

                    elif classification == "pairwise_iterative" and info:
                        replacement = self._build_optimized_gates(
                            group, info["optimizations"], info["remaining_patterns"]
                        )

                    elif classification == "pairwise" and info:
                        replacement = self._build_optimized_gates(group, info, [])

                if replacement:
                    optimizations_to_apply.append((group, replacement))

        for group, replacement in optimizations_to_apply:
            self._replace_gates_in_dag(dag, group, replacement)

        return dag
