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
        common_pos, common_vals = [], []
        for pos in range(len(patterns[0])):
            bits = set(p[pos] for p in patterns)
            if len(bits) == 1:
                common_pos.append(pos)
                common_vals.append(patterns[0][pos])
        return common_pos, "".join(common_vals)

    def _check_single_variable(self, patterns: List[str]) -> Optional[Tuple[int, str]]:
        """Check if patterns simplify to a single variable control."""
        n = self.num_qubits
        pattern_set = set(patterns)
        if len(pattern_set) != 2 ** (n - 1):
            return None
        for ctrl_pos in range(n):
            for val in ["0", "1"]:
                expected = set()
                for combo in range(2 ** (n - 1)):
                    p = ""
                    bit_idx = 0
                    for pos in range(n):
                        if pos == ctrl_pos:
                            p += val
                        else:
                            p += str((combo >> bit_idx) & 1)
                            bit_idx += 1
                    expected.add(p)
                if pattern_set == expected:
                    return (ctrl_pos, val)
        return None

    def simplify_patterns_pairwise(self, patterns: List[str]) -> Optional[List[dict]]:
        """Find ONE pairwise optimization (Hamming distance 1, 2, or n)."""
        if len(patterns) < 2:
            return None
        patterns_list = sorted(set(patterns))
        for i, p1 in enumerate(patterns_list):
            for p2 in patterns_list[i + 1 :]:
                diff = [k for k in range(len(p1)) if p1[k] != p2[k]]
                hamming = len(diff)
                if hamming == 1:  # Complementary pair
                    common = [k for k in range(len(p1)) if k != diff[0]]
                    return [
                        {
                            "type": "complementary",
                            "patterns": [p1, p2],
                            "control_positions": common,
                            "ctrl_state": "".join(p1[k] for k in common),
                        }
                    ]
                if hamming == 2:  # XOR pair
                    pi, pj = diff
                    bits = (p1[pi] + p1[pj], p2[pi] + p2[pj])
                    if bits in [("10", "01"), ("01", "10")]:
                        xtype = "xor_standard"
                    elif bits in [("11", "00"), ("00", "11")]:
                        xtype = "xor_with_x"
                    else:
                        continue
                    common = [k for k in range(len(p1)) if k not in diff]
                    ctrl_pos = sorted(common + [pj])
                    return [
                        {
                            "type": xtype,
                            "patterns": [p1, p2],
                            "control_positions": ctrl_pos,
                            "ctrl_state": "".join(
                                "1" if idx == pj else p1[idx] for idx in ctrl_pos
                            ),
                            "xor_qubits": [pi, pj],
                        }
                    ]
                if hamming == len(p1) >= 2:  # XOR chain
                    return [
                        {
                            "type": "xor_chain",
                            "patterns": [p1, p2],
                            "control_positions": diff[1:],
                            "ctrl_state": p1[diff[0]] * (hamming - 1),
                            "xor_anchor": diff[0],
                            "xor_targets": diff[1:],
                        }
                    ]
        return None

    def simplify_patterns_iterative(self, patterns: List[str]) -> Tuple:
        """Iteratively simplify patterns using pairwise optimizations."""
        remaining = sorted(set(patterns))
        all_opts = []
        while len(remaining) >= 2:
            result = self.simplify_patterns_pairwise(remaining)
            if not result:
                break
            all_opts.append(result[0])
            matched = set(result[0]["patterns"])
            remaining = [p for p in remaining if p not in matched]
        if not all_opts:
            return (None, None, None)
        return (
            "pairwise_iterative",
            {"optimizations": all_opts, "remaining_patterns": remaining},
            None,
        )

    def simplify_patterns(self, patterns: List[str]) -> Tuple:
        """Main entry point for pattern simplification."""
        if not patterns or len(set(len(p) for p in patterns)) > 1:
            return ("no_optimization", None, None)
        unique = sorted(set(patterns))
        n = self.num_qubits
        # Complete partition
        if len(unique) == 2**n:
            return ("unconditional", [], "")
        # Complement: all but one pattern
        if len(unique) == 2**n - 1:
            all_p = {format(i, f"0{n}b") for i in range(2**n)}
            missing = list(all_p - set(unique))
            if len(missing) == 1:
                return ("complement", list(range(n)), missing[0])
        # Iterative pairwise
        if len(unique) > 2:
            result = self.simplify_patterns_iterative(unique)
            if result[0] == "pairwise_iterative":
                return result
        # Single pairwise
        pairwise = self.simplify_patterns_pairwise(unique)
        if pairwise:
            return ("pairwise", pairwise, None)
        # Single variable
        single = self._check_single_variable(unique)
        if single:
            return ("single", [single[0]], single[1])
        # Common bits (AND)
        common_pos, common_vals = self._find_common_bits(unique)
        if common_pos and len(common_pos) < n:
            varying = [i for i in range(n) if i not in common_pos]
            if len(unique) == 2 ** len(varying):
                return ("single" if len(common_pos) == 1 else "and", common_pos, common_vals)
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
    """Simplify multi-controlled gates using Boolean algebraic pattern matching."""

    def __init__(self, tolerance=1e-10):
        super().__init__()
        self.tolerance = tolerance

    def _extract_pattern(self, gate: ControlledGate, num_ctrl: int) -> str:
        """Extract control pattern as binary string (LSB-first internally)."""
        cs = gate.ctrl_state
        if cs is None:
            return "1" * num_ctrl
        if isinstance(cs, str):
            return cs[::-1]
        if isinstance(cs, int):
            return format(cs, f"0{num_ctrl}b")[::-1]
        return "1" * num_ctrl

    def _params_match(self, p1: Tuple, p2: Tuple) -> bool:
        """Check if two parameter tuples match within tolerance."""
        if len(p1) != len(p2):
            return False
        return all(
            (
                np.isclose(a, b, atol=self.tolerance)
                if isinstance(a, (int, float)) and isinstance(b, (int, float))
                else a == b
            )
            for a, b in zip(p1, p2)
        )

    def _collect_gates(self, dag: DAGCircuit) -> List[List[ControlledGateInfo]]:
        """Collect runs of consecutive controlled gates from the DAG."""
        runs, current = [], []
        for node in dag.topological_op_nodes():
            if isinstance(node.op, ControlledGate):
                nc = node.op.num_ctrl_qubits
                qargs = [dag.find_bit(q).index for q in node.qargs]
                current.append(
                    ControlledGateInfo(
                        node=node,
                        operation=node.op,
                        control_qubits=qargs[:nc],
                        target_qubits=qargs[nc:],
                        ctrl_state=self._extract_pattern(node.op, nc),
                        params=tuple(node.op.params) if node.op.params else (),
                    )
                )
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        return runs

    def _group_same_controls(
        self, gates: List[ControlledGateInfo]
    ) -> List[List[ControlledGateInfo]]:
        """Group gates with same control qubits."""
        if len(gates) < 2:
            return []
        groups, i = [], 0
        while i < len(gates):
            group, base = [gates[i]], gates[i]
            j = i + 1
            while j < len(gates):
                c = gates[j]
                if (
                    c.operation.base_gate.name == base.operation.base_gate.name
                    and c.target_qubits == base.target_qubits
                    and set(c.control_qubits) == set(base.control_qubits)
                    and self._params_match(c.params, base.params)
                ):
                    group.append(c)
                    j += 1
                else:
                    break
            if len(group) >= 2:
                groups.append(group)
            i = j if j > i + 1 else i + 1
        return groups

    def _group_mixed_controls(
        self, gates: List[ControlledGateInfo]
    ) -> List[List[ControlledGateInfo]]:
        """Group gates with subset/superset control qubits."""
        if len(gates) < 2:
            return []
        groups, used = [], set()
        for i, base in enumerate(gates):
            if i in used:
                continue
            group, base_c = [base], set(base.control_qubits)
            for j, cand in enumerate(gates[i + 1 :], start=i + 1):
                if j in used:
                    continue
                cand_c = set(cand.control_qubits)
                if (
                    cand.operation.base_gate.name == base.operation.base_gate.name
                    and cand.target_qubits == base.target_qubits
                    and self._params_match(cand.params, base.params)
                    and (base_c <= cand_c or cand_c <= base_c)
                ):
                    group.append(cand)
                    used.add(j)
            if len(group) >= 2 and len(set(len(g.control_qubits) for g in group)) > 1:
                groups.append(group)
                used.add(i)
        return groups

    def _expand_pattern(
        self, pattern: str, gate_ctrls: List[int], superset: List[int]
    ) -> List[str]:
        """Expand pattern to superset of control qubits."""
        missing = [q for q in superset if q not in gate_ctrls]
        if not missing:
            return [pattern]
        expanded = []
        for combo in range(2 ** len(missing)):
            p, pi, ci = "", 0, 0
            for q in superset:
                if q in gate_ctrls:
                    p += pattern[pi]
                    pi += 1
                else:
                    p += str((combo >> ci) & 1)
                    ci += 1
            expanded.append(p)
        return expanded

    def _build_gate(self, base_gate, params, ctrl_qubits, target_qubits, ctrl_state) -> Tuple:
        """Build a controlled gate."""
        gate = base_gate(*params) if params else base_gate()
        if not ctrl_qubits:
            return (gate, target_qubits)
        return (
            gate.control(len(ctrl_qubits), ctrl_state=ctrl_state[::-1]),
            ctrl_qubits + target_qubits,
        )

    def _build_optimized(self, group, optimizations, remaining) -> List[Tuple]:
        """Build optimized gates for pairwise optimizations."""
        bg = type(group[0].operation.base_gate)
        params, tgt, all_c = group[0].params, group[0].target_qubits, group[0].control_qubits
        gates = []
        for opt in optimizations:
            t, cp, cs = opt["type"], opt["control_positions"], opt["ctrl_state"]
            cq = [all_c[p] for p in cp]
            if t == "complementary":
                gates.append(self._build_gate(bg, params, cq, tgt, cs))
            elif t == "xor_standard":
                qi, qj = opt["xor_qubits"]
                gates.extend(
                    [
                        (CXGate(), [all_c[qi], all_c[qj]]),
                        self._build_gate(bg, params, cq, tgt, cs),
                        (CXGate(), [all_c[qi], all_c[qj]]),
                    ]
                )
            elif t == "xor_with_x":
                qi, qj = opt["xor_qubits"]
                qic, qjc = all_c[qi], all_c[qj]
                gates.extend(
                    [
                        (XGate(), [qjc]),
                        (CXGate(), [qic, qjc]),
                        self._build_gate(bg, params, cq, tgt, cs),
                        (CXGate(), [qic, qjc]),
                        (XGate(), [qjc]),
                    ]
                )
            elif t == "xor_chain":
                anc, tgts = all_c[opt["xor_anchor"]], [all_c[x] for x in opt["xor_targets"]]
                for tc in tgts:
                    gates.append((CXGate(), [anc, tc]))
                gates.append(self._build_gate(bg, params, cq, tgt, cs))
                for tc in reversed(tgts):
                    gates.append((CXGate(), [anc, tc]))
        # Remaining patterns
        rem_int = {int(p, 2) for p in remaining}
        for g in group:
            if int(g.ctrl_state, 2) in rem_int:
                gates.append((g.operation, g.control_qubits + g.target_qubits))
        return gates or None

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the ControlPatternSimplification pass."""
        to_apply = []
        for run in self._collect_gates(dag):
            used = set()
            # Mixed control counts
            for group in self._group_mixed_controls(run):
                superset = sorted(set().union(*(set(g.control_qubits) for g in group)))
                expanded = []
                for g in group:
                    expanded.extend(self._expand_pattern(g.ctrl_state, g.control_qubits, superset))
                if len(set(expanded)) == 2 ** len(superset):
                    bg = type(group[0].operation.base_gate)
                    to_apply.append(
                        (
                            group,
                            [self._build_gate(bg, group[0].params, [], group[0].target_qubits, "")],
                        )
                    )
                    used.update(id(g) for g in group)
            # Same control qubits
            remaining = [g for g in run if id(g) not in used]
            for group in self._group_same_controls(remaining):
                patterns = [g.ctrl_state for g in group]
                nq, unique = len(group[0].control_qubits), set(patterns)
                g0 = group[0]
                bg, params, ctrls, tgt = (
                    type(g0.operation.base_gate),
                    g0.params,
                    g0.control_qubits,
                    g0.target_qubits,
                )
                repl = None
                if len(unique) == 1:  # Merge angles
                    repl = [
                        self._build_gate(
                            bg,
                            (sum(g.params[0] for g in group),) if params else (),
                            ctrls,
                            tgt,
                            g0.ctrl_state,
                        )
                    ]
                else:
                    cls, info, cs = BitwisePatternAnalyzer(nq).simplify_patterns(patterns)
                    if cls == "single" and info:
                        repl = [self._build_gate(bg, params, [ctrls[info[0]]], tgt, cs)]
                    elif cls == "and" and info:
                        repl = [self._build_gate(bg, params, [ctrls[i] for i in info], tgt, cs)]
                    elif cls == "unconditional":
                        repl = [self._build_gate(bg, params, [], tgt, "")]
                    elif cls == "complement" and info:
                        neg = tuple(-p for p in params) if params else ()
                        repl = [
                            self._build_gate(bg, params, [], tgt, ""),
                            self._build_gate(bg, neg, [ctrls[i] for i in info], tgt, cs),
                        ]
                    elif cls == "pairwise_iterative" and info:
                        repl = self._build_optimized(
                            group, info["optimizations"], info["remaining_patterns"]
                        )
                    elif cls == "pairwise" and info:
                        repl = self._build_optimized(group, info, [])
                if repl:
                    to_apply.append((group, repl))
        for group, repl in to_apply:
            for g in group:
                dag.remove_op_node(g.node)
            for gate, qargs in repl:
                dag.apply_operation_back(gate, [dag.qubits[i] for i in qargs])
        return dag
