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

"""Simplify multi-controlled gates by Boolean algebraic reduction of control patterns."""

import numpy as np

from qiskit.circuit import ControlledGate
from qiskit.circuit.annotated_operation import AnnotatedOperation
from qiskit.circuit.library import CXGate, XGate
from qiskit.transpiler.basepasses import TransformationPass


class BitwisePatternAnalyzer:
    """Analyze control-state bit patterns for Boolean simplification opportunities."""

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def _find_common_bits(self, patterns):
        """Find bit positions that have the same value across all patterns."""
        if not patterns:
            return [], 0
        # AND all patterns together to find bits that are 1 in all
        all_ones = patterns[0]
        # AND all inverted patterns to find bits that are 0 in all
        all_zeros = ~patterns[0]
        for p in patterns[1:]:
            all_ones &= p
            all_zeros &= ~p
        mask = (1 << self.num_qubits) - 1
        all_zeros &= mask

        common_pos = []
        common_val = 0
        for pos in range(self.num_qubits):
            bit_mask = 1 << pos
            if all_ones & bit_mask:
                common_pos.append(pos)
                common_val |= bit_mask
            elif all_zeros & bit_mask:
                common_pos.append(pos)
                # bit is 0 in common_val already
        return common_pos, common_val

    def _check_single_variable(self, patterns):
        """Check if the pattern set simplifies to controlling on a single qubit."""
        n = self.num_qubits
        pattern_set = set(patterns)
        if len(pattern_set) != 2 ** (n - 1):
            return None
        for ctrl_pos in range(n):
            for val in (0, 1):
                expected = set()
                bit_mask = 1 << ctrl_pos
                for combo in range(2 ** (n - 1)):
                    p = 0
                    bit_idx = 0
                    for pos in range(n):
                        if pos == ctrl_pos:
                            if val:
                                p |= bit_mask
                        else:
                            if (combo >> bit_idx) & 1:
                                p |= 1 << pos
                            bit_idx += 1
                    expected.add(p)
                if pattern_set == expected:
                    return (ctrl_pos, val)
        return None

    def simplify_patterns_pairwise(self, patterns):
        """Find one pairwise simplification (complementary, XOR, or XOR-chain)."""
        if len(patterns) < 2:
            return None
        patterns_list = sorted(set(patterns))
        n = self.num_qubits

        # Hash-based search for Hamming distance 1 (complementary pairs)
        pattern_set = set(patterns_list)
        for p in patterns_list:
            for bit_pos in range(n):
                neighbor = p ^ (1 << bit_pos)
                if neighbor > p and neighbor in pattern_set:
                    # Found complementary pair differing at bit_pos
                    common_pos = [k for k in range(n) if k != bit_pos]
                    # Reindex ctrl_state to only include common positions
                    ctrl_state = 0
                    for idx, k in enumerate(common_pos):
                        if p & (1 << k):
                            ctrl_state |= 1 << idx
                    return [
                        {
                            "type": "complementary",
                            "patterns": [p, neighbor],
                            "control_positions": common_pos,
                            "ctrl_state": ctrl_state,
                        }
                    ]

        # O(n^2) search for Hamming distance 2 (XOR) and n (full chain)
        for i, p1 in enumerate(patterns_list):
            for p2 in patterns_list[i + 1 :]:
                xor_val = p1 ^ p2
                hamming = bin(xor_val).count("1")
                if hamming == 2:
                    diff = []
                    for k in range(n):
                        if xor_val & (1 << k):
                            diff.append(k)
                    pi, pj = diff
                    b1_i = (p1 >> pi) & 1
                    b1_j = (p1 >> pj) & 1
                    b2_i = (p2 >> pi) & 1
                    b2_j = (p2 >> pj) & 1
                    bits = ((b1_i, b1_j), (b2_i, b2_j))
                    if bits in [((1, 0), (0, 1)), ((0, 1), (1, 0))]:
                        xtype = "xor_standard"
                    elif bits in [((1, 1), (0, 0)), ((0, 0), (1, 1))]:
                        xtype = "xor_with_x"
                    else:
                        continue
                    common = [k for k in range(n) if k not in diff]
                    ctrl_pos = sorted(common + [pj])
                    ctrl_state = 0
                    for idx, k in enumerate(ctrl_pos):
                        if k == pj:
                            ctrl_state |= 1 << idx
                        elif p1 & (1 << k):
                            ctrl_state |= 1 << idx
                    return [
                        {
                            "type": xtype,
                            "patterns": [p1, p2],
                            "control_positions": ctrl_pos,
                            "ctrl_state": ctrl_state,
                            "xor_qubits": [pi, pj],
                        }
                    ]
                if hamming == n >= 2:
                    diff = []
                    for k in range(n):
                        if xor_val & (1 << k):
                            diff.append(k)
                    anchor = diff[0]
                    targets = diff[1:]
                    anchor_val = (p1 >> anchor) & 1
                    ctrl_state = 0
                    for idx in range(len(targets)):
                        if anchor_val:
                            ctrl_state |= 1 << idx
                    return [
                        {
                            "type": "xor_chain",
                            "patterns": [p1, p2],
                            "control_positions": targets,
                            "ctrl_state": ctrl_state,
                            "xor_anchor": anchor,
                            "xor_targets": targets,
                        }
                    ]
        return None

    def simplify_patterns_iterative(self, patterns):
        """Apply pairwise simplifications repeatedly until no more are found."""
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

    def simplify_patterns(self, patterns):
        """Determine the best simplification for a set of control patterns."""
        if not patterns:
            return ("no_optimization", None, None)
        unique = sorted(set(patterns))
        n = self.num_qubits
        # Complete partition
        if len(unique) == 2**n:
            return ("unconditional", [], 0)
        # Complement: all but one pattern
        if len(unique) == 2**n - 1:
            all_p = set(range(2**n))
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
        common_pos, common_val = self._find_common_bits(unique)
        if common_pos and len(common_pos) < n:
            varying = [i for i in range(n) if i not in common_pos]
            if len(unique) == 2 ** len(varying):
                # Reindex ctrl_state for just the common positions
                ctrl_state = 0
                for idx, pos in enumerate(common_pos):
                    if common_val & (1 << pos):
                        ctrl_state |= 1 << idx
                cls = "single" if len(common_pos) == 1 else "and"
                return (cls, common_pos, ctrl_state)
        return ("no_optimization", None, None)


class ControlPatternSimplification(TransformationPass):
    """Simplify groups of multi-controlled gates that share the same base gate and target
    by applying Boolean algebra to their control-state patterns."""

    def __init__(self, tolerance=1e-10):
        """ControlPatternSimplification initializer.

        Args:
            tolerance (float): numerical tolerance for comparing gate parameters.
        """
        super().__init__()
        self.tolerance = tolerance

    def _get_controlled_info(self, op):
        """Return ``(base_gate, num_ctrl_qubits, ctrl_state)`` or ``None``."""
        if isinstance(op, ControlledGate):
            return (op.base_gate, op.num_ctrl_qubits, op.ctrl_state)
        if isinstance(op, AnnotatedOperation):
            from qiskit.circuit import ControlModifier

            ctrl_mods = [m for m in op.modifiers if isinstance(m, ControlModifier)]
            if ctrl_mods:
                nc = sum(m.num_ctrl_qubits for m in ctrl_mods)
                return (op.base_op, nc, ctrl_mods[0].ctrl_state)
        return None

    def _params_match(self, p1, p2):
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

    def _collect_gates(self, dag):
        """Collect runs of consecutive controlled gates from the DAG."""
        runs, current = [], []
        for node in dag.topological_op_nodes():
            info = self._get_controlled_info(node.op)
            if info is not None:
                base_gate, nc, ctrl_state = info
                qargs = [dag.find_bit(q).index for q in node.qargs]
                ctrl_qubits = qargs[:nc]
                tgt_qubits = qargs[nc:]
                params = tuple(node.op.params) if node.op.params else ()
                # Normalize control qubit ordering and remap ctrl_state
                if ctrl_qubits != sorted(ctrl_qubits):
                    sort_order = sorted(range(nc), key=lambda k: ctrl_qubits[k])
                    new_cs = 0
                    for new_idx, old_idx in enumerate(sort_order):
                        if ctrl_state & (1 << old_idx):
                            new_cs |= 1 << new_idx
                    ctrl_qubits = [ctrl_qubits[k] for k in sort_order]
                    ctrl_state = new_cs
                current.append((node, base_gate, ctrl_qubits, tgt_qubits, ctrl_state, params))
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        return runs

    def _group_same_controls(self, gates):
        """Group gates sharing the same base gate, control qubits, and target."""
        if len(gates) < 2:
            return []
        groups = []
        used = set()
        for i in range(len(gates)):
            if i in used:
                continue
            _, bg_i, ctrl_i, tgt_i, _, params_i = gates[i]
            group = [gates[i]]
            group_indices = [i]
            for j in range(i + 1, len(gates)):
                if j in used:
                    continue
                _, bg_j, ctrl_j, tgt_j, _, params_j = gates[j]
                if (
                    bg_j.name == bg_i.name
                    and tgt_j == tgt_i
                    and ctrl_j == ctrl_i
                    and self._params_match(params_j, params_i)
                ):
                    group.append(gates[j])
                    group_indices.append(j)
                else:
                    # Two controlled gates commute if their target qubits
                    # are disjoint (shared controls don't break commutativity).
                    # For safety, also require no target-control overlap.
                    group_tgts = set(tgt_i)
                    gate_tgts = set(tgt_j)
                    if (
                        group_tgts.isdisjoint(gate_tgts)
                        and group_tgts.isdisjoint(set(ctrl_j))
                        and gate_tgts.isdisjoint(set(ctrl_i))
                    ):
                        continue  # skip over commuting gate
                    else:
                        break
            if len(group) >= 2:
                groups.append(group)
                used.update(group_indices)
        return groups

    def _group_mixed_controls(self, gates):
        """Group gates with subset/superset control qubit relationships."""
        if len(gates) < 2:
            return []
        groups, used = [], set()
        for i in range(len(gates)):
            if i in used:
                continue
            _, bg_i, ctrl_i, tgt_i, _, params_i = gates[i]
            group = [gates[i]]
            base_c = set(ctrl_i)
            for j in range(i + 1, len(gates)):
                if j in used:
                    continue
                _, bg_j, ctrl_j, tgt_j, _, params_j = gates[j]
                cand_c = set(ctrl_j)
                if (
                    bg_j.name == bg_i.name
                    and tgt_j == tgt_i
                    and self._params_match(params_j, params_i)
                    and (base_c <= cand_c or cand_c <= base_c)
                ):
                    group.append(gates[j])
                    used.add(j)
            if len(group) >= 2 and len(set(len(g[2]) for g in group)) > 1:
                groups.append(group)
                used.add(i)
        return groups

    def _expand_pattern(self, ctrl_state, gate_ctrls, superset):
        """Expand a control pattern to a superset of control qubits."""
        missing = [q for q in superset if q not in gate_ctrls]
        if not missing:
            return [ctrl_state]
        expanded = []
        gate_ctrl_set = set(gate_ctrls)
        for combo in range(2 ** len(missing)):
            p = 0
            gate_bit_idx = 0
            miss_bit_idx = 0
            for sup_idx, q in enumerate(superset):
                if q in gate_ctrl_set:
                    if ctrl_state & (1 << gate_bit_idx):
                        p |= 1 << sup_idx
                    gate_bit_idx += 1
                else:
                    if (combo >> miss_bit_idx) & 1:
                        p |= 1 << sup_idx
                    miss_bit_idx += 1
            expanded.append(p)
        return expanded

    def _build_gate(self, base_gate, params, ctrl_qubits, target_qubits, ctrl_state):
        """Build a ``(gate, qargs)`` tuple from the given parameters."""
        gate = base_gate(*params) if params else base_gate()
        if not ctrl_qubits:
            return (gate, target_qubits)
        return (
            gate.control(len(ctrl_qubits), ctrl_state=ctrl_state, annotated=False),
            ctrl_qubits + target_qubits,
        )

    def _build_optimized(self, group, optimizations, remaining):
        """Translate pairwise optimization descriptions into gate operations."""
        bg = type(group[0][1])  # base_gate type
        params = group[0][5]  # params
        tgt = group[0][3]  # target_qubits
        all_c = group[0][2]  # control_qubits
        gates = []
        for opt in optimizations:
            t = opt["type"]
            cp = opt["control_positions"]
            cs = opt["ctrl_state"]
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
                anc = all_c[opt["xor_anchor"]]
                tgts = [all_c[x] for x in opt["xor_targets"]]
                for tc in tgts:
                    gates.append((CXGate(), [anc, tc]))
                gates.append(self._build_gate(bg, params, cq, tgt, cs))
                for tc in reversed(tgts):
                    gates.append((CXGate(), [anc, tc]))
        # Remaining unmatched patterns
        rem_set = set(remaining)
        for g in group:
            if g[4] in rem_set:  # ctrl_state
                gates.append((g[0].op, [g[2][i] for i in range(len(g[2]))] + g[3]))
        return gates or None

    def run(self, dag):
        """Run the ControlPatternSimplification pass on ``dag``.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        # Identify groups of gates to optimize and their replacements.
        # Use _node_id (stable DAG node identifier) as keys.
        nodes_to_replace = {}
        for run in self._collect_gates(dag):
            used = set()
            # Mixed control counts
            for group in self._group_mixed_controls(run):
                all_ctrls = sorted(set().union(*(set(g[2]) for g in group)))
                expanded = []
                for g in group:
                    expanded.extend(self._expand_pattern(g[4], g[2], all_ctrls))
                if len(set(expanded)) == 2 ** len(all_ctrls):
                    bg = type(group[0][1])
                    repl = [self._build_gate(bg, group[0][5], [], group[0][3], 0)]
                    nodes_to_replace[group[0][0]._node_id] = repl
                    for g in group[1:]:
                        nodes_to_replace[g[0]._node_id] = None
                    used.update(g[0]._node_id for g in group)

            # Same control qubits
            remaining_gates = [g for g in run if g[0]._node_id not in used]
            for group in self._group_same_controls(remaining_gates):
                patterns = [g[4] for g in group]
                nq = len(group[0][2])
                unique = set(patterns)
                g0 = group[0]
                bg = type(g0[1])
                params = g0[5]
                ctrls = g0[2]
                tgt = g0[3]

                repl = None
                if len(unique) == 1:
                    # All gates have the same ctrl_state: merge angles
                    merged_params = (sum(g[5][0] for g in group),) if params else ()
                    repl = [self._build_gate(bg, merged_params, ctrls, tgt, g0[4])]
                else:
                    # Check for duplicate patterns: merge them first
                    from collections import Counter

                    counts = Counter(patterns)
                    has_duplicates = any(c > 1 for c in counts.values())
                    if has_duplicates:
                        # Merge duplicate patterns by summing angles, keep
                        # unique patterns as-is. Only proceed with pattern
                        # simplification on the unique-count subset.
                        repl = []
                        for cs_val, cnt in counts.items():
                            if cnt > 1 and params:
                                mp = (params[0] * cnt,) + params[1:]
                            else:
                                mp = params
                            repl.append(self._build_gate(bg, mp, ctrls, tgt, cs_val))
                    else:
                        # Each pattern appears exactly once: safe to simplify
                        cls, info, cs = BitwisePatternAnalyzer(nq).simplify_patterns(patterns)
                        if cls == "single" and info:
                            repl = [self._build_gate(bg, params, [ctrls[info[0]]], tgt, cs)]
                        elif cls == "and" and info:
                            repl = [self._build_gate(bg, params, [ctrls[i] for i in info], tgt, cs)]
                        elif cls == "unconditional":
                            repl = [self._build_gate(bg, params, [], tgt, 0)]
                        elif cls == "complement" and info:
                            neg = tuple(-p for p in params) if params else ()
                            repl = [
                                self._build_gate(bg, params, [], tgt, 0),
                                self._build_gate(bg, neg, [ctrls[i] for i in info], tgt, cs),
                            ]
                        elif cls == "pairwise_iterative" and info:
                            repl = self._build_optimized(
                                group, info["optimizations"], info["remaining_patterns"]
                            )
                        elif cls == "pairwise" and info:
                            repl = self._build_optimized(group, info, [])

                if repl:
                    nodes_to_replace[group[0][0]._node_id] = repl
                    for g in group[1:]:
                        nodes_to_replace[g[0]._node_id] = None

        # Build a new DAG preserving topological order
        new_dag = dag.copy_empty_like()
        for node in dag.topological_op_nodes():
            nid = node._node_id
            if nid in nodes_to_replace:
                replacement = nodes_to_replace[nid]
                if replacement is not None:
                    for gate, qargs in replacement:
                        new_dag.apply_operation_back(gate, [new_dag.qubits[i] for i in qargs])
            else:
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        return new_dag
