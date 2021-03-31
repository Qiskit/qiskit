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

"""Cancel the redundant (self-adjoint) gates through commutation relations."""

import math
from collections import deque
import numpy as np
import pandas as pd
from qiskit.quantum_info import Operator
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.optimization.axis_angle_analysis import (AxisAngleAnalysis,
                                                                       _su2_axis_angle)
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import Gate


_CUTOFF_PRECISION = 1E-5


class AxisAngleReduction(TransformationPass):
    """Reduce runs of single qubit gates with common axes.
    """

    def __init__(self, basis_gates=None):
        """
        AxisAngleReduction initializer.
        """
        super().__init__()
        if basis_gates:
            self.basis = set(basis_gates)
        else:
            self.basis = set()
        self._ctrl_axis = (1.0, 0., 0.)
        self.requires.append(AxisAngleAnalysis())

    def run(self, dag):
        """Run the AxisAngleReduction pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        self._reduction_analysis()
        dfprop = self.property_set['axis-angle']
        total_del_list = list()
        new_del_nodes = list()
        while True:
            new_del_nodes = self._run(dag, dfprop, total_del_list)
            if new_del_nodes:
                total_del_list += new_del_nodes
            else:
                break
        for node in total_del_list:
            dag.remove_op_node(node)
        return dag

    def _run(self, dag, dfprop, total_del_list):
        del_list = list()
        for wire in dag.wires:
            node_it = dag.nodes_on_wire(wire)
            stack = list()  # list of (node, dfprop index)
            for node in node_it:
                num_qargs = len(node.qargs)
                if node in total_del_list:
                    # for cycles of reduction
                    continue
                if node.type != 'op' or not isinstance(node.op, Gate):
                    # stack done, evaluate
                    del_list += self._eval_stack(stack, dag)
                    stack = list()
                    continue
                # just doing 1q and 2q
                if num_qargs > 2:
                    # stack done, evaluate
                    del_list += self._eval_stack(stack, dag)
                    stack = list()
                    continue
                this_node = node
                try:
                    this_index = self._get_index(this_node._node_id)
                except IndexError as ierr:
                    # This is probably a custom 2-qubit gate; stop
                    del_list += self._eval_stack(stack, dag)
                    stack = list()
                    continue
                else:
                    this_group = dfprop.iloc[this_index].basis_group
                if not stack:
                    # add first node to stack
                    stack.append((node, self._get_index(node._node_id)))
                    continue
                # add to stack if commuting
                top_node = stack[-1][0]
                top_index = self._get_index(top_node._node_id)
                top_group = dfprop.iloc[top_index].basis_group
                if top_group == this_group:
                    if num_qargs == 1:
                        stack.append((this_node, this_index))
                    elif num_qargs == 2:
                        if self._check_next_2q_commuting(dag, top_node, this_node):
                            stack.append((this_node, this_index))
                elif len(stack) > 1:
                    del_list += self._eval_stack(stack, dag)
                    stack = [(this_node, this_index)]  # start new stack with this valid op
                else:
                    stack = [(this_node, this_index)]  # start new stack with this valid op
        return del_list

    def _print_stack(self, stack):
        print('-'*4)
        for node, ind in stack:
            qrg = ' '.join([str(qarg.index) for qarg in node.qargs])
            print(f'{node.name} {qrg} ({hex(id(node))}) {node._node_id}')

    def _check_next_2q_commuting(self, dag, node1, node2):
        """checks that node2 is same type as node1 with only commuting
        single qubit ops in between"""
        from qiskit.circuit import ControlledGate
        dfprop = self.property_set['axis-angle'].set_index('id')
        successors = {(node2 in connode): connode[2] for connode in dag.edges(node1)}
        is_direct_successor = all(successors.keys())
        if is_direct_successor:
            if node1.name == node2.name and node1.qargs == node2.qargs:
                return True
            else:
                return False
        non_direct_qubit = successors[False]
        if isinstance(node1.op, ControlledGate) and node1.qargs[0] == non_direct_qubit:
            axis1 = (0, 0, 1)
        else:
            axis1 = dfprop.loc[node1._node_id].axis
        next_node = self._next_node_on_qubit(dag, node1, non_direct_qubit)
        while next_node is not node2:
            if len(next_node.qargs) == 1:
                try:
                    this_axis = dfprop.loc[next_node._node_id].axis
                except KeyError as kerr:
                    return False
                if math.isclose(abs(sum([a * b for a, b in zip(axis1, this_axis)])), 1):
                    next_node = self._next_node_on_qubit(dag, next_node, non_direct_qubit)
                else:
                    return False
            else:
                return False
        return bool(next_node)

    def _next_node_on_qubit(self, dag, node, qubit):
        """Returns next node on qubit."""
        anode = dag._multi_graph.find_adjacent_node_by_edge(
            node._node_id,
            lambda edge, target_edge=qubit: edge == target_edge)
        return anode

    def _eval_stack(self, stack, dag):
        dfprop = self.property_set['axis-angle']
        if len(stack) <= 1:
            return []
        top_node = stack[-1][0]
        top_index = self._get_index(top_node._node_id)
        var_gate = dfprop.iloc[top_index].var_gate
        if var_gate and not self._symmetry_complete(stack):
            del_list = self._reduce_stack(stack, var_gate, dag)
        else:
            del_list = self._symmetry_cancellation(stack, dag)
        return del_list

    def _get_index(self, idnode):
        """return the index in dfprop where idop occurs"""
        dfprop = self.property_set['axis-angle']
        return dfprop.index[dfprop.id == idnode][0]

    def _symmetry_cancellation(self, stack, dag):
        """Elliminate gates by symmetry. This doesn't require a
        variable rotation gate for the axis.
        Args:
            stack (list(DAGNode, int)): All nodes share a rotation axis and the int
                indexes the node in the dataframe.
            dag (DAGCircuit): the whole dag. Will not be modified.

        Returns:
            list(DAGNode): List of dag nodes to delete from dag ultimately.
        """
        if len(stack) <= 1:
            return []
        dfprop = self.property_set['axis-angle']
        stack_nodes, stack_indices = zip(*stack)
        del_list = []
        del_list_stack_indices = []
        # get contiguous symmetry groups
        dfsubset = dfprop.iloc[list(stack_indices)]
        symmetry_groups = dfsubset.groupby(
            (dfsubset.symmetry_order.shift() != dfsubset.symmetry_order).cumsum())
        for _, dfsym in symmetry_groups:
            sym_order = dfsym.iloc[0].symmetry_order
            if sym_order == 1:
                # no rotational symmetry
                continue
            num_cancellation_groups = len(dfsym) // sym_order
            groups_phase = dfsym.phase.iloc[0:num_cancellation_groups * sym_order].sum()
            if num_cancellation_groups == 0:
                # not enough members to satisfy symmetry cancellation
                continue
            is_1q = dfsym.qubit1.isnull().all()
            if num_cancellation_groups % 2 and is_1q:  # double cover (todo:improve conditionals)
                dag.global_phase += np.pi
            if math.cos(groups_phase) == -1 and is_1q:
                dag.global_phase += np.pi
            del_ids = dfsym.iloc[0:num_cancellation_groups * sym_order].id
            this_del_list = [dag.node(delId) for delId in del_ids]
            del_list += this_del_list
            # get indices of nodes in stack and remove from stack
            del_list_stack_indices += [stack_nodes.index(node)
                                       for node in this_del_list]
        red_stack = [nodepair for inode, nodepair in enumerate(stack)
                     if inode not in del_list_stack_indices]
        if len(red_stack) < len(stack):
            # stack modified; attempt further cancellation recursively
            del_list += self._symmetry_cancellation(red_stack, dag)
        return del_list

    def _reduce_stack(self, stack, var_gate_name, dag):
        """reduce common axis rotations to single rotation. This requires
        a single parameter rotation gate for the axis. Multiple parameter would
        be possible (e.g. RGate, RVGate) if one had a generic way of identifying how to rotate
        by the specified angle if the angle is not explicitly denoted in the gate arguments."""
        if not stack:
            return []
        dfprop = self.property_set['axis-angle']
        _, stack_indices = zip(*stack)
        smask = list(stack_indices)
        if not dfprop.iloc[smask].qubit1.all() and var_gate_name:
            period = 2 * np.pi
        else:
            period = 4 * np.pi
        del_list = []
        dfsubset = dfprop.iloc[smask].copy()
        dfsubset['var_gate_angle'] = dfsubset.angle * dfsubset.rotation_sense
        params = dfsubset[['var_gate_angle', 'phase']].sum()
        if np.mod(params.var_gate_angle, period) > _CUTOFF_PRECISION:
            var_gate = self.property_set['var_gate_class'][var_gate_name](
                params.var_gate_angle % period)
            new_qarg = QuantumRegister(var_gate.num_qubits, 'q')
            new_dag = DAGCircuit()
            # the variable gate for the axis may not be in this stack
            df_gate = dfprop[dfprop.name == var_gate_name]
            df_gate_phase = df_gate.phase
            df_gate_angle = df_gate.angle
            df_gate_phase_factor = df_gate_phase / df_gate_angle
            phase_factor_uni = df_gate_phase_factor.unique()
            if len(phase_factor_uni) == 1 and np.isfinite(phase_factor_uni[0]):
                gate_phase_factor = phase_factor_uni[0]
            else:
                _, _, gate_phase_factor = _su2_axis_angle(Operator(var_gate).data)
            new_dag.global_phase = params.phase - params.var_gate_angle * gate_phase_factor
            new_dag.add_qreg(new_qarg)
            new_dag.apply_operation_back(var_gate, new_qarg[:])
            try:
                dag.substitute_node_with_dag(stack[0][0], new_dag)
            except Exception as err:
                breakpoint()
            del_list += [node for node, _ in stack[1:]]
        else:
            del_list += [node for node, _ in stack]
        return del_list

    def _reduction_analysis(self, rel_tol=1e-9, abs_tol=0.0):
        dfprop = self.property_set['axis-angle']
        if dfprop.empty:
            return
        buniq = dfprop.axis.unique()  # basis unique
        # merge collinear axes iff either contains a variable rotation
        naxes = len(buniq)
        # index pairs of buniq which are vectors in opposite directions
        buniq_inverses = list()
        buniq_parallel = list()
        if naxes > 1:
            vdot = np.full((naxes, naxes), np.nan)
            for v1_ind in range(naxes):
                v1 = buniq[v1_ind]
                for v2_ind in range(v1_ind + 1, naxes):
                    v2 = buniq[v2_ind]
                    vdot[v1_ind, v2_ind] = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
            buniq_parallel = list(zip(*np.where(np.isclose(vdot, 1, rtol=rel_tol, atol=abs_tol))))
            buniq_inverses = list(zip(*np.where(np.isclose(vdot, -1, rtol=rel_tol, atol=abs_tol))))
            buniq_common = buniq_parallel + buniq_inverses
            grouped_common = [list(group) for group in join_if_intersect(buniq_common)]
        else:
            grouped_common = [[0]]
        dfprop['basis_group'] = None
        # "rotation sense" is used to indicate sense of rotation wrt : +1=ccw, -1=cw
        dfprop['rotation_sense'] = 1
        # name of variable rotation gate for the basis group if it exists, else None
        dfprop['var_gate'] = None
        # count the number of independent bases
        basis_counter = 0
        unlabeled_axes = list(range(naxes))
        # determine if inverses have arbitrary single parameter rotation
        mask_1p = dfprop.nparams == 1
        mask_1q = dfprop.qubit1.isnull()
        mask_2q = ~mask_1q
        # loop through parallel axis groups
        for group in grouped_common:
            lead = group[0]  # this will be the reference direction for the group
            mask_axis = pd.Series(False, index=range(dfprop.shape[0]))
            for member in group:
                # use 'almost_equal'?
                mask_axis |= dfprop.axis == buniq[member]
                unlabeled_axes.remove(member)
            mask_1q_axis = mask_axis & mask_1q
            mask_2q_axis = mask_axis & mask_2q
            if mask_1q_axis.any():
                dfprop.loc[mask_1q_axis, 'basis_group'] = basis_counter
                mask_1q_1p = mask_1q_axis & mask_1p
                if mask_1q_1p.any():
                    var_gate_name = dfprop.loc[mask_1q_1p].name.iloc[0]
                    dfprop.loc[mask_1q_axis, 'var_gate'] = var_gate_name
                basis_counter += 1
            if mask_2q_axis.any():
                dfprop.loc[mask_2q_axis, 'basis_group'] = basis_counter
                mask_2q_1p = mask_2q_axis & mask_1p
                if mask_2q_1p.any():
                    var_gate_name = dfprop.loc[mask_2q_1p].name.iloc[0]
                    dfprop.loc[mask_2q_axis, 'var_gate'] = var_gate_name
                basis_counter += 1
            # create mask for inverses to lead
            mask_axis[:] = False
            for pair in buniq_inverses:
                try:
                    mask_axis |= dfprop.axis == buniq[pair[int(not pair.index(lead))]]
                except ValueError:
                    # positive sense lead is not in pair; skip
                    pass
            dfprop.loc[mask_axis, 'rotation_sense'] = -1
        # index lone bases
        for bindex in unlabeled_axes[:]:
            mask_axis = dfprop.axis == buniq[bindex]
            mask_1q_axis = mask_axis & mask_1q
            mask_2q_axis = mask_axis & mask_2q
            if mask_1q_axis.any():
                dfprop.loc[mask_1q_axis, 'basis_group'] = basis_counter
                if (dfprop[mask_1q_axis].nparams == 1).any():
                    var_gate_name = dfprop.loc[mask_1q_axis].name.iloc[0]
                    dfprop.loc[mask_1q_axis, 'var_gate'] = var_gate_name
                if bindex in unlabeled_axes:
                    unlabeled_axes.remove(bindex)
                basis_counter += 1
            if mask_2q_axis.any():
                dfprop.loc[mask_2q_axis, 'basis_group'] = basis_counter
                if (dfprop[mask_2q_axis].nparams == 1).any():
                    var_gate_name = dfprop.loc[mask_2q_axis].name.iloc[0]
                    dfprop.loc[mask_2q_axis, 'var_gate'] = var_gate_name
                if bindex in unlabeled_axes:
                    unlabeled_axes.remove(bindex)
                basis_counter += 1

    def _symmetry_complete(self, stack):
        """Determine whether complete cancellation is possible due to symmetry"""
        dfprop = self.property_set['axis-angle']
        _, stack_indices = zip(*stack)
        sym_order = dfprop.iloc[list(stack_indices)].symmetry_order
        sym_order_zero = sym_order.iloc[0]
        return (sym_order_zero == len(sym_order)) and all(sym_order_zero == sym_order)


def join_if_intersect(lists):
    """This is from user 'agf' on stackoverflow
    https://stackoverflow.com/questions/9110837/python-simple-list-merging-based-on-intersections
    """
    results = []
    if not lists:
        return results
    sets = deque(set(lst) for lst in lists if lst)
    disjoint = 0
    current = sets.pop()
    while True:
        merged = False
        newsets = deque()
        for _ in range(disjoint, len(sets)):
            this = sets.pop()
            if not current.isdisjoint(this):
                current.update(this)
                merged = True
                disjoint = 0
            else:
                newsets.append(this)
                disjoint += 1
        if sets:
            newsets.extendleft(sets)
        if not merged:
            results.append(current)
            try:
                current = newsets.pop()
            except IndexError:
                break
            disjoint = 0
        sets = newsets
    return results
