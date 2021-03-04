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
import numpy as np

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.optimization.axis_angle_analysis import AxisAngleAnalysis
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.library.standard_gates.rx import RXGate
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.library.standard_gates.rz import RZGate


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

        self._var_z_map = {'rz': RZGate, 'p': PhaseGate, 'u1': U1Gate}
        self.requires.append(AxisAngleAnalysis())


    def run(self, dag):
        """Run the AxisAngleReduction pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        self._commutation_analysis()
        dfprop = self.property_set['axis-angle']
        delList = list()
        for wire in dag.wires:
            node_it = dag.nodes_on_wire(wire)
            stack = list()  # list of (node, dfprop index)
            for node in node_it:
                if node.type != 'op':
                    delList += self._eval_stack(stack, dag)
                    stack = list()
                    continue
                # just doing 1q for now
                if len(node.qargs) != 1:
                    delList += self._eval_stack(stack, dag)
                    stack = list()
                    continue
                if not stack:
                    stack.append((node, self._get_index(node._node_id)))
                    continue
                topnode = stack[-1][0]
                topIndex = self._get_index(topnode._node_id)
                thisNode = node
                thisIndex = self._get_index(thisNode._node_id)
                topGroup = dfprop.iloc[topIndex].basis_group
                thisGroup = dfprop.iloc[thisIndex].basis_group
                if topGroup == thisGroup:
                    stack.append((thisNode, thisIndex))
                elif len(stack) > 1:
                    delList += self._eval_stack(stack, dag)
                    stack = [(thisNode, thisIndex)]  # start new stack with this valid op
            delList += self._eval_stack(stack, dag)
        for node in delList:
            dag.remove_op_node(node)
        return dag

    def _eval_stack(self, stack, dag):
        dfprop = self.property_set['axis-angle']
        if len(stack) <= 1:
            return []
        topnode = stack[-1][0]
        topIndex = self._get_index(topnode._node_id)
        var_gate = dfprop.iloc[topIndex].var_gate
        if var_gate:
            delList = self._reduce_stack(stack, var_gate, dag)
        else:
            delList = self._symmetry_cancellation(stack, dag)
            
        return delList

    def _get_index(self, idnode):
        """return the index in dfprop where idop occurs"""
        dfprop = self.property_set['axis-angle']
        dfprop.index[dfprop.id == idnode][0]
        return dfprop.index[dfprop.id == idnode][0]

    def _symmetry_cancellation(self, stack, dag):
        """elliminate gates by symmetry. This doesn't require a
        variable rotation gate for the axis"""
        if len(stack) <= 1:
            return []
        dfprop = self.property_set['axis-angle']
        stack_nodes, stack_indices = zip(*stack)
        delList = []
        delListStackIndices = []
        # get contiguous symmetry groups
        dfsubset = dfprop.iloc[list(stack_indices)]
        symmetry_groups = dfsubset.groupby(
            (dfsubset.symmetry_order.shift() != dfsubset.symmetry_order).cumsum())
        for _, dfsym in symmetry_groups:
            sym_order = dfsym.iloc[0].symmetry_order
            num_cancellation_groups, _ = divmod(len(dfsym), sym_order)
            if num_cancellation_groups == 0:
                continue
            del_ids = dfsym.iloc[0:num_cancellation_groups * sym_order].id
            thisDelList = [dag.node(delId) for delId in del_ids]
            delList += thisDelList
            # get indices of nodes in stack and remove from stack
            delListStackIndices += [stack_nodes.index(node) for node in thisDelList]
        redStack = [nodepair for inode, nodepair in enumerate(stack) if inode not in delListStackIndices]
        if len(redStack) < len(stack):
            # stack modified; attempt further cancellation recursively
            delList += self._symmetry_cancellation(redStack, dag)
        return delList

    def _reduce_stack(self, stack, var_gate, dag):
        """reduce common axis rotations to single rotation. This requires
        a single parameter rotation gate for the axis. Multiple parameter would
        be possible (e.g. RGate, RVGate) if one had a generic way of identifying how to rotate
        by the specified angle if the angle is not explicitly denoted in the gate arguments."""
        if not stack:
            return []
        dfprop = self.property_set['axis-angle']
        stack_nodes, stack_indices = zip(*stack)
        delList = []
        dfsubset = dfprop.iloc[list(stack_indices)]
        dfsubset['var_gate_angle'] = dfsubset.angle * dfsubset.rotation_sense
        params = dfsubset[['var_gate_angle', 'phase']].sum()
        if np.mod(params.var_gate_angle, (2 * np.pi)) > _CUTOFF_PRECISION:
            var_gate = self.property_set['var_gate_class'][var_gate](params.var_gate_angle)
            new_qarg = QuantumRegister(1, 'q')
            new_dag = DAGCircuit()
            new_dag.add_qreg(new_qarg)
            new_dag.apply_operation_back(var_gate, [new_qarg[0]])
            dag.substitute_node_with_dag(stack[0][0], new_dag)
            delList += [node for node, _ in stack[1:]]
        else:
            delList += [node for node, _ in stack]
        return delList

    def _commutation_analysis(self, global_basis=True):
        if not global_basis:
            raise CircuitError('not implemented')
        var_gate_class = dict()
        
        dfprop = self.property_set['axis-angle']
        dfaxis = dfprop.groupby('axis')
        buniq = dfprop.axis.unique()  # basis unique
        # merge collinear axes iff either contains a variable rotation
        naxes = len(buniq)
        # index pairs of buniq which are vectors in opposite directions
        buniq_inverses = list()
        for v1_ind, v1 in enumerate(buniq[:-1]):
            for v2_ind, v2 in enumerate(buniq[v1_ind+1:]):
                if math.isclose(abs(v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]), 1):
                    buniq_inverses.append((v1_ind, v1_ind + 1 + v2_ind))
        dfprop['basis_group'] = None
        # "rotation sense" is used to indicate sense of rotation wrt : +1=ccw, -1=cw
        dfprop['rotation_sense'] = 1
        # name of variable rotation gate for the basis group if it exists, else None
        dfprop['var_gate'] = None
        # count the number of independent bases
        basis_counter = 0
        basis_group_dict = dict()
        unlabeled_axes = list(range(naxes))
        # determine if inverses have arbitrary single parameter rotation
        mask_1p = dfprop.nparams == 1
        for pair in buniq_inverses:
            mask0 = dfprop.axis == buniq[pair[0]]
            mask1 = dfprop.axis == buniq[pair[1]]
            inv_pair_mask = mask0 | mask1
            if (dfprop[inv_pair_mask].nparams == 1).any():
                dfprop.loc[inv_pair_mask, 'basis_group'] = basis_counter
                unlabeled_axes.remove(pair[0])
                unlabeled_axes.remove(pair[1])
                basis_counter += 1
                # the basis with the single parameter variable gate gets the +1 sense of rotation
                if (dfprop[mask0].nparams == 1).any():
                    var_gate_name = dfprop[mask0 & mask_1p].name.iloc[0]  # arb. taking 0, but maybe prioritize zero phase
                    dfprop.loc[inv_pair_mask, 'var_gate'] = var_gate_name
                    dfprop.loc[mask1, 'rotation_sense'] = -1                    
                else:  # the single variable rotation gate must be in mask1
                    var_gate_name = dfprop[mask1 & mask_1p].name.iloc[0]  # arb. taking 0, but maybe prioritize zero phase
                    dfprop.loc[inv_pair_mask, 'var_gate'] = var_gate_name
                    dfprop.loc[mask0, 'rotation_sense'] = -1
            else:
                dfprop.loc[dfprop.axis == buniq[pair[0]], 'basis_group'] = basis_counter
                unlabeled_axes.remove(pair[0])
                basis_counter += 1
                dfprop.loc[dfprop.axis == buniq[pair[1]], 'basis_group'] = basis_counter
                unlabeled_axes.remove(pair[1])
                basis_counter += 1
        # index non-inverse bases
        for bindex in unlabeled_axes[:]:
            mask = dfprop.axis == buniq[bindex]
            dfprop.loc[mask, 'basis_group'] = basis_counter
            if (dfprop[mask].nparams == 1).any():
                var_gate_name = dfprop.loc[mask & mask_1p].name.iloc[0]
                dfprop.loc[mask, 'var_gate'] = var_gate_name
            unlabeled_axes.remove(bindex)
            basis_counter += 1
                                             
            
        
