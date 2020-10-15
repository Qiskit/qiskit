# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The module for Quantum the Fisher Information."""
import copy
from typing import List, Union, Optional

import numpy as np
from qiskit.aqua.operators import OperatorBase, ListOp, CircuitOp
from qiskit.aqua.operators.expectations import PauliExpectation
from qiskit.aqua.operators.operator_globals import I, Z, Y, X, Zero
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn
from qiskit.circuit import ParameterVector, ParameterExpression
from qiskit.circuit.library import RZGate, RXGate, RYGate
from qiskit.converters import dag_to_circuit, circuit_to_dag

from .circuit_qfi import CircuitQFI
from ..derivative_base import DerivativeBase


class OverlapDiag(CircuitQFI):
    r"""Compute the diagonal of the QFI given a pure, parametrized quantum state.

    See also :class:`~qiskit.aqua.operators.QFI`.
    """

    def convert(self,
                operator: Union[CircuitOp, CircuitStateFn],
                params: Optional[Union[ParameterExpression, ParameterVector,
                                       List[ParameterExpression]]] = None
                ) -> ListOp:
        r"""
        Args:
            operator: The operator corresponding to the quantum state :math:`|\psi(\omega)\rangle`
                for which we compute the QFI.
            params: The parameters :math:`\omega` with respect to which we are computing the QFI.

        Returns:
            A ``ListOp[ListOp]`` where the operator at position ``[k][l]`` corresponds to the matrix
            element :math:`k, l` of the QFI.

        Raises:
            NotImplementedError: If ``operator`` is neither ``CircuitOp`` nor ``CircuitStateFn``.

        """

        if not isinstance(operator, CircuitStateFn):
            raise NotImplementedError('operator must be a CircuitStateFn')

        return self._diagonal_approx(operator=operator, params=params)

    # TODO, for some reason diagonal_approx doesn't use the same get_parameter_expression method.
    # This should be fixed.
    def _diagonal_approx(self,
                         operator: Union[CircuitOp, CircuitStateFn],
                         params: Union[ParameterExpression, ParameterVector, List] = None
                         ) -> OperatorBase:
        """
        Args:
            operator: The operator corresponding to the quantum state |ψ(ω)〉for which we compute
                the QFI
            params: The parameters we are computing the QFI wrt: ω

        Returns:
            ListOp where the operator at position k corresponds to QFI_k,k

        Raises:
            NotImplementedError: If a circuit is found such that one parameter controls multiple
                                 gates, or one gate contains multiple parameters.
            TypeError: If a circuit is found that includes more than one parameter as they are
                       currently not supported for the overlap diagonal QFI method.

        """

        if not isinstance(operator, CircuitStateFn):
            raise NotImplementedError('operator must be a CircuitStateFn')

        circuit = operator.primitive

        # Partition the circuit into layers, and build the circuits to prepare $\psi_i$
        layers = _partition_circuit(circuit)
        if layers[-1].num_parameters == 0:
            layers.pop(-1)

        psis = [CircuitOp(layer) for layer in layers]
        for i, psi in enumerate(psis):
            if i == 0:
                continue
            psis[i] = psi @ psis[i - 1]

        # TODO: make this work for other types of rotations
        # NOTE: This assumes that each parameter only affects one rotation.
        # we need to think more about what happens if multiple rotations
        # are controlled with a single parameter.
        generators = _get_generators(params, circuit)

        diag = []
        for param in params:
            if len(circuit._parameter_table[param]) > 1:
                raise NotImplementedError("OverlapDiag does not yet support multiple "
                                          "gates parameterized by a single parameter. For such "
                                          "circuits use LinCombFull")

            gate = circuit._parameter_table[param][0][0]

            if len(gate.params) != 1:
                raise TypeError("OverlapDiag cannot yet support gates with more than one "
                                "parameter.")

            param_value = gate.params[0]
            generator = generators[param]
            meas_op = ~StateFn(generator)

            # get appropriate psi_i
            psi = [(psi) for psi in psis if param in psi.primitive.parameters][0]

            op = meas_op @ psi @ Zero
            if type(param_value) == ParameterExpression:  # pylint: disable=unidiomatic-typecheck
                expr_grad = DerivativeBase.parameter_expression_grad(param_value, param)
                op *= expr_grad
            rotated_op = PauliExpectation().convert(op)
            diag.append(rotated_op)

        grad_op = ListOp(diag, combo_fn=lambda x: np.diag(np.real([1 - y ** 2 for y in x])))
        return grad_op


def _partition_circuit(circuit):
    dag = circuit_to_dag(circuit)
    dag_layers = ([i['graph'] for i in dag.serial_layers()])
    num_qubits = circuit.num_qubits
    layers = list(
        zip(dag_layers, [{x: False for x in range(0, num_qubits)} for layer in dag_layers]))

    # initialize the ledger
    # The ledger tracks which qubits in each layer are available to have
    # gates from subsequent layers shifted backward.
    # The idea being that all parameterized gates should have
    # no descendants within their layer
    for i, (layer, ledger) in enumerate(layers):
        op_node = layer.op_nodes()[0]
        is_param = op_node.op.is_parameterized()
        qargs = op_node.qargs
        indices = [qarg.index for qarg in qargs]
        if is_param:
            for index in indices:
                ledger[index] = True

    def apply_node_op(node, dag, back=True):
        op = copy.copy(node.op)
        qargs = copy.copy(node.qargs)
        cargs = copy.copy(node.cargs)
        condition = copy.copy(node.condition)
        if back:
            dag.apply_operation_back(op, qargs, cargs, condition)
        else:
            dag.apply_operation_front(op, qargs, cargs, condition)

    converged = False

    for _ in range(dag.depth() + 1):
        if converged:
            break

        converged = True

        for i, (layer, ledger) in enumerate(layers):
            if i == len(layers) - 1:
                continue

            (next_layer, next_ledger) = layers[i + 1]
            for next_node in next_layer.op_nodes():
                is_param = next_node.op.is_parameterized()
                qargs = next_node.qargs
                indices = [qarg.index for qarg in qargs]

                # If the next_node can be moved back a layer without
                # without becoming the descendant of a parameterized gate,
                # then do it.
                if not any([ledger[x] for x in indices]):

                    apply_node_op(next_node, layer)
                    next_layer.remove_op_node(next_node)

                    if is_param:
                        for index in indices:
                            ledger[index] = True
                            next_ledger[index] = False

                    converged = False

            # clean up empty layers left behind.
            if len(next_layer.op_nodes()) == 0:
                layers.pop(i + 1)

    partitioned_circs = [dag_to_circuit(layer[0]) for layer in layers]
    return partitioned_circs


def _get_generators(params, circuit):
    dag = circuit_to_dag(circuit)
    layers = list(dag.serial_layers())

    generators = {}
    num_qubits = dag.num_qubits()

    for layer in layers:
        instr = layer['graph'].op_nodes()[0].op
        if len(instr.params) == 0:
            continue
        assert len(instr.params) == 1, "Circuit was not properly decomposed"
        param_value = instr.params[0]
        for param in params:
            if param in param_value.parameters:

                if isinstance(instr, RYGate):
                    generator = Y
                elif isinstance(instr, RZGate):
                    generator = Z
                elif isinstance(instr, RXGate):
                    generator = X
                else:
                    raise NotImplementedError

                # get all qubit indices in this layer where the param parameterizes
                # an operation.
                indices = [[q.index for q in qreg] for qreg in layer['partition']]
                indices = [item for sublist in indices for item in sublist]

                if len(indices) > 1:
                    raise NotImplementedError
                index = indices[0]
                generator = (I ^ (index)) ^ generator ^ (I ^ (num_qubits - index - 1))
                generators[param] = generator

    return generators
