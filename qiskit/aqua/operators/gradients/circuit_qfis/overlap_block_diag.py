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

from typing import List, Union, Optional

import numpy as np
from scipy.linalg import block_diag
from qiskit.aqua import AquaError
from qiskit.aqua.operators import ListOp, CircuitOp
from qiskit.aqua.operators.expectations import PauliExpectation
from qiskit.aqua.operators.operator_globals import Zero
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression

from .circuit_qfi import CircuitQFI
from ..derivative_base import DerivativeBase
from .overlap_diag import _get_generators, _partition_circuit


class OverlapBlockDiag(CircuitQFI):
    r"""Compute the block-diagonal of the QFI given a pure, parametrized quantum state.

    The blocks are given by all parameterized gates in quantum circuit layer.
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
        if not isinstance(operator, (CircuitStateFn)):
            raise NotImplementedError('operator must be a CircuitOp or CircuitStateFn')
        return self._block_diag_approx(operator=operator, params=params)

    def _block_diag_approx(self,
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
            NotImplementedError: If a circuit is found such that one parameter controls multiple
                gates, or one gate contains multiple parameters.
            AquaError: If there are more than one parameter.

        """

        circuit = operator.primitive
        # Partition the circuit into layers, and build the circuits to prepare $\psi_i$
        layers = _partition_circuit(circuit)
        if layers[-1].num_parameters == 0:
            layers.pop(-1)

        block_params = [list(layer.parameters) for layer in layers]
        # Remove any parameters found which are not in params
        block_params = [[param for param in block if param in params] for block in block_params]

        # Determine the permutation needed to ensure that the final
        # operator is consistent with the ordering of the input parameters
        perm = [params.index(param) for block in block_params for param in block]

        psis = [CircuitOp(layer) for layer in layers]
        for i, psi in enumerate(psis):
            if i == 0:
                continue
            psis[i] = psi @ psis[i - 1]

        # Get generators
        # TODO: make this work for other types of rotations
        # NOTE: This assumes that each parameter only affects one rotation.
        # we need to think more about what happens if multiple rotations
        # are controlled with a single parameter.

        generators = _get_generators(params, circuit)

        blocks = []

        # Psi_i = layer_i @ layer_i-1 @ ... @ layer_0 @ Zero
        for k, psi_i in enumerate(psis):
            params = block_params[k]
            block = np.zeros((len(params), len(params))).tolist()

            # calculate all single-operator terms <psi_i|generator_i|psi_i>
            single_terms = np.zeros(len(params)).tolist()
            for i, p_i in enumerate(params):
                generator = generators[p_i]
                psi_gen_i = ~StateFn(generator) @ psi_i @ Zero
                psi_gen_i = PauliExpectation().convert(psi_gen_i)
                single_terms[i] = psi_gen_i

            def get_parameter_expression(circuit, param):
                if len(circuit._parameter_table[param]) > 1:
                    raise NotImplementedError("OverlapDiag does not yet support multiple "
                                              "gates parameterized by a single parameter. For such "
                                              "circuits use LinCombFull")
                gate = circuit._parameter_table[param][0][0]
                if len(gate.params) > 1:
                    raise AquaError("OverlapDiag cannot yet support gates with more than one "
                                    "parameter.")

                param_value = gate.params[0]
                return param_value

            # Calculate all double-operator terms <psi_i|generator_j @ generator_i|psi_i>
            # and build composite operators for each matrix entry
            for i, p_i in enumerate(params):
                generator_i = generators[p_i]
                param_expr_i = get_parameter_expression(circuit, p_i)

                for j, p_j in enumerate(params):
                    if i == j:
                        block[i][i] = ListOp([single_terms[i]], combo_fn=lambda x: 1 - x[0] ** 2)
                        if isinstance(param_expr_i, ParameterExpression) and not isinstance(
                                param_expr_i, Parameter):
                            expr_grad_i = DerivativeBase.parameter_expression_grad(
                                param_expr_i, p_i)
                            block[i][j] *= (expr_grad_i) * (expr_grad_i)
                        continue

                    generator_j = generators[p_j]
                    generator = ~generator_j @ generator_i
                    param_expr_j = get_parameter_expression(circuit, p_j)

                    psi_gen_ij = ~StateFn(generator) @ psi_i @ Zero
                    psi_gen_ij = PauliExpectation().convert(psi_gen_ij)
                    cross_term = ListOp([single_terms[i], single_terms[j]], combo_fn=np.prod)
                    block[i][j] = psi_gen_ij - cross_term

                    # pylint: disable=unidiomatic-typecheck
                    if type(param_expr_i) == ParameterExpression:
                        expr_grad_i = DerivativeBase.parameter_expression_grad(param_expr_i, p_i)
                        block[i][j] *= expr_grad_i
                    if type(param_expr_j) == ParameterExpression:
                        expr_grad_j = DerivativeBase.parameter_expression_grad(param_expr_j, p_j)
                        block[i][j] *= expr_grad_j

            wrapped_block = ListOp([ListOp(row) for row in block])
            blocks.append(wrapped_block)

        block_diagonal_qfi = ListOp(oplist=blocks,
                                    combo_fn=lambda x: np.real(block_diag(*x))[:, perm][perm, :])
        return block_diagonal_qfi
