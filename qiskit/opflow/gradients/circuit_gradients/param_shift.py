# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The module to compute the state gradient with the parameter shift rule."""

from collections.abc import Iterable
from copy import deepcopy
from functools import partial
from typing import List, Union, Tuple, Dict

import scipy
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector
from qiskit.utils.deprecation import deprecate_func
from .circuit_gradient import CircuitGradient
from ...operator_base import OperatorBase
from ...state_fns.state_fn import StateFn
from ...operator_globals import Zero, One
from ...state_fns.circuit_state_fn import CircuitStateFn
from ...primitive_ops.circuit_op import CircuitOp
from ...list_ops.summed_op import SummedOp
from ...list_ops.list_op import ListOp
from ...list_ops.composed_op import ComposedOp
from ...state_fns.dict_state_fn import DictStateFn
from ...state_fns.vector_state_fn import VectorStateFn
from ...state_fns.sparse_vector_state_fn import SparseVectorStateFn
from ...exceptions import OpflowError
from ..derivative_base import _coeff_derivative


class ParamShift(CircuitGradient):
    """Deprecated: Compute the gradient d⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω respectively the gradients of the sampling
    probabilities of the basis states of a state |ψ(ω)〉w.r.t. ω with the parameter shift
    method.
    """

    SUPPORTED_GATES = {"x", "y", "z", "h", "rx", "ry", "rz", "p", "u", "cx", "cy", "cz"}

    @deprecate_func(
        since="0.24.0",
        package_name="qiskit-terra",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(self, analytic: bool = True, epsilon: float = 1e-6):
        r"""
        Args:
            analytic: If True use the parameter shift rule to compute analytic gradients,
                      else use a finite difference approach
            epsilon: The offset size to use when computing finite difference gradients.
                     Ignored if analytic == True

        Raises:
            ValueError: If method != ``fin_diff`` and ``epsilon`` is not None.
        """
        super().__init__()
        self._analytic = analytic
        self._epsilon = epsilon

    @property
    def analytic(self) -> bool:
        """Returns ``analytic`` flag.

        Returns:
             ``analytic`` flag.

        """
        return self._analytic

    @property
    def epsilon(self) -> float:
        """Returns ``epsilon``.

        Returns:
            ``epsilon``.

        """
        return self._epsilon

    # pylint: disable=signature-differs
    def convert(
        self,
        operator: OperatorBase,
        params: Union[
            ParameterExpression,
            ParameterVector,
            List[ParameterExpression],
            Tuple[ParameterExpression, ParameterExpression],
            List[Tuple[ParameterExpression, ParameterExpression]],
        ],
    ) -> OperatorBase:
        """
        Args:
            operator: The operator corresponding to our quantum state we are taking the
                      gradient of: |ψ(ω)〉
            params: The parameters we are taking the gradient wrt: ω
                    If a ParameterExpression, ParameterVector or List[ParameterExpression] is given,
                    then the 1st order derivative of the operator is calculated.
                    If a Tuple[ParameterExpression, ParameterExpression] or
                    List[Tuple[ParameterExpression, ParameterExpression]]
                    is given, then the 2nd order derivative of the operator is calculated.

        Returns:
            An operator corresponding to the gradient resp. Hessian. The order is in accordance with
            the order of the given parameters.

        Raises:
            OpflowError: If the parameters are given in an invalid format.

        """
        if isinstance(params, (ParameterExpression, ParameterVector)):
            return self._parameter_shift(operator, params)
        elif isinstance(params, tuple):
            return self._parameter_shift(self._parameter_shift(operator, params[0]), params[1])
        elif isinstance(params, Iterable):
            if all(isinstance(param, ParameterExpression) for param in params):
                return self._parameter_shift(operator, params)
            elif all(isinstance(param, tuple) for param in params):
                return ListOp(
                    [
                        self._parameter_shift(self._parameter_shift(operator, pair[0]), pair[1])
                        for pair in params
                    ]
                )
            else:
                raise OpflowError(
                    "The linear combination gradient does only support "
                    "the computation "
                    "of 1st gradients and 2nd order gradients."
                )
        else:
            raise OpflowError(
                "The linear combination gradient does only support the computation "
                "of 1st gradients and 2nd order gradients."
            )

    # pylint: disable=too-many-return-statements
    def _parameter_shift(
        self, operator: OperatorBase, params: Union[ParameterExpression, ParameterVector, List]
    ) -> OperatorBase:
        r"""
        Args:
            operator: The operator containing circuits we are taking the derivative of.
            params: The parameters (ω) we are taking the derivative with respect to. If
                    a ParameterVector is provided, each parameter will be shifted.

        Returns:
            param_shifted_op: An operator object which evaluates to the respective gradients.

        Raises:
            ValueError: If the given parameters do not occur in the provided operator
            TypeError: If the operator has more than one circuit representing the quantum state
        """
        if isinstance(params, (ParameterVector, list)):
            param_grads = [self._parameter_shift(operator, param) for param in params]
            absent_params = [
                params[i] for i, grad_ops in enumerate(param_grads) if grad_ops is None
            ]
            if len(absent_params) > 0:
                raise ValueError(
                    "The following parameters do not appear in the provided operator: ",
                    absent_params,
                )
            return ListOp(absent_params)

        # By this point, it's only one parameter
        param = params

        if isinstance(operator, ListOp) and not isinstance(operator, ComposedOp):
            return_op = operator.traverse(partial(self._parameter_shift, params=param))

            # Remove any branch of the tree where the relevant parameter does not occur
            trimmed_oplist = [op for op in return_op.oplist if op is not None]
            # If all branches are None, remove the parent too
            if len(trimmed_oplist) == 0:
                return None
            # Rebuild the operator with the trimmed down oplist
            properties = {"coeff": return_op._coeff, "abelian": return_op._abelian}
            if return_op.__class__ == ListOp:
                properties["combo_fn"] = return_op.combo_fn
            return return_op.__class__(oplist=trimmed_oplist, **properties)

        else:
            circs = self.get_unique_circuits(operator)

            if len(circs) > 1:
                raise TypeError(
                    "Please define an operator with a single circuit representing "
                    "the quantum state."
                )
            if len(circs) == 0:
                return operator
            circ = circs[0]

            if self.analytic:
                # Unroll the circuit into a gate set for which the gradient may be computed
                # using pi/2 shifts.
                circ = ParamShift._transpile_to_supported_operations(circ, self.SUPPORTED_GATES)
                operator = ParamShift._replace_operator_circuit(operator, circ)

            if param not in circ._parameter_table:
                return ~Zero @ One

            shifted_ops = []
            summed_shifted_op = None

            iref_to_data_index = {id(inst.operation): idx for idx, inst in enumerate(circ.data)}

            for param_reference in circ._parameter_table[param]:
                original_gate, param_index = param_reference
                m = iref_to_data_index[id(original_gate)]

                pshift_op = deepcopy(operator)
                mshift_op = deepcopy(operator)

                # We need the circuit objects of the newly instantiated operators
                pshift_circ = self.get_unique_circuits(pshift_op)[0]
                mshift_circ = self.get_unique_circuits(mshift_op)[0]

                pshift_gate = pshift_circ.data[m].operation
                mshift_gate = mshift_circ.data[m].operation

                p_param = pshift_gate.params[param_index]
                m_param = mshift_gate.params[param_index]
                # For analytic gradients the circuit parameters are shifted once by +pi/2 and
                # once by -pi/2.
                if self.analytic:
                    shift_constant = 0.5
                    pshift_gate.params[param_index] = p_param + (np.pi / (4 * shift_constant))
                    mshift_gate.params[param_index] = m_param - (np.pi / (4 * shift_constant))
                # For finite difference gradients the circuit parameters are shifted once by
                # +epsilon and once by -epsilon.
                else:
                    shift_constant = 1.0 / (2 * self._epsilon)
                    pshift_gate.params[param_index] = p_param + self._epsilon
                    mshift_gate.params[param_index] = m_param - self._epsilon
                # The results of the shifted operators are now evaluated according the parameter
                # shift / finite difference formula.
                if isinstance(operator, ComposedOp):
                    shifted_op = shift_constant * (pshift_op - mshift_op)
                # If the operator represents a quantum state then we apply a special combo
                # function to evaluate probability gradients.
                elif isinstance(operator, StateFn):
                    shifted_op = ListOp(
                        [pshift_op, mshift_op],
                        combo_fn=partial(self._prob_combo_fn, shift_constant=shift_constant),
                    )
                else:
                    raise TypeError(
                        "Probability gradients are not supported for the given operator type"
                    )

                if isinstance(p_param, ParameterExpression) and not isinstance(p_param, Parameter):
                    expr_grad = _coeff_derivative(p_param, param)
                    shifted_op *= expr_grad
                if not summed_shifted_op:
                    summed_shifted_op = shifted_op
                else:
                    summed_shifted_op += shifted_op

            shifted_ops.append(summed_shifted_op)

            if not SummedOp(shifted_ops).reduce():
                return ~StateFn(Zero) @ One
            else:
                return SummedOp(shifted_ops).reduce()

    @staticmethod
    def _prob_combo_fn(
        x: Union[
            DictStateFn,
            VectorStateFn,
            SparseVectorStateFn,
            List[Union[DictStateFn, VectorStateFn, SparseVectorStateFn]],
        ],
        shift_constant: float,
    ) -> Union[Dict, np.ndarray]:
        """Implement the combo_fn used to evaluate probability gradients

        Args:
            x: Output of an operator evaluation
            shift_constant: Shifting constant factor needed for proper rescaling

        Returns:
            Array representing the probability gradients w.r.t. the given operator and parameters

        Raises:
            TypeError: if ``x`` is not DictStateFn, VectorStateFn or their list.

        """
        # Note: In the probability gradient case, the amplitudes still need to be converted
        # into sampling probabilities.

        def get_primitives(item):
            if isinstance(item, (DictStateFn, SparseVectorStateFn)):
                item = item.primitive
            if isinstance(item, VectorStateFn):
                item = item.primitive.data
            return item

        is_statefn = False
        if isinstance(x, list):
            # Check if all items in x are a StateFn items
            if all(isinstance(item, StateFn) for item in x):
                is_statefn = True
            items = [get_primitives(item) for item in x]
        else:
            # Check if x is a StateFn item
            if isinstance(x, StateFn):
                is_statefn = True
            items = [get_primitives(x)]
        if isinstance(items[0], dict):
            prob_dict: Dict[str, float] = {}
            for i, item in enumerate(items):
                for key, prob_counts in item.items():
                    prob_dict[key] = (
                        prob_dict.get(key, 0) + shift_constant * ((-1) ** i) * prob_counts
                    )
            return prob_dict
        elif isinstance(items[0], scipy.sparse.spmatrix):
            # If x was given as StateFn the state amplitudes need to be multiplied in order to
            # evaluate the sampling probabilities which are then subtracted according to the
            # parameter shift rule.
            if is_statefn:
                return shift_constant * np.subtract(
                    items[0].multiply(np.conj(items[0])), items[1].multiply(np.conj(items[1]))
                )
            # If x was not given as a StateFn the state amplitudes were already converted into
            # sampling probabilities which are then only subtracted according to the
            # parameter shift rule.
            else:
                return shift_constant * np.subtract(items[0], items[1])
        elif isinstance(items[0], Iterable):
            # If x was given as StateFn the state amplitudes need to be multiplied in order to
            # evaluate the sampling probabilities which are then subtracted according to the
            # parameter shift rule.
            if is_statefn:
                return shift_constant * np.subtract(
                    np.multiply(items[0], np.conj(items[0])),
                    np.multiply(items[1], np.conj(items[1])),
                )
            # If x was not given as a StateFn the state amplitudes were already converted into
            # sampling probabilities which are then only subtracted according to the
            # parameter shift rule.
            else:
                return shift_constant * np.subtract(items[0], items[1])
        raise TypeError(
            "Probability gradients can only be evaluated from VectorStateFs or DictStateFns."
        )

    @staticmethod
    def _replace_operator_circuit(operator: OperatorBase, circuit: QuantumCircuit) -> OperatorBase:
        """Replace a circuit element in an operator with a single element given as circuit

        Args:
            operator: Operator for which the circuit representing the quantum state shall be
                      replaced
            circuit: Circuit which shall replace the circuit in the given operator

        Returns:
            Operator with replaced circuit quantum state function

        """
        if isinstance(operator, CircuitStateFn):
            return CircuitStateFn(circuit, coeff=operator.coeff)
        elif isinstance(operator, CircuitOp):
            return CircuitOp(circuit, coeff=operator.coeff)
        elif isinstance(operator, (ComposedOp, ListOp)):
            return operator.traverse(partial(ParamShift._replace_operator_circuit, circuit=circuit))
        else:
            return operator

    @classmethod
    def get_unique_circuits(cls, operator: OperatorBase) -> List[QuantumCircuit]:
        """Traverse the operator and return all unique circuits

        Args:
            operator: An operator that potentially includes QuantumCircuits

        Returns:
            A list of all unique quantum circuits that appear in the operator

        """
        if isinstance(operator, CircuitStateFn):
            return [operator.primitive]

        def get_circuit(op):
            return op.primitive if isinstance(op, (CircuitStateFn, CircuitOp)) else None

        unrolled_op = cls.unroll_operator(operator)
        circuits = []
        for ops in unrolled_op:
            if not isinstance(ops, list):
                ops = [ops]
            for op in ops:
                if isinstance(op, (CircuitStateFn, CircuitOp, QuantumCircuit)):
                    c = get_circuit(op)
                    if c and c not in circuits:
                        circuits.append(c)
        return circuits

    @classmethod
    def unroll_operator(cls, operator: OperatorBase) -> Union[OperatorBase, List[OperatorBase]]:
        """Traverse the operator and return all OperatorBase objects flattened
           into a single list. This is used as a subroutine to extract all
           circuits within a large composite operator.

        Args:
            operator: An OperatorBase type object

        Returns:
            A single flattened list of all OperatorBase objects within the
            input operator

        """
        if isinstance(operator, ListOp):
            return [cls.unroll_operator(op) for op in operator]
        if hasattr(operator, "primitive") and isinstance(operator.primitive, ListOp):
            return [operator.__class__(op) for op in operator.primitive]
        return operator
