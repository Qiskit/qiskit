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

"""The module to compute the state gradient with the linear combination method."""

from collections.abc import Iterable
from copy import deepcopy
from functools import partial
from itertools import product
from typing import List, Optional, Tuple, Union, Callable

import scipy
import numpy as np

from qiskit.circuit import Gate, Instruction
from qiskit.circuit import (
    CircuitInstruction,
    QuantumCircuit,
    QuantumRegister,
    ParameterVector,
    ParameterExpression,
    Parameter,
)
from qiskit.circuit.parametertable import ParameterReferences, ParameterTable
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.library import SGate, SdgGate, XGate
from qiskit.circuit.library.standard_gates import (
    CXGate,
    CYGate,
    CZGate,
    IGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZXGate,
    RZZGate,
    PhaseGate,
    UGate,
    ZGate,
)
from qiskit.quantum_info import partial_trace
from qiskit.utils.deprecation import deprecate_func
from ...operator_base import OperatorBase
from ...list_ops.list_op import ListOp
from ...list_ops.composed_op import ComposedOp
from ...list_ops.summed_op import SummedOp
from ...operator_globals import Z, I, Y, One, Zero
from ...primitive_ops.primitive_op import PrimitiveOp
from ...state_fns.state_fn import StateFn
from ...state_fns.circuit_state_fn import CircuitStateFn
from ...state_fns.dict_state_fn import DictStateFn
from ...state_fns.vector_state_fn import VectorStateFn
from ...state_fns.sparse_vector_state_fn import SparseVectorStateFn
from ...exceptions import OpflowError
from .circuit_gradient import CircuitGradient
from ...converters import PauliBasisChange


class LinComb(CircuitGradient):
    """Deprecated: Compute the state gradient d⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω respectively the gradients of the
    sampling probabilities of the basis states of
    a state |ψ(ω)〉w.r.t. ω.
    This method employs a linear combination of unitaries,
    see e.g. https://arxiv.org/pdf/1811.11184.pdf
    """

    SUPPORTED_GATES = {
        "rx",
        "ry",
        "rz",
        "rzx",
        "rzz",
        "ryy",
        "rxx",
        "p",
        "u",
        "controlledgate",
        "cx",
        "cy",
        "cz",
        "ccx",
        "swap",
        "iswap",
        "t",
        "s",
        "sdg",
        "x",
        "y",
        "z",
    }

    # pylint: disable=signature-differs, arguments-differ
    @deprecate_func(
        since="0.24.0",
        package_name="qiskit-terra",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(self, aux_meas_op: OperatorBase = Z):
        """
        Args:
            aux_meas_op: The operator that the auxiliary qubit is measured with respect to.
                For ``aux_meas_op = Z`` we compute 2Re[(dω⟨ψ(ω)|)O(θ)|ψ(ω)〉],
                for ``aux_meas_op = -Y`` we compute 2Im[(dω⟨ψ(ω)|)O(θ)|ψ(ω)〉], and
                for ``aux_meas_op = Z - 1j * Y`` we compute 2(dω⟨ψ(ω)|)O(θ)|ψ(ω)〉.
        Raises:
            ValueError: If the provided auxiliary measurement operator is not supported.
        """
        super().__init__()
        if aux_meas_op not in [Z, -Y, (Z - 1j * Y)]:
            raise ValueError(
                "This auxiliary measurement operator is currently not supported. Please choose "
                "either Z, -Y, or Z - 1j * Y. "
            )
        self._aux_meas_op = aux_meas_op

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
        """Convert ``operator`` into an operator that represents the gradient w.r.t. ``params``.

        Args:
            operator: The operator we are taking the gradient of: ⟨ψ(ω)|O(θ)|ψ(ω)〉
            params: The parameters we are taking the gradient wrt: ω
                    If a ParameterExpression, ParameterVector or List[ParameterExpression] is given,
                    then the 1st order derivative of the operator is calculated.
                    If a Tuple[ParameterExpression, ParameterExpression] or
                    List[Tuple[ParameterExpression, ParameterExpression]]
                    is given, then the 2nd order derivative of the operator is calculated.
        Returns:
            An operator corresponding to the gradient resp. Hessian. The order is in accordance with
            the order of the given parameters.
        """
        return self._prepare_operator(operator, params)

    # pylint: disable=too-many-return-statements
    def _prepare_operator(
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
        """Traverse ``operator`` to get back the adapted operator representing the gradient.

        Args:
            operator: The operator we are taking the gradient of: ⟨ψ(ω)|O(θ)|ψ(ω)〉.
            params: The parameters we are taking the gradient wrt: ω.
                If a ``ParameterExpression```, ``ParameterVector`` or ``List[ParameterExpression]``
                is given, then the 1st order derivative of the operator is calculated.
                If a ``Tuple[ParameterExpression, ParameterExpression]`` or
                ``List[Tuple[ParameterExpression, ParameterExpression]]``
                is given, then the 2nd order derivative of the operator is calculated.
        Returns:
            The adapted operator.
            Measurement operators are attached with an additional Z term acting
            on an additional working qubit.
            Quantum states - which must be given as circuits - are adapted. An additional
            working qubit controls intercepting gates.
            See e.g. [1].

        Raises:
            ValueError: If ``operator`` does not correspond to an expectation value.
            TypeError: If the ``StateFn`` corresponding to the quantum state could not be extracted
                       from ``operator``.
            OpflowError: If third or higher order gradients are requested.

        References:
            [1]: Evaluating analytic gradients on quantum hardware
                 Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, and Nathan Killoran
                 Phys. Rev. A 99, 032331 – Published 21 March 2019

        """

        if isinstance(operator, ComposedOp):
            # Get the measurement and the state operator
            if not isinstance(operator[0], StateFn) or not operator[0].is_measurement:
                raise ValueError("The given operator does not correspond to an expectation value")
            if not isinstance(operator[-1], StateFn) or operator[-1].is_measurement:
                raise ValueError("The given operator does not correspond to an expectation value")
            if operator[0].is_measurement:
                meas = deepcopy(operator.oplist[0])
                meas = meas.primitive * meas.coeff
                if len(operator.oplist) == 2:
                    state_op = operator[1]
                    if not isinstance(state_op, StateFn):
                        raise TypeError(
                            "The StateFn representing the quantum state could not be extracted."
                        )
                    if isinstance(params, (ParameterExpression, ParameterVector)) or (
                        isinstance(params, list)
                        and all(isinstance(param, ParameterExpression) for param in params)
                    ):

                        return self._gradient_states(
                            state_op,
                            meas_op=(2 * meas),
                            target_params=params,
                        )
                    elif isinstance(params, tuple) or (
                        isinstance(params, list)
                        and all(isinstance(param, tuple) for param in params)
                    ):
                        return self._hessian_states(
                            state_op,
                            meas_op=(4 * (I ^ meas)),
                            target_params=params,
                        )  # type: ignore
                    else:
                        raise OpflowError(
                            "The linear combination gradient does only support the "
                            "computation of 1st gradients and 2nd order gradients."
                        )
                else:
                    state_op = deepcopy(operator)
                    state_op.oplist.pop(0)
                    if not isinstance(state_op, StateFn):
                        raise TypeError(
                            "The StateFn representing the quantum state could not be extracted."
                        )

                    if isinstance(params, (ParameterExpression, ParameterVector)) or (
                        isinstance(params, list)
                        and all(isinstance(param, ParameterExpression) for param in params)
                    ):
                        return state_op.traverse(
                            partial(
                                self._gradient_states,
                                meas_op=(2 * meas),
                                target_params=params,
                            )
                        )
                    elif isinstance(params, tuple) or (
                        isinstance(params, list)
                        and all(isinstance(param, tuple) for param in params)
                    ):
                        return state_op.traverse(
                            partial(
                                self._hessian_states,
                                meas_op=(4 * I ^ meas),
                                target_params=params,
                            )
                        )

                    raise OpflowError(
                        "The linear combination gradient only supports the "
                        "computation of 1st and 2nd order gradients."
                    )
            else:
                return operator.traverse(partial(self._prepare_operator, params=params))
        elif isinstance(operator, ListOp):
            return operator.traverse(partial(self._prepare_operator, params=params))
        elif isinstance(operator, StateFn):
            if operator.is_measurement:
                return operator.traverse(partial(self._prepare_operator, params=params))
            else:
                if isinstance(params, (ParameterExpression, ParameterVector)) or (
                    isinstance(params, list)
                    and all(isinstance(param, ParameterExpression) for param in params)
                ):
                    return self._gradient_states(operator, target_params=params)
                elif isinstance(params, tuple) or (
                    isinstance(params, list) and all(isinstance(param, tuple) for param in params)
                ):
                    return self._hessian_states(operator, target_params=params)  # type: ignore
                else:
                    raise OpflowError(
                        "The linear combination gradient does only support the computation "
                        "of 1st gradients and 2nd order gradients."
                    )
        elif isinstance(operator, PrimitiveOp):
            return operator
        return operator

    @staticmethod
    def _grad_combo_fn(x, state_op):
        def get_result(item):
            if isinstance(item, (DictStateFn, SparseVectorStateFn)):
                item = item.primitive
            if isinstance(item, VectorStateFn):
                item = item.primitive.data

            if isinstance(item, dict):
                prob_dict = {}
                for key, val in item.items():
                    prob_counts = val * np.conj(val)
                    if int(key[0]) == 1:
                        prob_counts *= -1
                    suffix = key[1:]
                    prob_dict[suffix] = prob_dict.get(suffix, 0) + prob_counts
                for key in prob_dict:
                    prob_dict[key] *= 2
                return prob_dict
            elif isinstance(item, scipy.sparse.spmatrix):
                # Generate the operator which computes the linear combination
                trace = _z_exp(item)
                return trace
            elif isinstance(item, Iterable):
                # Generate the operator which computes the linear combination
                lin_comb_op = 2 * Z ^ (I ^ state_op.num_qubits)
                lin_comb_op = lin_comb_op.to_matrix()
                outer = np.outer(item, item.conj())
                return list(
                    np.diag(partial_trace(lin_comb_op.dot(outer), [state_op.num_qubits]).data)
                )
            else:
                raise TypeError(
                    "The state result should be either a DictStateFn or a VectorStateFn."
                )

        if not isinstance(x, Iterable):
            return get_result(x)
        elif len(x) == 1:
            return get_result(x[0])
        else:
            result = []
            for item in x:
                result.append(get_result(item))
            return result

    @staticmethod
    def _hess_combo_fn(x, state_op):
        def get_result(item):
            if isinstance(item, DictStateFn):
                item = item.primitive
            if isinstance(item, VectorStateFn):
                item = item.primitive.data
            if isinstance(item, Iterable):
                # Generate the operator which computes the linear combination
                lin_comb_op = 4 * (I ^ (state_op.num_qubits + 1)) ^ Z
                lin_comb_op = lin_comb_op.to_matrix()
                return list(
                    np.diag(
                        partial_trace(lin_comb_op.dot(np.outer(item, np.conj(item))), [0, 1]).data
                    )
                )
            elif isinstance(item, scipy.sparse.spmatrix):
                # Generate the operator which computes the linear combination
                trace = _z_exp(item)
                return trace
            elif isinstance(item, dict):
                prob_dict = {}
                for key, val in item.values():
                    prob_counts = val * np.conj(val)
                    if int(key[-1]) == 1:
                        prob_counts *= -1
                    prefix = key[:-2]
                    prob_dict[prefix] = prob_dict.get(prefix, 0) + prob_counts
                for key in prob_dict:
                    prob_dict[key] *= 4
                return prob_dict
            else:
                raise TypeError(
                    "The state result should be either a DictStateFn or a VectorStateFn."
                )

        if not isinstance(x, Iterable):
            return get_result(x)
        elif len(x) == 1:
            return get_result(x[0])
        else:
            result = []
            for item in x:
                result.append(get_result(item))
            return result

    @staticmethod
    def _gate_gradient_dict(gate: Gate) -> List[Tuple[List[complex], List[Instruction]]]:
        r"""Given a parameterized gate U(theta) with derivative
        dU(theta)/dtheta = sum_ia_iU(theta)V_i.
        This function returns a:=[a_0, ...] and V=[V_0, ...]
        Suppose U takes multiple parameters, i.e., U(theta^0, ... theta^k).
        The returned coefficients and gates are ordered accordingly.
        Only parameterized Qiskit gates are supported.

        Args:
            gate: The gate for which the derivative is being computed.

        Returns:
            The coefficients and the gates used for the metric computation for each parameter of
            the respective gates ``[([a^0], [V^0]) ..., ([a^k], [V^k])]``.

        Raises:
            OpflowError: If the input gate is controlled by another state but '|1>^{\otimes k}'
            TypeError: If the input gate is not a supported parameterized gate.
        """

        # pylint: disable=too-many-return-statements
        if isinstance(gate, PhaseGate):
            # theta
            return [([0.5j, -0.5j], [IGate(), CZGate()])]
        if isinstance(gate, UGate):
            # theta, lambda, phi
            return [([-0.5j], [CZGate()]), ([-0.5j], [CZGate()]), ([-0.5j], [CZGate()])]
        if isinstance(gate, RXGate):
            # theta
            return [([-0.5j], [CXGate()])]
        if isinstance(gate, RYGate):
            # theta
            return [([-0.5j], [CYGate()])]
        if isinstance(gate, RZGate):
            # theta
            return [([-0.5j], [CZGate()])]
        if isinstance(gate, RXXGate):
            # theta
            cxx_circ = QuantumCircuit(3)
            cxx_circ.cx(0, 1)
            cxx_circ.cx(0, 2)
            cxx = cxx_circ.to_instruction()
            return [([-0.5j], [cxx])]
        if isinstance(gate, RYYGate):
            # theta
            cyy_circ = QuantumCircuit(3)
            cyy_circ.cy(0, 1)
            cyy_circ.cy(0, 2)
            cyy = cyy_circ.to_instruction()
            return [([-0.5j], [cyy])]
        if isinstance(gate, RZZGate):
            # theta
            czz_circ = QuantumCircuit(3)
            czz_circ.cz(0, 1)
            czz_circ.cz(0, 2)
            czz = czz_circ.to_instruction()
            return [([-0.5j], [czz])]
        if isinstance(gate, RZXGate):
            # theta
            czx_circ = QuantumCircuit(3)
            czx_circ.cx(0, 2)
            czx_circ.cz(0, 1)
            czx = czx_circ.to_instruction()
            return [([-0.5j], [czx])]
        if isinstance(gate, ControlledGate):
            # TODO support arbitrary control states
            if gate.ctrl_state != 2**gate.num_ctrl_qubits - 1:
                raise OpflowError(
                    "Function only support controlled gates with control state `1` on all control "
                    "qubits."
                )

            base_coeffs_gates = LinComb._gate_gradient_dict(gate.base_gate)
            coeffs_gates = []
            # The projectors needed for the gradient of a controlled gate are integrated by a sum
            # of gates.
            # The following line generates the decomposition gates.

            proj_gates_controlled = [
                [(-1) ** p.count(ZGate()), p]
                for p in product([IGate(), ZGate()], repeat=gate.num_ctrl_qubits)
            ]
            for base_coeffs, base_gates in base_coeffs_gates:  # loop over parameters
                coeffs = []
                gates = []
                for phase, proj_gates in proj_gates_controlled:
                    coeffs.extend([phase * c / (2**gate.num_ctrl_qubits) for c in base_coeffs])
                    for base_gate in base_gates:
                        controlled_circ = QuantumCircuit(gate.num_ctrl_qubits + gate.num_qubits)
                        for i, proj_gate in enumerate(proj_gates):
                            if isinstance(proj_gate, ZGate):
                                controlled_circ.cz(0, i + 1)
                        if not isinstance(base_gate, IGate):
                            controlled_circ.append(
                                base_gate,
                                [
                                    0,
                                    range(
                                        gate.num_ctrl_qubits + 1,
                                        gate.num_ctrl_qubits + gate.num_qubits,
                                    ),
                                ],
                            )
                        gates.append(controlled_circ.to_instruction())
                c_g = (coeffs, gates)
                coeffs_gates.append(c_g)
            return coeffs_gates
        raise TypeError(f"Unrecognized parameterized gate, {gate}")

    @staticmethod
    def apply_grad_gate(
        circuit,
        gate,
        param_index,
        grad_gate,
        grad_coeff,
        qr_superpos,
        open_ctrl=False,
        trim_after_grad_gate=False,
    ):
        """Util function to apply a gradient gate for the linear combination of unitaries method.
        Replaces the ``gate`` instance in ``circuit`` with ``grad_gate`` using ``qr_superpos`` as
        superposition qubit. Also adds the appropriate sign-fix gates on the superposition qubit.

        Args:
            circuit (QuantumCircuit): The circuit in which to do the replacements.
            gate (Gate): The gate instance to replace.
            param_index (int): The index of the parameter in ``gate``.
            grad_gate (Gate): A controlled gate encoding the gradient of ``gate``.
            grad_coeff (float): A coefficient to the gradient component. Might not be one if the
                gradient contains multiple summed terms.
            qr_superpos (QuantumRegister): A ``QuantumRegister`` of size 1 contained in ``circuit``
                that is used as control for ``grad_gate``.
            open_ctrl (bool): If True use an open control for ``grad_gate`` instead of closed.
            trim_after_grad_gate (bool): If True remove all gates after the ``grad_gate``. Can
                be used to reduce the circuit depth in e.g. computing an overlap of gradients.

        Returns:
            QuantumCircuit: A copy of the original circuit with the gradient gate added.

        Raises:
            RuntimeError: If ``gate`` is not in ``circuit``.
        """
        qr_superpos_qubits = tuple(qr_superpos)
        # copy the input circuit taking the gates by reference
        out = QuantumCircuit(*circuit.qregs)
        out._data = circuit._data.copy()
        out._parameter_table = ParameterTable(
            {param: values.copy() for param, values in circuit._parameter_table.items()}
        )

        # get the data index and qubits of the target gate  TODO use built-in
        gate_idx, gate_qubits = None, None
        for i, instruction in enumerate(out._data):
            if instruction.operation is gate:
                gate_idx, gate_qubits = i, instruction.qubits
                break
        if gate_idx is None:
            raise RuntimeError("The specified gate could not be found in the circuit data.")

        # initialize replacement instructions
        replacement = []

        # insert the phase fix before the target gate better documentation
        sign = np.sign(grad_coeff)
        is_complex = np.iscomplex(grad_coeff)

        if sign < 0 and is_complex:
            replacement.append(CircuitInstruction(SdgGate(), qr_superpos_qubits, ()))
        elif sign < 0:
            replacement.append(CircuitInstruction(ZGate(), qr_superpos_qubits, ()))
        elif is_complex:
            replacement.append(CircuitInstruction(SGate(), qr_superpos_qubits, ()))
        # else no additional gate required

        # open control if specified
        if open_ctrl:
            replacement += [CircuitInstruction(XGate(), qr_superpos_qubits, [])]

        # compute the replacement
        if isinstance(gate, UGate) and param_index == 0:
            theta = gate.params[2]
            rz_plus, rz_minus = RZGate(theta), RZGate(-theta)
            replacement += [CircuitInstruction(rz_plus, (qubit,), ()) for qubit in gate_qubits]
            replacement += [
                CircuitInstruction(RXGate(np.pi / 2), (qubit,), ()) for qubit in gate_qubits
            ]
            replacement.append(CircuitInstruction(grad_gate, qr_superpos_qubits + gate_qubits, []))
            replacement += [
                CircuitInstruction(RXGate(-np.pi / 2), (qubit,), ()) for qubit in gate_qubits
            ]
            replacement += [CircuitInstruction(rz_minus, (qubit,), ()) for qubit in gate_qubits]

            # update parametertable if necessary
            if isinstance(theta, ParameterExpression):
                # This dangerously subverts ParameterTable by abusing the fact that binding will
                # mutate the exact instruction instance, and relies on all instances of `rz_plus`
                # that were added before being the same in memory, which QuantumCircuit usually
                # ensures is not the case.  I'm leaving this as close to its previous form as
                # possible, to avoid introducing further complications, but this whole method
                # accesses internal attributes of `QuantumCircuit` and needs rewriting.
                # - Jake Lishman, 2022-03-02.
                out._update_parameter_table(CircuitInstruction(rz_plus, (gate_qubits[0],), ()))
                out._update_parameter_table(CircuitInstruction(rz_minus, (gate_qubits[0],), ()))

            if open_ctrl:
                replacement.append(CircuitInstruction(XGate(), qr_superpos_qubits, ()))

            if not trim_after_grad_gate:
                replacement.append(CircuitInstruction(gate, gate_qubits, ()))

        elif isinstance(gate, UGate) and param_index == 1:
            # gradient gate is applied after the original gate in this case
            replacement.append(CircuitInstruction(gate, gate_qubits, ()))
            replacement.append(CircuitInstruction(grad_gate, qr_superpos_qubits + gate_qubits, ()))
            if open_ctrl:
                replacement.append(CircuitInstruction(XGate(), qr_superpos_qubits, ()))

        else:
            replacement.append(CircuitInstruction(grad_gate, qr_superpos_qubits + gate_qubits, ()))
            if open_ctrl:
                replacement.append(CircuitInstruction(XGate(), qr_superpos_qubits, ()))
            if not trim_after_grad_gate:
                replacement.append(CircuitInstruction(gate, gate_qubits, ()))

        # replace the parameter we compute the derivative of with the replacement
        # TODO can this be done more efficiently?
        if trim_after_grad_gate:  # remove everything after the gradient gate
            out._data[gate_idx:] = replacement
            # reset parameter table
            table = ParameterTable()
            for instruction in out._data:
                for idx, param_expression in enumerate(instruction.operation.params):
                    if isinstance(param_expression, ParameterExpression):
                        for param in param_expression.parameters:
                            if param not in table.keys():
                                table[param] = ParameterReferences(((instruction.operation, idx),))
                            else:
                                table[param].add((instruction.operation, idx))

            out._parameter_table = table

        else:
            out._data[gate_idx : gate_idx + 1] = replacement

        return out

    def _aux_meas_basis_trafo(
        self, aux_meas_op: OperatorBase, state: StateFn, state_op: StateFn, combo_fn: Callable
    ) -> ListOp:
        """
        This function applies the necessary basis transformation to measure the quantum state in
        a different basis -- given by the auxiliary measurement operator ``aux_meas_op``.

        Args:
            aux_meas_op: The auxiliary measurement operator defines the necessary measurement basis.
            state: This operator represents the gradient or Hessian before the basis transformation.
            state_op: The operator representing the quantum state for which we compute the gradient
                or Hessian.
            combo_fn: This ``combo_fn`` defines whether the target is a gradient or Hessian.


        Returns:
            Operator representing the gradient or Hessian.

        Raises:
            ValueError: If ``aux_meas_op`` is neither ``Z`` nor ``-Y`` nor ``Z - 1j * Y``.

        """
        if aux_meas_op == Z - 1j * Y:
            state_z = ListOp(
                [state],
                combo_fn=partial(combo_fn, state_op=state_op),
            )
            pbc = PauliBasisChange(replacement_fn=PauliBasisChange.measurement_replacement_fn)
            pbc = pbc.convert(-Y ^ (I ^ (state.num_qubits - 1)))
            state_y = pbc[-1] @ state
            state_y = ListOp(
                [state_y],
                combo_fn=partial(combo_fn, state_op=state_op),
            )
            return state_z - 1j * state_y

        elif aux_meas_op == -Y:
            pbc = PauliBasisChange(replacement_fn=PauliBasisChange.measurement_replacement_fn)
            pbc = pbc.convert(aux_meas_op ^ (I ^ (state.num_qubits - 1)))
            state = pbc[-1] @ state
            return -1 * ListOp(
                [state],
                combo_fn=partial(combo_fn, state_op=state_op),
            )
        elif aux_meas_op == Z:
            return ListOp(
                [state],
                combo_fn=partial(combo_fn, state_op=state_op),
            )
        else:
            raise ValueError(
                f"The auxiliary measurement operator passed {aux_meas_op} is not supported. "
                "Only Y, Z, or Z - 1j * Y are valid."
            )

    def _gradient_states(
        self,
        state_op: StateFn,
        meas_op: Optional[OperatorBase] = None,
        target_params: Optional[Union[Parameter, List[Parameter]]] = None,
        open_ctrl: bool = False,
        trim_after_grad_gate: bool = False,
    ) -> ListOp:
        """Generate the gradient states.

        Args:
            state_op: The operator representing the quantum state for which we compute the gradient.
            meas_op: The operator representing the observable for which we compute the gradient.
            target_params: The parameters we are taking the gradient wrt: ω
            open_ctrl: If True use an open control for ``grad_gate`` instead of closed.
            trim_after_grad_gate: If True remove all gates after the ``grad_gate``. Can
                be used to reduce the circuit depth in e.g. computing an overlap of gradients.

        Returns:
            ListOp of StateFns as quantum circuits which are the states w.r.t. which we compute the
            gradient. If a parameter appears multiple times, one circuit is created per
            parameterized gates to compute the product rule.

        Raises:
            QiskitError: If one of the circuits could not be constructed.
            TypeError: If the operators is of unsupported type.
            ValueError: If the auxiliary operator preparation fails.
        """
        # unroll separately from the H gate since we need the H gate to be the first
        # operation in the data attributes of the circuit
        unrolled = self._transpile_to_supported_operations(state_op.primitive, self.SUPPORTED_GATES)
        qr_superpos = QuantumRegister(1)
        state_qc = QuantumCircuit(*state_op.primitive.qregs, qr_superpos)
        state_qc.h(qr_superpos)

        state_qc.compose(unrolled, inplace=True)

        # Define the working qubit to realize the linear combination of unitaries
        if not isinstance(target_params, (list, np.ndarray)):
            target_params = [target_params]

        oplist = []
        for param in target_params:
            if param not in state_qc.parameters:
                oplist += [~Zero @ One]
            else:
                param_gates = state_qc._parameter_table[param]
                sub_oplist = []
                for gate, idx in param_gates:
                    grad_coeffs, grad_gates = self._gate_gradient_dict(gate)[idx]

                    # construct the states
                    for grad_coeff, grad_gate in zip(grad_coeffs, grad_gates):
                        grad_circuit = self.apply_grad_gate(
                            state_qc,
                            gate,
                            idx,
                            grad_gate,
                            grad_coeff,
                            qr_superpos,
                            open_ctrl,
                            trim_after_grad_gate,
                        )
                        # apply final Hadamard on superposition qubit
                        grad_circuit.h(qr_superpos)

                        # compute the correct coefficient and append to list of circuits
                        coeff = np.sqrt(np.abs(grad_coeff)) * state_op.coeff
                        state = CircuitStateFn(grad_circuit, coeff=coeff)

                        # apply the chain rule if the parameter expression if required
                        param_expression = gate.params[idx]

                        if isinstance(meas_op, OperatorBase):
                            state = (
                                StateFn(self._aux_meas_op ^ meas_op, is_measurement=True) @ state
                            )

                        else:
                            state = self._aux_meas_basis_trafo(
                                self._aux_meas_op, state, state_op, self._grad_combo_fn
                            )

                        if param_expression != param:  # parameter is not identity, apply chain rule
                            param_grad = param_expression.gradient(param)
                            state *= param_grad

                        sub_oplist += [state]

                oplist += [SummedOp(sub_oplist) if len(sub_oplist) > 1 else sub_oplist[0]]

        return ListOp(oplist) if len(oplist) > 1 else oplist[0]

    def _hessian_states(
        self,
        state_op: StateFn,
        meas_op: Optional[OperatorBase] = None,
        target_params: Optional[
            Union[
                Tuple[ParameterExpression, ParameterExpression],
                List[Tuple[ParameterExpression, ParameterExpression]],
            ]
        ] = None,
    ) -> OperatorBase:
        """Generate the operator states whose evaluation returns the Hessian (items).

        Args:
            state_op: The operator representing the quantum state for which we compute the Hessian.
            meas_op: The operator representing the observable for which we compute the gradient.
            target_params: The parameters we are computing the Hessian wrt: ω

        Returns:
            Operators which give the Hessian. If a parameter appears multiple times, one circuit is
            created per parameterized gates to compute the product rule.

        Raises:
            QiskitError: If one of the circuits could not be constructed.
            TypeError: If ``operator`` is of unsupported type.
            ValueError: If the auxiliary operator preparation fails.
        """
        if not isinstance(target_params, list):
            target_params = [target_params]

        if not all(isinstance(params, tuple) for params in target_params):
            raise TypeError(
                "Please define in the parameters for which the Hessian is evaluated "
                "either as parameter tuple or a list of parameter tuples"
            )

        # create circuit with two additional qubits
        qr_add0 = QuantumRegister(1, "s0")
        qr_add1 = QuantumRegister(1, "s1")
        state_qc = QuantumCircuit(*state_op.primitive.qregs, qr_add0, qr_add1)

        # add Hadamards
        state_qc.h(qr_add0)
        state_qc.h(qr_add1)

        # compose with the original circuit
        state_qc.compose(state_op.primitive, inplace=True)

        # create a copy of the original circuit with an additional working qubit register
        oplist = []
        for param_a, param_b in target_params:
            if param_a not in state_qc.parameters or param_b not in state_qc.parameters:
                oplist += [~Zero @ One]
            else:
                sub_oplist = []
                param_gates_a = state_qc._parameter_table[param_a]
                param_gates_b = state_qc._parameter_table[param_b]
                for gate_a, idx_a in param_gates_a:
                    grad_coeffs_a, grad_gates_a = self._gate_gradient_dict(gate_a)[idx_a]

                    for grad_coeff_a, grad_gate_a in zip(grad_coeffs_a, grad_gates_a):
                        grad_circuit = self.apply_grad_gate(
                            state_qc, gate_a, idx_a, grad_gate_a, grad_coeff_a, qr_add0
                        )

                        for gate_b, idx_b in param_gates_b:
                            grad_coeffs_b, grad_gates_b = self._gate_gradient_dict(gate_b)[idx_b]

                            for grad_coeff_b, grad_gate_b in zip(grad_coeffs_b, grad_gates_b):
                                hessian_circuit = self.apply_grad_gate(
                                    grad_circuit, gate_b, idx_b, grad_gate_b, grad_coeff_b, qr_add1
                                )

                                # final Hadamards and CZ
                                hessian_circuit.h(qr_add0)
                                hessian_circuit.cz(qr_add1[0], qr_add0[0])
                                hessian_circuit.h(qr_add1)

                                coeff = state_op.coeff
                                coeff *= np.sqrt(np.abs(grad_coeff_a) * np.abs(grad_coeff_b))
                                state = CircuitStateFn(hessian_circuit, coeff=coeff)

                                if meas_op is not None:
                                    state = (
                                        StateFn(self._aux_meas_op ^ meas_op, is_measurement=True)
                                        @ state
                                    )
                                else:
                                    state = self._aux_meas_basis_trafo(
                                        self._aux_meas_op, state, state_op, self._hess_combo_fn
                                    )

                                # Chain Rule Parameter Expression
                                param_grad = 1
                                for gate, idx, param in zip(
                                    [gate_a, gate_b], [idx_a, idx_b], [param_a, param_b]
                                ):
                                    param_expression = gate.params[idx]
                                    if param_expression != param:  # need to apply chain rule
                                        param_grad *= param_expression.gradient(param)

                                if param_grad != 1:
                                    state *= param_grad

                                sub_oplist += [state]

                oplist += [SummedOp(sub_oplist) if len(sub_oplist) > 1 else sub_oplist[0]]

        return ListOp(oplist) if len(oplist) > 1 else oplist[0]


def _z_exp(spmatrix):
    """Compute the sampling probabilities of the qubits after applying measurement on the
    auxiliary qubit."""

    dok = spmatrix.todok()
    num_qubits = int(np.log2(dok.shape[1]))
    exp = scipy.sparse.dok_matrix((1, 2 ** (num_qubits - 1)))

    for index, amplitude in dok.items():
        binary = bin(index[1])[2:].zfill(num_qubits)
        sign = -1 if binary[0] == "1" else 1
        new_index = int(binary[1:], 2)
        exp[(0, new_index)] = exp[(0, new_index)] + 2 * sign * np.abs(amplitude) ** 2

    return exp
