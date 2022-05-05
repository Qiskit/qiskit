# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The module for Quantum the Fisher Information."""

from typing import List, Union

import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterVector, ParameterExpression
from qiskit.utils.arithmetic import triu_to_dense

from ...operator_base import OperatorBase
from ...list_ops.list_op import ListOp
from ...list_ops.summed_op import SummedOp
from ...operator_globals import I, Z, Y
from ...state_fns.state_fn import StateFn
from ...state_fns.circuit_state_fn import CircuitStateFn
from ..circuit_gradients.lin_comb import LinComb
from .circuit_qfi import CircuitQFI


class LinCombFull(CircuitQFI):
    r"""Compute the full Quantum Fisher Information (QFI).

    Given a pure, parameterized quantum state this class uses the linear combination of unitaries
    See also :class:`~qiskit.opflow.QFI`.
    """

    # pylint: disable=signature-differs, arguments-differ
    def __init__(
        self,
        aux_meas_op: OperatorBase = Z,
        phase_fix: bool = True,
    ):
        """
        Args:
            aux_meas_op: The operator that the auxiliary qubit is measured with respect to.
                For ``aux_meas_op = Z`` we compute 4Re[(dω⟨ψ(ω)|)O(θ)|ψ(ω)〉],
                for ``aux_meas_op = -Y`` we compute 4Im[(dω⟨ψ(ω)|)O(θ)|ψ(ω)〉], and
                for ``aux_meas_op = Z - 1j * Y`` we compute 4(dω⟨ψ(ω)|)O(θ)|ψ(ω)〉.
            phase_fix: Whether or not to compute and add the additional phase fix term
                Re[(dω⟨<ψ(ω)|)|ψ(ω)><ψ(ω)|(dω|ψ(ω))>].
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
        self._phase_fix = phase_fix

    def convert(
        self,
        operator: CircuitStateFn,
        params: Union[ParameterExpression, ParameterVector, List[ParameterExpression]],
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
            TypeError: If ``operator`` is an unsupported type.
        """
        # QFI & phase fix observable
        qfi_observable = StateFn(
            4 * self._aux_meas_op ^ (I ^ operator.num_qubits), is_measurement=True
        )

        # Check if the given operator corresponds to a quantum state given as a circuit.
        if not isinstance(operator, CircuitStateFn):
            raise TypeError(
                "LinCombFull is only compatible with states that are given as "
                f"CircuitStateFn, not {type(operator)}"
            )

        # If a single parameter is given wrap it into a list.
        if isinstance(params, ParameterExpression):
            params = [params]
        elif isinstance(params, ParameterVector):
            params = params[:]  # unroll to list

        if self._phase_fix:
            # First, the operators are computed which can compensate for a potential phase-mismatch
            # between target and trained state, i.e.〈ψ|∂lψ〉
            phase_fix_observable = I ^ operator.num_qubits
            gradient_states = LinComb(aux_meas_op=(Z - 1j * Y))._gradient_states(
                operator,
                meas_op=phase_fix_observable,
                target_params=params,
                open_ctrl=False,
                trim_after_grad_gate=True,
            )

            # pylint: disable=unidiomatic-typecheck
            if type(gradient_states) == ListOp:
                phase_fix_states = gradient_states.oplist
            else:
                phase_fix_states = [gradient_states]

        # Get  4 * Re[〈∂kψ|∂lψ]
        qfi_operators = []
        # Add a working qubit
        qr_work = QuantumRegister(1, "work_qubit")
        state_qc = QuantumCircuit(*operator.primitive.qregs, qr_work)
        state_qc.h(qr_work)
        # unroll separately from the H gate since we need the H gate to be the first
        # operation in the data attributes of the circuit
        unrolled = LinComb._transpile_to_supported_operations(
            operator.primitive, LinComb.SUPPORTED_GATES
        )
        state_qc.compose(unrolled, inplace=True)

        # Get the circuits needed to compute〈∂iψ|∂jψ〉
        for i, param_i in enumerate(params):  # loop over parameters
            qfi_ops = []
            for j, param_j in enumerate(params[i:], i):
                # Get the gates of the quantum state which are parameterized by param_i
                qfi_op = []
                param_gates_i = state_qc._parameter_table[param_i]
                for gate_i, idx_i in param_gates_i:
                    grad_coeffs_i, grad_gates_i = LinComb._gate_gradient_dict(gate_i)[idx_i]

                    # get the location of gate_i, used for trimming
                    location_i = None
                    for idx, (op, _, _) in enumerate(state_qc._data):
                        if op is gate_i:
                            location_i = idx
                            break

                    for grad_coeff_i, grad_gate_i in zip(grad_coeffs_i, grad_gates_i):

                        # Get the gates of the quantum state which are parameterized by param_j
                        param_gates_j = state_qc._parameter_table[param_j]
                        for gate_j, idx_j in param_gates_j:
                            grad_coeffs_j, grad_gates_j = LinComb._gate_gradient_dict(gate_j)[idx_j]

                            # get the location of gate_j, used for trimming
                            location_j = None
                            for idx, (op, _, _) in enumerate(state_qc._data):
                                if op is gate_j:
                                    location_j = idx
                                    break

                            for grad_coeff_j, grad_gate_j in zip(grad_coeffs_j, grad_gates_j):

                                grad_coeff_ij = np.conj(grad_coeff_i) * grad_coeff_j
                                qfi_circuit = LinComb.apply_grad_gate(
                                    state_qc,
                                    gate_i,
                                    idx_i,
                                    grad_gate_i,
                                    grad_coeff_ij,
                                    qr_work,
                                    open_ctrl=True,
                                    trim_after_grad_gate=(location_j < location_i),
                                )

                                # create a copy of the original circuit with the same registers
                                qfi_circuit = LinComb.apply_grad_gate(
                                    qfi_circuit,
                                    gate_j,
                                    idx_j,
                                    grad_gate_j,
                                    1,
                                    qr_work,
                                    open_ctrl=False,
                                    trim_after_grad_gate=(location_j >= location_i),
                                )

                                qfi_circuit.h(qr_work)
                                # Convert the quantum circuit into a CircuitStateFn and add the
                                # coefficients i, j and the original operator coefficient
                                coeff = operator.coeff
                                coeff *= np.sqrt(np.abs(grad_coeff_i) * np.abs(grad_coeff_j))
                                state = CircuitStateFn(qfi_circuit, coeff=coeff)

                                param_grad = 1
                                for gate, idx, param in zip(
                                    [gate_i, gate_j], [idx_i, idx_j], [param_i, param_j]
                                ):
                                    param_expression = gate.params[idx]
                                    param_grad *= param_expression.gradient(param)
                                meas = param_grad * qfi_observable

                                term = meas @ state

                                qfi_op.append(term)

                # Compute −4 * Re(〈∂kψ|ψ〉〈ψ|∂lψ〉)
                def phase_fix_combo_fn(x):
                    return -4 * np.real(x[0] * np.conjugate(x[1]))

                if self._phase_fix:
                    phase_fix_op = ListOp(
                        [phase_fix_states[i], phase_fix_states[j]], combo_fn=phase_fix_combo_fn
                    )
                    # Add the phase fix quantities to the entries of the QFI
                    # Get 4 * Re[〈∂kψ|∂lψ〉−〈∂kψ|ψ〉〈ψ|∂lψ〉]
                    qfi_ops += [SummedOp(qfi_op) + phase_fix_op]
                else:
                    qfi_ops += [SummedOp(qfi_op)]

            qfi_operators.append(ListOp(qfi_ops))

        # Return estimate of the full QFI -- A QFI is by definition positive semi-definite.
        return ListOp(qfi_operators, combo_fn=triu_to_dense)
