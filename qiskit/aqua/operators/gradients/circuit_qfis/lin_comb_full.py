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

from copy import deepcopy
from typing import List, Union, Optional, Tuple

import numpy as np
from qiskit.aqua import AquaError
from qiskit.aqua.operators import ListOp
from qiskit.aqua.operators.operator_globals import I, Z, Y, X
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn
from qiskit.circuit import Gate, Qubit
from qiskit.circuit import (QuantumCircuit, QuantumRegister, ParameterVector,
                            ParameterExpression)
from qiskit.circuit.library import RZGate, RXGate, HGate, XGate, SdgGate, SGate, ZGate, UGate

from .circuit_qfi import CircuitQFI
from ..circuit_gradients.lin_comb import LinComb
from ..derivative_base import DerivativeBase


class LinCombFull(CircuitQFI):
    r"""Compute the full Quantum Fisher Information (QFI).

    Given a pure, parametrized quantum state this class uses the linear combination of unitaries
    approach, requiring one additional working qubit.
    See also :class:`~qiskit.aqua.operators.QFI`.
    """

    def convert(self,
                operator: CircuitStateFn,
                params: Optional[Union[ParameterExpression, ParameterVector,
                                       List[ParameterExpression]]] = None,
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
            AquaError: If one of the circuits could not be constructed.
            TypeError: If ``operator`` is an unsupported type.
        """

        # QFI & phase fix observable
        qfi_observable = ~StateFn(4 * Z ^ (I ^ operator.num_qubits))
        phase_fix_observable = ~StateFn((X + 1j * Y) ^ (I ^ operator.num_qubits))
        # see https://arxiv.org/pdf/quant-ph/0108146.pdf

        if not isinstance(operator, CircuitStateFn):
            raise TypeError(
                'LinCombFull is only compatible with states that are given as CircuitStateFn')

        if not isinstance(params, (list, np.ndarray)):
            params = [params]
        state_qc = operator.primitive

        # First, the operators are computed which can compensate for a potential phase-mismatch
        # between target and trained state, i.e.〈ψ|∂lψ〉
        phase_fix_states = None
        qr_work = QuantumRegister(1, 'work_qubit')
        work_q = qr_work[0]
        additional_qubits: Tuple[List[Qubit], List[Qubit]] = ([work_q], [])
        # create a copy of the original state with an additional work_q register
        for param in params:
            param_gates = state_qc._parameter_table[param]
            for m, param_occurence in enumerate(param_gates):
                coeffs_i, gates_i = LinComb._gate_gradient_dict(param_occurence[0])[
                    param_occurence[1]]
                for k, gate_to_insert_i in enumerate(gates_i):
                    grad_state = state_qc.copy()
                    grad_state.add_register(qr_work)

                    # apply Hadamard on work_q
                    LinComb.insert_gate(grad_state, param_occurence[0], HGate(),
                                        qubits=[work_q])
                    # Fix work_q phase
                    coeff_i = coeffs_i[k]
                    sign = np.sign(coeff_i)
                    is_complex = np.iscomplex(coeff_i)
                    if sign == -1:
                        if is_complex:
                            LinComb.insert_gate(grad_state,
                                                param_occurence[0],
                                                SdgGate(),
                                                qubits=[work_q])
                        else:
                            LinComb.insert_gate(grad_state,
                                                param_occurence[0],
                                                ZGate(),
                                                qubits=[work_q])
                    else:
                        if is_complex:
                            LinComb.insert_gate(grad_state,
                                                param_occurence[0],
                                                SGate(),
                                                qubits=[work_q])

                    # Insert controlled, intercepting gate - controlled by |0>

                    if isinstance(param_occurence[0], UGate):
                        if param_occurence[1] == 0:
                            LinComb.insert_gate(grad_state, param_occurence[0],
                                                RZGate(param_occurence[0].params[2]))
                            LinComb.insert_gate(grad_state, param_occurence[0],
                                                RXGate(np.pi / 2))
                            LinComb.insert_gate(grad_state, param_occurence[0],
                                                gate_to_insert_i,
                                                additional_qubits=additional_qubits)
                            LinComb.insert_gate(grad_state, param_occurence[0],
                                                RXGate(-np.pi / 2))
                            LinComb.insert_gate(grad_state, param_occurence[0],
                                                RZGate(-param_occurence[0].params[2]))

                        elif param_occurence[1] == 1:
                            LinComb.insert_gate(grad_state, param_occurence[0],
                                                gate_to_insert_i, after=True,
                                                additional_qubits=additional_qubits)
                        else:
                            LinComb.insert_gate(grad_state, param_occurence[0],
                                                gate_to_insert_i,
                                                additional_qubits=additional_qubits)
                    else:
                        LinComb.insert_gate(grad_state, param_occurence[0],
                                            gate_to_insert_i,
                                            additional_qubits=additional_qubits)

                    grad_state = self.trim_circuit(grad_state, param_occurence[0])

                    grad_state.h(work_q)

                    state = np.sqrt(np.abs(coeff_i)) * operator.coeff * CircuitStateFn(grad_state)

                    # Chain Rule parameter expressions
                    gate_param = param_occurence[0].params[param_occurence[1]]
                    if gate_param == param:
                        state = phase_fix_observable @ state
                    else:
                        if isinstance(gate_param, ParameterExpression):
                            expr_grad = DerivativeBase.parameter_expression_grad(gate_param, param)
                            state = (expr_grad * phase_fix_observable) @ state
                        else:
                            state *= 0

                    if m == 0 and k == 0:
                        phase_fix_state = state
                    else:
                        phase_fix_state += state
            if not phase_fix_states:
                phase_fix_states = [phase_fix_state]
            else:
                phase_fix_states += [phase_fix_state]

        # Get  4 * Re[〈∂kψ|∂lψ]
        qfi_operators = []
        qr_work_qubit = QuantumRegister(1, 'work_qubit')
        work_qubit = qr_work_qubit[0]
        additional_qubits = ([work_qubit], [])
        # create a copy of the original circuit with an additional work_qubit register
        circuit = state_qc.copy()
        circuit.add_register(qr_work_qubit)
        LinComb.insert_gate(circuit, state_qc._parameter_table[params[0]][0][0], HGate(),
                            qubits=[work_qubit])

        # Get the circuits needed to compute〈∂iψ|∂jψ〉
        for i, param_i in enumerate(params):  # loop over parameters
            qfi_ops = None
            for j, param_j in enumerate(params):
                # Construct the circuits
                param_gates_i = state_qc._parameter_table[param_i]
                for m_i, param_occurence_i in enumerate(param_gates_i):
                    coeffs_i, gates_i = LinComb._gate_gradient_dict(param_occurence_i[0])[
                        param_occurence_i[1]]
                    # apply Hadamard on work_qubit

                    for k_i, gate_to_insert_i in enumerate(gates_i):
                        coeff_i = coeffs_i[k_i]
                        param_gates_j = state_qc._parameter_table[param_j]

                        for m_j, param_occurence_j in enumerate(param_gates_j):
                            coeffs_j, gates_j = \
                                LinComb._gate_gradient_dict(param_occurence_j[0])[
                                    param_occurence_j[1]]
                            for k_j, gate_to_insert_j in enumerate(gates_j):
                                coeff_j = coeffs_j[k_j]

                                # create a copy of the original circuit with the same registers
                                qfi_circuit = QuantumCircuit(*circuit.qregs)
                                qfi_circuit.data = circuit.data

                                # Fix work_qubit phase
                                sign = np.sign(np.conj(coeff_i) * coeff_j)
                                is_complex = np.iscomplex(np.conj(coeff_i) * coeff_j)
                                if sign == -1:
                                    if is_complex:
                                        LinComb.insert_gate(qfi_circuit,
                                                            param_occurence_i[0],
                                                            SdgGate(),
                                                            qubits=[work_qubit])
                                    else:
                                        LinComb.insert_gate(qfi_circuit,
                                                            param_occurence_i[0],
                                                            ZGate(),
                                                            qubits=[work_qubit])
                                else:
                                    if is_complex:
                                        LinComb.insert_gate(qfi_circuit,
                                                            param_occurence_i[0],
                                                            SGate(),
                                                            qubits=[work_qubit])

                                LinComb.insert_gate(qfi_circuit,
                                                    param_occurence_i[0],
                                                    XGate(),
                                                    qubits=[work_qubit])

                                # Insert controlled, intercepting gate - controlled by |1>
                                if isinstance(param_occurence_i[0], UGate):
                                    if param_occurence_i[1] == 0:
                                        LinComb.insert_gate(
                                            qfi_circuit, param_occurence_i[0],
                                            RZGate(param_occurence_i[0].params[2]))
                                        LinComb.insert_gate(
                                            qfi_circuit, param_occurence_i[0],
                                            RXGate(np.pi / 2))
                                        LinComb.insert_gate(
                                            qfi_circuit, param_occurence_i[0], gate_to_insert_i,
                                            additional_qubits=additional_qubits)
                                        LinComb.insert_gate(
                                            qfi_circuit, param_occurence_i[0],
                                            RXGate(-np.pi / 2))
                                        LinComb.insert_gate(
                                            qfi_circuit, param_occurence_i[0],
                                            RZGate(-param_occurence_i[0].params[2]))

                                    elif param_occurence_i[1] == 1:
                                        LinComb.insert_gate(
                                            qfi_circuit, param_occurence_i[0],
                                            gate_to_insert_i, after=True,
                                            additional_qubits=additional_qubits)
                                    else:
                                        LinComb.insert_gate(
                                            qfi_circuit, param_occurence_i[0],
                                            gate_to_insert_i, additional_qubits=additional_qubits)
                                else:
                                    LinComb.insert_gate(
                                        qfi_circuit, param_occurence_i[0],
                                        gate_to_insert_i, additional_qubits=additional_qubits)

                                LinComb.insert_gate(
                                    qfi_circuit, gate_to_insert_i,
                                    XGate(), qubits=[work_qubit], after=True)

                                # Insert controlled, intercepting gate - controlled by |0>
                                if isinstance(param_occurence_j[0], UGate):
                                    if param_occurence_j[1] == 0:
                                        LinComb.insert_gate(
                                            qfi_circuit, param_occurence_j[0],
                                            RZGate(param_occurence_j[0].params[2]))
                                        LinComb.insert_gate(
                                            qfi_circuit, param_occurence_j[0],
                                            RXGate(np.pi / 2))
                                        LinComb.insert_gate(
                                            qfi_circuit, param_occurence_j[0],
                                            gate_to_insert_j, additional_qubits=additional_qubits)
                                        LinComb.insert_gate(
                                            qfi_circuit, param_occurence_j[0],
                                            RXGate(-np.pi / 2))
                                        LinComb.insert_gate(
                                            qfi_circuit, param_occurence_j[0],
                                            RZGate(-param_occurence_j[0].params[2]))

                                    elif param_occurence_j[1] == 1:
                                        LinComb.insert_gate(
                                            qfi_circuit, param_occurence_j[0],
                                            gate_to_insert_j, after=True,
                                            additional_qubits=additional_qubits)
                                    else:
                                        LinComb.insert_gate(
                                            qfi_circuit, param_occurence_j[0],
                                            gate_to_insert_j, additional_qubits=additional_qubits)
                                else:
                                    LinComb.insert_gate(
                                        qfi_circuit, param_occurence_j[0],
                                        gate_to_insert_j, additional_qubits=additional_qubits)

                                # Remove redundant gates

                                if j <= i:
                                    qfi_circuit = self.trim_circuit(
                                        qfi_circuit, param_occurence_i[0]
                                    )
                                else:
                                    qfi_circuit = self.trim_circuit(
                                        qfi_circuit, param_occurence_j[0]
                                    )

                                qfi_circuit.h(work_qubit)
                                # Convert the quantum circuit into a CircuitStateFn
                                term = np.sqrt(np.abs(coeff_i) * np.abs(coeff_j)) * operator.coeff
                                term = term * CircuitStateFn(qfi_circuit)

                                # Chain Rule Parameter Expression
                                gate_param_i = param_occurence_i[0].params[param_occurence_i[1]]
                                gate_param_j = param_occurence_j[0].params[param_occurence_j[1]]

                                meas = deepcopy(qfi_observable)
                                if isinstance(gate_param_i, ParameterExpression):
                                    expr_grad = DerivativeBase.parameter_expression_grad(
                                        gate_param_i, param_i)
                                    meas *= expr_grad
                                if isinstance(gate_param_j, ParameterExpression):
                                    expr_grad = DerivativeBase.parameter_expression_grad(
                                        gate_param_j, param_j)
                                    meas *= expr_grad
                                term = meas @ term

                                if m_i == 0 and k_i == 0 and m_j == 0 and k_j == 0:
                                    qfi_op = term
                                else:
                                    # Product Rule
                                    qfi_op += term

                # Compute −4 * Re(〈∂kψ|ψ〉〈ψ|∂lψ〉)
                def phase_fix_combo_fn(x):
                    return 4 * (-0.5) * (x[0] * np.conjugate(x[1]) + x[1] * np.conjugate(x[0]))

                phase_fix = ListOp([phase_fix_states[i], phase_fix_states[j]],
                                   combo_fn=phase_fix_combo_fn)
                # Add the phase fix quantities to the entries of the QFI
                # Get 4 * Re[〈∂kψ|∂lψ〉−〈∂kψ|ψ〉〈ψ|∂lψ〉]
                if not qfi_ops:
                    qfi_ops = [qfi_op + phase_fix]
                else:
                    qfi_ops += [qfi_op + phase_fix]
            qfi_operators.append(ListOp(qfi_ops))
        # Return the full QFI
        return ListOp(qfi_operators)

    @staticmethod
    def trim_circuit(circuit: QuantumCircuit,
                     reference_gate: Gate) -> QuantumCircuit:
        """Trim the given quantum circuit before the reference gate.

        Args:
            circuit: The circuit to be trimmed.
            reference_gate: The gate where the circuit is supposed to be trimmed.

        Returns:
            The trimmed circuit.

        Raises:
            AquaError: If the reference gate is not part of the given circuit.
        """
        parameterized_gates = []
        for _, elements in circuit._parameter_table.items():
            for element in elements:
                parameterized_gates.append(element[0])

        for i, op in enumerate(circuit.data):
            if op[0] == reference_gate:
                trimmed_circuit = QuantumCircuit(*circuit.qregs)
                trimmed_circuit.data = circuit.data[:i]
                return trimmed_circuit

        raise AquaError('The reference gate is not in the given quantum circuit.')
