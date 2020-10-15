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

"""The module to compute the state gradient with the linear combination method."""

from collections.abc import Iterable
from copy import deepcopy
from functools import partial
from itertools import product
from typing import List, Optional, Tuple, Union

import numpy as np
from qiskit.aqua.aqua_globals import AquaError
from qiskit.aqua.operators import OperatorBase, ListOp, ComposedOp
from qiskit.aqua.operators.operator_globals import Z, I, One, Zero
from qiskit.aqua.operators.primitive_ops.primitive_op import PrimitiveOp
from qiskit.aqua.operators.state_fns import StateFn, CircuitStateFn, DictStateFn, VectorStateFn
from qiskit.circuit import Gate, Instruction, Qubit
from qiskit.circuit import (QuantumCircuit, QuantumRegister, ParameterVector,
                            ParameterExpression)
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.library import SGate, SdgGate, HGate
from qiskit.circuit.library.standard_gates import (CXGate, CYGate, CZGate,
                                                   IGate, RXGate, RXXGate,
                                                   RYGate, RYYGate, RZGate,
                                                   RZXGate, RZZGate, PhaseGate,
                                                   UGate, ZGate)
from qiskit.quantum_info import partial_trace

from .circuit_gradient import CircuitGradient
from ..derivative_base import DerivativeBase


class LinComb(CircuitGradient):
    """Compute the state gradient d⟨ψ(ω)|O(θ)|ψ(ω)〉/ dω respectively the gradients of the
    sampling probabilities of the basis states of
    a state |ψ(ω)〉w.r.t. ω.
    This method employs a linear combination of unitaries,
    see e.g. https://arxiv.org/pdf/1811.11184.pdf
    """

    def convert(self,
                operator: OperatorBase,
                params: Optional[Union[ParameterExpression, ParameterVector,
                                       List[ParameterExpression],
                                       Tuple[ParameterExpression, ParameterExpression],
                                       List[Tuple[ParameterExpression, ParameterExpression]]]]
                = None,
                ) -> OperatorBase:
        """ Convert the given operator into an operator object that represents the gradient w.r.t.
            params

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
    def _prepare_operator(self,
                          operator: OperatorBase,
                          params: Optional[Union[ParameterExpression, ParameterVector,
                                                 List[ParameterExpression],
                                                 Tuple[ParameterExpression, ParameterExpression],
                                                 List[Tuple[ParameterExpression,
                                                            ParameterExpression]]]] = None
                          ) -> OperatorBase:
        """ Traverse through the given operator to get back the adapted operator representing the
            gradient

        Args:
            operator: The operator we are taking the gradient of: ⟨ψ(ω)|O(θ)|ψ(ω)〉
            params: The parameters we are taking the gradient wrt: ω
                    If a ParameterExpression, ParameterVector or List[ParameterExpression] is given,
                    then the 1st order derivative of the operator is calculated.
                    If a Tuple[ParameterExpression, ParameterExpression] or
                    List[Tuple[ParameterExpression, ParameterExpression]]
                    is given, then the 2nd order derivative of the operator is calculated.

        Returns:
            Adapted operator.
                Measurement operators are attached with an additional Z term acting
                on an additional working qubit.
                Quantum states - which must be given as circuits - are adapted. An additional
                working qubit controls intercepting gates.
                See e.g. Evaluating analytic gradients on quantum hardware
                    Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, and Nathan Killoran
                    Phys. Rev. A 99, 032331 – Published 21 March 2019

        Raises:
            ValueError: If ``operator`` does not correspond to an expectation value.
            TypeError: If the ``StateFn`` corresponding to the quantum state could not be extracted
                       from ``operator``.
            AquaError: If third or higher order gradients are requested.
        """

        if isinstance(operator, ComposedOp):
            # Get the measurement and the state operator
            if not isinstance(operator[0], StateFn) or not operator[0].is_measurement:
                raise ValueError("The given operator does not correspond to an expectation value")
            if not isinstance(operator[-1], StateFn) or operator[-1].is_measurement:
                raise ValueError("The given operator does not correspond to an expectation value")
            if operator[0].is_measurement:
                if len(operator.oplist) == 2:
                    state_op = operator[1]
                    if not isinstance(state_op, StateFn):
                        raise TypeError('The StateFn representing the quantum state could not be'
                                        'extracted.')
                    if isinstance(params, (ParameterExpression, ParameterVector)) or \
                            (isinstance(params, list) and all(isinstance(param, ParameterExpression)
                                                              for param in params)):

                        return self._gradient_states(state_op, meas_op=(2 * ~StateFn(Z) ^
                                                                        operator[0]),
                                                     target_params=params)
                    elif isinstance(params, tuple) or \
                            (isinstance(params, list) and all(isinstance(param, tuple)
                                                              for param in params)):
                        return self._hessian_states(state_op,
                                                    meas_op=(4 * ~StateFn(Z ^ I) ^ operator[0]),
                                                    target_params=params)  # type: ignore
                    else:
                        raise AquaError('The linear combination gradient does only support the '
                                        'computation of 1st gradients and 2nd order gradients.')
                else:
                    state_op = deepcopy(operator)
                    state_op.oplist.pop(0)
                    if not isinstance(state_op, StateFn):
                        raise TypeError('The StateFn representing the quantum state could not be'
                                        'extracted.')

                    if isinstance(params, (ParameterExpression, ParameterVector)) or \
                            (isinstance(params, list) and all(isinstance(param, ParameterExpression)
                                                              for param in params)):
                        return state_op.traverse(partial(self._gradient_states,
                                                         meas_op=(2 * ~StateFn(Z) ^ operator[0]),
                                                         target_params=params))
                    elif isinstance(params, tuple) or \
                        (isinstance(params, list) and all(isinstance(param, tuple)
                                                          for param in params)):
                        return state_op.traverse(
                            partial(self._hessian_states,
                                    meas_op=(4 * ~StateFn(Z ^ I) ^ operator[0]),
                                    target_params=params))

                    raise AquaError(
                        'The linear combination gradient does only support the computation '
                        'of 1st gradients and 2nd order gradients.')
            else:
                return operator.traverse(partial(self._prepare_operator, params=params))
        elif isinstance(operator, ListOp):
            return operator.traverse(partial(self._prepare_operator, params=params))
        elif isinstance(operator, StateFn):
            if operator.is_measurement:
                return operator.traverse(partial(self._prepare_operator, params=params))
            else:
                if isinstance(params, (ParameterExpression, ParameterVector)) or \
                        (isinstance(params, list) and all(isinstance(param, ParameterExpression)
                                                          for param in params)):
                    return self._gradient_states(operator, target_params=params)
                elif isinstance(params, tuple) or \
                        (isinstance(params, list) and all(isinstance(param, tuple)
                                                          for param in params)):
                    return self._hessian_states(operator, target_params=params)  # type: ignore
                else:
                    raise AquaError(
                        'The linear combination gradient does only support the computation '
                        'of 1st gradients and 2nd order gradients.')
        elif isinstance(operator, PrimitiveOp):
            return operator
        return operator

    def _gradient_states(self,
                         state_op: StateFn,
                         meas_op: Optional[OperatorBase] = None,
                         target_params: Optional[
                             Union[ParameterExpression, ParameterVector,
                                   List[ParameterExpression]]] = None
                         ) -> ListOp:
        """Generate the gradient states.

        Args:
            state_op: The operator representing the quantum state for which we compute the gradient.
            meas_op: The operator representing the observable for which we compute the gradient.
            target_params: The parameters we are taking the gradient wrt: ω

        Returns:
            ListOp of StateFns as quantum circuits which are the states w.r.t. which we compute the
            gradient. If a parameter appears multiple times, one circuit is created per
            parameterized gates to compute the product rule.

        Raises:
            AquaError: If one of the circuits could not be constructed.
            TypeError: If the operators is of unsupported type.
        """
        state_qc = deepcopy(state_op.primitive)

        # Define the working qubit to realize the linear combination of unitaries
        qr_work = QuantumRegister(1, 'work_qubit_lin_comb_grad')
        work_q = qr_work[0]

        if not isinstance(target_params, (list, np.ndarray)):
            target_params = [target_params]

        if len(target_params) > 1:
            states = None

        additional_qubits: Tuple[List[Qubit], List[Qubit]] = ([work_q], [])

        for param in target_params:
            if param not in state_qc._parameter_table.get_keys():
                op = ~Zero @ One
            else:
                param_gates = state_qc._parameter_table[param]
                for m, param_occurence in enumerate(param_gates):
                    coeffs, gates = self._gate_gradient_dict(param_occurence[0])[param_occurence[1]]

                    # construct the states
                    for k, gate_to_insert in enumerate(gates):
                        grad_state = QuantumCircuit(*state_qc.qregs, qr_work)
                        grad_state.compose(state_qc, inplace=True)

                        # apply Hadamard on work_q
                        self.insert_gate(grad_state, param_occurence[0], HGate(), qubits=[work_q])

                        # Fix work_q phase
                        coeff_i = coeffs[k]
                        sign = np.sign(coeff_i)
                        is_complex = np.iscomplex(coeff_i)
                        if sign == -1:
                            if is_complex:
                                self.insert_gate(grad_state, param_occurence[0],
                                                 SdgGate(), qubits=[work_q])
                            else:
                                self.insert_gate(grad_state, param_occurence[0],
                                                 ZGate(), qubits=[work_q])
                        else:
                            if is_complex:
                                self.insert_gate(grad_state, param_occurence[0],
                                                 SGate(), qubits=[work_q])

                        # Insert controlled, intercepting gate - controlled by |0>
                        if isinstance(param_occurence[0], UGate):
                            if param_occurence[1] == 0:
                                self.insert_gate(grad_state, param_occurence[0],
                                                 RZGate(param_occurence[0].params[2]))
                                self.insert_gate(grad_state, param_occurence[0],
                                                 RXGate(np.pi / 2))
                                self.insert_gate(grad_state, param_occurence[0],
                                                 gate_to_insert,
                                                 additional_qubits=additional_qubits)
                                self.insert_gate(grad_state, param_occurence[0],
                                                 RXGate(-np.pi / 2))
                                self.insert_gate(grad_state, param_occurence[0],
                                                 RZGate(-param_occurence[0].params[2]))

                            elif param_occurence[1] == 1:
                                self.insert_gate(grad_state, param_occurence[0],
                                                 gate_to_insert, after=True,
                                                 additional_qubits=additional_qubits)
                            else:
                                self.insert_gate(grad_state, param_occurence[0],
                                                 gate_to_insert,
                                                 additional_qubits=additional_qubits)
                        else:
                            self.insert_gate(grad_state, param_occurence[0],
                                             gate_to_insert,
                                             additional_qubits=additional_qubits)
                        grad_state.h(work_q)

                        state = np.sqrt(np.abs(coeff_i)) * state_op.coeff * CircuitStateFn(
                            grad_state)
                        # Chain Rule parameter expressions
                        gate_param = param_occurence[0].params[param_occurence[1]]
                        if meas_op:
                            if gate_param == param:
                                state = meas_op @ state
                            else:
                                if isinstance(gate_param, ParameterExpression):
                                    expr_grad = DerivativeBase.parameter_expression_grad(gate_param,
                                                                                         param)
                                    state = (expr_grad * meas_op) @ state
                                else:
                                    state = ~Zero @ One
                        else:
                            if gate_param == param:
                                state = ListOp([state],
                                               combo_fn=partial(self._grad_combo_fn,
                                                                state_op=state_op))
                            else:
                                if isinstance(gate_param, ParameterExpression):
                                    expr_grad = DerivativeBase.parameter_expression_grad(gate_param,
                                                                                         param)
                                    state = expr_grad * ListOp(
                                        [state],
                                        combo_fn=partial(self._grad_combo_fn, state_op=state_op))
                                else:
                                    state = ~Zero @ One

                        if m == 0 and k == 0:
                            op = state
                        else:
                            # Product Rule
                            op += state
                if len(target_params) > 1:
                    if not states:
                        states = [op]
                    else:
                        states += [op]
                else:
                    return op
        if len(target_params) > 1:
            return ListOp(states)
        else:
            return op

    def _hessian_states(self,
                        state_op: StateFn,
                        meas_op: Optional[OperatorBase] = None,
                        target_params: Optional[Union[Tuple[ParameterExpression,
                                                            ParameterExpression],
                                                      List[Tuple[ParameterExpression,
                                                                 ParameterExpression]]]] = None
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
            AquaError: If one of the circuits could not be constructed.
            TypeError: If ``operator`` is of unsupported type.
        """
        state_qc = deepcopy(state_op.primitive)
        if isinstance(target_params, list) and isinstance(target_params[0], tuple):
            tuples_list = deepcopy(target_params)
            target_params = []
            for tuples in tuples_list:
                if all([param in state_qc._parameter_table.get_keys() for param in tuples]):
                    for param in tuples:
                        if param not in target_params:
                            target_params.append(param)
        elif isinstance(target_params, tuple):
            tuples_list = deepcopy([target_params])
            target_params = []
            for tuples in tuples_list:
                if all([param in state_qc._parameter_table.get_keys() for param in tuples]):
                    for param in tuples:
                        if param not in target_params:
                            target_params.append(param)
        else:
            raise TypeError(
                'Please define in the parameters for which the Hessian is evaluated either '
                'as parameter tuple or a list of parameter tuples')

        qr_add0 = QuantumRegister(1, 'work_qubit0')
        work_q0 = qr_add0[0]
        qr_add1 = QuantumRegister(1, 'work_qubit1')
        work_q1 = qr_add1[0]
        # create a copy of the original circuit with an additional working qubit register
        circuit = state_qc.copy()
        circuit.add_register(qr_add0, qr_add1)
        # Get the circuits needed to compute the Hessian
        hessian_ops = None
        for param_a, param_b in tuples_list:

            if param_a not in state_qc._parameter_table.get_keys() or param_b \
                    not in state_qc._parameter_table.get_keys():
                hessian_op = ~Zero @ One
            else:
                param_gates_a = state_qc._parameter_table[param_a]
                param_gates_b = state_qc._parameter_table[param_b]
                for i, param_occurence_a in enumerate(param_gates_a):
                    coeffs_a, gates_a = self._gate_gradient_dict(param_occurence_a[0])[
                        param_occurence_a[1]]
                    # apply Hadamard on working qubit
                    self.insert_gate(circuit, param_occurence_a[0], HGate(),
                                     qubits=[work_q0])
                    self.insert_gate(circuit, param_occurence_a[0], HGate(),
                                     qubits=[work_q1])
                    for j, gate_to_insert_a in enumerate(gates_a):

                        coeff_a = coeffs_a[j]
                        hessian_circuit_temp = QuantumCircuit(*circuit.qregs)
                        hessian_circuit_temp.data = circuit.data
                        # Fix working qubit 0 phase
                        sign = np.sign(coeff_a)
                        is_complex = np.iscomplex(coeff_a)
                        if sign == -1:
                            if is_complex:
                                self.insert_gate(hessian_circuit_temp,
                                                 param_occurence_a[0],
                                                 SdgGate(),
                                                 qubits=[work_q0])
                            else:
                                self.insert_gate(hessian_circuit_temp,
                                                 param_occurence_a[0],
                                                 ZGate(),
                                                 qubits=[work_q0])
                        else:
                            if is_complex:
                                self.insert_gate(hessian_circuit_temp,
                                                 param_occurence_a[0],
                                                 SGate(),
                                                 qubits=[work_q0])

                        # Insert controlled, intercepting gate - controlled by |1>
                        if isinstance(param_occurence_a[0], UGate):
                            if param_occurence_a[1] == 0:
                                self.insert_gate(hessian_circuit_temp, param_occurence_a[0],
                                                 RZGate(param_occurence_a[0].params[2]))
                                self.insert_gate(hessian_circuit_temp, param_occurence_a[0],
                                                 RXGate(np.pi / 2))
                                self.insert_gate(hessian_circuit_temp, param_occurence_a[0],
                                                 gate_to_insert_a,
                                                 additional_qubits=([work_q0], []))
                                self.insert_gate(hessian_circuit_temp, param_occurence_a[0],
                                                 RXGate(-np.pi / 2))
                                self.insert_gate(hessian_circuit_temp, param_occurence_a[0],
                                                 RZGate(-param_occurence_a[0].params[2]))

                            elif param_occurence_a[1] == 1:
                                self.insert_gate(hessian_circuit_temp, param_occurence_a[0],
                                                 gate_to_insert_a, after=True,
                                                 additional_qubits=([work_q0], []))
                            else:
                                self.insert_gate(hessian_circuit_temp, param_occurence_a[0],
                                                 gate_to_insert_a,
                                                 additional_qubits=([work_q0], []))
                        else:
                            self.insert_gate(hessian_circuit_temp, param_occurence_a[0],
                                             gate_to_insert_a, additional_qubits=([work_q0], []))

                        for m, param_occurence_b in enumerate(param_gates_b):
                            coeffs_b, gates_b = self._gate_gradient_dict(param_occurence_b[0])[
                                param_occurence_b[1]]
                            for n, gate_to_insert_b in enumerate(gates_b):
                                coeff_b = coeffs_b[n]
                                # create a copy of the original circuit with the same registers
                                hessian_circuit = QuantumCircuit(*hessian_circuit_temp.qregs)
                                hessian_circuit.data = hessian_circuit_temp.data

                                # Fix working qubit 1 phase
                                sign = np.sign(coeff_b)
                                is_complex = np.iscomplex(coeff_b)
                                if sign == -1:
                                    if is_complex:
                                        self.insert_gate(hessian_circuit,
                                                         param_occurence_b[0],
                                                         SdgGate(),
                                                         qubits=[work_q1])
                                    else:
                                        self.insert_gate(hessian_circuit,
                                                         param_occurence_b[0],
                                                         ZGate(),
                                                         qubits=[work_q1])
                                else:
                                    if is_complex:
                                        self.insert_gate(hessian_circuit,
                                                         param_occurence_b[0],
                                                         SGate(),
                                                         qubits=[work_q1])

                                # Insert controlled, intercepting gate - controlled by |1>

                                if isinstance(param_occurence_b[0], UGate):
                                    if param_occurence_b[1] == 0:
                                        self.insert_gate(hessian_circuit, param_occurence_b[0],
                                                         RZGate(param_occurence_b[0].params[2]))
                                        self.insert_gate(hessian_circuit, param_occurence_b[0],
                                                         RXGate(np.pi / 2))
                                        self.insert_gate(hessian_circuit, param_occurence_b[0],
                                                         gate_to_insert_b,
                                                         additional_qubits=([work_q1], []))
                                        self.insert_gate(hessian_circuit, param_occurence_b[0],
                                                         RXGate(-np.pi / 2))
                                        self.insert_gate(hessian_circuit, param_occurence_b[0],
                                                         RZGate(-param_occurence_b[0].params[2]))

                                    elif param_occurence_b[1] == 1:
                                        self.insert_gate(hessian_circuit, param_occurence_b[0],
                                                         gate_to_insert_b, after=True,
                                                         additional_qubits=([work_q1], []))
                                    else:
                                        self.insert_gate(hessian_circuit, param_occurence_b[0],
                                                         gate_to_insert_b,
                                                         additional_qubits=([work_q1], []))
                                else:
                                    self.insert_gate(hessian_circuit, param_occurence_b[0],
                                                     gate_to_insert_b,
                                                     additional_qubits=([work_q1], []))

                                hessian_circuit.h(work_q0)
                                hessian_circuit.cz(work_q1, work_q0)
                                hessian_circuit.h(work_q1)

                                term = state_op.coeff * np.sqrt(np.abs(coeff_a) * np.abs(coeff_b)) \
                                                      * CircuitStateFn(hessian_circuit)

                                # Chain Rule Parameter Expression
                                gate_param_a = param_occurence_a[0].params[param_occurence_a[1]]
                                gate_param_b = param_occurence_b[0].params[param_occurence_b[1]]

                                if meas_op:
                                    meas = deepcopy(meas_op)
                                    if isinstance(gate_param_a, ParameterExpression):
                                        expr_grad = DerivativeBase.parameter_expression_grad(
                                            gate_param_a,
                                            param_a)
                                        meas *= expr_grad
                                    if isinstance(gate_param_b, ParameterExpression):
                                        expr_grad = DerivativeBase.parameter_expression_grad(
                                            gate_param_a,
                                            param_a)
                                        meas *= expr_grad
                                    term = meas @ term
                                else:
                                    term = ListOp([term],
                                                  combo_fn=partial(self._hess_combo_fn,
                                                                   state_op=state_op))
                                    if isinstance(gate_param_a, ParameterExpression):
                                        expr_grad = DerivativeBase.parameter_expression_grad(
                                            gate_param_a,
                                            param_a)
                                        term *= expr_grad
                                    if isinstance(gate_param_b, ParameterExpression):
                                        expr_grad = DerivativeBase.parameter_expression_grad(
                                            gate_param_a,
                                            param_a)
                                        term *= expr_grad

                                if i == 0 and j == 0 and m == 0 and n == 0:
                                    hessian_op = term
                                else:
                                    # Product Rule
                                    hessian_op += term
            # Create a list of Hessian elements w.r.t. the given parameter tuples
            if len(tuples_list) == 1:
                return hessian_op
            else:
                if not hessian_ops:
                    hessian_ops = [hessian_op]
                else:
                    hessian_ops += [hessian_op]
        return ListOp(hessian_ops)

    @staticmethod
    def _grad_combo_fn(x, state_op):
        def get_result(item):
            if isinstance(item, DictStateFn):
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
            elif isinstance(item, Iterable):
                # Generate the operator which computes the linear combination
                lin_comb_op = 2 * (I ^ state_op.num_qubits) ^ Z
                lin_comb_op = lin_comb_op.to_matrix()
                return list(np.diag(
                    partial_trace(lin_comb_op.dot(np.outer(item, np.conj(item))), [0]).data))
            else:
                raise TypeError(
                    'The state result should be either a DictStateFn or a VectorStateFn.')

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
                return list(np.diag(
                    partial_trace(lin_comb_op.dot(np.outer(item, np.conj(item))), [0, 1]).data))
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
                    'The state result should be either a '
                    'DictStateFn or a VectorStateFn.')

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
                the respective gates.
                [([a^0], [V^0]) ..., ([a^k], [V^k])]


        Raises:
            AquaError: If the input gate is controlled by another state but '|1>^{\otimes k}'
            TypeError: If the input gate is not a supported parametrized gate.
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
            if gate.ctrl_state != 2 ** gate.num_ctrl_qubits - 1:
                raise AquaError(
                    'Function only support controlled gates with control state `1` on all control '
                    'qubits.')

            base_coeffs_gates = LinComb._gate_gradient_dict(gate.base_gate)
            coeffs_gates = []
            # The projectors needed for the gradient of a controlled gate are integrated by a sum
            # of gates.
            # The following line generates the decomposition gates.

            proj_gates_controlled = [[(-1) ** p.count(ZGate()), p] for p in
                                     product([IGate(), ZGate()],
                                             repeat=gate.num_ctrl_qubits)]
            for base_coeffs, base_gates in base_coeffs_gates:  # loop over parameters
                coeffs = []
                gates = []
                for phase, proj_gates in proj_gates_controlled:
                    coeffs.extend([phase * c / (2 ** gate.num_ctrl_qubits) for c in base_coeffs])
                    for base_gate in base_gates:
                        controlled_circ = QuantumCircuit(gate.num_ctrl_qubits + gate.num_qubits)
                        for i, proj_gate in enumerate(proj_gates):
                            if isinstance(proj_gate, ZGate):
                                controlled_circ.cz(0, i + 1)
                        if not isinstance(base_gate, IGate):
                            controlled_circ.append(base_gate, [0, range(gate.num_ctrl_qubits + 1,
                                                                        gate.num_ctrl_qubits +
                                                                        gate.num_qubits)])
                        gates.append(controlled_circ.to_instruction())
                c_g = (coeffs, gates)
                coeffs_gates.append(c_g)
            return coeffs_gates

        raise TypeError('Unrecognized parametrized gate, {}'.format(gate))

    @staticmethod
    def insert_gate(circuit: QuantumCircuit,
                    reference_gate: Gate,
                    gate_to_insert: Instruction,
                    qubits: Optional[List[Qubit]] = None,
                    additional_qubits: Optional[Tuple[List[Qubit], List[Qubit]]] = None,
                    after: bool = False):
        """Insert a gate into the circuit.

        Args:
            circuit: The circuit onto which the gate is added.
            reference_gate: A gate instance before or after which a gate is inserted.
            gate_to_insert: The gate to be inserted.
            qubits: The qubits on which the gate is inserted. If None, the qubits of the
                reference_gate are used.
            additional_qubits: If qubits is None and the qubits of the reference_gate are
                used, this can be used to specify additional qubits before (first list in
                tuple) or after (second list in tuple) the qubits.
            after: If the gate_to_insert should be inserted after the reference_gate set True.

        Raises:
            AquaError: Gate insertion fail
        """

        if isinstance(gate_to_insert, IGate):
            return
        else:
            for i, op in enumerate(circuit.data):
                if op[0] == reference_gate:
                    qubits = qubits or op[1]
                    if additional_qubits:
                        qubits = additional_qubits[0] + qubits + additional_qubits[1]
                    if after:
                        insertion_index = i + 1
                    else:
                        insertion_index = i
                    circuit.data.insert(insertion_index, (gate_to_insert, qubits, []))
                    return
            raise AquaError('Could not insert the controlled gate, something went wrong!')
