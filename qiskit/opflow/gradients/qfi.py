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
import functools

from qiskit.circuit.quantumcircuit import _compare_parameters
from qiskit.circuit import ParameterExpression, ParameterVector
from ..list_ops.list_op import ListOp
from ..expectations.pauli_expectation import PauliExpectation
from ..state_fns.circuit_state_fn import CircuitStateFn
from .qfi_base import QFIBase


class QFI(QFIBase):
    r"""Compute the Quantum Fisher Information (QFI).

    Computes the QFI given a pure, parameterized quantum state, where QFI is:

    .. math::

        \mathrm{QFI}_{kl}= 4 \mathrm{Re}[\langle \partial_k \psi | \partial_l \psi \rangle
            − \langle\partial_k \psi | \psi \rangle \langle\psi | \partial_l \psi \rangle].

    """

    # pylint: disable=signature-differs
    def convert(
        self,
        operator: CircuitStateFn,
        params: Optional[
            Union[ParameterExpression, ParameterVector, List[ParameterExpression]]
        ] = None,
    ) -> ListOp:
        r"""
        Args:
            operator: The operator corresponding to the quantum state \|ψ(ω)〉for which we compute
                the QFI
            params: The parameters we are computing the QFI wrt: ω
                If not explicitly passed, they are inferred from the operator and sorted by name.

        Returns:
            ListOp[ListOp] where the operator at position k,l corresponds to QFI_kl

        Raises:
            ValueError: If operator is not parameterized.
        """
        if len(operator.parameters) == 0:
            raise ValueError("The operator we are taking the gradient of is not parameterized!")

        expec_op = PauliExpectation(group_paulis=False).convert(operator).reduce()
        cleaned_op = self._factor_coeffs_out_of_composed_op(expec_op)

        if params is None:
            params = sorted(operator.parameters, key=functools.cmp_to_key(_compare_parameters))
        return self.qfi_method.convert(cleaned_op, params)
