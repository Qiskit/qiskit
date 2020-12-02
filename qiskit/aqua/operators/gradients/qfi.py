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

from qiskit.aqua.operators import ListOp
from qiskit.aqua.operators.expectations import PauliExpectation
from qiskit.aqua.operators.state_fns import CircuitStateFn
from qiskit.circuit import (ParameterExpression, ParameterVector)

from .qfi_base import QFIBase


class QFI(QFIBase):
    r"""Compute the Quantum Fisher Information (QFI).

    Computes the QFI given a pure, parametrized quantum state, where QFI is:

    .. math::

        \mathrm{QFI}_{kl}= 4 \mathrm{Re}[\langle \partial_k \psi | \partial_l \psi \rangle
            − \langle\partial_k \psi | \psi \rangle \langle\psi | \partial_l \psi \rangle].

    """

    def convert(self,
                operator: CircuitStateFn,
                params: Optional[Union[ParameterExpression, ParameterVector,
                                       List[ParameterExpression]]] = None
                ) -> ListOp:
        r"""
        Args:
            operator: The operator corresponding to the quantum state \|ψ(ω)〉for which we compute
                the QFI
            params: The parameters we are computing the QFI wrt: ω

        Returns:
            ListOp[ListOp] where the operator at position k,l corresponds to QFI_kl
        """
        expec_op = PauliExpectation(group_paulis=False).convert(operator).reduce()
        cleaned_op = self._factor_coeffs_out_of_composed_op(expec_op)

        return self.qfi_method.convert(cleaned_op, params)
