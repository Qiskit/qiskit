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

""" CircuitQFI Class """

from abc import abstractmethod
from typing import List, Union

from qiskit.circuit import ParameterExpression, ParameterVector
from ...converters.converter_base import ConverterBase
from ...operator_base import OperatorBase


class CircuitQFI(ConverterBase):
    r"""Circuit to Quantum Fisher Information operator converter.

    Converter for changing parameterized circuits into operators
    whose evaluation yields Quantum Fisher Information metric tensor
    with respect to the given circuit parameters

    This is distinct from DerivativeBase converters which take gradients of composite
    operators and handle things like differentiating combo_fn's and enforcing product rules
    when operator coefficients are parameterized.

    CircuitQFI - uses quantum techniques to get the QFI of circuits
    DerivativeBase - uses classical techniques to differentiate opflow data structures
    """

    # pylint: disable=arguments-differ
    @abstractmethod
    def convert(
        self,
        operator: OperatorBase,
        params: Union[ParameterExpression, ParameterVector, List[ParameterExpression]],
    ) -> OperatorBase:
        r"""
        Args:
            operator: The operator corresponding to the quantum state :math:`|\psi(\omega)\rangle`
                for which we compute the QFI.
            params: The parameters :math:`\omega` with respect to which we are computing the QFI.

        Returns:
            An operator whose evaluation yields the QFI metric tensor.

        Raises:
            ValueError: If ``params`` contains a parameter not present in ``operator``.
        """
        raise NotImplementedError
