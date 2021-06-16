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

"""CircuitGradient Class """

from abc import abstractmethod
from typing import List, Union, Optional, Tuple

from qiskit.circuit import ParameterExpression, ParameterVector
from ...converters.converter_base import ConverterBase
from ...operator_base import OperatorBase


class CircuitGradient(ConverterBase):
    r"""Circuit to gradient operator converter.

    Converter for changing parameterized circuits into operators
    whose evaluation yields the gradient with respect to the circuit parameters.

    This is distinct from DerivativeBase converters which take gradients of composite
    operators and handle things like differentiating combo_fn's and enforcing product rules
    when operator coefficients are parameterized.

    CircuitGradient - uses quantum techniques to get derivatives of circuits
    DerivativeBase - uses classical techniques to differentiate operator flow data structures
    """

    # pylint: disable=arguments-differ
    @abstractmethod
    def convert(
        self,
        operator: OperatorBase,
        params: Optional[
            Union[
                ParameterExpression,
                ParameterVector,
                List[ParameterExpression],
                Tuple[ParameterExpression, ParameterExpression],
                List[Tuple[ParameterExpression, ParameterExpression]],
            ]
        ] = None,
    ) -> OperatorBase:
        r"""
        Args:
            operator: The operator we are taking the gradient of
            params: The parameters we are taking the gradient wrt: ω
                    If a ParameterExpression, ParameterVector or List[ParameterExpression] is given,
                    then the 1st order derivative of the operator is calculated.
                    If a Tuple[ParameterExpression, ParameterExpression] or
                    List[Tuple[ParameterExpression, ParameterExpression]]
                    is given, then the 2nd order derivative of the operator is calculated.

        Returns:
            An operator whose evaluation yields the Gradient.

        Raises:
            ValueError: If ``params`` contains a parameter not present in ``operator``.
        """
        raise NotImplementedError
