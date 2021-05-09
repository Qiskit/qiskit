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

"""DictToCircuitSum Class """

from qiskit.opflow.converters.converter_base import ConverterBase
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.state_fns.circuit_state_fn import CircuitStateFn
from qiskit.opflow.state_fns.dict_state_fn import DictStateFn
from qiskit.opflow.state_fns.vector_state_fn import VectorStateFn


class DictToCircuitSum(ConverterBase):
    r"""
    Converts ``DictStateFns`` or ``VectorStateFns`` to equivalent ``CircuitStateFns`` or sums
    thereof. The behavior of this class can be mostly replicated by calling ``to_circuit_op`` on
    an Operator, but with the added control of choosing whether to convert only ``DictStateFns``
    or ``VectorStateFns``, rather than both.
    """

    def __init__(
        self, traverse: bool = True, convert_dicts: bool = True, convert_vectors: bool = True
    ) -> None:
        """
        Args:
            traverse: Whether to recurse down into Operators with internal sub-operators for
                conversion.
            convert_dicts: Whether to convert VectorStateFn.
            convert_vectors: Whether to convert DictStateFns.
        """
        self._traverse = traverse
        self._convert_dicts = convert_dicts
        self._convert_vectors = convert_vectors

    def convert(self, operator: OperatorBase) -> OperatorBase:
        """Convert the Operator to ``CircuitStateFns``, recursively if ``traverse`` is True.

        Args:
            operator: The Operator to convert

        Returns:
            The converted Operator.
        """

        if isinstance(operator, DictStateFn) and self._convert_dicts:
            return CircuitStateFn.from_dict(operator.primitive)
        if isinstance(operator, VectorStateFn) and self._convert_vectors:
            return CircuitStateFn.from_vector(operator.to_matrix(massive=True))
        elif isinstance(operator, ListOp) and "Dict" in operator.primitive_strings():
            return operator.traverse(self.convert)
        else:
            return operator
