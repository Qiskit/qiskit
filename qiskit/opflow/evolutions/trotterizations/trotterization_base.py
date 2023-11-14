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

"""Trotterization Algorithm Base"""

from abc import abstractmethod

from qiskit.opflow.evolutions.evolution_base import EvolutionBase
from qiskit.opflow.operator_base import OperatorBase
from qiskit.utils.deprecation import deprecate_func

# TODO centralize handling of commuting groups


class TrotterizationBase(EvolutionBase):
    """Deprecated: A base for Trotterization methods, algorithms for approximating exponentiations of
    operator sums by compositions of exponentiations.
    """

    @deprecate_func(
        since="0.24.0",
        package_name="qiskit-terra",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(self, reps: int = 1) -> None:
        super().__init__()
        self._reps = reps

    @property
    def reps(self) -> int:
        """The number of repetitions to use in the Trotterization, improving the approximation
        accuracy.
        """
        return self._reps

    @reps.setter
    def reps(self, reps: int) -> None:
        r"""Set the number of repetitions to use in the Trotterization."""
        self._reps = reps

    @abstractmethod
    def convert(self, operator: OperatorBase) -> OperatorBase:
        r"""
        Convert a ``SummedOp`` into a ``ComposedOp`` or ``CircuitOp`` representing an
        approximation of e^-i*``op_sum``.

        Args:
            operator: The ``SummedOp`` to evolve.

        Returns:
            The Operator approximating op_sum's evolution.

        Raises:
            TypeError: A non-SummedOps Operator is passed into ``convert``.

        """
        raise NotImplementedError

    # TODO @abstractmethod - trotter_error_bound
