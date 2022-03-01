# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An algorithm to implement a Trotterization real time-evolution."""

from typing import Union, Dict

from qiskit.algorithms.quantum_time_evolution.real.implementations.trotterization\
    .trotter_ops_validator import (
    _validate_input,
    _is_op_bound,
)
from qiskit.algorithms.quantum_time_evolution.real.qrte import Qrte
from qiskit.circuit import Parameter
from qiskit.opflow import (
    OperatorBase,
    StateFn,
    SummedOp,
    PauliSumOp,
    PauliOp,
    CircuitOp,
)
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli
from qiskit.synthesis import ProductFormula, LieTrotter


class TrotterQrte(Qrte):
    """ Class for performing Quantum Real Time Evolution using Trotterization.
    Type of Trotterization is defined by a ProductFormula provided.

    Examples:

        .. jupyter-execute::

            from qiskit.opflow import X, Y, Zero
            from qiskit.algorithms.quantum_time_evolution.real.implementations.\
                trotterization.trotter_qrte import TrotterQrte

            operator = X + Z
            # LieTrotter with 1 rep
            trotter_qrte = TrotterQrte()
            initial_state = Zero
            time = 1
            evolved_state = trotter_qrte.evolve(operator, time, initial_state)
    """

    def __init__(self, product_formula: ProductFormula = LieTrotter()) -> None:
        """
        Args:
            product_formula: A Lie-Trotter-Suzuki product formula. The default is the Lie-Trotter
                            first order product formula with a single repetition.
        """
        self.product_formula = product_formula

    def evolve(
        self,
        hamiltonian: Union[Pauli, PauliOp, PauliSumOp],
        time: float,
        initial_state: StateFn = None,
        observable: OperatorBase = None,
        t_param: Parameter = None,
        hamiltonian_value_dict: Dict[Parameter, Union[float, complex]] = None,
    ) -> StateFn:
        """
        Evolves a quantum state or an observable for a given time using the Trotterization method
        based on a product formula provided.
        Time-dependent Hamiltonians are not yet supported.

        Args:
            hamiltonian:
                The operator to evolve. Can also be provided as list of non-commuting
                operators where the elements are sums of commuting operators.
                For example: ``[XY + YX, ZZ + ZI + IZ, YY]``.
            time: Total time of evolution.
            initial_state: If interested in a quantum state time evolution, a quantum state to be
                evolved.
            observable: If interested in a quantum observable time evolution, a quantum observable
                to be evolved.
            t_param: Not supported by this algorithm.
            hamiltonian_value_dict: Dictionary that maps all parameters in a Hamiltonian to
                certain values.

        Returns:
            The evolved hamiltonian applied to either an initial state or an observable.

        Raises:
            ValueError: If t_param is not set to None (feature not currently supported).
        """
        if t_param is not None:
            raise ValueError(
                "TrotterQrte does not accept a time dependent hamiltonian,"
                "t_param should be set to None."
            )

        hamiltonian = self._try_binding_params(hamiltonian, hamiltonian_value_dict)
        _validate_input(initial_state, observable)
        # the evolution gate
        evolution_gate = CircuitOp(
            PauliEvolutionGate(hamiltonian, time, synthesis=self.product_formula)
        )

        if initial_state is not None:
            return (evolution_gate @ initial_state).eval()
        if observable is not None:
            # TODO Temporary patch due to terra bug
            evolution_gate_adjoint = CircuitOp(
                PauliEvolutionGate(hamiltonian[::-1], -time, synthesis=self.product_formula)
            )
            return evolution_gate_adjoint @ observable @ evolution_gate

        raise ValueError("Either initial_state or observable must be provided.")

    @staticmethod
    def _try_binding_params(
        hamiltonian: Union[SummedOp, PauliOp, OperatorBase],
        hamiltonian_value_dict: Dict[Parameter, Union[float, complex]],
    ) -> Union[SummedOp, PauliOp, OperatorBase]:
        """
        Tries binding parameters in a Hamiltonian.

        Args:
            hamiltonian:
                The operator to evolve. Can also be provided as list of non-commuting
                operators where the elements are sums of commuting operators.
                For example: ``[XY + YX, ZZ + ZI + IZ, YY]``.
            hamiltonian_value_dict: Dictionary that maps all parameters in a Hamiltonian to
                certain values.

        Returns:
            Bound Hamiltonian.

        Raises:
            ValueError: If a Hamiltonian is not of an expected type.
        """
        # PauliSumOp does not allow parametrized coefficients but after binding the parameters
        # we need to convert it into a PauliSumOp for the PauliEvolutionGate.
        if isinstance(hamiltonian, SummedOp):
            op_list = []
            for op in hamiltonian.oplist:
                if hamiltonian_value_dict is not None:
                    op_bound = op.bind_parameters(hamiltonian_value_dict)
                else:
                    op_bound = op
                _is_op_bound(op_bound)
                op_list.append(op_bound)
            return sum(op_list)
        elif isinstance(
            hamiltonian, (PauliOp, OperatorBase)
        ):  # in case there is only a single summand
            if hamiltonian_value_dict is not None:
                op_bound = hamiltonian.bind_parameters(hamiltonian_value_dict)
            else:
                op_bound = hamiltonian

            _is_op_bound(op_bound)
            return op_bound
        else:
            raise ValueError(
                f"Provided a Hamiltonian of an unsupported type: {type(hamiltonian)}. Only "
                f"SummedOp, PauliOp, and OperatorBase base are supported by TrotterQrte."
            )

