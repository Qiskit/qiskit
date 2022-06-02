# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class for a Real McLachlan's Variational Principle."""

from typing import Union, Dict, List, Callable, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import (
    StateFn,
    SummedOp,
    Y,
    I,
    PauliExpectation,
    CircuitSampler,
    OperatorBase,
    QFI,
)
from ..calculators import (
    evolution_grad_calculator,
)
from .real_variational_principle import (
    RealVariationalPrinciple,
)


class RealMcLachlanPrinciple(RealVariationalPrinciple):
    """Class for an Imaginary McLachlan's Variational Principle. It aims to minimize the distance
    between both sides of the Schrödinger equation with a quantum state given as a parametrized
    trial state. The principle leads to a system of linear equations handled by the
    `~qiskit.algorithms.evolvers.variational.solvers.VarQTELinearSolver` class. The real variant
    means that we consider real time dynamics.
    """

    def create_qfi(
        self,
    ) -> QFI:
        """
        Creates a QFI instance according to the rules of this variational principle. It is used
        to calculate a metric tensor required in the ODE.

        Returns:
            QFI instance.
        """
        return QFI(self._qfi_method)

    def calc_evolution_grad(
        self,
        hamiltonian: OperatorBase,
        ansatz: Union[StateFn, QuantumCircuit],
        parameters: List[Parameter],
    ) -> Callable:
        """
        Calculates an evolution gradient according to the rules of this variational principle.

        Args:
            hamiltonian: Hamiltonian for which an evolution gradient should be calculated.
            ansatz: Quantum state in the form of a parametrized quantum circuit to be used for
                calculating an evolution gradient.
            parameters: Parameters with respect to which gradients should be computed.

        Returns:
            Transformed evolution gradient.
        """

        def raw_evolution_grad_imag(
            param_dict: Dict[Parameter, complex],
            circuit_sampler: Optional[CircuitSampler] = None,
        ) -> OperatorBase:
            """
            Args:
                param_dict: Dictionary which relates parameter values to the parameters in the
                    ansatz.
                circuit_sampler: Samples circuits using an underlying backend.

            Returns:
                Calculated evolution gradient, according to the variational principle.
            """

            energy = StateFn(hamiltonian, is_measurement=True) @ StateFn(ansatz)
            energy = PauliExpectation().convert(energy)

            if circuit_sampler is not None:
                energy = circuit_sampler.convert(energy, param_dict).eval()
            else:
                energy = energy.assign_parameters(param_dict).eval()

            energy_term = I ^ hamiltonian.num_qubits
            energy_term *= -1
            energy_term *= energy
            hamiltonian_ = SummedOp([hamiltonian, energy_term]).reduce()
            basis_operator = -Y
            grad = (
                evolution_grad_calculator.calculate(
                    hamiltonian_, ansatz, parameters, self._grad_method, basis=basis_operator
                )
                * 0.5
            )  # Im(...)
            return grad

        return raw_evolution_grad_imag
