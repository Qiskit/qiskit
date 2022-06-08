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
from typing import Union

import numpy as np

from qiskit.opflow import (
    StateFn,
    SummedOp,
    Y,
    I,
    PauliExpectation,
    CircuitQFI,
)
from qiskit.opflow.gradients.circuit_gradients import LinComb
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

    def __init__(
        self,
        qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
    ) -> None:
        """
        Args:
            qfi_method: The method used to compute the QFI. Can be either
                ``'lin_comb_full'`` or ``'overlap_block_diag'`` or ``'overlap_diag'`` or
                ``CircuitQFI``.
        """
        self._grad_method = LinComb(aux_meas_op=-Y)

        super().__init__(qfi_method)

    def calc_evolution_grad(
        self,
        hamiltonian,
        ansatz,
        circuit_sampler,
        param_dict,
        bind_params,
        gradient_params,
        quantum_instance,
        param_values,
    ) -> np.ndarray:
        """
        Calculates an evolution gradient according to the rules of this variational principle.

        Returns:
            Transformed evolution gradient.
        """

        # TODO quick fix for a bug that will be addressed in another PR
        return (-1) * super().calc_evolution_grad(
            hamiltonian,
            ansatz,
            circuit_sampler,
            param_dict,
            bind_params,
            gradient_params,
            quantum_instance,
            param_values,
        )

    def modify_hamiltonian(self, hamiltonian, ansatz, circuit_sampler, param_dict):
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
        operator = StateFn(hamiltonian_, is_measurement=True) @ StateFn(ansatz)
        return operator
