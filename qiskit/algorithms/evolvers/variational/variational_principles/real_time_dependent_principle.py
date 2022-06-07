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

"""Class for a Real Time Dependent Variational Principle."""

from qiskit.opflow import Y, StateFn, QFI, Gradient
from qiskit.opflow.gradients.circuit_gradients import LinComb
from qiskit.opflow.gradients.circuit_qfis import LinCombFull
from .real_variational_principle import (
    RealVariationalPrinciple,
)


class RealTimeDependentPrinciple(RealVariationalPrinciple):
    """Class for a Real Time Dependent Variational Principle. It works by evaluating the Lagrangian
    corresponding the given system at a parametrized trial state and applying the Euler-Lagrange
    equation. The principle leads to a system of linear equations handled by the
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
        qfi_method = self._qfi_method
        if self._qfi_method == "lin_comb_full" or isinstance(self._qfi_method, LinCombFull):
            qfi_method = LinCombFull(aux_meas_op=-Y)

        return QFI(qfi_method)

    def calc_evolution_grad(
        self,
    ) -> Gradient:
        """
        Calculates an evolution gradient according to the rules of this variational principle.

        Returns:
            Transformed evolution gradient.
        """
        if self._grad_method == "lin_comb":
            self._grad_method = LinComb()
        evolution_grad_real = Gradient(self._grad_method)  # *0.5

        return evolution_grad_real

    def modify_hamiltonian(self, hamiltonian, ansatz, circuit_sampler, param_dict):
        return StateFn(hamiltonian, is_measurement=True) @ StateFn(ansatz)
