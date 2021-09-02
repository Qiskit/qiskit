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
from ctypes import Union
from typing import List

import numpy as np
from scipy.integrate import OdeSolver

from qiskit import QiskitError


class VarQteOdeSolver:
    def __init__(
            self,
            init_params: Union[List, np.ndarray],
            ode_function_generator,
    ):
        """
        Initialize ODE Solver
        Args:
            t: Evolution time.
            init_params: Set of initial parameters for time 0.
        """
        self._init_params = init_params
        self._ode_function_generator = ode_function_generator
        self._ode_function = ode_function_generator.var_qte_ode_function

    def _run(self, evolution_time: float):
        """
        Find numerical solution with ODE Solver.
        """
        ode_solver = OdeSolver(
            self._ode_function, t_bound=evolution_time, t0=0, y0=self._init_params,
            vectorized=False, support_complex=False
        )  # TODO added vectorized, support_complex, how to handle this?
        param_values = None

        _ = ode_solver.fun(ode_solver.t, ode_solver.y)
        while ode_solver.t < evolution_time:
            ode_solver.step()
            if ode_solver.t <= evolution_time:
                _ = ode_solver.fun(ode_solver.t, ode_solver.y)
            param_values = ode_solver.y[:-1]
            if ode_solver.status == "finished":
                break
            elif ode_solver.status == "failed":
                raise QiskitError("ODESolver failed.")

        return param_values
