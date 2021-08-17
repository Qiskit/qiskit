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


class OdeSolver:
    def __init__(self, t: float, init_params: Union[List, np.ndarray]):
        """
        Initialize ODE Solver
        Args:
            t: Evolution time.
            init_params: Set of initial parameters for time 0.
        """
        self._t = t
        self._init_params = init_params

    def _run(self, t: float, init_params: Union[List, np.ndarray]):
        """
        Find numerical solution with ODE Solver
        Args:
            t: Evolution time
            init_params: Set of initial parameters for time 0
        """
        self._init_ode_solver(t, np.append(init_params, 0))
        if isinstance(self._ode_solver, OdeSolver):
            self._store_now = True
            _ = self._ode_solver.fun(self._ode_solver.t, self._ode_solver.y)
            while self._ode_solver.t < t:
                self._store_now = False
                self._ode_solver.step()
                if self._snapshot_dir is not None and self._ode_solver.t <= t:
                    self._store_now = True
                    _ = self._ode_solver.fun(self._ode_solver.t, self._ode_solver.y)
                print("ode time", self._ode_solver.t)
                param_values = self._ode_solver.y[:-1]
                print("ode parameters", self._ode_solver.y[:-1])
                print("ode error", self._ode_solver.y[-1])
                print("ode step size", self._ode_solver.step_size)
                if self._ode_solver.status == "finished":
                    break
                elif self._ode_solver.status == "failed":
                    raise Warning("ODESolver failed")

        else:
            raise TypeError("Please provide a scipy ODESolver or ode type object.")
        print("Parameter Values ", param_values)

        return param_values
