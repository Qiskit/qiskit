# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Solvers (:mod:`qiskit.algorithms.time_evolvers.variational.solvers`)
====================================================================

This package contains the necessary classes to solve systems of equations arising in the
Variational Quantum Time Evolution. They include ordinary differential equations (ODE) which
describe ansatz parameter propagation and systems of linear equations.


Systems of Linear Equations Solver
----------------------------------

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

    VarQTELinearSolver


ODE Solver
----------
.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

    VarQTEOdeSolver
"""

from qiskit.algorithms.time_evolvers.variational.solvers.ode.var_qte_ode_solver import (
    VarQTEOdeSolver,
)
from qiskit.algorithms.time_evolvers.variational.solvers.var_qte_linear_solver import (
    VarQTELinearSolver,
)

__all__ = ["VarQTELinearSolver", "VarQTEOdeSolver"]
