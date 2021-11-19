# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Linear solvers (:mod:`qiskit.algorithms.linear_solvers`)
=========================================================
It  contains a variety of classical optimizers for use by quantum variational algorithms,
such as :class:`~qiskit.algorithms.VQE`.
Logically, these optimizers can be divided into two categories:

`Local Optimizers`_
  Given an optimization problem, a **local optimizer** is a function
  that attempts to find an optimal value within the neighboring set of a candidate solution.

`Global Optimizers`_
  Given an optimization problem, a **global optimizer** is a function
  that attempts to find an optimal value among all possible solutions.

.. currentmodule:: qiskit.algorithms.optimizers

Linear Solvers
====================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   LinearSolver
   LinearSolverResult
   HHL
   NumPyLinearSolver

Matrices
================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   LinearSystemMatrix
   NumPyMatrix
   TridiagonalToeplitz

Observables
=================

.. autosummary::
    :toctree: ../stubs/

    LinearSystemObservable
    AbsoluteAverage
    MatrixFunctional

"""

from .hhl import HHL
from .numpy_linear_solver import NumPyLinearSolver
from .linear_solver import LinearSolver, LinearSolverResult
from .matrices import LinearSystemMatrix, NumPyMatrix, TridiagonalToeplitz
from .observables import LinearSystemObservable, AbsoluteAverage, MatrixFunctional

__all__ = [
    "HHL",
    "NumPyLinearSolver",
    "LinearSolver",
    "LinearSolverResult",
    "LinearSystemMatrix",
    "NumPyMatrix",
    "TridiagonalToeplitz",
    "LinearSystemObservable",
    "AbsoluteAverage",
    "MatrixFunctional",
]
