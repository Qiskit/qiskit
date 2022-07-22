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
It  contains classical and quantum algorithms to solve systems of linear equations such as
:class:`~qiskit.algorithms.HHL`.
Although the quantum algorithm accepts a general Hermitian matrix as input, Qiskit's default
Hamiltonian evolution is exponential in such cases and therefore the quantum linear solver will
not achieve an exponential speedup.
Furthermore, the quantum algorithm can find a solution exponentially faster in the size of the
system than their classical counterparts (i.e. logarithmic complexity instead of polynomial),
meaning that reading the full solution vector would kill such speedup (since this would take
linear time in the size of the system).
Therefore, to achieve an exponential speedup we can only compute functions from the solution
vector (the so called observables) to learn information about the solution.
Known efficient implementations of Hamiltonian evolutions or observables are contained in the
following subfolders:

`Matrices`_
  A placeholder for efficient implementations of the Hamiltonian evolution of particular types of
  matrices.

`Observables`_
  A placeholder for efficient implementations of functions that can be computed from the solution
  vector to a system of linear equations.

.. currentmodule:: qiskit.algorithms.linear_solvers

Linear Solvers
==============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   LinearSolver
   LinearSolverResult
   HHL
   NumPyLinearSolver

Matrices
========

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   LinearSystemMatrix
   NumPyMatrix
   TridiagonalToeplitz

Observables
===========

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
