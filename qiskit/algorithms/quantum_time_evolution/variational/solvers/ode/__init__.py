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

"""
Solvers (:mod:`qiskit.algorithms.quantum_time_evolution.variational.solvers.ode`)
=====================================================
This package contains the necessary classes to solve ordinary differential equations (ODE) which
describe ansatz parameter propagation that implements a variational time evolution. An ODE salver
uses one of ODE functions generators. OdeFunctionGenerator utilizes a natural gradient and
ErrorBasedOdeFunctionGenerator uses a calculated error to steer the evolution.

ODE functions generators
====================
.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

    AbstractOdeFunctionGenerator
    OdeFunctionGenerator
    ErrorBasedOdeFunctionGenerator

ODE Solver
================
.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

    VarQteOdeSolver
"""
