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
Variational Quantum Time Evolutions (:mod:`qiskit.algorithms.time_evolvers.variational`)
========================================================================================

Algorithms for performing Variational Quantum Time Evolution of quantum states,
which can be tailored to near-term devices.
:class:`~qiskit.algorithms.time_evolvers.variational.VarQTE` base class exposes an interface, compliant
with the Quantum Time Evolution Framework in Qiskit Terra, that is implemented by
:class:`~qiskit.algorithms.VarQRTE` and :class:`~qiskit.algorithms.VarQITE` classes for real and
imaginary time evolution respectively. The variational approach is taken according to a variational
principle chosen by a user.

Example:

    .. code-block:: python

        import numpy as np

        from qiskit.algorithms import TimeEvolutionProblem, VarQITE
        from qiskit.algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple
        from qiskit.circuit.library import EfficientSU2
        from qiskit.quantum_info import SparsePauliOp

        observable = SparsePauliOp.from_list(
            [
                ("II", 0.2252),
                ("ZZ", 0.5716),
                ("IZ", 0.3435),
                ("ZI", -0.4347),
                ("YY", 0.091),
                ("XX", 0.091),
            ]
        )

        ansatz = EfficientSU2(observable.num_qubits, reps=1)
        init_param_values = np.zeros(len(ansatz.parameters))
        for i in range(len(ansatz.parameters)):
            init_param_values[i] = np.pi / 2
        var_principle = ImaginaryMcLachlanPrinciple()
        time = 1
        evolution_problem = TimeEvolutionProblem(observable, time)
        var_qite = VarQITE(ansatz, var_principle, init_param_values)
        evolution_result = var_qite.evolve(evolution_problem)

.. currentmodule:: qiskit.algorithms.time_evolvers.variational

Variational Principles
----------------------

With variational principles we can project time evolution of a quantum state
onto the parameters of a model, in our case a variational quantum circuit.

They can be divided into two categories: Variational Quantum _Real_ Time Evolution, which evolves
the variational ansatz under the standard Schroediger equation and
Variational Quantum _Imaginary_ Time Evolution, which evolves under the normalized
Wick-rotated Schroedinger equation.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

    VariationalPrinciple
    RealVariationalPrinciple
    ImaginaryVariationalPrinciple
    RealMcLachlanPrinciple
    ImaginaryMcLachlanPrinciple

ODE solvers
-----------
ODE solvers that implement the SciPy ODE Solver interface. The Forward Euler Solver is
a preferred choice in the presence of noise. One might also use solvers provided by SciPy directly,
e.g. RK45.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   ForwardEulerSolver

"""
from .solvers.ode.forward_euler_solver import ForwardEulerSolver
from .var_qrte import VarQRTE
from .var_qite import VarQITE

from .var_qte import VarQTE
from .var_qte_result import VarQTEResult
from .variational_principles import (
    VariationalPrinciple,
    RealVariationalPrinciple,
    ImaginaryVariationalPrinciple,
    ImaginaryMcLachlanPrinciple,
    RealMcLachlanPrinciple,
)

__all__ = [
    "ForwardEulerSolver",
    "VarQTE",
    "VarQTEResult",
    "VariationalPrinciple",
    "RealVariationalPrinciple",
    "ImaginaryVariationalPrinciple",
    "RealMcLachlanPrinciple",
    "ImaginaryMcLachlanPrinciple",
    "VarQITE",
    "VarQRTE",
]
