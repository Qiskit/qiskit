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
"""
Variational Quantum Time Evolutions (:mod:`qiskit.algorithms.evolvers.variational`)
===================================================================================

Algorithms for performing Variational Quantum Time Evolution of quantum states,
which can be tailored to near-term devices.
:class:`~qiskit.algorithms.evolvers.variational.VarQTE` base class exposes an interface, compliant
with the Quantum Time Evolution Framework in Qiskit Terra, that is implemented by
:class:`~qiskit.algorithms.VarQRTE` and :class:`~qiskit.algorithms.VarQITE` classes for real and
imaginary time evolution respectively. The variational approach is taken according to a variational
principle chosen by a user.

Examples:

.. code-block::

    from qiskit import BasicAer
    from qiskit.circuit.library import EfficientSU2
    from qiskit.opflow import SummedOp, I, Z, Y, X
    from qiskit.algorithms.evolvers.variational import (
        ImaginaryMcLachlanPrinciple,
    )
    from qiskit.algorithms import EvolutionProblem
    from qiskit.algorithms import VarQITE

    # define a Hamiltonian
    observable = SummedOp(
        [
            0.2252 * (I ^ I),
            0.5716 * (Z ^ Z),
            0.3435 * (I ^ Z),
            -0.4347 * (Z ^ I),
            0.091 * (Y ^ Y),
            0.091 * (X ^ X),
        ]
    ).reduce()

    # define a parametrized initial state to be evolved

    ansatz = EfficientSU2(observable.num_qubits, reps=1)
    parameters = ansatz.parameters

    # define values of initial parameters
    init_param_values = np.zeros(len(ansatz.parameters))
    for i in range(len(ansatz.parameters)):
        init_param_values[i] = np.pi / 2
    param_dict = dict(zip(parameters, init_param_values))

    # define a variational principle
    var_principle = ImaginaryMcLachlanPrinciple()

    # optionally define a backend
    backend = BasicAer.get_backend("statevector_simulator")

    # define evolution time
    time = 1

    # define evolution problem
    evolution_problem = EvolutionProblem(observable, time)

    # instantiate the algorithm
    var_qite = VarQITE(ansatz, var_principle, param_dict, quantum_instance=backend)

    # run the algorithm/evolve the state
    evolution_result = var_qite.evolve(evolution_problem)

.. currentmodule:: qiskit.algorithms.evolvers.variational

Variational Principles
----------------------

Variational principles can be used to simulate quantum time evolution by propagating the parameters
of a parameterized quantum circuit.

They can be divided into two categories:

    1) Variational Quantum Imaginary Time Evolution
        Given a Hamiltonian, a time and a variational ansatz, the variational principle describes a
        variational principle according to the normalized Wick-rotated Schroedinger equation.

    2) Variational Quantum Real Time Evolution
        Given a Hamiltonian, a time and a variational ansatz, the variational principle describes a
        variational principle according to the Schroedinger equation.

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
from .var_qte import VarQTE
from .variational_principles.variational_principle import VariationalPrinciple
from .variational_principles import RealVariationalPrinciple, ImaginaryVariationalPrinciple
from .variational_principles.imaginary_mc_lachlan_principle import (
    ImaginaryMcLachlanPrinciple,
)
from .variational_principles.real_mc_lachlan_principle import (
    RealMcLachlanPrinciple,
)


__all__ = [
    "ForwardEulerSolver",
    "VarQTE",
    "VariationalPrinciple",
    "RealVariationalPrinciple",
    "ImaginaryVariationalPrinciple",
    "RealMcLachlanPrinciple",
    "ImaginaryMcLachlanPrinciple",
]
