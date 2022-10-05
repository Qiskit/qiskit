# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=====================================
Algorithms (:mod:`qiskit.algorithms`)
=====================================
It contains a collection of quantum algorithms, for use with quantum computers, to
carry out research and investigate how to solve problems in different domains on
near-term quantum devices with short depth circuits.

Algorithms configuration includes the use of :mod:`~qiskit.algorithms.optimizers` which
were designed to be swappable sub-parts of an algorithm. Any component and may be exchanged for
a different implementation of the same component type in order to potentially alter the behavior
and outcome of the algorithm.

Quantum algorithms are run via a :class:`~qiskit.algorithms.QuantumInstance`
which must be set with the
desired backend where the algorithm's circuits will be executed and be configured with a number of
compile and runtime parameters controlling circuit compilation and execution. It ultimately uses
`Terra <https://www.qiskit.org/terra>`__ for the actual compilation and execution of the quantum
circuits created by the algorithm and its components.

.. currentmodule:: qiskit.algorithms

Algorithms
==========

It contains a variety of quantum algorithms and these have been grouped by logical function such
as minimum eigensolvers and amplitude amplifiers.


Amplitude Amplifiers
--------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   AmplificationProblem
   AmplitudeAmplifier
   Grover
   GroverResult


Amplitude Estimators
--------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   AmplitudeEstimator
   AmplitudeEstimatorResult
   AmplitudeEstimation
   AmplitudeEstimationResult
   EstimationProblem
   FasterAmplitudeEstimation
   FasterAmplitudeEstimationResult
   IterativeAmplitudeEstimation
   IterativeAmplitudeEstimationResult
   MaximumLikelihoodAmplitudeEstimation
   MaximumLikelihoodAmplitudeEstimationResult


Eigen Solvers
-------------

Algorithms to find eigenvalues of an operator. For chemistry these can be used to find excited
states of a molecule, and qiskit-nature has some algorithms that leverage chemistry specific
knowledge to do this in that application domain. These algorithms are pending deprecation.
One should instead make use of the Eigensolver classes in the section below, which leverage
Runtime primitives.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Eigensolver
   EigensolverResult

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   NumPyEigensolver
   VQD
   VQDResult

Eigensolvers
------------

Algorithms to find eigenvalues of an operator. For chemistry these can be used to find excited
states of a molecule, and qiskit-nature has some algorithms that leverage chemistry specific
knowledge to do this in that application domain.

.. autosummary::
   :toctree: ../stubs/

   eigensolvers

Evolvers
--------

Pending deprecation: This package has been superseded by the package below. It will be
deprecated in a future release and subsequently removed after that:

`Time Evolvers`_

Algorithms to evolve quantum states in time. Both real and imaginary time evolution is possible
with algorithms that support them. For machine learning, Quantum Imaginary Time Evolution might be
used to train Quantum Boltzmann Machine Neural Networks for example.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    RealEvolver
    ImaginaryEvolver
    TrotterQRTE
    EvolutionResult
    EvolutionProblem


Time Evolvers
-------------

Primitives-enabled algorithms to evolve quantum states in time. Both real and imaginary time
evolution is possible with algorithms that support them. For machine learning, Quantum Imaginary
Time Evolution might be used to train Quantum Boltzmann Machine Neural Networks for example.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    RealTimeEvolver
    ImaginaryTimeEvolver
    PVQD
    PVQDResult
    TimeEvolutionResult
    TimeEvolutionProblem


Trotterization-based Quantum Real Time Evolution
------------------------------------------------

Package for primitives-enabled Trotterization-based quantum time evolution algorithm - TrotterQRTE.

.. autosummary::
   :toctree: ../stubs/

   time_evolvers.trotterization


Factorizers
-----------

Algorithms to find factors of a number.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Shor
   ShorResult


Gradients
----------

Algorithms to calculate the gradient of a quantum circuit.

.. autosummary::
   :toctree: ../stubs/

   gradients


Linear Solvers
--------------

Algorithms to solve linear systems of equations.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   linear_solvers


Minimum Eigen Solvers
---------------------

Algorithms that can find the minimum eigenvalue of an operator.
These algorithms are pending deprecation. One should instead make use of the
Minimum Eigensolver classes in the section below, which leverage Runtime primitives.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MinimumEigensolver
   MinimumEigensolverResult

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   NumPyMinimumEigensolver
   QAOA
   VQE

Minimum Eigensolvers
--------------------

Algorithms that can find the minimum eigenvalue of an operator and leverage primitives.

.. autosummary::
   :toctree: ../stubs/

   minimum_eigensolvers


Optimizers
----------

Classical optimizers for use by quantum variational algorithms.

.. autosummary::
   :toctree: ../stubs/

   optimizers


Phase Estimators
----------------

Algorithms that estimate the phases of eigenstates of a unitary.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   HamiltonianPhaseEstimation
   HamiltonianPhaseEstimationResult
   PhaseEstimationScale
   PhaseEstimation
   PhaseEstimationResult
   IterativePhaseEstimation


State Fidelities
----------------

Algorithms that compute the fidelity of pairs of quantum states.

.. autosummary::
   :toctree: ../stubs/

   state_fidelities


Exceptions
----------

.. autosummary::
   :toctree: ../stubs/

   AlgorithmError


Utility methods
---------------

Utility methods used by algorithms.

.. autosummary::
   :toctree: ../stubs/

   eval_observables
   estimate_observables


Utility classes
---------------

Utility classes used by algorithms (mainly for type-hinting purposes).

.. autosummary::
   :toctree: ../stubs/

   AlgorithmJob

"""
from .algorithm_job import AlgorithmJob
from .algorithm_result import AlgorithmResult
from .evolvers import EvolutionResult, EvolutionProblem
from .evolvers.real_evolver import RealEvolver
from .evolvers.imaginary_evolver import ImaginaryEvolver
from .time_evolvers.imaginary_time_evolver import ImaginaryTimeEvolver
from .time_evolvers.real_time_evolver import RealTimeEvolver
from .time_evolvers.time_evolution_problem import TimeEvolutionProblem
from .time_evolvers.time_evolution_result import TimeEvolutionResult
from .variational_algorithm import VariationalAlgorithm, VariationalResult
from .amplitude_amplifiers import Grover, GroverResult, AmplificationProblem, AmplitudeAmplifier
from .amplitude_estimators import (
    AmplitudeEstimator,
    AmplitudeEstimatorResult,
    AmplitudeEstimation,
    AmplitudeEstimationResult,
    FasterAmplitudeEstimation,
    FasterAmplitudeEstimationResult,
    IterativeAmplitudeEstimation,
    IterativeAmplitudeEstimationResult,
    MaximumLikelihoodAmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimationResult,
    EstimationProblem,
)
from .eigen_solvers import NumPyEigensolver, Eigensolver, EigensolverResult, VQD, VQDResult
from .factorizers import Shor, ShorResult
from .linear_solvers import HHL, LinearSolver, NumPyLinearSolver, LinearSolverResult
from .minimum_eigen_solvers import (
    VQE,
    VQEResult,
    QAOA,
    NumPyMinimumEigensolver,
    MinimumEigensolver,
    MinimumEigensolverResult,
)
from .phase_estimators import (
    HamiltonianPhaseEstimation,
    HamiltonianPhaseEstimationResult,
    PhaseEstimationScale,
    PhaseEstimation,
    PhaseEstimationResult,
    IterativePhaseEstimation,
)
from .exceptions import AlgorithmError
from .aux_ops_evaluator import eval_observables
from .observables_evaluator import estimate_observables
from .evolvers.trotterization import TrotterQRTE

from .time_evolvers.pvqd import PVQD, PVQDResult

__all__ = [
    "AlgorithmJob",
    "AlgorithmResult",
    "VariationalAlgorithm",
    "VariationalResult",
    "AmplitudeAmplifier",
    "AmplificationProblem",
    "Grover",
    "GroverResult",
    "AmplitudeEstimator",
    "AmplitudeEstimatorResult",
    "AmplitudeEstimation",
    "AmplitudeEstimationResult",
    "FasterAmplitudeEstimation",
    "FasterAmplitudeEstimationResult",
    "IterativeAmplitudeEstimation",
    "IterativeAmplitudeEstimationResult",
    "MaximumLikelihoodAmplitudeEstimation",
    "MaximumLikelihoodAmplitudeEstimationResult",
    "EstimationProblem",
    "NumPyEigensolver",
    "RealEvolver",
    "ImaginaryEvolver",
    "RealTimeEvolver",
    "ImaginaryTimeEvolver",
    "TrotterQRTE",
    "EvolutionResult",
    "EvolutionProblem",
    "TimeEvolutionResult",
    "TimeEvolutionProblem",
    "LinearSolverResult",
    "Eigensolver",
    "EigensolverResult",
    "Shor",
    "ShorResult",
    "VQE",
    "VQEResult",
    "QAOA",
    "LinearSolver",
    "HHL",
    "NumPyLinearSolver",
    "NumPyMinimumEigensolver",
    "MinimumEigensolver",
    "MinimumEigensolverResult",
    "HamiltonianPhaseEstimation",
    "HamiltonianPhaseEstimationResult",
    "VQD",
    "VQDResult",
    "PhaseEstimationScale",
    "PhaseEstimation",
    "PhaseEstimationResult",
    "PVQD",
    "PVQDResult",
    "IterativePhaseEstimation",
    "AlgorithmError",
    "eval_observables",
    "estimate_observables",
]
