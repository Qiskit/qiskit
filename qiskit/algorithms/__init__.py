# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Algorithms (:mod:`qiskit.algorithms`)
==========================================
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
++++++++++++++++++++

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   AmplificationProblem
   Grover
   GroverResult

Amplitude Estimators
++++++++++++++++++++

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

Eigensolvers
++++++++++++
Algorithms to find eigenvalues of an operator. For chemistry these can be used to find excited
states of a molecule and qiskit.chemistry has some algorithms that leverage chemistry specific
knowledge to do this in that application domain.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Eigensolver
   EigensolverResult

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   NumPyEigensolver

Factorizers
+++++++++++
Algorithms to find factors of a number.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Shor
   ShorResult

Linear Solvers
++++++++++++++
Algorithms to solve linear systems of equations.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   HHL
   NumPyLinearSolver
   LinearSolver
   LinearSolverResult

Minimum Eigensolvers
++++++++++++++++++++
Algorithms that can find the minimum eigenvalue of an operator.

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

Optimizers
++++++++++
Classical optimizers for use by quantum variational algorithms.

.. autosummary::
   :toctree:

   optimizers

Phase Estimators
++++++++++++++++
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

Exceptions
==========

.. autosummary::
   :toctree: ../stubs/

   AlgorithmError
"""

from .algorithm_result import AlgorithmResult
from .variational_algorithm import VariationalAlgorithm, VariationalResult
from .amplitude_amplifiers import Grover, GroverResult, AmplificationProblem
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
from .eigen_solvers import NumPyEigensolver, Eigensolver, EigensolverResult
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

__all__ = [
    "AlgorithmResult",
    "VariationalAlgorithm",
    "VariationalResult",
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
    "PhaseEstimationScale",
    "PhaseEstimation",
    "PhaseEstimationResult",
    "IterativePhaseEstimation",
    "AlgorithmError",
]
