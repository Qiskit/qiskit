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

.. deprecated:: 0.25.0

    The :mod:`qiskit.algorithms` module has been migrated to an independent package:
    https://github.com/qiskit-community/qiskit-algorithms.
    The current import path is deprecated and will be removed no earlier
    than 3 months after the release date. If your code uses primitives, you can run
    ``pip install qiskit_algorithms`` and import ``from qiskit_algorithms`` instead.
    If you use opflow/quantum instance-based algorithms, please update your code to
    use primitives following: https://qisk.it/algo_migration before migrating to
    the new package.

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


Eigensolvers
------------

Algorithms to find eigenvalues of an operator. For chemistry these can be used to find excited
states of a molecule, and ``qiskit-nature`` has some algorithms that leverage chemistry specific
knowledge to do this in that application domain.

Primitive-based Eigensolvers
++++++++++++++++++++++++++++

These algorithms are based on the Qiskit Primitives, a new execution paradigm that replaces the use
of :class:`.QuantumInstance` in algorithms. To ensure continued support and development, we recommend
using the primitive-based Eigensolvers in place of the legacy :class:`.QuantumInstance`-based ones.

.. autosummary::
   :toctree: ../stubs/

   eigensolvers


Legacy Eigensolvers
+++++++++++++++++++

These algorithms, still based on the :class:`.QuantumInstance`, are superseded
by the primitive-based versions in the section above but are still supported for now.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Eigensolver
   EigensolverResult
   NumPyEigensolver
   VQD
   VQDResult


Time Evolvers
-------------

Algorithms to evolve quantum states in time. Both real and imaginary time evolution is possible
with algorithms that support them. For machine learning, Quantum Imaginary Time Evolution might be
used to train Quantum Boltzmann Machine Neural Networks for example.

Primitive-based Time Evolvers
+++++++++++++++++++++++++++++

These algorithms are based on the Qiskit Primitives, a new execution paradigm that replaces the use
of :class:`.QuantumInstance` in algorithms. To ensure continued support and development, we recommend
using the primitive-based Time Evolvers in place of the legacy :class:`.QuantumInstance`-based ones.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   RealTimeEvolver
   ImaginaryTimeEvolver
   TimeEvolutionResult
   TimeEvolutionProblem
   PVQD
   PVQDResult
   SciPyImaginaryEvolver
   SciPyRealEvolver
   VarQITE
   VarQRTE

Legacy Time Evolvers
++++++++++++++++++++

These algorithms, still based on the :class:`.QuantumInstance`, are superseded
by the primitive-based versions in the section above but are still supported for now.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    RealEvolver
    ImaginaryEvolver
    TrotterQRTE
    EvolutionResult
    EvolutionProblem


Variational Quantum Time Evolution
++++++++++++++++++++++++++++++++++

Classes used by variational quantum time evolution algorithms - :class:`.VarQITE` and
:class:`.VarQRTE`.

.. autosummary::
   :toctree: ../stubs/

   time_evolvers.variational


Trotterization-based Quantum Real Time Evolution
++++++++++++++++++++++++++++++++++++++++++++++++

Package for primitives-enabled Trotterization-based quantum time evolution
algorithm - :class:`~.time_evolvers.TrotterQRTE`.

.. autosummary::
   :toctree: ../stubs/

   time_evolvers.trotterization


Gradients
----------

Algorithms to calculate the gradient of a quantum circuit.

.. autosummary::
   :toctree: ../stubs/

   gradients


Minimum Eigensolvers
---------------------

Algorithms that can find the minimum eigenvalue of an operator.

Primitive-based Minimum Eigensolvers
++++++++++++++++++++++++++++++++++++

These algorithms are based on the Qiskit Primitives, a new execution paradigm that replaces the use
of :class:`.QuantumInstance` in algorithms. To ensure continued support and development, we recommend
using the primitive-based Minimum Eigensolvers in place of the legacy :class:`.QuantumInstance`-based
ones.

.. autosummary::
   :toctree: ../stubs/

   minimum_eigensolvers


Legacy Minimum Eigensolvers
+++++++++++++++++++++++++++

These algorithms, still based on the :class:`.QuantumInstance`, are superseded
by the primitive-based versions in the section above but are still supported for now.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MinimumEigensolver
   MinimumEigensolverResult
   NumPyMinimumEigensolver
   QAOA
   VQE


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

.. autoexception:: AlgorithmError

Utility classes
---------------

Utility classes used by algorithms (mainly for type-hinting purposes).

.. autosummary::
   :toctree: ../stubs/

   AlgorithmJob

Utility functions
-----------------

Utility functions used by algorithms.

.. autofunction:: eval_observables
.. autofunction:: estimate_observables

"""
import warnings

from .algorithm_job import AlgorithmJob
from .algorithm_result import AlgorithmResult
from .evolvers import EvolutionResult, EvolutionProblem
from .evolvers.real_evolver import RealEvolver
from .evolvers.imaginary_evolver import ImaginaryEvolver
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

from .time_evolvers import (
    ImaginaryTimeEvolver,
    RealTimeEvolver,
    TimeEvolutionProblem,
    TimeEvolutionResult,
    PVQD,
    PVQDResult,
    SciPyImaginaryEvolver,
    SciPyRealEvolver,
    VarQITE,
    VarQRTE,
    VarQTE,
    VarQTEResult,
)

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
    "Eigensolver",
    "EigensolverResult",
    "VQE",
    "VQEResult",
    "QAOA",
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
    "SciPyRealEvolver",
    "SciPyImaginaryEvolver",
    "IterativePhaseEstimation",
    "AlgorithmError",
    "eval_observables",
    "estimate_observables",
    "VarQITE",
    "VarQRTE",
    "VarQTE",
    "VarQTEResult",
]

warnings.warn(
    "``qiskit.algorithms`` has been migrated to an independent package: "
    "https://github.com/qiskit-community/qiskit-algorithms. "
    "The ``qiskit.algorithms`` import path is deprecated as of qiskit-terra 0.25.0 and "
    "will be removed no earlier than 3 months after the release date. "
    "Please run ``pip install qiskit_algorithms`` and use ``import qiskit_algorithms`` instead.",
    category=DeprecationWarning,
    stacklevel=2,
)
