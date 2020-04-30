# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Algorithms (:mod:`qiskit.aqua.algorithms`)
==========================================
Aqua contains a collection of quantum algorithms, for use with quantum computers, to
carry out research and investigate how to solve problems in different domains on
near-term quantum devices with short depth circuits.

Algorithms configuration includes the use of :mod:`~qiskit.aqua.components` which
were designed to be swappable sub-parts of an algorithm. Any component and may be exchanged for
a different implementation of the same component type in order to potentially alter the behavior
and outcome of the algorithm.

Quantum algorithms are run via a :class:`~qiskit.aqua.QuantumInstance` which must be set with the
desired backend where the algorithm's circuits will be executed and be configured with a number of
compile and runtime parameters controlling circuit compilation and execution. Aqua ultimately uses
`Terra <https://www.qiskit.org/terra>`__ for the actual compilation and execution of the quantum
circuits created by the algorithm and its components.

.. currentmodule:: qiskit.aqua.algorithms

Algorithms Base Class
=====================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QuantumAlgorithm
   ClassicalAlgorithm

Algorithms
==========

Aqua contains a variety of quantum algorithms and these have been grouped by logical function such
as minimum eigensolvers and amplitude amplifiers.

Additionally Aqua includes some classical algorithms. While these algorithms do not use a quantum
device or simulator, and rely on purely classical approaches, they may be useful in the near term
to generate reference values while experimenting with, developing and testing quantum algorithms.

The classical algorithms are designed to take the same input data as the quantum algorithms so that
behavior, data validity and output can be evaluated and compared to a quantum result.

Amplitude Amplifiers
++++++++++++++++++++

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Grover

Amplitude Estimators
++++++++++++++++++++
Algorithms that estimate a value.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   AmplitudeEstimation
   IterativeAmplitudeEstimation
   MaximumLikelihoodAmplitudeEstimation

Classifiers
+++++++++++
Algorithms for data classification.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QSVM
   VQC
   SklearnSVM

Distribution Learners
+++++++++++++++++++++

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QGAN

Education
+++++++++
Algorithms whose main role is educational. These are provided as Aqua algorithms so they can be
run in the same framework but their existence here is principally for educational reasons.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BernsteinVazirani
   DeutschJozsa
   EOH
   Simon

Eigensolvers
++++++++++++
Algorithms to find eigenvalues of an operator. For chemistry these can be used to find excited
states of a molecule and qiskit.chemistry has some algorithms that leverage chemistry specific
knowledge to do this in that application domain.

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

Linear Solvers
++++++++++++++
Algorithms to find solutions for linear equations of equations.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   HHL
   NumPyLSsolver

Minimum Eigensolvers
++++++++++++++++++++
Algorithms that can find the minimum eigenvalue of an operator.

Note: The :class:`ClassicalCPLEX` algorithm requires `IBM ILOG CPLEX Optimization Studio
<https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/COS_KC_home.html>`__
and its Python API to be installed. See the following for more information:

.. toctree::
   :maxdepth: 1

   qiskit.aqua.algorithms.minimum_eigen_solvers.cplex

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MinimumEigensolver
   MinimumEigensolverResult

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ClassicalCPLEX
   IQPE
   NumPyMinimumEigensolver
   QAOA
   QPE
   VQE

"""

from .algorithm_result import AlgorithmResult
from .quantum_algorithm import QuantumAlgorithm
from .classical_algorithm import ClassicalAlgorithm
from .vq_algorithm import VQAlgorithm, VQResult
from .amplitude_amplifiers import Grover
from .amplitude_estimators import (AmplitudeEstimation,
                                   IterativeAmplitudeEstimation,
                                   MaximumLikelihoodAmplitudeEstimation)
from .classifiers import VQC, QSVM, SklearnSVM, SVM_Classical
from .distribution_learners import QGAN
from .eigen_solvers import NumPyEigensolver, ExactEigensolver, EigensolverResult
from .factorizers import Shor
from .linear_solvers import HHL, NumPyLSsolver, ExactLSsolver
from .minimum_eigen_solvers import (VQE, VQEResult, QAOA, IQPE, IQPEResult, QPE, QPEResult,
                                    ClassicalCPLEX, CPLEX_Ising, NumPyMinimumEigensolver,
                                    MinimumEigensolver, MinimumEigensolverResult)
from .education import EOH, Simon, DeutschJozsa, BernsteinVazirani

__all__ = [
    'AlgorithmResult',
    'QuantumAlgorithm',
    'VQE',
    'VQEResult',
    'QAOA',
    'VQC',
    'QGAN',
    'ClassicalAlgorithm',
    'VQAlgorithm',
    'VQResult',
    'NumPyEigensolver',
    'ExactEigensolver',
    'NumPyLSsolver',
    'EigensolverResult',
    'ExactLSsolver',
    'NumPyMinimumEigensolver',
    'MinimumEigensolver',
    'MinimumEigensolverResult',
    'SklearnSVM',
    'SVM_Classical',
    'ClassicalCPLEX',
    'CPLEX_Ising',
    'EOH',
    'QSVM',
    'Grover',
    'IQPE',
    'IQPEResult',
    'QPE',
    'QPEResult',
    'AmplitudeEstimation',
    'IterativeAmplitudeEstimation',
    'MaximumLikelihoodAmplitudeEstimation',
    'Simon',
    'DeutschJozsa',
    'BernsteinVazirani',
    'HHL',
    'Shor',
]
