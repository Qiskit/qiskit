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

Algorithms are run via a :class:`~qiskit.aqua.QuantumInstance` which must be set with the desired
backend where the algorithm's circuits will be executed and be configured with a number of compile
and runtime parameters controlling circuit compilation and execution. Aqua ultimately uses
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

Quantum Algorithms
==================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   VQE
   QAOA
   VQC
   QGAN
   EOH
   QSVM
   Grover
   IQPE
   QPE
   AmplitudeEstimation
   IterativeAmplitudeEstimation
   MaximumLikelihoodAmplitudeEstimation
   Simon
   DeutschJozsa
   BernsteinVazirani
   HHL
   Shor

Classical Algorithms
====================
Aqua includes some classical algorithms. While these algorithms do not use a quantum device or
simulator, and rely on purely classical approaches, they may be useful in the near term to
generate reference values while experimenting with, developing and testing quantum algorithms.

The algorithms are designed to take the same input data as the quantum algorithms so that
behavior, data validity and output can be evaluated and compared to a quantum result.

Note: The :class:`ClassicalCPLEX` algorithm requires `IBM ILOG CPLEX Optimization Studio
<https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/COS_KC_home.html>`__
and its Python API to be installed. See the following for more information:

.. toctree::
   :maxdepth: 1

   qiskit.aqua.algorithms.minimum_eigen_solvers.cplex

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ClassicalEigensolver
   ClassicalMinimumEigensolver
   ClassicalLSsolver
   ClassicalSVM
   ClassicalCPLEX

"""

from .algorithm_result import AlgorithmResult
from .quantum_algorithm import QuantumAlgorithm
from .classical_algorithm import ClassicalAlgorithm
from .vq_algorithm import VQAlgorithm
from .amplitude_amplifiers import Grover
from .amplitude_estimators import (AmplitudeEstimation,
                                   IterativeAmplitudeEstimation,
                                   MaximumLikelihoodAmplitudeEstimation)
from .classifiers import VQC, QSVM, ClassicalSVM, SVM_Classical
from .distribution_learners import QGAN
from .eigen_solvers import ClassicalEigensolver, ExactEigensolver, EigensolverResult
from .factorizers import Shor
from .linear_solvers import HHL, ClassicalLSsolver, ExactLSsolver
from .minimum_eigen_solvers import (VQE, QAOA, IQPE, QPE, ClassicalCPLEX, CPLEX_Ising,
                                    ClassicalMinimumEigensolver, MinimumEigensolverResult)
from .education import EOH, Simon, DeutschJozsa, BernsteinVazirani

__all__ = [
    'AlgorithmResult',
    'QuantumAlgorithm',
    'VQE',
    'QAOA',
    'VQC',
    'QGAN',
    'ClassicalAlgorithm',
    'VQAlgorithm',
    'ClassicalEigensolver',
    'ExactEigensolver',
    'ClassicalLSsolver',
    'EigensolverResult',
    'ExactLSsolver',
    'ClassicalMinimumEigensolver',
    'MinimumEigensolverResult',
    'ClassicalSVM',
    'SVM_Classical',
    'ClassicalCPLEX',
    'CPLEX_Ising',
    'EOH',
    'QSVM',
    'Grover',
    'IQPE',
    'QPE',
    'AmplitudeEstimation',
    'IterativeAmplitudeEstimation',
    'MaximumLikelihoodAmplitudeEstimation',
    'Simon',
    'DeutschJozsa',
    'BernsteinVazirani',
    'HHL',
    'Shor',
]
