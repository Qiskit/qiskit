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
near-term quantum devices with short depth circuits. Aqua uses
`Terra <https://www.qiskit.org/terra>`__ for compilation and execution
of the quantum circuits required by the algorithm for the specific problems.

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

Note: The :class:`CPLEX_Ising` algorithm requires `IBM ILOG CPLEX Optimization Studio
<https://www.ibm.com/support/knowledgecenter/SSSA5P_12.10.0/COS_KC_home.html>`__
and its Python API to be installed. See the following for more information:

.. toctree::
   :maxdepth: 1

   qiskit.aqua.algorithms.classical.cplex

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ExactEigensolver
   ExactLSsolver
   SVM_Classical
   CPLEX_Ising

"""

from .quantum_algorithm import QuantumAlgorithm
from .adaptive import VQE, QAOA, VQC, QGAN
from .classical import (ClassicalAlgorithm, ExactEigensolver, ExactLSsolver,
                        SVM_Classical, CPLEX_Ising)
from .many_sample import EOH, QSVM
from .single_sample import (Grover, IQPE, QPE, AmplitudeEstimation,
                            Simon, DeutschJozsa, BernsteinVazirani, HHL, Shor,
                            IterativeAmplitudeEstimation,
                            MaximumLikelihoodAmplitudeEstimation)


__all__ = [
    'QuantumAlgorithm',
    'VQE',
    'QAOA',
    'VQC',
    'QGAN',
    'ClassicalAlgorithm',
    'ExactEigensolver',
    'ExactLSsolver',
    'SVM_Classical',
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
