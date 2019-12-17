# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" algorithms package """

from .quantum_algorithm import QuantumAlgorithm
from .adaptive import VQE, QAOA, VQC, QGAN
from .classical import ExactEigensolver, ExactLSsolver, SVM_Classical
from .many_sample import EOH, QSVM
from .single_sample import Grover, IQPE, QPE, AmplitudeEstimation, \
    Simon, DeutschJozsa, BernsteinVazirani, HHL, Shor, \
    IterativeAmplitudeEstimation, MaximumLikelihoodAmplitudeEstimation


__all__ = [
    'QuantumAlgorithm',
    'VQE',
    'QAOA',
    'VQC',
    'QGAN',
    'ExactEigensolver',
    'ExactLSsolver',
    'SVM_Classical',
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

try:
    from .classical import CPLEX_Ising
    __all__ += ['CPLEX_Ising']
except ImportError:
    pass
