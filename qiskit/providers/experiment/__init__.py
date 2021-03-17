# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=========================================================
Experiment Interface (:mod:`qiskit.providers.experiment`)
=========================================================

.. currentmodule:: qiskit.providers.experiment

This module contains the classes used to build experiment services, which
allow users to store experiment data and metadata in databases. An experiment
typically has one or more jobs, analysis results, and graphs associated with it.

An experiment service provider can inherit the
:class:`~qiskit.providers.experiment.ExperimentService` class. An
experiment service consumer can inherit the
:class:`~qiskit.providers.experiment.ExperimentData` class, which already
has methods that interacts with the service implemented.


Abstract Classes
================

Service Provider
----------------

.. autosummary::
   :toctree: ../stubs/

   ExperimentService
   ExperimentServiceV1
   LocalExperimentService

Service Consumer
----------------

.. autosummary::
   :toctree: ../stubs/

   ExperimentData
   ExperimentDataV1
   AnalysisResult
   AnalysisResultV1


Exceptions
==========

.. autosummary::
   :toctree: ../stubs/

   ExperimentError
   ExperimentDataNotFound
   ExperimentDataExists
"""

from .constants import ResultQuality
from .experiment_data import ExperimentData, ExperimentDataV1
from .analysis_result import AnalysisResult, AnalysisResultV1
from .experiment_service import ExperimentService, ExperimentServiceV1
from .exceptions import ExperimentError, ExperimentEntryExists, ExperimentEntryNotFound
