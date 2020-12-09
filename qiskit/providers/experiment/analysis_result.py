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

"""Analysis result abstract interface."""

from abc import ABC, abstractmethod
from typing import Optional, List, Any, Union, Tuple, Callable, Dict
import uuid

import qiskit.providers.experiment.experiment_data as experiment_data

from .constants import ResultQuality


class AnalysisResult:
    """Base common type for all versioned AnalysisResult abstract classes.

    Note this class should not be inherited from directly, it is intended
    to be used for type checking. When implementing a provider you should use
    the versioned abstract classes as the parent class and not this class
    directly.
    """
    version = 0


class AnalysisResultV1(AnalysisResult, ABC):
    def __init__(
            self,
            experiment: experiment_data.ExperimentDataV1,
            quality: Union[ResultQuality, str] = ResultQuality.NO_INFORMATION,
            target_components: Optional[List[str]] = None,
            tags: Optional[List[str]] = None,
            result_id: Optional[str] = None,
            **result_data
    ):
        # Data to be stored in DB.
        self._data = result_data
        self._type = f"{self.__class__.__module__}.{self.__class__.__name__}"
        self._quality = quality
        self.tags = tags or []
        self._id = result_id or str(uuid.uuid4())
        self._target_components = target_components

        # Other metadata
        self._experiment = experiment
        self._created_local = False
        self._created_remote = False

    @abstractmethod
    def serialize_data(self):
        return self._data

    @abstractmethod
    def deserialize_data(self, data):
        self._data = data

    def save(self):
        self._experiment.save_analysis_result(self)

    @property
    def id(self):
        return self._id

    @property
    def type(self):
        return self._type

    @property
    def quality(self):
        return self._quality

    @quality.setter
    def quality(self, new_quality):
        self._quality = new_quality
        if self._experiment.auto_save:
            self.save()

    @property
    def target_components(self):
        return self._target_components

    @target_components.setter
    def target_components(self, new_target):
        self._target_components = new_target
        if self._experiment.auto_save:
            self.save()
