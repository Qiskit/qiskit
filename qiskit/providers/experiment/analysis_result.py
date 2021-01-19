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

import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Union, Dict
import uuid
from datetime import datetime
import json

import qiskit.providers.experiment.experiment_data as experiment_data
import qiskit.providers.experiment.experiment_service as experiment_service

from .constants import ResultQuality
from .json import NumpyEncoder, NumpyDecoder
from .utils import MonitoredList, MonitoredDict, save_data
from .exceptions import ExperimentError
from .local_service import LocalExperimentService

LOG = logging.getLogger(__name__)


class AnalysisResult:
    """Base common type for all versioned AnalysisResult abstract classes.

    Note this class should not be inherited from directly, it is intended
    to be used for type checking. When implementing a provider you should use
    the versioned abstract classes as the parent class and not this class
    directly.
    """
    version = 0


class AnalysisResultV1(AnalysisResult, ABC):
    """Class representing an analysis result for an experiment."""

    version = 1
    data_version = 1

    def __init__(
            self,
            experiment: Optional[experiment_data.ExperimentDataV1] = None,
            experiment_id: Optional[str] = None,
            result_type: Optional[str] = None,
            quality: Union[ResultQuality, int] = ResultQuality.AVERAGE,
            tags: Optional[List[str]] = None,
            data: Optional[Dict] = None
    ):
        """AnalysisResult constructor.

        Args:
            experiment: Experiment this analysis result is for. If ``None``,
                `experiment_id` is required.
            experiment_id: ID of the experiment. Required if `experiment` is ``None``.
            result_type: Result type. If ``None``, the class name is used.
            quality: Quality of the analysis.
            tags: Tags for this analysis.
            data: Additional analysis result data. Note that ``_source_path`` and
                ``_data_version`` are reserved keys and cannot be in `data`.

        Raises:
            ExperimentError: If an input argument is invalid.
        """
        if not experiment and not experiment_id:
            raise ExperimentError("Experiment or experiment ID is required when "
                                  "creating a new AnalysisResult.")
        if experiment and experiment_id and experiment.id != experiment_id:
            raise ExperimentError("Both experiment and experiment ID are specified, "
                                  "but the IDs don't match.")

        # Data to be stored in DB.
        self._source = {'_source_path': f"{self.__class__.__module__}.{self.__class__.__name__}",
                        '_data_version': self.data_version}
        data = data or {}
        for key in self._source:
            if key in data and data[key] != self._source[key]:
                raise ExperimentError(f"{key} is reserved and cannot be in data.")

        self._experiment_id = experiment_id or experiment.id
        self._id = str(uuid.uuid4())
        self._type = result_type or self.__class__.__name__
        self._data = MonitoredDict.create_with_callback(
            callback=self._monitored_callback, init_data=data)
        self._type = f"{self.__class__.__module__}.{self.__class__.__name__}"
        self._quality = ResultQuality(quality)
        self._tags = MonitoredList.create_with_callback(
            callback=self._monitored_callback, init_data=tags)
        self._creation_date = datetime.now()

        # Other metadata
        self._experiment = experiment
        self._created_local = False
        self._created_remote = False
        if experiment:
            self.auto_save = experiment.auto_save
            self.save_local = experiment.save_local
            self.save_remote = experiment.save_remote
            self._local_service = experiment.local_service
            self._remote_service = experiment.remote_service
        else:
            self.auto_save = True
            self.save_local = True
            self.save_remote = True
            self._local_service = LocalExperimentService()
            self._remote_service = None

        if self.auto_save:
            self.save()

    def _monitored_callback(self):
        """Callback function invoked when a monitored collection changes."""
        if self.auto_save:
            self.save()

    def to_dict(self) -> Dict:
        """Return a dictionary format representation of this analysis result.

        Returns:
            A dictionary format representation of this analysis result.
        """
        data = {'experiment_id': self._experiment_id,
                'type': self._type,
                'data': self.data,
                'quality': self.quality,
                'tags': self.tags,
                'result_id': self.id,
                'creation_date': self.creation_date
                }
        return data

    @classmethod
    def from_stored_data(cls, **kwargs: Any) -> AnalysisResult:
        """Return an ``AnalysisResult`` instance from the stored data.

        Args:
            **kwargs: Dictionary that contains the stored data.

        Returns:
            An ``AnalysisResult`` instance.
        """
        obj = cls(
            experiment=kwargs.get('experiment', None),
            experiment_id=kwargs.get('experiment_id', None),
            result_type=kwargs.get('type', None),
            tags=kwargs.get('tags', []),
        )
        obj._id = kwargs.get('result_id', obj.id)
        obj._quality = kwargs.get('quality', obj._quality)

        _data = obj.deserialize_data(json.dumps(kwargs.get('data', {})))
        obj._source = {'_source_path', _data.pop('_source_path', ''),
                       '_data_version', _data.pop('_data_version', cls.data_version)}
        obj._data.update(_data)
        obj._creation_date = kwargs.get('creation_date', obj._creation_date)
        return obj

    @abstractmethod
    def serialize_data(self, encoder: Optional[json.JSONEncoder] = NumpyEncoder) -> str:
        """Serialize experiment data into JSON string.

        Args:
            encoder: Custom JSON encoder to use.

        Returns:
            Serialized JSON string.
        """
        return json.dumps(self._data, cls=encoder)

    @abstractmethod
    def deserialize_data(
            self,
            data: str,
            decoder: Optional[json.JSONDecoder] = NumpyDecoder
    ) -> Any:
        """Deserialize experiment from JSON string.

        Args:
            data: Data to be deserialized.
            decoder: Custom decoder to use.

        Returns:
            Deserialized data.
        """
        return json.loads(data, cls=decoder)

    def save(
            self,
            save_local: Optional[bool] = None,
            save_remote: Optional[bool] = None,
            remote_service: Optional[experiment_service.ExperimentServiceV1] = None
    ) -> None:
        """Save this analysis result in the database.

        Args:
            save_local: ``True`` if data should be saved locally. If ``None``, the
                ``save_local`` attribute of this object is used.
            save_remote: ``True`` if data should be saved remotely. If ``None``, the
                ``save_remote`` attribute of this object is used.
            remote_service: Remote experiment service to be used to save the data.
                Not applicable if `local_only` is set to ``True``. If ``None``, the
                default, if any, is used.

        Raises:
            ExperimentError: If the analysis result contains invalid data.
        """
        _data = json.loads(self.serialize_data())
        for key in self._source:
            if key in _data and _data[key] != self._source[key]:
                raise ExperimentError(f"{key} is reserved and cannot be in data.")
        _data.update(self._source)

        new_data = {'experiment_id': self._experiment_id, 'result_type': self._type}
        update_data = {'result_id': self.id, 'data': _data,
                       'tags': self.tags, 'quality': self.quality}

        save_local = save_local if save_local is not None else self.save_local
        if save_local:
            self._created_local, _ = save_data(
                is_new=self._created_local,
                new_func=self._local_service.create_analysis_result,
                update_func=self._local_service.update_analysis_result,
                new_data=new_data, update_data=update_data)

        if self._is_access_remote(save_remote, remote_service):
            remote_service = remote_service or self.remote_service
            self._created_remote, _ = save_data(
                is_new=self._created_remote,
                new_func=remote_service.create_experiment,
                update_func=remote_service.update_experiment,
                new_data=new_data, update_data=update_data)

    def _is_access_remote(
            self,
            save_remote: Optional[bool] = None,
            remote_service: Optional[experiment_service.ExperimentServiceV1] = None
    ) -> bool:
        """Determine whether data should be saved in the remote database.

        Args:
            save_remote: Used to overwrite the default ``save_remote`` option.
                ``None`` if the default should be used.
            remote_service: Used to overwrite the default remote database
                service. ``None`` if the default should be used.

        Returns:
            ``True`` if data should be saved in the remote database. ``False``
            otherwise.
        """
        _save_remote = save_remote if save_remote is not None else self.save_remote
        if _save_remote:
            if remote_service or self.remote_service:
                return True
            self.save_remote = False
            err_msg = "Unable to access the remote experiment database " \
                      "because no suitable service is found."
            if save_remote:
                # Raise if user explicitly asked for save_remote.
                raise ExperimentError(err_msg)
            LOG.warning(err_msg)
            return False
        return False

    @property
    def id(self):
        """Return analysis result ID.

        Returns:
            ID for this analysis result.
        """
        return self._id

    @property
    def type(self) -> str:
        """Return analysis result type.

        Returns:
            Analysis result type.
        """
        return self._type

    @property
    def quality(self) -> ResultQuality:
        """Return the quality of this analysis.

        Returns:
            Quality of this analysis.
        """
        return self._quality

    @quality.setter
    def quality(self, new_quality: Union[ResultQuality, int]) -> None:
        """Set the quality of this analysis.

        Args:
            new_quality: New analysis quality.
        """
        self._quality = ResultQuality(new_quality)
        if self.auto_save:
            self.save()

    @property
    def experiment(self) -> Optional[experiment_data.ExperimentDataV1]:
        """Return the experiment associated with this analysis result.

        Returns:
            Experiment associated with this analysis result. ``None`` if unknown.
        """
        return self._experiment

    @property
    def experiment_id(self) -> str:
        """Return the ID of the experiment associated with this analysis result.

        Returns:
            ID of experiment associated with this analysis result.
        """
        return self._experiment_id

    @property
    def data(self) -> Dict:
        """Return analysis result data.

        Returns:
            Analysis result data.
        """
        return self._data

    @property
    def tags(self):
        """Return tags associated with this result."""
        return self._tags

    @tags.setter
    def tags(self, new_tags: List[str]) -> None:
        """Set tags for this result.

        Args:
            new_tags: New tags for the result.
        """
        self._tags = MonitoredList.create_with_callback(
            callback=self._monitored_callback, init_data=new_tags)
        if self.auto_save:
            self.save()

    @property
    def creation_date(self) -> datetime:
        """Return analysis result creation date.

        Returns:
            Analysis result creation date in local timezone.
        """
        return self._creation_date

    @property
    def local_service(self) -> LocalExperimentService:
        """Return the local database service.

        Returns:
            Service that can be used to access this analysis in a remote database.
        """
        return self._local_service

    @property
    def remote_service(self) -> Optional[experiment_service.ExperimentServiceV1]:
        """Return the remote database service.

        Returns:
            Service that can be used to access this analysis in a remote database.
            ``None`` if not available.
        """
        return self._remote_service

    @remote_service.setter
    def remote_service(self, service: experiment_service.ExperimentServiceV1) -> None:
        """Set the service to be used for storing result data remotely.

        Args:
            service: Service to be used.

        Raises:
            ExperimentError: If a remote experiment service is already being used.
        """
        if self._remote_service:
            raise ExperimentError("A remote experiment service is already being used.")
        self._remote_service = service
