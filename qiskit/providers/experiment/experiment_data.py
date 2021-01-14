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

"""Experiment data abstract interface."""

import logging
import uuid
import json
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Union, Tuple, Callable, Dict, Set

import qiskit.providers.experiment.experiment_service as experiment_service
from .local_service import LocalExperimentService
from .exceptions import ExperimentError, ExperimentDataNotFound, ExperimentDataExists
from .analysis_result import AnalysisResultV1 as AnalysisResult
from .json import NumpyEncoder, NumpyDecoder
from .utils import MonitoredList, MonitoredDict
from ..job import JobV1 as Job

LOG = logging.getLogger(__name__)


class ExperimentData:
    """Base common type for all versioned ExperimentData abstract classes.

    Note this class should not be inherited from directly, it is intended
    to be used for type checking. When implementing a provider you should use
    the versioned abstract classes as the parent class and not this class
    directly.
    """
    version = 0


class ExperimentDataV1(ExperimentData, ABC):
    """Class to handle experiment data."""

    version = 1

    def __init__(
            self,
            tags: Optional[List[str]] = None,
            jobs: Optional[List[Job]] = None,
            share_level: Optional[str] = None,
            backend_name: Optional[str] = None,
            data: Optional[Dict] = None,
            graph_names: Optional[List[str]] = None,
            auto_save: bool = True,
            save_local: bool = True,
            save_remote: bool = True,
    ):
        """Initializes the experiment data.

        Args:
            tags: Tags to be associated with the experiment.
            jobs: Experiment jobs.
            share_level: Whether this experiment can be shared with others. This
                is applicable only if the experiment service supports sharing. See
                the specific service provider's documentation on valid values.
            backend_name: Name of the backend this experiment is for.
            data: Additional experiment data.
            graph_names: Name of graphs associated with this experiment.
            auto_save: ``True`` if changes to the experiment data, including newly
                generated analysis results and graphs, should be automatically saved.
            save_local: ``True`` if changes should be saved in the local database.
            save_remote: ``True`` if changes should be saved in the remote database.
                Data will not be saved remotely if no remote experiment
                service can be found.
        """
        # Data to be saved in DB.
        self._id = str(uuid.uuid4())
        self._tags = MonitoredList.create_with_callback(
            callback=self._monitored_callback, init_data=tags)
        self._job_ids = {job.job_id() for job in jobs} if jobs else set()
        self._share_level = share_level
        self._data = MonitoredDict.create_with_callback(
            callback=self._monitored_callback, init_data=data)
        self._type = f"{self.__class__.__module__}.{self.__class__.__name__}"
        self._backend_name = None
        if backend_name:
            self._backend_name = backend_name
        elif jobs:
            self._backend_name = jobs[0].backend().name()
        self._creation_date = datetime.now()
        self._graph_names = MonitoredList.create_with_callback(
            callback=self._jobs_list_callback, init_data=graph_names)

        # Other metadata
        self.save_local = save_local
        self.save_remote = save_remote
        self._local_service = LocalExperimentService()
        self._remote_service = None  # Determined after jobs are submitted.
        self._analysis_results = []
        self.auto_save = auto_save
        self._jobs = MonitoredList.create_with_callback(
            callback=self._jobs_list_callback, init_data=jobs)
        self._created_local = False
        self._created_remote = False

    def _jobs_list_callback(self):
        """Callback function invoked when a monitored job list changes."""
        self._job_ids = {job.job_id() for job in self._jobs}
        if self.auto_save:
            self.save()

    def _monitored_callback(self):
        """Callback function invoked when a monitored collection changes."""
        if self.auto_save:
            self.save()

    @classmethod
    def from_stored_data(cls, **kwargs) -> ExperimentData:
        """Return an ``ExperimentData`` instance from the stored data.

        Args:
            kwargs: Dictionary that contains the stored data.

        Returns:
            An ``ExperimentData`` instance.
        """
        obj = cls(
            tags=kwargs.get('tags', []),
            share_level=kwargs.get('share_level', None),
            backend_name=kwargs.get('backend_name', None),
            graph_names=kwargs.get('graph_names', []))
        # Turn off auto_save during initialization.
        saved_auto_save = obj.auto_save
        obj.auto_save = False
        obj._id = kwargs['experiment_id']
        obj._data.update(obj.deserialize_data(json.dumps(kwargs.get('data', {}))))
        obj._job_ids = kwargs.get('job_ids', set())
        obj._creation_date = kwargs.get('creation_date', obj._creation_date)
        obj._type = kwargs.get('type', obj._type)
        obj._jobs.append(kwargs.get('jobs'), [])
        obj._analysis_results.append(kwargs.get('analysis_results', []))
        obj.auto_save = saved_auto_save

        return obj

    def save(
            self,
            save_local: Optional[bool] = None,
            save_remote: Optional[bool] = None,
            remote_service: Optional[experiment_service.ExperimentServiceV1] = None
    ) -> None:
        """Save this experiment in the database.

        Args:
            save_local: ``True`` if data should be saved locally. If ``None``, the
                ``save_local`` attribute of this object is used.
            save_remote: ``True`` if data should be saved remotely. If ``None``, the
                ``save_remote`` attribute of this object is used.
            remote_service: Remote experiment service to be used to save the data.
                Not applicable if `local_only` is set to ``True``. If ``None``, the
                provider used to submit jobs will be used.
        """
        if not self._backend_name and not self._jobs:
            raise ExperimentError("Experiment can only be saved after jobs are submitted.")
        self._backend_name = self._backend_name or self._jobs[0].backend().name()

        update_data = {'experiment_id': self._id, 'data': json.loads(self.serialize_data()),
                       'job_ids': self.job_ids, 'tags': self.tags}
        new_data = {'experiment_type': self._type, 'backend_name': self._backend_name,
                    'creation_date': self.creation_date}
        if self.share_level:
            update_data['share_level'] = self.share_level

        save_local = save_local if save_local is not None else self.save_local
        if save_local:
            self._created_local, _ = self._save(
                is_new=self._created_local,
                new_func=self._local_service.create_experiment,
                update_func=self._local_service.update_experiment,
                new_data=new_data, update_data=update_data)

        if self._is_access_remote(save_remote=save_remote, remote_service=remote_service):
            remote_service = remote_service or self.remote_service
            self._created_remote, _ = self._save(
                is_new=self._created_remote,
                new_func=remote_service.create_experiment,
                update_func=remote_service.update_experiment,
                new_data=new_data, update_data=update_data)

    def _save(
            self,
            is_new: bool,
            new_func: Callable,
            update_func: Callable,
            new_data: Dict,
            update_data: Dict
    ) -> Tuple[bool, Any]:
        """Save data in the database.

        Args:
            is_new: ``True`` if `new_func` should be called. Otherwise `update_func` is called.
            new_func: Function to create new entry in the database.
            update_func: Function to update an existing entry in the database.
            new_data: In addition to `update_data`, this data will be stored if creating
                a new entry.
            update_data: Data to be stored if updating an existing entry.

        Returns:
            A tuple of whether the data was saved and the function return value.
        """
        attempts = 0
        try:
            # Attempt 3x for the unlikely scenario wherein is_new=False but the
            # entry doesn't actually exists. The second try might also fail if an entry
            # with the same ID somehow got created in the meantime.
            while attempts < 3:
                attempts += 1
                if is_new:
                    try:
                        return True, new_func(**{**new_data, **update_data})
                    except ExperimentDataExists:
                        is_new = False
                else:
                    try:
                        return True, update_func(**update_data)
                    except ExperimentDataNotFound:
                        is_new = True
            raise ExperimentError("Unable to determine the existence of the entry.")
        except Exception as ex:
            # Don't fail the experiment just because its data cannot be saved.
            LOG.error(f"Unable to save the experiment data: {str(ex)}")
            return False, None

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

    def save_graph(
            self,
            graph: Union[str, bytes],
            graph_name: str = None,
            overwrite=False,
            remote_service: experiment_service.ExperimentServiceV1 = None
    ) -> Tuple[str, int]:
        """Save the experiment graph in the remote database.

        Note that graphs are not saved in the local database because they are
        not structured data.

        Args:
            graph: Name of the graph file or graph data to upload.
            graph_name: Name of the graph. If ``None``, use the graph file name, if
                given, or a generated name.
            overwrite: Whether to overwrite the graph if one already exists with
                the same name.
            remote_service: Remote experiment service to be used to save the data.
                If ``None``, the  provider used to submit jobs will be used.

        Returns:
            A tuple of the name and size of the saved graph.
        """
        if not graph_name:
            if isinstance(graph, str):
                graph_name = graph
            else:
                graph_name = f"graph_{self.id}_{len(self.graph_names)}"

        existing_graph = graph_name in self.graph_names
        if existing_graph and not overwrite:
            raise ExperimentError(f"A graph with the name {graph_name} for this experiment "
                                  f"already exists. Specify overwrite=True if you "
                                  f"want to overwrite it.")

        out = {'', 0}
        if self._is_access_remote(save_remote=True, remote_service=remote_service):
            data = {'experiment_id': self.id, 'graph': graph, 'graph_name': graph_name}
            remote_service = remote_service or self.remote_service
            _, out = self._save(is_new=not existing_graph,
                                new_func=remote_service.create_graph,
                                update_func=remote_service.update_graph,
                                new_data={},
                                update_data=data)
        if not existing_graph:
            self._graph_names.append(graph_name)

        return out

    def graph(
            self,
            graph_name: str,
            file_name: Optional[str] = None
    ) -> Union[int, bytes]:
        """Retrieve the specified experiment graph from the remote database.

        Args:
            graph_name: Name of the graph.
            file_name: Name of the local file to save the graph to. If ``None``,
                the content of the graph is returned instead.

        Returns:
            The size of the graph if `file_name` is specified. Otherwise the
            content of the graph in bytes.

        Raises:
            ExperimentDataNotFound: If the graph cannot be found.
        """
        if self._is_access_remote(save_remote=True):
            return self.remote_service.graph(self.id, graph_name, file_name)

        raise ExperimentDataNotFound("Unable to retrieve graph because there is "
                                     "no access to remote database.")

    def save_analysis_result(
            self,
            result: AnalysisResult,
            save_local: Optional[bool] = None,
            save_remote: Optional[bool] = None,
            remote_service: experiment_service.ExperimentServiceV1 = None
    ) -> None:
        """Save the analysis result.

        Args:
            result: Analysis result to be saved.
            save_local: ``True`` if data should be saved locally. If ``None``, the
                ``save_local`` attribute of this object is used.
            save_remote: ``True`` if data should be saved remotely. If ``None``, the
                ``save_remote`` attribute of this object is used.
            remote_service: Remote experiment service to be used to save the data.
                Not applicable if `local_only` is set to ``True``. If ``None``, the
                provider used to submit jobs will be used.
        """
        result.save(save_local=save_local, save_remote=save_remote, remote_service=remote_service)
        if result.id not in [res.id for res in self._analysis_results]:
            self._analysis_results.append(result)

    @property
    def id(self) -> str:
        """Return experiment ID

        Returns:
            Experiment ID.
        """
        return self._id

    @property
    def job_ids(self) -> Set[str]:
        """Return experiment job IDs.

        Returns: A list of jobs IDs for this experiment.
        """
        return self._job_ids

    @property
    def tags(self):
        """Return tags associated with this experiment."""
        return self._tags

    @tags.setter
    def tags(self, new_tags: List[str]) -> None:
        """Set tags for this experiment.

        Args:
            new_tags: New tags for the experiment.
        """
        self._tags = MonitoredList.create_with_callback(
            callback=self._monitored_callback, init_data=new_tags)
        if self.auto_save:
            self.save()

    @property
    def graph_names(self) -> List[str]:
        """Return names of the graphs associated with this experiment.

        Returns:
            Names of graphs associated with this experiment.
        """
        return self._graph_names

    @property
    def share_level(self) -> str:
        """Return the share level fo this experiment.

        Returns:
            Experiment share level.
        """
        return self._share_level

    @share_level.setter
    def share_level(self, new_level: str) -> None:
        """Set the experiment share level.

        Args:
            new_level: New experiment share level. Valid share levels are provider-
                specified. For example, IBMQ allows "global", "hub", "group",
                "project", and "private".
        """
        self._share_level = new_level
        if self.auto_save:
            self.save()

    @property
    def data(self) -> Dict:
        """Return experiment data

        Returns:
            Experiment data.
        """
        return self._data

    @property
    def creation_date(self) -> datetime:
        """Return experiment creation date.

        Returns:
            Experiment creation date in local timezone.
        """
        return self._creation_date

    @property
    def local_service(self) -> LocalExperimentService:
        """Return the local database service.

        Returns:
            Service that can be used to access this experiment in a remote database.
        """
        return self._local_service

    @property
    def remote_service(self) -> Optional[experiment_service.ExperimentServiceV1]:
        """Return the remote database service.

        Returns:
            Service that can be used to access this experiment in a remote database.
        """
        if self._remote_service:
            return self._remote_service
        if self._jobs:
            try:
                provider = self._jobs[0].backend().provider()
                self._remote_service = provider.get_service('experiment')
                return self._remote_service
            except Exception:
                pass
        return None

    @remote_service.setter
    def remote_service(self, service: experiment_service.ExperimentServiceV1) -> None:
        """Set the service to be used for storing experiment data remotely.

        Args:
            service: Service to be used.

        Raises:
            ExperimentError: If a remote experiment service is already being used.
        """
        if self._remote_service:
            raise ExperimentError("A remote experiment service is already being used.")
        self._remote_service = service

    @property
    def analysis_results(self) -> Optional[List[AnalysisResult]]:
        """Return analysis results associated with this experiment.

        Returns:
            Analysis results for this experiment, or ``None`` if they
            cannot be retrieved.
        """
        if self._analysis_results is None and self.remote_service:
            self._analysis_results = self.remote_service.analysis_results(
                    experiment_id=self.id, limit=None)

        return self._analysis_results
