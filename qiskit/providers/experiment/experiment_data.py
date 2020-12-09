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
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Union, Tuple, Callable, Dict

import qiskit.providers.experiment.experiment_service as experiment_service
from .local_service import LocalExperimentService
from .exceptions import ExperimentError, ExperimentDataNotFound, ExperimentDataExists
from .analysis_result import AnalysisResultV1 as AnalysisResult
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
            jobs: Optional[List[str]] = None,
            share_level: str = None,
            auto_save: bool = True,
            save_local: bool = True,
            save_remote: Optional[bool] = None,
            **kwargs: Any
    ):
        """Initializes the experiment data.

        Args:
            tags: Tags to be associated with the experiment.
            job_ids: IDs of experiment jobs.
            share_level: Whether this experiment can be shared with others. This
                is applicable only if the experiment service supports sharing. See
                the specific service provider's documentation on valid values.
            auto_save: ``True`` if changes to the experiment data, including newly
                generated analysis results and graphs, should be automatically saved.
            save_local: ``True`` if changes should be saved in the local database.
            save_remote: ``True`` if changes should be saved in the remote database.
                If ``None``, data will be saved remotely only if a remote experiment
                service can be found.
            kwargs: Additional experiment data.
        """
        # Data to be saved in DB.
        self._id = str(uuid.uuid4())
        self._tags = tags or []  # TODO need to monitor changes
        self._jobs = jobs or []  # type: List[Job]
        self._share_level = share_level
        self._data = kwargs
        self._type = f"{self.__class__.__module__}.{self.__class__.__name__}"

        # Other metadata
        self.save_local = save_local
        self.save_remote = save_remote
        self._local_service = LocalExperimentService() if save_local else None
        self._remote_service = None  # Determined after jobs are submitted.
        self._analysis_results = []
        self._graph_names = []
        self.auto_save = auto_save
        self._job_ids = set()
        self._created_local = False
        self._created_remote = False

    @classmethod
    def from_stored_data(cls, **kwargs) -> ExperimentData:
        """Return an ``ExperimentData`` instance from the stored data.

        Args:
            kwargs: Dictionary that contains the stored data.

        Returns:
            An ``ExperimentData`` instance.
        """
        obj = cls()
        obj._from_stored_data(**kwargs)
        return obj

    def _from_stored_data(self, **kwargs) -> None:
        """Update the object attributes with stored data.

        Args:
            kwargs: Dictionary that contains the stored data.
        """
        self._id = kwargs.get('experiment_id', self._id)
        self.deserialize_data(kwargs.get('data', {}))
        self._tags = kwargs.get('tags', [])
        self._job_ids = kwargs.get('job_ids', set())
        self._share_level = kwargs.get('share_level', None)
        self._created = True

    def save(
            self,
            save_local: Optional[bool] = None,
            save_remote: Optional[bool] = None,
            remote_service: experiment_service.ExperimentServiceV1 = None
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
        update_data = {'experiment_id': self._id, 'data': self.serialize_data(),
                       'job_ids': self.job_ids, 'tags': self._tags}
        new_data = {'experiment_type': self._type, 'backend_name': self.backend_name}
        if self.share_level:
            update_data['share_level'] = self.share_level

        save_local = save_local if save_local is not None else self.save_local
        if save_local:
            self._created_local, _ = self._save(
                is_new=self._created_local,
                new_func=self._local_service.create_experiment,
                update_func=self._local_service.update_experiment,
                new_data=new_data, update_data=update_data)

        remote_service = remote_service or self._remote_service
        if self._is_access_remote(save_remote, remote_service):
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
            # Attempt 3x for the unlikely scenario where new_entry=False but the
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

    def _is_access_remote(self, save_remote=None, remote_service=None):
        remote_service = remote_service or self.remote_service

        if save_remote is None:
            if self.save_remote is None:
                save_remote = True if remote_service else False
            else:
                save_remote = self.save_remote

        if save_remote:
            if not remote_service:
                LOG.warning("Unable to access the remote experiment database "
                            "because no suitable service is found.")
                return False
            return True
        return False

    def refresh(self):
        """Obtain the latest experiment attributes from the remote database."""
        if not self._created_remote or not self.remote_service:
            return
        retrieved_data = self.remote_service.experiment(self._id)
        self._from_stored_data(**retrieved_data)

    @abstractmethod
    def serialize_data(self):
        return self._data

    @abstractmethod
    def deserialize_data(self, data: Dict):
        self._data = data

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

        new_graph = graph_name in self.graph_names
        if new_graph and not overwrite:
            raise ExperimentError(f"A graph with the name {graph_name} for this experiment "
                                  f"already exists. Specify overwrite=True if you "
                                  f"want to overwrite it.")

        remote_service = remote_service or self._remote_service
        if self._is_access_remote(save_remote=True, remote_service=remote_service):
            data = {'experiment_id': self.id, 'graph': graph, 'graph_name': graph_name}
            new_graph, out = self._save(is_new=new_graph,
                                        new_func=remote_service.create_graph,
                                        update_func=remote_service.update_graph,
                                        new_data={},
                                        update_data=data)
            return out

        return '', 0

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
        """

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
        new_data = {'experiment_id': self._id, 'result_type': result.type}
        update_data = {'result_id': result.id, 'data': result.serialize_data(),
                       'tags': result.tags, 'quality': result.quality}
        if self.share_level:
            update_data['share_level'] = self.share_level

        save_local = save_local if save_local is not None else self.save_local
        if save_local:
            created_local, _ = self._save(
                is_new=result._created_local,
                new_func=self._local_service.create_analysis_result,
                update_func=self._local_service.update_analysis_result,
                new_data=new_data, update_data=update_data)
            result._created_local = created_local

        remote_service = remote_service or self._remote_service
        if self._is_access_remote(save_remote, remote_service):
            created_remote, _ = self._save(
                is_new=result._created_remote,
                new_func=remote_service.create_experiment,
                update_func=remote_service.update_experiment,
                new_data=new_data, update_data=update_data)
            result._created_remote = created_remote

    @property
    def id(self):
        return self._id

    @property
    def backend_name(self):
        if not self._jobs:
            raise
        return self._jobs[0].backend().name()

    @property
    def job_ids(self) -> List[str]:
        """Return experiment job IDs.

        Returns: A list of jobs IDs for this experiment.
        """
        self._job_ids = self._job_ids.union(set([job.job_id() for job in self._jobs]))
        return self._job_ids

    @property
    def jobs(self) -> List[Job]:
        """Return the jobs for this experiment.

        Returns:
            A list of jobs for this experiment.
        """
        return self._jobs

    @property
    def graph_names(self):
        return self._graph_names

    @property
    def share_level(self) -> str:
        return self._share_level

    @share_level.setter
    def share_level(self, new_level) -> None:
        self._share_level = new_level
        if self.auto_save:
            self.save()

    @property
    def remote_service(self) -> Optional[experiment_service.ExperimentServiceV1]:
        if not self._remote_service and self.jobs:
            try:
                provider = self.jobs[0].backend().provider()
                self._remote_service = provider.get_service('experiment')
                return self._remote_service
            except AttributeError:
                self.save_remote = False
        return None

    @remote_service.setter
    def remote_service(self, service):
        if self._remote_service:
            raise ExperimentError("A remote experiment service is already being used.")
        self._remote_service = service
