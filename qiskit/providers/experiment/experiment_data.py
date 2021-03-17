# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Experiment data class."""

import logging
import uuid
import json
from abc import abstractmethod
from typing import Optional, List, Any, Union, Tuple, Callable, Dict
import copy
from concurrent import futures
import queue
from collections import OrderedDict, defaultdict
from functools import wraps
import traceback

from qiskit.version import __qiskit_version__
from qiskit.providers import Job, BaseJob, Backend, BaseBackend
from qiskit.result import Result
from qiskit.providers import JobStatus
from qiskit.exceptions import QiskitError

from .exceptions import ExperimentError, ExperimentEntryNotFound, ExperimentEntryExists
from .analysis_result import AnalysisResultV1 as AnalysisResult
from .json import NumpyEncoder, NumpyDecoder
from .utils import save_data

LOG = logging.getLogger(__name__)


def auto_save(func: Callable):
    """Decorate the input function."""
    @wraps(func)
    def _wrapped(self, *args, **kwargs):
        return_val = func(self, *args, **kwargs)
        if self.auto_save:
            self.save()
        return return_val
    return _wrapped


class ExperimentData:
    """Base common type for all versioned ExperimentData classes.

    Note this class should not be inherited from directly, it is intended
    to be used for type checking. When implementing a provider you should use
    the versioned abstract classes as the parent class and not this class
    directly.
    """
    version = 0


class ExperimentDataV1(ExperimentData):
    """Class to handle experiment data.

    This class serves as a data container for experiment data and any analysis
    results or figures associated with the experiment. It also provides methods
    used to interact with the database, such as saving the experiment data in
    the database or retrieving its figured.
    """

    version = 1
    _metadata_version = 1
    _executor = futures.ThreadPoolExecutor()
    """Threads used for asynchronous processing."""

    _json_encoder = NumpyEncoder
    _json_decoder = NumpyDecoder

    def __init__(
            self,
            backend: Union[Backend, BaseBackend],
            experiment_type: str,
            experiment_id: Optional[str] = None,
            tags: Optional[List[str]] = None,
            job_ids: Optional[List[str]] = None,
            share_level: Optional[str] = None,
            metadata: Optional[Dict] = None,
            figure_names: Optional[List[str]] = None,
            notes: Optional[str] = None
    ):
        """Initializes the experiment data.

        Args:
            backend: Backend the experiment runs on. It can either be a
                :class:`~qiskit.providers.Backend` instance or just backend name.
            experiment_type: Experiment type.
            experiment_id: Experiment ID. One will be generated if not supplied.
            tags: Tags to be associated with the experiment.
            job_ids: IDs of jobs submitted for the experiment.
            share_level: Whether this experiment can be shared with others. This
                is applicable only if the experiment service supports sharing. See
                the specific service provider's documentation on valid values.
            metadata: Additional experiment metadata.
            figure_names: Name of figures associated with this experiment.
            notes: Freeform notes about the experiment.

        Raises:
            ExperimentError: If an input argument is invalid.
        """
        metadata = metadata or {}
        self._source = metadata.pop(
            "_source",
            {"class": f"{self.__class__.__module__}.{self.__class__.__name__}",
             "data_version": self._metadata_version,
             "_qiskit_version": __qiskit_version__})

        self._service = None
        self._backend = backend
        self.auto_save = False
        try:
            self._service = backend.provider().service('experiment')
            self.auto_save = self._service.option['auto_save']
        except Exception:  # pylint: disable=broad-except
            pass

        self._id = experiment_id or str(uuid.uuid4())
        self._type = experiment_type
        self._tags = tags or []
        self._share_level = share_level
        self._notes = notes or ""
        self._metadata = copy.deepcopy(metadata)

        job_ids = job_ids or []
        self._jobs = OrderedDict((k, None) for k in job_ids)
        self._job_futures = []
        self._errors = []

        self._data_queue = queue.Queue()
        self._data = []

        figure_names = figure_names or []
        self._figures = dict.fromkeys(figure_names)
        self._figures_queue = queue.Queue()
        self._figure_names = figure_names  # Only used by add_figure.

        self._analysis_results_queue = queue.Queue()
        self._analysis_results = []

        self._created_in_db = False

    def add_data(
            self,
            data: Union[Result, List[Result], Job, List[Job], Dict, List[Dict]],
            post_processing_callback: Optional[Callable] = None
    ):
        """Add experiment data.

        Args:
            data: Experiment data to add.
                Several types are accepted for convenience:

                    * Result: Add data from this ``Result`` object.
                    * List[Result]: Add data from the ``Result`` objects.
                    * Job: Add data from the job result.
                    * List[Job]: Add data from the job results.
                    * Dict: Add this data.
                    * List[Dict]: Add this list of data.
            post_processing_callback: Callback function invoked when all pending
                jobs finish. This ``ExperimentData`` object is the only argument
                to be passed to the callback function.
        """
        if isinstance(data, dict):
            self._add_single_data(data)
        elif isinstance(data, Result):
            self._add_result_data(data)
        elif isinstance(data, (Job, BaseJob)):
            self._jobs[data.job_id()] = data
            self._job_futures.append(
                (data, self._executor.submit(self._wait_for_job, data, post_processing_callback)))
            if self.auto_save:
                self.save()
        elif isinstance(data, list):
            for dat in data:
                self.add_data(dat)
        else:
            raise QiskitError(f"Invalid data type {type(data)}.")

    def _wait_for_job(
            self, job: Union[Job, BaseJob],
            job_done_callback: Optional[Callable] = None
    ) -> None:
        """Wait for a job to finish.

        Args:
            job: Job to wait for.
            job_done_callback: Callback function to invoke when job finishes.
        """
        LOG.debug(f"Waiting for job {job.job_id()} to finish.")
        self._add_result_data(job.result())
        if job_done_callback:
            job_done_callback(self)

    def _add_result_data(self, result: Result) -> None:
        """Add data from a Result object

        Args:
            result: Result object containing data to be added.
        """
        for i in range(len(result.results)):
            data = result.data(i)
            data['job_id'] = result.job_id
            if 'counts' in data:
                # Format to Counts object rather than hex dict
                data['counts'] = result.get_counts(i)
            self._add_single_data(data)

    def _add_single_data(self, data: Dict[str, any]) -> None:
        """Add a single data dictionary to the experiment.

        Args:
            data: Data to be added.
        """
        self._data_queue.put_nowait(data)

    def data(
            self,
            index: Optional[Union[int, slice, str]] = None
    ) -> Union[Dict, List[Dict]]:
        """Return the experiment data at the specified index.

        Args:
            index: Index of the data to be returned.
                Several types are accepted for convenience:

                    * None: Return all experiment data.
                    * int: Specific index of the data.
                    * slice: A list slice of data indexes.
                    * str: ID of the job that produced the data.

        Returns:
            Experiment data.
        """
        # Get job results if missing experiment data.
        if not self._data and self._backend.provider():
            for jid in self._jobs:
                if self._jobs[jid] is None:
                    try:
                        self._jobs[jid] = self._backend.provider().retrieve_job(jid)
                    except Exception:  # pylint: disable=broad-except
                        pass
                if self._jobs[jid] is not None:
                    self._add_result_data(self._jobs[jid].result())
        self._collect_from_queues()

        if index is None:
            return self._data
        if isinstance(index, (int, slice)):
            return self._data[index]
        if isinstance(index, str):
            return [data for data in self._data if data.get("job_id") == index]
        raise QiskitError(f"Invalid index type {type(index)}.")

    def _collect_from_queues(self):
        """Collect entries from queues."""
        while not self._data_queue.empty():
            try:
                self._data.append(self._data_queue.get_nowait())
            except queue.Empty:
                pass

        while not self._analysis_results_queue.empty():
            try:
                self._analysis_results.append(self._analysis_results_queue.get_nowait())
            except queue.Empty:
                pass

        while not self._figures_queue.empty():
            try:
                fig = self._figures_queue.get_nowait()
                self._figures[fig[0]] = fig[1]
            except queue.Empty:
                pass

    @auto_save
    def add_figure(
            self,
            figure: Union[str, bytes],
            figure_name: Optional[str] = None,
            overwrite: bool = False,
            service: Optional['ExperimentServiceV1'] = None
    ) -> Tuple[str, int]:
        """Save the experiment figure.

        Args:
            figure: Name of the figure file or figure data to upload.
            figure_name: Name of the figure. If ``None``, use the figure file name, if
                given, or a generated name.
            overwrite: Whether to overwrite the figure if one already exists with
                the same name.
            service: Experiment service to be used to save the figure.

        Returns:
            A tuple of the name and size of the saved figure. Returned size
            is 0 if there is no experiment service to use.

        Raises:
            ExperimentEntryExists: If the figure with the same name already exists,
                and `overwrite=True` is not specified.
        """
        if not figure_name:
            if isinstance(figure, str):
                figure_name = figure
            else:
                figure_name = f"figure_{self.id}_{len(self.figure_names)}"

        existing_figure = figure_name in self._figure_names
        if existing_figure and not overwrite:
            raise ExperimentEntryExists(f"A figure with the name {figure_name} for this experiment "
                                        f"already exists. Specify overwrite=True if you "
                                        f"want to overwrite it.")

        out = [figure_name, 0]
        service = service or self._service
        if service:
            data = {'experiment_id': self.id, 'figure': figure, 'figure_name': figure_name}
            _, out = save_data(is_new=not existing_figure,
                               new_func=service.create_figure,
                               update_func=service.update_figure,
                               new_data={},
                               update_data=data)

        self._figures_queue.put_nowait((figure_name, figure))
        self._figure_names.append(figure_name)
        return out

    def figure(
            self,
            figure_name: str,
            file_name: Optional[str] = None
    ) -> Union[int, bytes]:
        """Retrieve the specified experiment figure.

        Args:
            figure_name: Name of the figure.
            file_name: Name of the local file to save the figure to. If ``None``,
                the content of the figure is returned instead.

        Returns:
            The size of the figure if `file_name` is specified. Otherwise the
            content of the figure in bytes.

        Raises:
            ExperimentDataNotFound: If the figure cannot be found.
        """
        self._collect_from_queues()

        figure_data = self._figures.get(figure_name, None)
        if figure_data is not None:
            if isinstance(figure_data, str):
                with open(figure_data, 'rb') as file:
                    figure_data = file.read()
            if file_name:
                with open(file_name, 'wb') as output:
                    num_bytes = output.write(figure_data)
                    return num_bytes
            return figure_data
        elif self.service:
            return self.service.figure(experiment_id=self.id,
                                       figure_name=figure_name, file_name=file_name)
        raise ExperimentEntryNotFound(f"Figure {figure_name} not found.")

    @auto_save
    def add_analysis_result(
            self,
            result: AnalysisResult,
            service: 'ExperimentServiceV1' = None
    ) -> None:
        """Save the analysis result.

        Args:
            result: Analysis result to be saved.
            service: Experiment service to be used to save the data.
                If ``None``, the default service is used.
        """
        self._analysis_results_queue.put_nowait(result)
        if self.auto_save:
            result.save(service=service)

    def analysis_result(
            self,
            index: Optional[Union[int, slice, str]],
            refresh: bool = False) -> Union[AnalysisResult, List[AnalysisResult]]:
        """Return analysis results associated with this experiment.

        Args:
            index: Index of the analysis result to be returned.
                Several types are accepted for convenience:

                    * None: Return all analysis results.
                    * int: Specific index of the analysis results.
                    * slice: A list slice of indexes.
                    * str: ID of the analysis result.
            refresh: Retrieve the latest analysis results from the server, if
                an experiment service is available.

        Returns:
            Analysis results for this experiment.
        """
        self._collect_from_queues()

        if self.service and (not self._analysis_results or refresh):
            self._analysis_results = self.service.analysis_results(
                experiment_id=self.id, limit=None)

        if index is None:
            return self._analysis_results
        if isinstance(index, (int, slice)):
            return self._analysis_results[index]
        if isinstance(index, str):
            for res in self._analysis_results:
                if res.id == index:
                    return res
            raise QiskitError(f"Analysis result {index} not found.")
        raise QiskitError(f"Invalid index type {type(index)}.")

    def save(
            self,
            service: Optional['ExperimentServiceV1'] = None
    ) -> None:
        """Save this experiment in the database.

        Args:
            service: Experiment service to be used to save the data.
                If ``None``, the provider used to submit jobs will be used.

        Raises:
            ExperimentError: If the experiment contains invalid data.
        """
        service = service or self._service
        if not service:
            LOG.warning("Experiment cannot be saved because no experiment service is available.")
            return

        self._collect_from_queues()
        metadata = json.loads(self._serialize_metadata())
        metadata.update(self._source)

        update_data = {'experiment_id': self._id, 'metadata': metadata,
                       'job_ids': self.job_ids, 'tags': self.tags}
        new_data = {'experiment_type': self._type, 'backend_name': self._backend.name()}
        if self.share_level:
            update_data['share_level'] = self.share_level

        self._created_in_db, _ = save_data(
            is_new=self._created_in_db,
            new_func=service.create_experiment,
            update_func=service.update_experiment,
            new_data=new_data, update_data=update_data)

    def _serialize_metadata(self) -> str:
        """Serialize experiment data into JSON string.

        Returns:
            Serialized JSON string.
        """
        return json.dumps(self._metadata, cls=self._json_encoder)

    @abstractmethod
    def deserialize_data(self, data: str) -> Any:
        """Deserialize experiment from JSON string.

        Args:
            data: Data to be deserialized.

        Returns:
            Deserialized data.
        """
        return json.loads(data, cls=self._json_decoder)

    def status(self) -> str:
        """Return the data processing status.

        Returns:
            Data processing status.
        """
        job_stats = defaultdict(list)
        for idx, item in enumerate(self._job_futures):
            job, fut = item
            if not fut.done():
                job_stats[job.status()].append(job)
            elif fut.exception():
                ex = fut.exception()
                self._errors.append(traceback.print_exception(type(ex), ex, ex.__traceback__))
            else:
                self._job_futures[idx] = None

        self._job_futures = list(filter(None, self._job_futures))

        for stat in [JobStatus.INITIALIZING, JobStatus.VALIDATING, JobStatus.QUEUED,
                     JobStatus.RUNNING, JobStatus.CANCELLED]:
            if stat in job_stats:
                return stat.name

        if JobStatus.ERROR in job_stats:
            self._errors.extend([f"Job {bad_job.job_id()} failed" for bad_job
                                 in job_stats[JobStatus.ERROR]])

        if not self._errors:
            return "ERROR"

        if self._job_futures:
            return "POST_PROCESSING"

        return "DONE"

    def tags(self) -> List[str]:
        """Return tags assigned to this experiment data.

        Returns:
            A list of tags assigned to this experiment data.

        """
        return self._tags

    @auto_save
    def update_tags(self, new_tags: List[str]) -> None:
        """Set tags for this experiment.

        Args:
            new_tags: New tags for the experiment.
        """
        self._tags = new_tags

    @property
    def id(self) -> str:
        """Return experiment ID

        Returns:
            Experiment ID.
        """
        return self._id

    @property
    def job_ids(self) -> List[str]:
        """Return experiment job IDs.

        Returns: IDs of jobs submitted for this experiment.
        """
        return list(self._jobs.keys())

    @property
    def backend(self) -> Union[BaseBackend, Backend]:
        """Return backend.

        Returns:
            Backend this experiment is for.
        """
        return self._backend

    @property
    def metadata(self) -> Dict:
        """Return experiment metadata.

        Returns:
            Experiment metadata.
        """
        return self._metadata

    @property
    def type(self) -> str:
        """Return experiment type.

        Returns:
            Experiment type.
        """
        return self._type

    @property
    def figure_names(self) -> List[str]:
        """Return names of the figures associated with this experiment.

        Returns:
            Names of figures associated with this experiment.
        """
        return list(self._figures.keys())

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
    def notes(self) -> str:
        """Return experiment notes.

        Returns:
            Experiment notes.
        """
        return self._notes

    @notes.setter
    def notes(self, new_notes: str) -> None:
        """Update experiment notes.

        Args:
            new_notes: New experiment notes.
        """
        self._notes = new_notes
        if self.auto_save:
            self.save()

    @property
    def service(self) -> Optional['ExperimentServiceV1']:
        """Return the database service.

        Returns:
            Service that can be used to access this experiment in a database.
        """
        return self._service

    @service.setter
    def service(self, service: 'ExperimentServiceV1') -> None:
        """Set the service to be used for storing experiment data remotely.

        Args:
            service: Service to be used.

        Raises:
            ExperimentError: If a remote experiment service is already being used.
        """
        if self._service:
            raise ExperimentError("An experiment service is already being used.")
        self._service = service

    @property
    def source(self) -> Dict:
        """Return the class name and version."""
        return self._source

    def __str__(self):
        line = 51 * '-'
        n_res = len(self._analysis_results)
        status = self.status()
        ret = line
        ret += f'\nExperiment: {self.type}'
        ret += f'\nExperiment ID: {self.id}'
        ret += f'\nStatus: {status}'
        if status == "ERROR":
            ret += "\n".join(self._errors)
        ret += f'\nCircuits: {len(self._data)}'
        ret += f'\nAnalysis Results: {n_res}'
        ret += '\n' + line
        if n_res:
            ret += '\nLast Analysis Result'
            ret += f'\n{str(self._analysis_results[-1])}'
        return ret
