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
from typing import Optional, List, Any, Union, Callable, Dict
import copy
from concurrent import futures
from functools import wraps
import traceback
import contextlib
from collections import deque
from datetime import datetime
import io

from qiskit.providers import Job, BaseJob, Backend, BaseBackend, Provider
from qiskit.result import Result
from qiskit.providers.jobstatus import JobStatus, JOB_FINAL_STATES
from qiskit.providers.exceptions import JobError
from qiskit.visualization import HAS_MATPLOTLIB

from .exceptions import ExperimentError, ExperimentEntryNotFound, ExperimentEntryExists
from .analysis_result import AnalysisResultV1 as AnalysisResult
from .json import NumpyEncoder, NumpyDecoder
from .utils import save_data, qiskit_version, ThreadSafeOrderedDict, ThreadSafeList

LOG = logging.getLogger(__name__)


def auto_save(func: Callable):
    """Decorate the input function to auto save data."""

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
        experiment_type: str,
        backend: Optional[Union[Backend, BaseBackend]] = None,
        experiment_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        job_ids: Optional[List[str]] = None,
        share_level: Optional[str] = None,
        metadata: Optional[Dict] = None,
        figure_names: Optional[List[str]] = None,
        notes: Optional[str] = None,
        **kwargs,
    ):
        """Initializes the experiment data.

        Args:
            experiment_type: Experiment type.
            backend: Backend the experiment runs on.
            experiment_id: Experiment ID. One will be generated if not supplied.
            tags: Tags to be associated with the experiment.
            job_ids: IDs of jobs submitted for the experiment.
            share_level: Whether this experiment can be shared with others. This
                is applicable only if the experiment service supports sharing. See
                the specific service provider's documentation on valid values.
            metadata: Additional experiment metadata.
            figure_names: Name of figures associated with this experiment.
            notes: Freeform notes about the experiment.
            **kwargs: Additional experiment attributes.

        Raises:
            ExperimentError: If an input argument is invalid.
        """
        metadata = metadata or {}
        self._metadata = copy.deepcopy(metadata)
        self._source = self._metadata.pop(
            "_source",
            {
                "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
                "metadata_version": self._metadata_version,
                "qiskit_version": qiskit_version(),
            },
        )

        self._service = None
        self._backend = backend
        self.auto_save = False
        self._set_service_from_backend(backend)

        self._id = experiment_id or str(uuid.uuid4())
        self._type = experiment_type
        self._tags = tags or []
        self._share_level = share_level
        self._notes = notes or ""

        self._jobs = ThreadSafeOrderedDict(job_ids or [])
        self._job_futures = ThreadSafeList()
        self._errors = []

        self._data = ThreadSafeList()
        self._figures = ThreadSafeOrderedDict(figure_names or [])
        self._analysis_results = ThreadSafeOrderedDict()

        self._deleted_figures = deque()
        self._deleted_analysis_results = deque()

        self._created_in_db = False
        self._extra_data = kwargs

    def _set_service_from_backend(self, backend: Union[Backend, BaseBackend]) -> None:
        """Set the service to be used from the input backend.

        Args:
            backend: Backend whose provider may offer experiment service.
        """
        with contextlib.suppress(Exception):
            self._service = backend.provider().service("experiment")
        with contextlib.suppress(Exception):
            self.auto_save = self._service.option("auto_save")

    def add_data(
        self,
        data: Union[Result, List[Result], Job, List[Job], Dict, List[Dict]],
        post_processing_callback: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Add experiment data.

        Note:
            This method is not thread safe and should not be called by the
            `post_processing` function.

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
            **kwargs: Keyword arguments to be passed to the callback function.
        Raises:
            TypeError: If the input data type is invalid.
        """
        if isinstance(data, dict):
            self._add_single_data(data)
        elif isinstance(data, Result):
            self._add_result_data(data)
        elif isinstance(data, (Job, BaseJob)):
            if self.backend and self.backend != data.backend():
                LOG.warning(
                    "Adding a job from a backend (%s) that is different "
                    "than the current ExperimentData backend (%s). "
                    "The new backend will be used, but "
                    "service is not changed if one already exists.",
                    data.backend(),
                    self.backend,
                )
            self._backend = data.backend()
            if not self._service:
                self._set_service_from_backend(self._backend)

            self._jobs[data.job_id()] = data
            self._job_futures.append(
                (
                    data,
                    self._executor.submit(
                        self._wait_for_job, data, post_processing_callback, **kwargs
                    ),
                )
            )
            if self.auto_save:
                self.save()
        elif isinstance(data, list):
            # TODO use loop instead of recursion for fewer save()
            for dat in data:
                self.add_data(dat)
        else:
            raise TypeError(f"Invalid data type {type(data)}.")

    def _wait_for_job(
        self,
        job: Union[Job, BaseJob],
        job_done_callback: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        """Wait for a job to finish.

        Args:
            job: Job to wait for.
            job_done_callback: Callback function to invoke when job finishes.
            **kwargs: Keyword arguments to be passed to the callback function.
        """
        LOG.debug("Waiting for job %s to finish.", job.job_id())
        try:
            self._add_result_data(job.result())
        except JobError as err:
            LOG.warning("Job %s failed: %s", job.job_id(), str(err))
            return
        if job_done_callback:
            job_done_callback(self, **kwargs)

    def _add_result_data(self, result: Result) -> None:
        """Add data from a Result object

        Args:
            result: Result object containing data to be added.
        """
        if result.job_id not in self._jobs:
            self._jobs[result.job_id] = None
        for i in range(len(result.results)):
            data = result.data(i)
            data["job_id"] = result.job_id
            if "counts" in data:
                # Format to Counts object rather than hex dict
                data["counts"] = result.get_counts(i)
            expr_result = result.results[i]
            if hasattr(expr_result, "header") and hasattr(expr_result.header, "metadata"):
                data["metadata"] = expr_result.header.metadata
            self._add_single_data(data)

    def _add_single_data(self, data: Dict[str, any]) -> None:
        """Add a single data dictionary to the experiment.

        Args:
            data: Data to be added.
        """
        self._data.append(data)

    def data(self, index: Optional[Union[int, slice, str]] = None) -> Union[Dict, List[Dict]]:
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

        Raises:
            TypeError: If the input `index` has an invalid type.
        """
        # Get job results if missing experiment data.
        if (not self._data) and self._provider:
            with self._jobs.lock:
                for jid in self._jobs:
                    if self._jobs[jid] is None:
                        try:
                            self._jobs[jid] = self._provider.retrieve_job(jid)
                        except Exception:  # pylint: disable=broad-except
                            pass
                    if self._jobs[jid] is not None:
                        self._add_result_data(self._jobs[jid].result())

        if index is None:
            return self._data.copy()
        if isinstance(index, (int, slice)):
            return self._data[index]
        if isinstance(index, str):
            return [data for data in self._data if data.get("job_id") == index]
        raise TypeError(f"Invalid index type {type(index)}.")

    @auto_save
    def add_figures(
        self,
        figures: Union[List[Union[str, bytes, "pyplot.Figure"]], str, bytes, "pyplot.Figure"],
        figure_names: Optional[Union[List[str], str]] = None,
        overwrite: bool = False,
        save_figure: Optional[bool] = None,
        service: Optional["ExperimentServiceV1"] = None,
    ) -> Union[str, List[str]]:
        """Add the experiment figure.

        Args:
            figures: Names of the figure files or figure data.
            figure_names: Names of the figures. If ``None``, use the figure file
                names, if given, or a generated name. If `figures` is a list, then
                `figure_names` must also be a list of the same length or ``None``.
            overwrite: Whether to overwrite the figure if one already exists with
                the same name.
            save_figure: Whether to save the figure in the database. If ``None``,
                the ``auto-save`` attribute is used.
            service: Experiment service to be used to update the database, if
                the figure is to be uploaded. If ``None``, the default service is used.

        Returns:
            Figure names.

        Raises:
            ExperimentEntryExists: If the figure with the same name already exists,
                and `overwrite=True` is not specified.
            ValueError: If an input parameter has an invalid value.
        """
        if (
            isinstance(figures, list)
            and figure_names is not None
            and (not isinstance(figure_names, list) or len(figures) != len(figure_names))
        ):
            raise ValueError(
                f"The parameter figure_names must be None or a list of "
                f"the same size as the parameter figures."
            )
        if not isinstance(figures, list):
            figures = [figures]
        if figure_names is not None and not isinstance(figure_names, list):
            figure_names = [figure_names]

        added_figs = []
        for idx, figure in enumerate(figures):
            if figure_names is None:
                if isinstance(figure, str):
                    fig_name = figure
                else:
                    fig_name = f"figure_{self.experiment_id}_{datetime.now().isoformat()}"
            else:
                fig_name = figure_names[idx]

            existing_figure = fig_name in self._figures
            if existing_figure and not overwrite:
                raise ExperimentEntryExists(
                    f"A figure with the name {fig_name} for this experiment "
                    f"already exists. Specify overwrite=True if you "
                    f"want to overwrite it."
                )
            figure_data = None
            if isinstance(figure, str):
                with open(figure, "rb") as file:
                    figure = file.read()
            elif HAS_MATPLOTLIB:
                from matplotlib import pyplot
                if isinstance(figure, pyplot.Figure):
                    buf = io.BytesIO()
                    figure.savefig(buf, format='svg')
                    buf.seek(0)
                    figure_data = buf.read()
                    buf.close()

            self._figures[fig_name] = figure

            service = service or self._service
            save = save_figure if save_figure is not None else self.auto_save
            if save and service:
                data = {
                    "experiment_id": self.experiment_id,
                    "figure": figure_data or figure,
                    "figure_name": fig_name,
                }
                save_data(
                    is_new=(not existing_figure),
                    new_func=service.create_figure,
                    update_func=service.update_figure,
                    new_data={},
                    update_data=data,
                )
            added_figs.append(fig_name)

        return added_figs if len(added_figs) > 1 else added_figs[0]

    @auto_save
    def delete_figure(
        self,
        figure_key: Union[str, int],
        service: Optional["ExperimentServiceV1"] = None,
    ) -> str:
        """Add the experiment figure.

        Args:
            figure_key: Name or index of the figure.
            service: Experiment service to be used to update the database. If ``None``,
                the default service is used.

        Returns:
            Figure name.

        Raises:
            ExperimentEntryNotFound: If the figure is not found.
        """
        if isinstance(figure_key, int):
            figure_key = self._figures.keys()[figure_key]
        elif figure_key not in self._figures:
            raise ExperimentEntryNotFound(f"Figure {figure_key} not found.")

        del self._figures[figure_key]
        self._deleted_figures.append(figure_key)

        service = service or self._service
        if service and self.auto_save:
            with contextlib.suppress(ExperimentEntryNotFound):
                self.service.delete_figure(experiment_id=self.experiment_id, figure_name=figure_key)
            self._deleted_figures.remove(figure_key)

        return figure_key

    def figure(
        self, figure_key: Union[str, int], file_name: Optional[str] = None
    ) -> Union[int, bytes]:
        """Retrieve the specified experiment figure.

        Args:
            figure_key: Name or index of the figure.
            file_name: Name of the local file to save the figure to. If ``None``,
                the content of the figure is returned instead.

        Returns:
            The size of the figure if `file_name` is specified. Otherwise the
            content of the figure in bytes.

        Raises:
            ExperimentEntryNotFound: If the figure cannot be found.
        """
        if isinstance(figure_key, int):
            figure_key = self._figures.keys()[figure_key]

        figure_data = self._figures.get(figure_key, None)
        if figure_data is None and self.service:
            figure_data = self.service.figure(
                experiment_id=self.experiment_id, figure_name=figure_key
            )
            self._figures[figure_key] = figure_data

        if figure_data is None:
            raise ExperimentEntryNotFound(f"Figure {figure_key} not found.")

        if file_name:
            with open(file_name, "wb") as output:
                num_bytes = output.write(figure_data)
                return num_bytes
        return figure_data

    @auto_save
    def add_analysis_results(
        self,
        results: Union[AnalysisResult, List[AnalysisResult]],
        service: "ExperimentServiceV1" = None,
    ) -> None:
        """Save the analysis result.

        Args:
            results: Analysis results to be saved.
            service: Experiment service to be used to update the database. If ``None``,
                the default service is used.
        """
        if not isinstance(results, list):
            results = [results]

        for result in results:
            self._analysis_results[result.result_id] = result

            with contextlib.suppress(ExperimentError):
                result.service = self.service
                result.auto_save = self.auto_save

            use_service = service or self.service
            if self.auto_save and use_service:
                result.save(service=service)

    @auto_save
    def delete_analysis_result(
        self, result_key: Union[int, str], service: "ExperimentServiceV1" = None
    ) -> str:
        """Delete the analysis result.

        Args:
            result_key: ID or index of the analysis result to be delete.
            service: Experiment service to be used to update the database. If ``None``,
                the default service is used.

        Returns:
            Analysis result ID.

        Raises:
            ExperimentEntryNotFound: If analysis result not found.
        """
        if isinstance(result_key, int):
            result_key = self._analysis_results.keys()[result_key]
        elif result_key not in self._analysis_results:
            raise ExperimentEntryNotFound(f"Analysis result {result_key} not found.")

        del self._analysis_results[result_key]
        self._deleted_analysis_results.append(result_key)

        service = service or self._service
        if service and self.auto_save:
            with contextlib.suppress(ExperimentEntryNotFound):
                self.service.delete_analysis_result(result_id=result_key)
            self._deleted_analysis_results.remove(result_key)

        return result_key

    def analysis_result(
        self, index: Optional[Union[int, slice, str]] = None, refresh: bool = False
    ) -> Union[AnalysisResult, List[AnalysisResult]]:
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

        Raises:
            TypeError: If the input `index` has an invalid type.
            ExperimentEntryNotFound: If the entry cannot be found.
        """
        if self.service and (not self._analysis_results or refresh):
            retrieved_results = self.service.analysis_results(
                experiment_id=self.experiment_id, limit=None
            )
            for result in retrieved_results:
                self._analysis_results[result.result_id] = result

        if index is None:
            return self._analysis_results.values()
        if isinstance(index, (int, slice)):
            return self._analysis_results.values()[index]
        if isinstance(index, str):
            if index not in self._analysis_results:
                raise ExperimentEntryNotFound(f"Analysis result {index} not found.")
            return self._analysis_results[index]

        raise TypeError(f"Invalid index type {type(index)}.")

    def save(self, service: Optional["ExperimentServiceV1"] = None) -> None:
        """Save this experiment in the database.

        Note:
            Not all experiment properties are saved.
            See :meth:`qiskit.providers.experiment.ExperimentServiceV1.create_experiment`
            for fields that are saved.

        Note:
            Note that this method does not save analysis results nor figures. Use
            ``save_all()`` if you want to save those.

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

        if not self._backend:
            LOG.warning("Experiment cannot be saved because backend is missing.")
            return

        metadata = json.loads(self.serialize_metadata())
        metadata["_source"] = self._source

        update_data = {
            "experiment_id": self._id,
            "metadata": metadata,
            "job_ids": self.job_ids,
            "tags": self.tags(),
            "notes": self.notes,
        }
        new_data = {"experiment_type": self._type, "backend_name": self._backend.name()}
        if self.share_level:
            update_data["share_level"] = self.share_level

        self._created_in_db, _ = save_data(
            is_new=(not self._created_in_db),
            new_func=service.create_experiment,
            update_func=service.update_experiment,
            new_data=new_data,
            update_data=update_data,
        )

    def save_all(self, service: Optional["ExperimentServiceV1"] = None) -> None:
        """Save this experiment and its analysis results and figures in the database.

        Note:
            Depending on the amount of data, this operation could take a while.

        Args:
            service: Experiment service to be used to save the data.
                If ``None``, the provider used to submit jobs will be used.

        Raises:
            ExperimentError: If the experiment contains invalid data.
        """
        # TODO - track changes
        use_service = service or self._service
        if not use_service:
            LOG.warning("Experiment cannot be saved because no experiment service is available.")
            return

        self.save(service=use_service)
        for result in self._analysis_results.values():
            result.save(service)

        for result in self._deleted_analysis_results.copy():
            try:
                use_service.delete_analysis_result(result_id=result)
            except ExperimentEntryNotFound:
                pass
            self._deleted_analysis_results.remove(result)

        with self._figures.lock:
            for name, figure in self._figures.items():
                data = {"experiment_id": self.experiment_id, "figure": figure, "figure_name": name}
                save_data(
                    is_new=True,
                    new_func=use_service.create_figure,
                    update_func=use_service.update_figure,
                    new_data={},
                    update_data=data,
                )

        for name in self._deleted_figures.copy():
            try:
                use_service.delete_figure(experiment_id=self.experiment_id, figure_name=name)
            except ExperimentEntryNotFound:
                pass
            self._deleted_figures.remove(name)

    def serialize_metadata(self) -> str:
        """Serialize experiment metadata into a JSON string.

        Returns:
            Serialized JSON string.
        """
        return json.dumps(self._metadata, cls=self._json_encoder)

    @classmethod
    def deserialize_metadata(cls, data: str) -> Any:
        """Deserialize experiment metadata from a JSON string.

        Args:
            data: Data to be deserialized.

        Returns:
            Deserialized data.
        """
        return json.loads(data, cls=cls._json_decoder)

    @classmethod
    def from_data(
        cls,
        experiment_type: str,
        experiment_id: str,
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> "ExperimentDataV1":
        """Reconstruct an ExperimentData using the input data.

        Args:
            experiment_type: Experiment type.
            experiment_id: Experiment ID. One will be generated if not supplied.
            metadata: Additional experiment metadata.
            **kwargs: Additional experiment attributes.

        Returns:
            An ExperimentData instance.
        """
        if metadata:
            metadata = cls.deserialize_metadata(json.dumps(metadata))
        return cls(
            experiment_type=experiment_type,
            experiment_id=experiment_id,
            metadata=metadata,
            **kwargs,
        )

    def cancel_jobs(self) -> None:
        """Cancel any running jobs."""
        for job, fut in self._job_futures.copy():
            if not fut.done() and job.status() not in JOB_FINAL_STATES:
                try:
                    job.cancel()
                except Exception as err:  # pylint: disable=broad-except
                    LOG.info("Unable to cancel job %s: %s", job.job_id(), err)

    def block_for_results(self) -> None:
        """Block until all pending jobs and their post processing finish."""
        for job, fut in self._job_futures.copy():
            LOG.info("Waiting for job %s and its post processing to finish.", job.job_id())
            with contextlib.suppress(Exception):
                fut.result()

    def status(self) -> str:
        """Return the data processing status.

        If the experiment consists of multiple jobs, the returned status is mapped
        in the following order:

                * INITIALIZING - if any job is being initialized.
                * VALIDATING - if any job is being validated.
                * QUEUED - if any job is queued.
                * RUNNING - if any job is still running.
                * ERROR - if any job incurred an error.
                * CANCELLED - if any job is cancelled.
                * POST_PROCESSING - if any of the post-processing functions is still running.
                * DONE - if all jobs and their post-processing functions finished.

        Returns:
            Data processing status.
        """
        if all(
            len(container) == 0
            for container in [self._data, self._jobs, self._figures, self._analysis_results]
        ):
            return "INITIALIZING"

        statuses = set()
        with self._job_futures.lock:
            for idx, item in enumerate(self._job_futures):
                job, fut = item
                job_status = job.status()
                statuses.add(job_status)
                if job_status == JobStatus.ERROR:
                    job_err = "."
                    if hasattr(job, "error_message"):
                        job_err = ": " + job.error_message()
                    self._errors.append(f"Job {job.job_id()} failed{job_err}")

                if fut.done():
                    self._job_futures[idx] = None
                    ex = fut.exception()
                    if ex:
                        self._errors.append(
                            f"Post processing for job {job.job_id()} failed: \n"
                            + "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
                        )
                        statuses.add(JobStatus.ERROR)

            self._job_futures = ThreadSafeList(list(filter(None, self._job_futures)))

        for stat in [
            JobStatus.INITIALIZING,
            JobStatus.VALIDATING,
            JobStatus.QUEUED,
            JobStatus.RUNNING,
            JobStatus.ERROR,
            JobStatus.CANCELLED,
        ]:
            if stat in statuses:
                return stat.name

        if self._job_futures:
            return "POST_PROCESSING"

        return "DONE"

    def errors(self) -> str:
        """Return errors encountered.

        Returns:
            Experiment errors.
        """
        self.status()  # Collect new errors.
        return "\n".join(self._errors)

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

    def metadata(self) -> Dict:
        """Return experiment metadata.

        Returns:
            Experiment metadata.
        """
        return self._metadata

    @auto_save
    def update_metadata(self, metadata: Dict) -> None:
        """Update metadata for this experiment.

        Args:
            metadata: New metadata for the experiment.
        """
        self._metadata = copy.deepcopy(metadata)

    @property
    def _provider(self) -> Optional[Provider]:
        """Return the provider.

        Returns:
            Provider used for the experiment, or ``None`` if unknown.
        """
        if self._backend is None:
            return None
        return self._backend.provider()

    @property
    def experiment_id(self) -> str:
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
        return self._jobs.keys()

    @property
    def backend(self) -> Optional[Union[BaseBackend, Backend]]:
        """Return backend.

        Returns:
            Backend this experiment is for, or ``None`` if backend is unknown.
        """
        return self._backend

    @property
    def experiment_type(self) -> str:
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
        return self._figures.keys()

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
    def service(self) -> Optional["ExperimentServiceV1"]:
        """Return the database service.

        Returns:
            Service that can be used to access this experiment in a database.
        """
        return self._service

    @service.setter
    def service(self, service: "ExperimentServiceV1") -> None:
        """Set the service to be used for storing experiment data remotely.

        Args:
            service: Service to be used.

        Raises:
            ExperimentError: If a remote experiment service is already being used.
        """
        if self._service:
            raise ExperimentError("An experiment service is already being used.")
        self._service = service
        with contextlib.suppress(Exception):
            self.auto_save = self._service.option("auto_save")

    @property
    def source(self) -> Dict:
        """Return the class name and version."""
        return self._source

    def __str__(self):
        line = 51 * "-"
        n_res = len(self._analysis_results)
        status = self.status()
        ret = line
        ret += f"\nExperiment: {self.experiment_type}"
        ret += f"\nExperiment ID: {self.experiment_id}"
        ret += f"\nStatus: {status}"
        if status == "ERROR":
            ret += "\n  "
            ret += "\n  ".join(self._errors)
        ret += f"\nData: {len(self._data)}"
        ret += f"\nAnalysis Results: {n_res}"
        ret += "\n" + line
        if n_res:
            ret += "\nLast Analysis Result:"
            ret += f"\n{str(self._analysis_results.values()[-1])}"
        return ret

    def __getattr__(self, name: str) -> Any:
        try:
            return self._extra_data[name]
        except KeyError:
            # pylint: disable=raise-missing-from
            raise AttributeError("Attribute %s is not defined" % name)
