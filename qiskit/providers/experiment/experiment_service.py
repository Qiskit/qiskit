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

"""Experiment service abstract interface."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, Union, Tuple

from .experiment_data import ExperimentDataV1 as ExperimentData
from .analysis_result import AnalysisResultV1 as AnalysisResult
from .constants import ResultQuality
from .device_component import DeviceComponent


class ExperimentService:
    """Base common type for all versioned ExperimentService abstract classes.

    Note this class should not be inherited from directly, it is intended
    to be used for type checking. When implementing a subclass you should use
    the versioned abstract classes as the parent class and not this class
    directly.
    """
    version = 0


class ExperimentServiceV1(ExperimentService, ABC):
    """Class to provide experiment service.

    The experiment service allows you to store experiment data and metadata
    in a database. An experiment can have one or more jobs, analysis results,
    and figures.

    Each implementation of this service may use different data structure and
    should issue a warning on unsupported keywords.
    """
    version = 1

    def __init__(self):
        """Initialize an ExperimentService instance."""
        self._options = self._default_options()

    @classmethod
    @abstractmethod
    def _default_options(cls) -> Dict:
        """Return the default options

        Returns:
            A dictionary of default options.
        """
        pass

    @abstractmethod
    def create_experiment(
            self,
            experiment_type: str,
            backend_name: str,
            metadata: Optional[Dict] = None,
            experiment_id: Optional[str] = None,
            job_ids: Optional[List[str]] = None,
            tags: Optional[List[str]] = None,
            notes: Optional[str] = None,
            **kwargs: Any
    ) -> str:
        """Create a new experiment in the database.

        Args:
            experiment_type: Experiment type.
            backend_name: Name of the backend the experiment ran on.
            metadata: Experiment metadata.
            experiment_id: Experiment ID. It must be in the ``uuid4`` format.
                One will be generated if not supplied.
            job_ids: IDs of experiment jobs.
            tags: Tags to be associated with the experiment.
            notes: Freeform notes about the experiment.
            kwargs: Additional keywords supported by the service provider.

        Returns:
            Experiment ID.

        Raises:
            ExperimentEntryExists: If the experiment already exits.
        """
        pass

    @abstractmethod
    def update_experiment(
            self,
            experiment_id: str,
            metadata: Optional[Dict] = None,
            job_ids: Optional[List[str]] = None,
            notes: Optional[str] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any
    ) -> None:
        """Update an existing experiment.

        Args:
            experiment_id: Experiment ID.
            metadata: Experiment metadata.
            job_ids: IDs of experiment jobs.
            notes: Freeform notes about the experiment.
            tags: Tags to be associated with the experiment.
            kwargs: Additional keywords supported by the service provider.

        Raises:
            ExperimentEntryNotFound: If the experiment does not exist.
        """
        pass

    @abstractmethod
    def experiment(self, experiment_id: str) -> ExperimentData:
        """Retrieve a previously stored experiment.

        Args:
            experiment_id: Experiment ID.

        Returns:
            Retrieved experiment.

        Raises:
            ExperimentEntryNotFound: If the experiment does not exist.
        """
        pass

    @abstractmethod
    def experiments(
            self,
            limit: Optional[int] = 10,
            device_components: Optional[Union[str, DeviceComponent]] = None,
            experiment_type: Optional[str] = None,
            backend_name: Optional[str] = None,
            tags: Optional[List[str]] = None,
            tags_operator: Optional[str] = "OR",
            **filters: Any) -> List[ExperimentData]:
        """Retrieve all experiment data, with optional filtering.

        Args:
            limit: Number of experiments to retrieve. ``None`` means no limit.
            device_components: Filter by device components. An experiment must have analysis
                results with device components matching the given list exactly to be included.
            experiment_type: Experiment type used for filtering.
            backend_name: Backend name used for filtering.
            tags: Filter by tags assigned to experiments. This can be used
                with `tags_operator` for granular filtering.
            tags_operator: Logical operator to use when filtering by tags. Valid
                values are "AND" and "OR":

                    * If "AND" is specified, then an experiment must have all of the tags
                      specified in `tags` to be included.
                    * If "OR" is specified, then an experiment only needs to have any
                      of the tags specified in `tags` to be included.

            **filters: Additional filtering keywords supported by the service provider.

        Returns:
            A list of experiment data.
        """
        pass

    @abstractmethod
    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment.

        Args:
            experiment_id: Experiment ID.

        Raises:
            ExperimentEntryNotFound: If the experiment does not exist.
        """
        pass

    @abstractmethod
    def create_analysis_result(
            self,
            experiment_id: str,
            data: Dict,
            result_type: str,
            device_components: Optional[Union[str, DeviceComponent]] = None,
            tags: Optional[List[str]] = None,
            quality: Union[ResultQuality, str] = ResultQuality.UNKNOWN,
            verified: bool = False,
            result_id: Optional[str] = None,
            **kwargs: Any
    ) -> str:
        """Create a new analysis result in the database.

        Args:
            experiment_id: ID of the experiment this result is for.
            data: Result data to be stored.
            result_type: Analysis result type.
            device_components: Target device components, such as qubits.
            tags: Tags to be associated with the analysis result.
            quality: Quality of this analysis.
            verified: Whether the result quality has been verified.
            result_id: Analysis result ID. It must be in the ``uuid4`` format.
                One will be generated if not supplied.
            kwargs: Additional keywords supported by the service provider.

        Returns:
            Analysis result ID.

        Raises:
            ExperimentEntryExists: If the analysis result already exits.
        """
        pass

    @abstractmethod
    def update_analysis_result(
            self,
            result_id: str,
            data: Optional[Dict] = None,
            tags: Optional[List[str]] = None,
            quality: Union[ResultQuality, str] = ResultQuality.UNKNOWN,
            verified: bool = False,
            **kwargs: Any
    ) -> None:
        """Update an existing analysis result.

        Args:
            result_id: Analysis result ID.
            data: Result data to be stored.
            quality: Quality of this analysis.
            verified: Whether the result quality has been verified.
            tags: Tags to be associated with the analysis result.
            kwargs: Additional keywords supported by the service provider.

        Raises:
            ExperimentEntryNotFound: If the analysis result does not exist.
        """
        pass

    @abstractmethod
    def analysis_result(self, analysis_result_id: str) -> AnalysisResult:
        """Retrieve a previously stored experiment.

        Args:
            analysis_result_id: Analysis result ID.

        Returns:
            Retrieved analysis result.

        Raises:
            ExperimentEntryNotFound: If the analysis result does not exist.
        """
        pass

    @abstractmethod
    def analysis_results(
            self,
            limit: Optional[int] = 10,
            device_components: Optional[Union[str, DeviceComponent]] = None,
            experiment_id: Optional[str] = None,
            result_type: Optional[str] = None,
            backend_name: Optional[str] = None,
            quality: Optional[Union[ResultQuality, str]] = None,
            verified: Optional[bool] = None,
            tags: Optional[List[str]] = None,
            tags_operator: Optional[str] = "OR",
            **filters: Any
    ) -> List[AnalysisResult]:
        """Retrieve all analysis results, with optional filtering.

        Args:
            limit: Number of analysis results to retrieve. ``None`` means no limit.
            device_components: Target device components, such as qubits.
            experiment_id: Experiment ID used for filtering.
            result_type: Analysis result type used for filtering.
            backend_name: Backend name used for filtering. If specified, analysis
                results associated with experiments on that backend are returned.
            quality: Quality value used for filtering.
            verified: Whether the result quality has been verified.
            tags: Filter by tags assigned to analysis results. This can be used
                with `tags_operator` for granular filtering.
            tags_operator: Logical operator to use when filtering by tags. Valid
                values are "AND" and "OR":

                    * If "AND" is specified, then an analysis result must have all of the tags
                      specified in `tags` to be included.
                    * If "OR" is specified, then an analysis result only needs to have any
                      of the tags specified in `tags` to be included.

            **filters: Additional filtering keywords supported by the service provider.

        Returns:
            A list of analysis results.
        """
        pass

    @abstractmethod
    def delete_analysis_result(self, analysis_result_id: str) -> None:
        """Delete an analysis result.

        Args:
            analysis_result_id: Analysis result ID.

        Raises:
            ExperimentEntryNotFound: If the analysis result does not exist.
        """
        pass

    @abstractmethod
    def create_figure(
            self,
            experiment_id: str,
            figure: Union[str, bytes],
            figure_name: Optional[str]
    ) -> Tuple[str, int]:
        """Store a new figure in the database.

        Args:
            experiment_id: ID of the experiment this figure is for.
            figure: Name of the figure file or figure data to store.
            figure_name: Name of the figure. If ``None``, the figure file name, if
                given, or a generated name is used.

        Returns:
            A tuple of the name and size of the saved figure.

        Raises:
            ExperimentEntryExists: If the figure already exits.
        """
        pass

    @abstractmethod
    def update_figure(
            self,
            experiment_id: str,
            figure: Union[str, bytes],
            figure_name: str
    ) -> Dict:
        """Update an existing figure.

        Args:
            experiment_id: Experiment ID.
            figure: Name of the figure file or figure data to store.
            figure_name: Name of the figure.

        Returns:
            A dictionary with name and size of the uploaded figure.

        Raises:
            ExperimentEntryNotFound: If the figure does not exist.
        """
        pass

    @abstractmethod
    def figure(
            self,
            experiment_id: str,
            figure_name: str,
            file_name: Optional[str] = None
    ) -> Union[int, bytes]:
        """Retrieve an existing figure.

        Args:
            experiment_id: Experiment ID.
            figure_name: Name of the figure.
            file_name: Name of the local file to save the figure to. If ``None``,
                the content of the figure is returned instead.

        Returns:
            The size of the figure if `file_name` is specified. Otherwise the
            content of the figure in bytes.

        Raises:
            ExperimentEntryNotFound: If the figure does not exist.
        """
        pass

    @abstractmethod
    def delete_figure(
            self,
            experiment_id: str,
            figure_name: str,
    ) -> None:
        """Delete an existing figure.

        Args:
            experiment_id: Experiment ID.
            figure_name: Name of the figure.

        Raises:
            ExperimentEntryNotFound: If the figure does not exist.
        """
        pass

    def set_options(self, **fields):
        """Set the options fields for the service.

        Args:
            fields: The fields to update the options

        Raises:
            AttributeError: If the field passed in is not part of the
                options
        """
        for field in fields:
            if field not in self._options:
                raise AttributeError(
                    "Options field %s is not valid for this "
                    "service." % field)
        self._options.update(**fields)

    def option(self, field: str) -> Any:
        """Get the value of the specified option.

        Args:
            field: Option field to retrieve.

        Returns:
            Option value.
        """
        if field not in self._options:
            raise AttributeError(
                f"Options field {field} is not valid for this service.")
        return self._options[field]
