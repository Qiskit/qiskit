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
import json

from .experiment_data import ExperimentData, AnalysisResult
from .constants import ResultQuality


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
    and graphs.

    Each implementation of this service may use different data structure and
    should issue a warning on unsupported keywords.
    """
    version = 1

    @abstractmethod
    def create_experiment(
            self,
            experiment_type: str,
            backend_name: str,
            data: Any,
            experiment_id: Optional[str] = None,
            job_ids: Optional[List[str]] = None,
            tags: Optional[List[str]] = None,
            **kwargs: Any
    ) -> str:
        """Create a new experiment in the database.

        Args:
            experiment_type: Experiment type.
            backend_name: Name of the backend the experiment ran on.
            data: Data to be saved in the database.
            experiment_id: Experiment ID.
            job_ids: IDs of experiment jobs.
            tags: Tags to be associated with the experiment.
            kwargs: Additional keywords supported by the service provider.

        Returns:
            Experiment ID.

        Raises:
            ExperimentDataExists: If the experiment already exits.
        """
        pass

    @abstractmethod
    def update_experiment(
            self,
            experiment_id: str,
            data: Any,
            job_ids: List[str],
            tags: List[str],
            **kwargs: Any
    ) -> None:
        """Update an existing experiment.

        Args:
            experiment_id: Experiment ID.
            data: Data to be saved in the database.
            job_ids: IDs of experiment jobs.
            tags: Tags to be associated with the experiment.
            kwargs: Additional keywords supported by the service provider.

        Raises:
            ExperimentDataNotFound: If the experiment does not exist.
        """
        pass

    @abstractmethod
    def experiment(self, experiment_id: str) -> Dict:
        """Retrieve a previously stored experiment.

        Args:
            experiment_id: Experiment ID.

        Returns:
            Retrieved experiment.

        Raises:
            ExperimentDataNotFound: If the experiment does not exist.
        """
        pass

    @abstractmethod
    def experiments(self, limit: Optional[int] = 10, **filters: Any) -> List[ExperimentData]:
        """Retrieve all experiments, with optional filtering.

        Args:
            limit: Number of experiments to retrieve. ``None`` means no limit.
            **filters: Additional filtering keywords supported by the service provider.

        Returns:
            A list of experiments.
        """
        pass

    @abstractmethod
    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment.

        Args:
            experiment_id: Experiment ID.
        """
        pass

    @abstractmethod
    def create_analysis_result(
            self,
            experiment_id: str,
            data: Dict,
            result_type: str,
            tags: Optional[List[str]] = None,
            quality: Union[ResultQuality, str] = ResultQuality.NO_INFORMATION,
            result_id: Optional[str] = None,
            **kwargs: Any
    ) -> str:
        """Create a new analysis result in the database.

        Args:
            experiment_id: ID of the experiment this result is for.
            data: Result data to be stored.
            result_type: Analysis result type.
            tags: Tags to be associated with the analysis result.
            quality: Quality of the analysis results. One of
                ``Human Bad``, ``Computer Bad``, ``No Information``,
                ``Human Good``, and ``Computer Good``.
            result_id: Analysis result ID.
            kwargs: Additional keywords supported by the service provider.

        Returns:
            Analysis result ID.

        Raises:
            ExperimentDataExists: If the analysis result already exits.
        """
        pass

    @abstractmethod
    def update_analysis_result(
            self,
            result_id: str,
            data: Optional[Dict] = None,
            tags: Optional[List[str]] = None,
            quality: Union[ResultQuality, str] = ResultQuality.NO_INFORMATION,
            **kwargs: Any
    ) -> None:
        """Update an existing analysis result.

        Args:
            result_id: Analysis result ID.
            data: Result data to be stored.
            quality: Quality of the analysis results. One of
                ``Human Bad``, ``Computer Bad``, ``No Information``,
                ``Human Good``, and ``Computer Good``.
            tags: Tags to be associated with the analysis result.
            kwargs: Additional keywords supported by the service provider.

        Raises:
            ExperimentDataNotFound: If the analysis result does not exist.
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
            ExperimentDataNotFound: If the analysis result does not exist.
        """
        pass

    @abstractmethod
    def analysis_results(
            self,
            limit: Optional[int] = 10,
            experiment_id: Optional[str] = None,
            **filters: Any
    ) -> List[AnalysisResult]:
        """Retrieve all analysis results, with optional filtering.

        Args:
            limit: Number of analysis results to retrieve. ``None`` means no limit.
            experiment_id: Experiment ID used for filtering.
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
        """
        pass

    @abstractmethod
    def create_graph(
            self,
            experiment_id: str,
            graph: Union[str, bytes],
            graph_name: Optional[str]
    ) -> Tuple[str, int]:
        """Store a new graph in the database.

        Args:
            experiment_id: ID of the experiment this graph is for.
            graph: Name of the graph file or graph data to store.
            graph_name: Name of the graph. If ``None``, the graph file name, if
                given, or a generated name is used.

        Returns:
            A tuple of the name and size of the saved graph.

        Raises:
            ExperimentDataExists: If the graph already exits.
        """
        pass

    @abstractmethod
    def update_graph(
            self,
            experiment_id: str,
            graph: Union[str, bytes],
            graph_name: str
    ) -> Dict:
        """Update an existing graph.

        Args:
            experiment_id: Experiment ID.
            graph: Name of the graph file or graph data to store.
            graph_name: Name of the graph.

        Returns:
            A dictionary with name and size of the uploaded graph.

        Raises:
            ExperimentDataNotFound: If the graph does not exist.
        """
        pass

    @abstractmethod
    def graph(
            self,
            experiment_id: str,
            graph_name: str,
            file_name: Optional[str] = None
    ) -> Union[int, bytes]:
        """Retrieve an existing graph.

        Args:
            experiment_id: Experiment ID.
            graph_name: Name of the graph.
            file_name: Name of the local file to save the graph to. If ``None``,
                the content of the graph is returned instead.

        Returns:
            The size of the graph if `file_name` is specified. Otherwise the
            content of the graph in bytes.

        Raises:
            ExperimentDataNotFound: If the graph does not exist.
        """
        pass

    @abstractmethod
    def delete_graph(
            self,
            experiment_id: str,
            graph_name: str,
    ) -> None:
        """Delete an existing graph.

        Args:
            experiment_id: Experiment ID.
            graph_name: Name of the graph.
        """
        pass
