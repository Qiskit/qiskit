# -*- coding: utf-8 -*-

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


class ResultData:
    """The result data class is a container class for results from an experiment.

    ``ResultData`` objects contain the result data from an execution and the
    associated metadata for it. There should be a single data type for the
    result, if an experiment returns more than 1 result data type (for example,
    counts and snapshots) a separate ``ResultData`` object should be returned
    for each type.
    """

    version = 1

    def __init__(self, experiment, data_type, data, **metadata):
        """Create a new ResultData object

        Args:
            experiment: The circuit or pulse schedule object that the result
                data is from
            data_type (str): The data type of the result. Can be used for
                filtering. Any type is acceptable, but 4 value ``counts``,
                ``statevector``, ``unitary``, and ``snapshot`` are special
                cases in the :class:`~qiskit.providers.v2.Job` base class
                because there are built-in methods to get those.
            data: The data from the run
            metadata: Any key value metadata to associate with the result data
        """
        self.experiment = experiment
        self.data = data
        self.metadata = metadata
