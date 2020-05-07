# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Schema and helper models for schema-conformant Results."""

import copy

from qiskit.qobj.utils import MeasReturnType, MeasLevel
from qiskit.qobj import QobjExperimentHeader
from qiskit.exceptions import QiskitError


class ExperimentResultData:
    """Class representing experiment result data"""

    def __init__(self, counts=None, snapshots=None, memory=None,
                 statevector=None, unitary=None):
        """Initialize an ExperimentalResult Data class

        Args:
            counts (dict): A dictionary where the keys are the result in
                hexadecimal as string of the format "0xff" and the value
                is the number of counts for that result
            snapshots (dict): A dictionary where the key is the snapshot
                slot and the value is a dictionary of the snapshots for
                that slot.
            memory (list): A list of results per shot if the run had
                memory enabled
            statevector (list or numpy.array): A list or numpy array of the
                statevector result
            unitary (list or numpy.array): A list or numpy arrray of the
                unitary result
        """

        if counts is not None:
            self.counts = counts
        if snapshots is not None:
            self.snapshots = snapshots
        if memory is not None:
            self.memory = memory
        if statevector is not None:
            self.statevector = statevector
        if unitary is not None:
            self.unitary = unitary

    def to_dict(self):
        """Return a dictionary format representation of the ExperimentResultData

        Returns:
            dict: The dictionary form of the ExperimentResultData
        """
        out_dict = {}
        for field in ['counts', 'snapshots', 'memory', 'statevector',
                      'unitary']:
            if hasattr(self, field):
                out_dict[field] = getattr(self, field)
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new ExperimentResultData object from a dictionary.

        Args:
            data (dict): A dictionary representing the ExperimentResultData to
                         create. It will be in the same format as output by
                         :meth:`to_dict`
        Returns:
            ExperimentResultData: The ``ExperimentResultData`` object from the
                                  input dictionary.
        """
        in_data = copy.copy(data)
        return cls(**in_data)


class ExperimentResult:
    """Class representing an Experiment Result.

    Attributes:
        shots (int or tuple): the starting and ending shot for this data.
        success (bool): if true, we can trust results for this experiment.
        data (ExperimentResultData): results information.
        meas_level (int): Measurement result level.
    """

    _metadata = {}

    def __init__(self, shots, success, data, meas_level=MeasLevel.CLASSIFIED,
                 status=None, seed=None, meas_return=None, header=None,
                 **kwargs):
        """Initialize an ExperimentResult object.

        Args:
            shots(int or tuple): if an integer the number of shots or if a
                tuple the starting and ending shot for this data
            success (bool): True if the experiment was successful
            data (ExperimentResultData): The data for the experiment's
                result
            meas_level (int): Measurement result level
            status (str): The status of the experiment
            seed (int): The seed used for simulation (if run on a simulator)
            meas_return (str): The type of measurement returned
            header (qiskit.qobj.QobjExperimentHeader): A free form dictionary
                header for the experiment
            kwargs: Arbitrary extra fields

        Raises:
            QiskitError: If meas_return or meas_level are not valid values
        """
        self._metadata = {}
        self.shots = shots
        self.success = success
        self.data = data
        self.meas_level = meas_level
        if header is not None:
            self.header = header
        if status is not None:
            self.status = status
        if seed is not None:
            self.seed = seed
        if meas_return is not None:
            if meas_return not in list(MeasReturnType):
                raise QiskitError('%s not a valid meas_return value')
            self.meas_return = meas_return
        self._metadata.update(kwargs)

    def __getattr__(self, name):
        try:
            return self._metadata[name]
        except KeyError:
            raise AttributeError('Attribute %s is not defined' % name)

    def to_dict(self):
        """Return a dictionary format representation of the ExperimentResult

        Returns:
            dict: The dictionary form of the ExperimentResult
        """
        out_dict = {
            'shots': self.shots,
            'success': self.success,
            'data': self.data.to_dict(),
            'meas_level': self.meas_level,
        }
        if hasattr(self, 'header'):
            out_dict['header'] = self.header.to_dict()
        if hasattr(self, 'status'):
            out_dict['status'] = self.status
        if hasattr(self, 'seed'):
            out_dict['seed'] = self.seed
        if hasattr(self, 'meas_return'):
            out_dict['meas_return'] = self.meas_return
        out_dict.update(self._metadata)
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new ExperimentResult object from a dictionary.

        Args:
            data (dict): A dictionary representing the ExperimentResult to
                         create. It will be in the same format as output by
                         :meth:`to_dict`

        Returns:
            ExperimentResult: The ``ExperimentResult`` object from the input
                              dictionary.
        """

        in_data = copy.copy(data)
        data_obj = ExperimentResultData.from_dict(in_data.pop('data'))
        if 'header' in in_data:
            in_data['header'] = QobjExperimentHeader.from_dict(
                in_data.pop('header'))
        shots = in_data.pop('shots')
        success = in_data.pop('success')

        return cls(shots, success, data_obj, **in_data)
