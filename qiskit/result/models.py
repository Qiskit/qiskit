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
from types import SimpleNamespace

from qiskit.qobj.utils import MeasReturnType, MeasLevel
from qiskit.validation.exceptions import ModelValidationError


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
            memory (list)
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
        out_dict = {}
        for field in ['counts', 'snapshots', 'memory', 'statevector',
                      'unitary']:
            if hasattr(self, field):
                out_dict[field] = getattr(self, field)
        return out_dict

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class ExperimentResult(SimpleNamespace):
    """Class representing an Experiment Result.

    Attributes:
        shots (int or tuple): the starting and ending shot for this data.
        success (bool): if true, we can trust results for this experiment.
        data (ExperimentResultData): results information.
        meas_level (int): Measurement result level.
    """

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
            header (dict): A free form dictionary header for the experiment
            kwargs: Arbitrary extra fields

        Raises:
            QiskitError: If meas_return or meas_level are not valid values
        """
        self.shots = shots
        self.success = success
        self.data = data
        self.meas_level = meas_level
        if status is not None:
            self.status = status
        if seed is not None:
            self.seed = seed
        if meas_return is not None:
            if meas_return not in list(MeasReturnType):
                raise ModelValidationError('%s not a valid meas_return value')
            self.meas_return = meas_return
        self.__dict__.update(kwargs)

    def to_dict(self):
        out_dict = {
            'shots': self.shots,
            'success': self.success,
            'data': self.data.to_dict(),
            'meas_level': self.meas_level,
        }
        for field in self.__dict__.keys():
            if field not in ['shots', 'success', 'data', 'meas_level']:
                out_dict[field] = getattr(self, field)
        return out_dict

    @classmethod
    def from_dict(cls, data):
        in_data = copy.copy()
        in_data['data'] = ExperimentResultData.from_dict(in_data.pop('data'))
        return cls(**in_data)

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        return self.from_dict(state)

    def __reduce__(self):
        return (self.__class__, (self.shots, self.success, self.data))
