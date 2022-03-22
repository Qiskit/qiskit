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

"""Model for schema-conformant Results."""

import copy
import warnings

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.pulse.schedule import Schedule
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states import statevector
from qiskit.result.models import ExperimentResult
from qiskit.result import postprocess
from qiskit.result.counts import Counts
from qiskit.qobj.utils import MeasLevel
from qiskit.qobj import QobjHeader


class Result:
    """Model for Results.

    Attributes:
        backend_name (str): backend name.
        backend_version (str): backend version, in the form X.Y.Z.
        qobj_id (str): user-generated Qobj id.
        job_id (str): unique execution id from the backend.
        success (bool): True if complete input qobj executed correctly. (Implies
            each experiment success)
        results (list[ExperimentResult]): corresponding results for array of
            experiments of the input qobj
    """

    _metadata = {}

    def __init__(
        self,
        backend_name,
        backend_version,
        qobj_id,
        job_id,
        success,
        results,
        date=None,
        status=None,
        header=None,
        **kwargs,
    ):
        self._metadata = {}
        self.backend_name = backend_name
        self.backend_version = backend_version
        self.qobj_id = qobj_id
        self.job_id = job_id
        self.success = success
        self.results = results
        if date is not None:
            self.date = date
        if status is not None:
            self.status = status
        if header is not None:
            self.header = header
        self._metadata.update(kwargs)

    def __repr__(self):
        out = (
            "Result(backend_name='%s', backend_version='%s', qobj_id='%s', "
            "job_id='%s', success=%s, results=%s"
            % (
                self.backend_name,
                self.backend_version,
                self.qobj_id,
                self.job_id,
                self.success,
                self.results,
            )
        )
        if hasattr(self, "date"):
            out += ", date=%s" % self.date
        if hasattr(self, "status"):
            out += ", status=%s" % self.status
        if hasattr(self, "header"):
            out += ", status=%s" % self.header
        for key in self._metadata:
            if isinstance(self._metadata[key], str):
                value_str = "'%s'" % self._metadata[key]
            else:
                value_str = repr(self._metadata[key])
            out += f", {key}={value_str}"
        out += ")"
        return out

    def to_dict(self):
        """Return a dictionary format representation of the Result

        Returns:
            dict: The dictionary form of the Result
        """
        out_dict = {
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "qobj_id": self.qobj_id,
            "job_id": self.job_id,
            "success": self.success,
            "results": [x.to_dict() for x in self.results],
        }
        if hasattr(self, "date"):
            out_dict["date"] = self.date
        if hasattr(self, "status"):
            out_dict["status"] = self.status
        if hasattr(self, "header"):
            out_dict["header"] = self.header.to_dict()
        out_dict.update(self._metadata)
        return out_dict

    def __getattr__(self, name):
        try:
            return self._metadata[name]
        except KeyError as ex:
            raise AttributeError(f"Attribute {name} is not defined") from ex

    @classmethod
    def from_dict(cls, data):
        """Create a new ExperimentResultData object from a dictionary.

        Args:
            data (dict): A dictionary representing the Result to create. It
                         will be in the same format as output by
                         :meth:`to_dict`.
        Returns:
            Result: The ``Result`` object from the input dictionary.

        """

        in_data = copy.copy(data)
        in_data["results"] = [ExperimentResult.from_dict(x) for x in in_data.pop("results")]
        if "header" in in_data:
            in_data["header"] = QobjHeader.from_dict(in_data.pop("header"))
        return cls(**in_data)

    def data(self, experiment=None):
        """Get the raw data for an experiment.

        Note this data will be a single classical and quantum register and in a
        format required by the results schema. We recommend that most users use
        the get_xxx method, and the data will be post-processed for the data type.

        Args:
            experiment (str or QuantumCircuit or Schedule or int or None): the index of the
                experiment. Several types are accepted for convenience::
                * str: the name of the experiment.
                * QuantumCircuit: the name of the circuit instance will be used.
                * Schedule: the name of the schedule instance will be used.
                * int: the position of the experiment.
                * None: if there is only one experiment, returns it.

        Returns:
            dict: A dictionary of results data for an experiment. The data
            depends on the backend it ran on and the settings of `meas_level`,
            `meas_return` and `memory`.

            QASM backends return a dictionary of dictionary with the key
            'counts' and  with the counts, with the second dictionary keys
            containing a string in hex format (``0x123``) and values equal to
            the number of times this outcome was measured.

            Statevector backends return a dictionary with key 'statevector' and
            values being a list[list[complex components]] list of 2^num_qubits
            complex amplitudes. Where each complex number is represented as a 2
            entry list for each component. For example, a list of
            [0.5+1j, 0-1j] would be represented as [[0.5, 1], [0, -1]].

            Unitary backends return a dictionary with key 'unitary' and values
            being a list[list[list[complex components]]] list of
            2^num_qubits x 2^num_qubits complex amplitudes in a two entry list for
            each component. For example if the amplitude is
            [[0.5+0j, 0-1j], ...] the value returned will be
            [[[0.5, 0], [0, -1]], ...].

            The simulator backends also have an optional key 'snapshots' which
            returns a dict of snapshots specified by the simulator backend.
            The value is of the form dict[slot: dict[str: array]]
            where the keys are the requested snapshot slots, and the values are
            a dictionary of the snapshots.

        Raises:
            QiskitError: if data for the experiment could not be retrieved.
        """
        try:
            return self._get_experiment(experiment).data.to_dict()
        except (KeyError, TypeError) as ex:
            raise QiskitError(f'No data for experiment "{repr(experiment)}"') from ex

    def get_memory(self, experiment=None):
        """Get the sequence of memory states (readouts) for each shot
        The data from the experiment is a list of format
        ['00000', '01000', '10100', '10100', '11101', '11100', '00101', ..., '01010']

        Args:
            experiment (str or QuantumCircuit or Schedule or int or None): the index of the
                experiment, as specified by ``data()``.

        Returns:
            List[str] or np.ndarray: Either the list of each outcome, formatted according to
            registers in circuit or a complex numpy np.ndarray with shape:

                ============  =============  =====
                `meas_level`  `meas_return`  shape
                ============  =============  =====
                0             `single`       np.ndarray[shots, memory_slots, memory_slot_size]
                0             `avg`          np.ndarray[memory_slots, memory_slot_size]
                1             `single`       np.ndarray[shots, memory_slots]
                1             `avg`          np.ndarray[memory_slots]
                2             `memory=True`  list
                ============  =============  =====

        Raises:
            QiskitError: if there is no memory data for the circuit.
        """
        exp_result = self._get_experiment(experiment)
        try:
            try:  # header is not available
                header = exp_result.header.to_dict()
            except (AttributeError, QiskitError):
                header = None

            meas_level = exp_result.meas_level

            memory = self.data(experiment)["memory"]

            if meas_level == MeasLevel.CLASSIFIED:
                return postprocess.format_level_2_memory(memory, header)
            elif meas_level == MeasLevel.KERNELED:
                return postprocess.format_level_1_memory(memory)
            elif meas_level == MeasLevel.RAW:
                return postprocess.format_level_0_memory(memory)
            else:
                raise QiskitError(f"Measurement level {meas_level} is not supported")

        except KeyError as ex:
            raise QiskitError(
                'No memory for experiment "{}". '
                "Please verify that you either ran a measurement level 2 job "
                'with the memory flag set, eg., "memory=True", '
                "or a measurement level 0/1 job.".format(repr(experiment))
            ) from ex

    def get_counts(self, experiment=None):
        """Get the histogram data of an experiment.

        Args:
            experiment (str or QuantumCircuit or Schedule or int or None): the index of the
                experiment, as specified by ``data([experiment])``.

        Returns:
            dict[str, int] or list[dict[str, int]]: a dictionary or a list of
            dictionaries. A dictionary has the counts for each qubit with
            the keys containing a string in binary format and separated
            according to the registers in circuit (e.g. ``0100 1110``).
            The string is little-endian (cr[0] on the right hand side).

        Raises:
            QiskitError: if there are no counts for the experiment.
        """
        if experiment is None:
            exp_keys = range(len(self.results))
        else:
            exp_keys = [experiment]

        dict_list = []
        for key in exp_keys:
            exp = self._get_experiment(key)
            try:
                header = exp.header.to_dict()
            except (AttributeError, QiskitError):  # header is not available
                header = None

            if "counts" in self.data(key).keys():
                if header:
                    counts_header = {
                        k: v
                        for k, v in header.items()
                        if k in {"time_taken", "creg_sizes", "memory_slots"}
                    }
                else:
                    counts_header = {}
                dict_list.append(Counts(self.data(key)["counts"], **counts_header))
            elif "statevector" in self.data(key).keys():
                vec = postprocess.format_statevector(self.data(key)["statevector"])
                dict_list.append(statevector.Statevector(vec).probabilities_dict(decimals=15))
            else:
                raise QiskitError(f'No counts for experiment "{repr(key)}"')

        # Return first item of dict_list if size is 1
        if len(dict_list) == 1:
            return dict_list[0]
        else:
            return dict_list

    def get_statevector(self, experiment=None, decimals=None):
        """Get the final statevector of an experiment.

        Args:
            experiment (str or QuantumCircuit or Schedule or int or None): the index of the
                experiment, as specified by ``data()``.
            decimals (int): the number of decimals in the statevector.
                If None, does not round.

        Returns:
            list[complex]: list of 2^num_qubits complex amplitudes.

        Raises:
            QiskitError: if there is no statevector for the experiment.
        """
        try:
            return postprocess.format_statevector(
                self.data(experiment)["statevector"], decimals=decimals
            )
        except KeyError as ex:
            raise QiskitError(f'No statevector for experiment "{repr(experiment)}"') from ex

    def get_unitary(self, experiment=None, decimals=None):
        """Get the final unitary of an experiment.

        Args:
            experiment (str or QuantumCircuit or Schedule or int or None): the index of the
                experiment, as specified by ``data()``.
            decimals (int): the number of decimals in the unitary.
                If None, does not round.

        Returns:
            list[list[complex]]: list of 2^num_qubits x 2^num_qubits complex
                amplitudes.

        Raises:
            QiskitError: if there is no unitary for the experiment.
        """
        try:
            return postprocess.format_unitary(self.data(experiment)["unitary"], decimals=decimals)
        except KeyError as ex:
            raise QiskitError(f'No unitary for experiment "{repr(experiment)}"') from ex

    def _get_experiment(self, key=None):
        """Return a single experiment result from a given key.

        Args:
            key (str or QuantumCircuit or Schedule or int or None): the index of the
                experiment, as specified by ``data()``.

        Returns:
            ExperimentResult: the results for an experiment.

        Raises:
            QiskitError: if there is no data for the experiment, or an unhandled
                error occurred while fetching the data.
        """
        # Automatically return the first result if no key was provided.
        if key is None:
            if len(self.results) != 1:
                raise QiskitError(
                    "You have to select a circuit or schedule when there is more than one available"
                )
            key = 0

        # Key is a QuantumCircuit/Schedule or str: retrieve result by name.
        if isinstance(key, (QuantumCircuit, Schedule)):
            key = key.name
        # Key is an integer: return result by index.
        if isinstance(key, int):
            try:
                exp = self.results[key]
            except IndexError as ex:
                raise QiskitError(f'Result for experiment "{key}" could not be found.') from ex
        else:
            # Look into `result[x].header.name` for the names.
            exp = [
                result
                for result in self.results
                if getattr(getattr(result, "header", None), "name", "") == key
            ]

            if len(exp) == 0:
                raise QiskitError('Data for experiment "%s" could not be found.' % key)
            if len(exp) == 1:
                exp = exp[0]
            else:
                warnings.warn(
                    'Result object contained multiple results matching name "%s", '
                    "only first match will be returned. Use an integer index to "
                    "retrieve results for all entries." % key
                )
                exp = exp[0]

        # Check that the retrieved experiment was successful
        if getattr(exp, "success", False):
            return exp
        # If unsuccessful check experiment and result status and raise exception
        result_status = getattr(self, "status", "Result was not successful")
        exp_status = getattr(exp, "status", "Experiment was not successful")
        raise QiskitError(result_status, ", ", exp_status)
