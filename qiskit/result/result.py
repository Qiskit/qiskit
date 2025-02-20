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

from collections.abc import Iterable
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

_MISSING = object()


class Result:
    """Model for Results.

    .. deprecated:: 1.4
        The use of positional arguments in the constructor of :class:`.Result`
        is deprecated as of Qiskit 1.4, and will be disabled in Qiskit 2.0.
        Please set all arguments using kwarg syntax, i.e: ``Result(backend_name="name", ....)``.
        In addition to this, the ``qobj_id`` argument is deprecated and will no longer
        be used in Qiskit 2.0. It will, however, still be possible to set ``qobj_id`` as a
        generic kwarg, which will land in the metadata field with the other generic kwargs.

    Args:
        backend_name (str): (REQUIRED) backend name.
        backend_version (str): (REQUIRED) backend version, in the form X.Y.Z.
        qobj_id (str): (REQUIRED) user-generated Qobj id.
        job_id (str): (REQUIRED) unique execution id from the backend.
        success (bool): (REQUIRED) True if complete input qobj executed correctly. (Implies
            each experiment success)
        results (list[ExperimentResult]): (REQUIRED) corresponding results for array of
            experiments of the input qobj
        date (str): (OPTIONAL) date of the experiment
        header(dict): (OPTIONAL)experiment header
        kwargs: generic keyword arguments. (OPTIONAL) These will be stored in the metadata field.
    """

    _metadata = {}

    def __init__(
        self,
        *args,
        date=None,
        status=None,
        header=None,
        **kwargs,
    ):
        # The following arguments are required.
        required_args = {
            "backend_name": _MISSING,
            "backend_version": _MISSING,
            "qobj_id": _MISSING,
            "job_id": _MISSING,
            "success": _MISSING,
            "results": _MISSING,
        }
        # Step 1: iterate over kwargs.
        # An item from required_args might be set as a kwarg, so we must separate
        # true kwargs from "required_args" kwargs.
        true_kwargs = {}
        for key, value in kwargs.items():
            if key in required_args:
                required_args[key] = value
            else:
                true_kwargs[key] = value
        # Step 2: iterate over args, which are expected in the order of the index_map below.
        index_map = ["backend_name", "backend_version", "qobj_id", "job_id", "success", "results"]
        raise_qobj = False
        missing_args = []
        for index, name in enumerate(index_map):
            try:
                value = args[index]
                required_args[name] = value
                # The use of args is deprecated in 1.4 and will be removed in 2.0.
                # Furthermore, qobj_id will be ignored if set as a kwarg in 2.0.
                if name == "qobj_id":
                    warnings.warn(
                        "The use of positional arguments in `qiskit.result.result.Result.__init__()` "
                        "is deprecated as of Qiskit 1.4, and will be disabled in Qiskit 2.0. "
                        "Please set this value using kwarg syntax, "
                        f"i.e: `Result(...,{name}={name}_value)`. "
                        "The `qobj_id` argument will no longer be used in Qiskit 2.0, "
                        "but it will still be possible to "
                        "set as a kwarg that will land in the metadata field.",
                        category=DeprecationWarning,
                        stacklevel=2,
                    )
                else:
                    warnings.warn(
                        "The use of positional arguments in `qiskit.result.result.Result.__init__()` "
                        "is deprecated as of Qiskit 1.4, and will be disabled in Qiskit 2.0. "
                        "Please set this value using kwarg syntax, "
                        f"i.e: `Result(...,{name}={name}_value)`. ",
                        category=DeprecationWarning,
                        stacklevel=2,
                    )
            except IndexError:
                if required_args[name] is _MISSING:
                    missing_args = [
                        key for (key, value) in required_args.items() if value is _MISSING
                    ]
                elif name == "qobj_id":
                    raise_qobj = True
                break

        # The deprecation warning should be raised outside of the try-except,
        # not to show a confusing trace that points to the IndexError
        if len(missing_args) > 1:
            raise TypeError(
                f"Result.__init__() missing {len(missing_args)} required arguments: {missing_args}"
            )
        if len(missing_args) == 1:
            raise TypeError(f"Result.__init__() missing a required argument: {missing_args[0]}")
        if raise_qobj:
            # qobj_id will be ignored if set as a kwarg in 2.0.
            warnings.warn(
                "The `qobj_id` argument will no longer be used in Qiskit 2.0, "
                "but it will still be possible to "
                "set as a kwarg that will land in the metadata field.",
                category=DeprecationWarning,
                stacklevel=2,
            )

        self._metadata = {}
        self.backend_name = required_args["backend_name"]
        self.backend_version = required_args["backend_version"]
        self.qobj_id = required_args["qobj_id"]
        self.job_id = required_args["job_id"]
        self.success = required_args["success"]
        self.results = (
            [required_args["results"]]
            if not isinstance(required_args["results"], Iterable)
            else required_args["results"]
        )
        self.date = date
        self.status = status
        self.header = header
        self._metadata.update(true_kwargs)

    def __repr__(self):
        out = (
            f"Result(backend_name='{self.backend_name}', backend_version='{self.backend_version}',"
            f" qobj_id='{self.qobj_id}', job_id='{self.job_id}', success={self.success},"
            f" results={self.results}"
        )
        out += f", date={self.date}, status={self.status}, header={self.header}"
        for key, value in self._metadata.items():
            if isinstance(value, str):
                value_str = f"'{value}'"
            else:
                value_str = repr(value)
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
            "date": self.date,
            "header": None if self.header is None else self.header.to_dict(),
            "qobj_id": self.qobj_id,
            "job_id": self.job_id,
            "status": self.status,
            "success": self.success,
            "results": [x.to_dict() for x in self.results],
        }
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
        if in_data.get("header") is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
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

            OpenQASM backends return a dictionary of dictionary with the key
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
                f'No memory for experiment "{repr(experiment)}". '
                "Please verify that you either ran a measurement level 2 job "
                'with the memory flag set, eg., "memory=True", '
                "or a measurement level 0/1 job."
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
                raise QiskitError(f'Data for experiment "{key}" could not be found.')
            if len(exp) == 1:
                exp = exp[0]
            else:
                warnings.warn(
                    f'Result object contained multiple results matching name "{key}", '
                    "only first match will be returned. Use an integer index to "
                    "retrieve results for all entries."
                )
                exp = exp[0]

        # Check that the retrieved experiment was successful
        if getattr(exp, "success", False):
            return exp
        # If unsuccessful check experiment and result status and raise exception
        result_status = getattr(self, "status", "Result was not successful")
        exp_status = getattr(exp, "status", "Experiment was not successful")
        raise QiskitError(result_status, ", ", exp_status)
