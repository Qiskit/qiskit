# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper class used to convert a user LO configuration into a list of frequencies."""

from qiskit.pulse.channels import DriveChannel, MeasureChannel
from qiskit.pulse.configuration import LoConfig
from qiskit.exceptions import QiskitError


class LoConfigConverter:
    """This class supports to convert LoConfig into ~`lo_freq` attribute of configs.
    The format of LO frequency setup can be easily modified by replacing
    ``get_qubit_los`` and ``get_meas_los`` to align with your backend.
    """

    def __init__(
        self,
        qobj_model,
        qubit_lo_freq=None,
        meas_lo_freq=None,
        qubit_lo_range=None,
        meas_lo_range=None,
        **run_config,
    ):
        """Create new converter.

        Args:
            qobj_model (Union[PulseQobjExperimentConfig, QasmQobjExperimentConfig): qobj model for
                experiment config.
            qubit_lo_freq (Optional[List[float]]): List of default qubit LO frequencies in Hz.
            meas_lo_freq (Optional[List[float]]): List of default meas LO frequencies in Hz.
            qubit_lo_range (Optional[List[List[float]]]): List of qubit LO ranges,
                each of form ``[range_min, range_max]`` in Hz.
            meas_lo_range (Optional[List[List[float]]]): List of measurement LO ranges,
                each of form ``[range_min, range_max]`` in Hz.
            n_qubits (int): Number of qubits in the system.
            run_config (dict): experimental configuration.
        """
        self.qobj_model = qobj_model
        self.qubit_lo_freq = qubit_lo_freq
        self.meas_lo_freq = meas_lo_freq
        self.run_config = run_config
        self.n_qubits = self.run_config.get("n_qubits", None)

        self.default_lo_config = LoConfig()

        if qubit_lo_range:
            for i, lo_range in enumerate(qubit_lo_range):
                self.default_lo_config.add_lo_range(DriveChannel(i), lo_range)

        if meas_lo_range:
            for i, lo_range in enumerate(meas_lo_range):
                self.default_lo_config.add_lo_range(MeasureChannel(i), lo_range)

    def __call__(self, user_lo_config):
        """Return experiment config w/ LO values property configured.

        Args:
            user_lo_config (LoConfig): A dictionary of LOs to format.

        Returns:
            Union[PulseQobjExperimentConfig, QasmQobjExperimentConfig]: Qobj experiment config.
        """
        lo_config = {}

        q_los = self.get_qubit_los(user_lo_config)
        if q_los:
            lo_config["qubit_lo_freq"] = [freq / 1e9 for freq in q_los]

        m_los = self.get_meas_los(user_lo_config)
        if m_los:
            lo_config["meas_lo_freq"] = [freq / 1e9 for freq in m_los]

        return self.qobj_model(**lo_config)

    def get_qubit_los(self, user_lo_config):
        """Set experiment level qubit LO frequencies. Use default values from job level if
        experiment level values not supplied. If experiment level and job level values not supplied,
        raise an error. If configured LO frequency is the same as default, this method returns
        ``None``.

        Args:
            user_lo_config (LoConfig): A dictionary of LOs to format.

        Returns:
            List[float]: A list of qubit LOs.

        Raises:
            QiskitError: When LO frequencies are missing and no default is set at job level.
        """
        _q_los = None

        # try to use job level default values
        if self.qubit_lo_freq:
            _q_los = self.qubit_lo_freq.copy()
        # otherwise initialize list with ``None`` entries
        elif self.n_qubits:
            _q_los = [None] * self.n_qubits

        # fill experiment level LO's
        if _q_los:
            for channel, lo_freq in user_lo_config.qubit_los.items():
                self.default_lo_config.check_lo(channel, lo_freq)
                _q_los[channel.index] = lo_freq

            if _q_los == self.qubit_lo_freq:
                return None

            # if ``None`` remains in LO's, indicates default not provided and user value is missing
            # raise error
            if None in _q_los:
                raise QiskitError(
                    "Invalid experiment level qubit LO's. Must either pass values "
                    "for all drive channels or pass 'default_qubit_los'."
                )

        return _q_los

    def get_meas_los(self, user_lo_config):
        """Set experiment level meas LO frequencies. Use default values from job level if experiment
        level values not supplied. If experiment level and job level values not supplied, raise an
        error. If configured LO frequency is the same as default, this method returns ``None``.

        Args:
            user_lo_config (LoConfig): A dictionary of LOs to format.

        Returns:
            List[float]: A list of measurement LOs.

        Raises:
            QiskitError: When LO frequencies are missing and no default is set at job level.
        """
        _m_los = None
        # try to use job level default values
        if self.meas_lo_freq:
            _m_los = self.meas_lo_freq.copy()
        # otherwise initialize list with ``None`` entries
        elif self.n_qubits:
            _m_los = [None] * self.n_qubits

        # fill experiment level LO's
        if _m_los:
            for channel, lo_freq in user_lo_config.meas_los.items():
                self.default_lo_config.check_lo(channel, lo_freq)
                _m_los[channel.index] = lo_freq

            if _m_los == self.meas_lo_freq:
                return None

            # if ``None`` remains in LO's, indicates default not provided and user value is missing
            # raise error
            if None in _m_los:
                raise QiskitError(
                    "Invalid experiment level measurement LO's. Must either pass "
                    "values for all measurement channels or pass 'default_meas_los'."
                )

        return _m_los
