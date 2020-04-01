# -*- coding: utf-8 -*-

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

"""Helper class used to convert a user lo configuration into a list of frequencies."""

from qiskit.pulse.channels import DriveChannel, MeasureChannel
from qiskit.pulse.configuration import LoConfig


class LoConfigConverter:
    """ This class supports to convert LoConfig into ~`lo_freq` attribute of configs.
    The format of LO frequency setup can be easily modified by replacing
    `get_qubit_los` and `get_meas_los` to align with your backend.
    """

    def __init__(self, qobj_model, qubit_lo_freq, meas_lo_freq,
                 qubit_lo_range=None, meas_lo_range=None, **run_config):
        """Create new converter.

        Args:
            qobj_model (PulseQobjExperimentConfig): qobj model for experiment config.
            qubit_lo_freq (list): List of default qubit lo frequencies in Hz.
            meas_lo_freq (list): List of default meas lo frequencies in Hz.
            qubit_lo_range (list): List of qubit lo ranges,
                each of form `[range_min, range_max]` in Hz.
            meas_lo_range (list): List of measurement lo ranges,
                each of form `[range_min, range_max]` in Hz.
            run_config (dict): experimental configuration.
        """
        self.qobj_model = qobj_model
        self.qubit_lo_freq = qubit_lo_freq
        self.meas_lo_freq = meas_lo_freq
        self.run_config = run_config

        self.default_lo_config = LoConfig()

        if qubit_lo_range:
            for i, lo_range in enumerate(qubit_lo_range):
                self.default_lo_config.add_lo_range(DriveChannel(i), lo_range)

        if meas_lo_range:
            for i, lo_range in enumerate(meas_lo_range):
                self.default_lo_config.add_lo_range(MeasureChannel(i), lo_range)

    def __call__(self, user_lo_config):
        """Return PulseQobjExperimentConfig

        Args:
            user_lo_config (LoConfig): A dictionary of LOs to format.

        Returns:
            PulseQobjExperimentConfig: qobj.
        """
        lo_config = {}

        q_los = self.get_qubit_los(user_lo_config)
        if q_los:
            lo_config['qubit_lo_freq'] = [freq/1e9 for freq in q_los]

        m_los = self.get_meas_los(user_lo_config)
        if m_los:
            lo_config['meas_lo_freq'] = [freq/1e9 for freq in m_los]

        return self.qobj_model(**lo_config)

    def get_qubit_los(self, user_lo_config):
        """Embed default qubit LO frequencies from backend and format them to list object.
        If configured lo frequency is the same as default, this method returns `None`.

        Args:
            user_lo_config (LoConfig): A dictionary of LOs to format.

        Returns:
            list: A list of qubit LOs.

        Raises:
            PulseError: when LO frequencies are missing.
        """
        _q_los = self.qubit_lo_freq.copy()

        for channel, lo_freq in user_lo_config.qubit_los.items():
            self.default_lo_config.check_lo(channel, lo_freq)
            _q_los[channel.index] = lo_freq

        if _q_los == self.qubit_lo_freq:
            return None

        return _q_los

    def get_meas_los(self, user_lo_config):
        """Embed default meas LO frequencies from backend and format them to list object.
        If configured lo frequency is the same as default, this method returns `None`.

        Args:
            user_lo_config (LoConfig): A dictionary of LOs to format.

        Returns:
            list: A list of meas LOs.

        Raises:
            PulseError: when LO frequencies are missing.
        """
        _m_los = self.meas_lo_freq.copy()

        for channel, lo_freq in user_lo_config.meas_los.items():
            self.default_lo_config.check_lo(channel, lo_freq)
            _m_los[channel.index] = lo_freq

        if _m_los == self.meas_lo_freq:
            return None

        return _m_los
