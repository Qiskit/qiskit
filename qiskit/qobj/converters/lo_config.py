# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper class used to convert a user lo configuration into lo_freq."""


from qiskit.pulse.exceptions import PulseError


class LoConfigConverter:
    """ This class supports to convert LoConfig into ~`lo_freq` attribute of configs.
    The format of LO frequency setup can be easily modified by replacing
    `get_qubit_los` and `get_meas_los` to align with your backend.
    """

    def __init__(self, qobj_model, **exp_config):
        """Create new converter.

        Args:
            qobj_model (PulseQobjExperimentConfig): qobj model for experiment config.
            exp_config (dict): experimental configuration.
        """
        self._qobj_model = qobj_model
        self._exp_config = exp_config

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
            lo_config['qubit_lo_freq'] = q_los
        m_los = self.get_meas_los(user_lo_config)
        if m_los:
            lo_config['meas_lo_freq'] = m_los

        return self._qobj_model(**lo_config)

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
        try:
            _q_los = self._exp_config['qubit_lo_freq'].copy()
        except KeyError:
            raise PulseError('Qubit default frequencies not exist.')

        for channel, lo_freq in user_lo_config.qubit_lo_dict():
            _q_los[channel.index] = lo_freq

        if _q_los != self._exp_config['qubit_lo_freq']:
            return _q_los
        return None

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
        try:
            _m_los = self._exp_config['meas_lo_freq'].copy()
        except KeyError:
            raise PulseError('Meas default frequencies not exist.')

        for channel, lo_freq in user_lo_config.meas_lo_dict():
            _m_los[channel.index] = lo_freq

        if _m_los != self._exp_config['meas_lo_freq']:
            return _m_los
        return None
