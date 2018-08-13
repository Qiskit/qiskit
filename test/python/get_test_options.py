# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Obtains and set the options in QISKIT_TESTS, used for running the tests."""

import os

def get_test_options(option_var='QISKIT_TESTS'):
    """
    Reads option_var from env and returns a dict in which the test options are set

    Args:
        option_var (str): The env var to read. Default: 'QISKIT_TESTS'

    Returns:
        dict: A dictionary with the format {<option>: (bool)<activated>}.
    """
    defaults = {
        'skip_online': False,
        'run_online': False,
        'mock_online': False,
        'skip_slow': True,
        'run_slow': False,
        'rec': False
    }

    def turn_true(option):
        """
        Turns an option to True
        Args:
            option (str): Turns defaults[option] to True

        Returns:
            bool: True, returns always True.
        """

        defaults[option] = True
        return True

    def turn_false(option):
        """
        Turns an option to False
        Args:
            option (str): Turns defaults[option] to False

        Returns:
            bool: True, returns always True.
        """

        defaults[option] = False
        return True

    if_true = {
        'skip_online': (lambda: turn_false('run_online') and turn_false('rec')),
        'run_online': (lambda: turn_false('skip_online')),
        'mock_online': (lambda: turn_true('run_online') and turn_false('skip_online')),
        'skip_slow': (lambda: turn_false('run_online')),
        'run_slow': (lambda: turn_false('skip_slow')),
        'rec': (lambda: turn_true('run_online') and turn_false('skip_online') and turn_false(
            'run_slow'))
    }

    opt_string = os.getenv(option_var, False)
    if not opt_string:
        return defaults

    for opt in opt_string.split(','):
        # This means, set the opt to True and flip all the opts that need to be rewritten.
        defaults[opt] = if_true[opt]()
    return defaults
