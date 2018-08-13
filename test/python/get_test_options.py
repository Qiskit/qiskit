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
        'mock_online': False,
        'run_slow': False,
        'rec': False
    }

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
        'skip_online': lambda: turn_false('rec'),
        'mock_online': lambda: turn_false('skip_online'),
        'run_slow': lambda: True,
        'rec': lambda: turn_false('skip_online') and turn_false('run_slow')
    }

    def set_opt_to_true(opt):
        """
        Set the opt to True and flip all the opts that need to be rewritten in defaults dict
        Args:
            opt (str): Option to be True
        """
        defaults[opt] = if_true[opt]()

    opt_string = os.getenv(option_var, False)
    if not opt_string:
        return defaults

    for opt in opt_string.split(','):
        set_opt_to_true(opt)
    return defaults
