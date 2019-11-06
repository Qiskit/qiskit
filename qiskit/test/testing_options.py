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

"""Obtain and set the options in QISKIT_TESTS, used for running the tests."""

import os
import logging

logger = logging.getLogger(__name__)


def get_test_options(option_var='QISKIT_TESTS'):
    """Read option_var from env and returns a dict in which the test options are set.

    Args:
        option_var (str): The env var to read. Default: 'QISKIT_TESTS'

    Returns:
        dict: A dictionary with the format {<option>: (bool)<activated>}.
    """
    tests_options = {
        'skip_online': False,
        'mock_online': False,
        'run_slow': False,
        'rec': False
    }

    def turn_false(option):
        """Turn an option to False.

        Args:
            option (str): Turns defaults[option] to False

        Returns:
            bool: True, returns always True.
        """

        tests_options[option] = False
        return True

    dependency_solvers = {
        'skip_online': lambda: turn_false('rec'),
        'mock_online': lambda: turn_false('skip_online'),
        'rec': lambda: turn_false('skip_online') and turn_false('run_slow')
    }

    def set_flag(flag_):
        """Set the flag to True and flip all the flags that need to be rewritten.

        Args:
            flag_ (str): Option to be True
        """
        tests_options[flag_] = True
        if flag_ in dependency_solvers:
            dependency_solvers[flag_]()

    flag_string = os.getenv(option_var, None)
    flags = flag_string.split(',') if flag_string else []
    for flag in flags:
        if flag not in tests_options:
            logger.error('Testing option "%s" unknown.', flag)
        set_flag(flag)

    if _is_ci_fork_pull_request():
        set_flag('skip_online')

    logger.debug(tests_options)
    return tests_options


def _is_ci_fork_pull_request():
    """Check if the tests are being run in a CI environment from a PR.

    Check if the tests are being run in a CI environment and if it is a pull
    request.

    Returns:
        bool: True if the tests are executed inside a CI tool, and the changes
            are not against the "master" branch.
    """
    if os.getenv('TRAVIS'):
        # Using Travis CI.
        if os.getenv('TRAVIS_PULL_REQUEST_BRANCH'):
            return True
    elif os.getenv('APPVEYOR'):
        # Using AppVeyor CI.
        if os.getenv('APPVEYOR_PULL_REQUEST_NUMBER'):
            return True
    return False
