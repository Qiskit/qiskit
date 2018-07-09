# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name, unused-argument

"""Tests for QISKit API registration"""
import os
import qiskit.wrapper._wrapper as wrap
from qiskit.wrapper.credentials._configrc import (has_qiskit_configrc,
                                                  generate_qiskitrc,
                                                  store_credentials,
                                                  get_credentials,
                                                  remove_credentials)
from .common import (QiskitTestCase, requires_ci, requires_qe_access)


class TestRegistration(QiskitTestCase):
    """Tests for QISKit API registration"""

    @requires_ci
    def test_amake_qiskitrc(self):
        """Generated qiskitrc file."""
        self.assertTrue(not has_qiskit_configrc())
        generate_qiskitrc()
        self.assertTrue(has_qiskit_configrc())

    @requires_ci
    def test_write_qiskitrc_data(self):
        """Add provider info to qiskitc"""
        generate_qiskitrc(overwrite=True)
        self.assertTrue(len(get_credentials()) == 0)
        store_credentials(token='abcdefg')
        self.assertTrue(len(get_credentials()) == 1)

    @requires_ci
    def test_get_qiskitrc_data(self):
        """Get provider info from qiskitc"""
        generate_qiskitrc(overwrite=True)
        store_credentials(token='abcdefg')
        _creds = get_credentials('ibmq')
        self.assertTrue(_creds['token'] == 'abcdefg')

    @requires_ci
    def test_remove_qiskitrc_data(self):
        """Remove provider info from qiskitc"""
        generate_qiskitrc(overwrite=True)
        store_credentials(token='abcdefg')
        self.assertTrue(len(get_credentials()) == 1)
        remove_credentials('ibmq')
        self.assertTrue(len(get_credentials()) == 0)

    @requires_ci
    @requires_qe_access
    def test_register_from_env(self, QE_TOKEN, QE_URL,
                               hub=None, group=None, project=None):
        """Register from env vars"""
        # Remove all providers execept core
        pro = wrap.available_providers()
        for p in pro:
            if p != 'terra':
                wrap.unregister(p)
        # Make sure blank qiskitrc
        generate_qiskitrc(overwrite=True)
        orig_back = len(wrap.available_backends())
        wrap.register()
        new_back = len(wrap.available_backends())
        self.assertTrue(new_back > orig_back)

    @requires_ci
    @requires_qe_access
    def test_register_from_qiskitrc(self, QE_TOKEN, QE_URL,
                                    hub=None, group=None, project=None):
        """Register from qiskitrc"""
        # Remove all providers execept core
        pro = wrap.available_providers()
        for p in pro:
            if p != 'terra':
                wrap.unregister(p)
        orig_back = len(wrap.available_backends())
        # Make sure blank qiskitrc
        generate_qiskitrc(overwrite=True)
        _token = os.getenv('QE_TOKEN')
        # Remove env var so that it does not trigger
        # env var registration
        del os.environ['QE_TOKEN']
        store_credentials(token=QE_TOKEN,
                          url=QE_URL,
                          hub=hub,
                          group=group,
                          project=project)
        wrap.register()
        new_back = len(wrap.available_backends())
        self.assertTrue(new_back > orig_back)
        # Put token back in env
        os.environ['QE_TOKEN'] = _token

    @requires_ci
    @requires_qe_access
    def test_save_from_register(self, QE_TOKEN, QE_URL,
                                hub=None, group=None, project=None):
        """Save credentials when calling register"""
        # Remove all providers execept core
        pro = wrap.available_providers()
        for p in pro:
            if p != 'terra':
                wrap.unregister(p)
        # Make sure blank qiskitrc
        generate_qiskitrc(overwrite=True)
        wrap.register(token=QE_TOKEN,
                      url=QE_URL,
                      hub=hub, group=group, project=project,
                      save_credentials=True)
        _creds = get_credentials()
        self.assertTrue(len(_creds) > 0)
