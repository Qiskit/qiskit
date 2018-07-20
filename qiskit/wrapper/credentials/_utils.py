# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Utilities for credentials.
"""


def get_account_name(provider_class):
    """
    Return the account name for a particular provider. This name is used by
    Qiskit internally and in the configuration file and uniquely identifies
    a provider.

    Args:
        provider_class (class): class for the account.

    Returns:
        str: the account name.
    """
    return '{}.{}'.format(provider_class.__module__, provider_class.__name__)
