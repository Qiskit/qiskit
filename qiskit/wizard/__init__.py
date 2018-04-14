# -*- coding: utf-8 -*-

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Helper for simplified QISKit usage."""

from qiskit import QISKitError
from qiskit.backends.ibmq.ibmqprovider import IBMQProvider
from qiskit.wizard.defaultqiskitprovider import DefaultQISKitProvider

# THIS IS A GLOBAL OBJECT - USE WITH CARE.
DEFAULT_PROVIDER = DefaultQISKitProvider()


def register(token, url, provider_name):
    """
    Authenticate against an online provider.
    """
    if provider_name == 'qiskit':
        DEFAULT_PROVIDER.add_provider(IBMQProvider(token, url))
    else:
        raise QISKitError('provider name %s is not recognized' % provider_name)


def available_backends():
    """
    Return available backends.
    """
    return DEFAULT_PROVIDER.available_backends()
