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

"""Meta-provider that aggregates several providers."""
import logging

from qiskit.backends.baseprovider import BaseProvider
from qiskit.backends.local.localprovider import LocalProvider

logger = logging.getLogger(__name__)


class DefaultQISKitProvider(BaseProvider):
    """
    Meta-provider that aggregates several providers.
    """
    def __init__(self):
        super().__init__()

        # List of providers.
        self.providers = [LocalProvider()]

    def get_backend(self, name):
        for provider in self.providers:
            try:
                return provider.get_backend(name)
            except KeyError:
                pass

    def available_backends(self):
        backends = []
        for provider in self.providers:
            backends.extend(provider.available_backends())

        return backends

    def add_provider(self, provider):
        """
        Add a new provider to the list of know providers.

        Args:
            provider (BaseProvider): Provider instance.
        """
        self.providers.append(provider)
