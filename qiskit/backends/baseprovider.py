# -*- coding: utf-8 -*-
# pylint: disable=redundant-returns-doc,missing-raises-doc

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

"""Base class for a backend provider."""


class BaseProvider(object):
    """
    Base class for a backend provider.
    """
    def __init__(self, *args, **kwargs):
        pass

    def available_backends(self):
        """
        Returns:
            list of str: a list of backend names available from this provider.
        """
        raise NotImplementedError

    def get_backend(self, name):
        """
        Return a backend instance.

        Args:
            name (str): identifier

        Returns:
            BaseBackend: a backend instance.

        Raises:
            KeyError: if `name` is not among the list of backends available
                from this provider.
        """
        raise NotImplementedError
