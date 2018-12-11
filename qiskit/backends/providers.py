# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Provider plugin manager."""

import logging

import stevedore

logger = logging.getLogger(__name__)


class ProviderPluginManager:
    """Qiskit provider plugin manager class

    This class is used to manage the lifecycle of external backend provider
    plugins. It provides functions for getting all backends
    """
    def __init__(self):
        self.ext_plugins = stevedore.ExtensionManager(
            'qiskit.providers', invoke_on_load=True,
            propagate_map_exceptions=True,
            on_load_failure_callback=self.failure_hook)

    @staticmethod
    def failure_hook(_, err_plugin, err):
        """Log errors on import and don't fail."""
        logger.error("Could not load provider plugin %r with error: %s",
                     err_plugin.name, err)

    def get_providers(self):
        """Return dict of all discovered provider plugins."""
        providers = {}
        for plug in self.ext_plugins:
            providers[plug.name] = plug.obj
        return providers

    def get_all_backends(self):
        """Return dict of lists of backends for all discovered provider plugins."""
        backends_dict = {}
        for plug in self.ext_plugins:
            backends_dict[plug.name] = plug.obj.backends()
        return backends_dict
