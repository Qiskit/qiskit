# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Provider for C++ simulator backends coming from the built-in Aer in versions prior to 0.7"""

import logging
import warnings
from collections import OrderedDict

from qiskit import QiskitError
from qiskit.backends.builtinsimulators import SimulatorsProvider

from .qasm_simulator import QasmSimulator, CliffordSimulator
from .statevector_simulator import StatevectorSimulator

logger = logging.getLogger(__name__)

LEGACY_SIMULATORS = [
    QasmSimulator,
    StatevectorSimulator,
    CliffordSimulator
]


class LegacyProvider(SimulatorsProvider):
    """Provider for legacy simulators backends."""

    def get_backend(self, name=None, **kwargs):
        warnings.warn('C++ simulators in the `LegacyProvider` are deprecated. Install '
                      '`qiskit-aer` and import `Aer` from `qiskit` if you want make the most '
                      'of performance. The `LegacyProvider` will be removed after 0.7',
                      DeprecationWarning)
        return super().get_backend(name=name, **kwargs)

    def _verify_backends(self):
        """
        Return the legacy simulators backends in `LEGACY_SIMULATORS` that are
        effectively available (as some of them might depend on the presence
        of an optional dependency or on the existence of a binary).

        Returns:
            dict[str:BaseBackend]: a dict of legacy simulators backend instances for
                the backends that could be instantiated, keyed by backend name.
        """
        ret = OrderedDict()
        for backend_cls in LEGACY_SIMULATORS:
            try:
                backend_instance = self._get_backend_instance(backend_cls)
                backend_name = backend_instance.name()
                ret[backend_name] = backend_instance
            except QiskitError as err:
                # Ignore backends that could not be initialized.
                logger.info('legacy simulator %s is not available: %s',
                            backend_cls, str(err))
        return ret

    @staticmethod
    def _deprecated_backend_names():
        """Returns deprecated backend names."""
        return {
            'local_qasm_simulator_cpp': 'qasm_simulator',
            'local_statevector_simulator_cpp': 'statevector_simulator',
            'local_qiskit_simulator': 'qasm_simulator',
            'local_qasm_simulator': 'qasm_simulator',
            'local_statevector_simulator': 'statevector_simulator'
        }

    def __str__(self):
        return 'LegacySimulators'
