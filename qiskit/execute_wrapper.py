# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Helper module for simplified Qiskit usage.

    This module includes
        execute_circuits: runs a list of quantum circuits.

    In general we recommend using the SDK functions directly. However, to get something
    running quickly we have provider this wrapper module.
"""

import logging
from qiskit.compiler import assemble_qobj, transpile
from qiskit.compiler import RunConfig, TranspileConfig
from qiskit.qobj import QobjHeader

logger = logging.getLogger(__name__)


def execute_circuits(circuits, backend, user_qobj_header=None, run_config=None,
                 transpile_config=None, **kwargs):
    """Executes a list of circuits.

    Args:
        circuits (QuantumCircuit or list[QuantumCircuit]): circuits to execute
        backend (BaseBackend): a backend to execute the circuits on
        user_qobj_header (QobjHeader): User input to go in the header
        run_config (RunConfig): Run Configuration
        transpile_config (TranspileConfig): Configurations for the transpiler
        kwargs: extra arguments used by AER for running configurable backends.
                Refer to the backend documentation for details on these arguments

    Returns:
        BaseJob: returns job instance derived from BaseJob
    """

    # HACK CODE TO BE REMOVED WHEN SYNTHESIZE DOES NOT USE BACKEND
    # ------
    transpile_config = transpile_config or TranspileConfig()
    transpile_config.backend = backend
    # ------

    # filling in the header with the backend name the qob was rune on
    user_qobj_header = user_qobj_header or QobjHeader()
    user_qobj_header.backend_name = backend.name()

    # default values
    if not run_config:
        # TODO remove max_credits from the default when it is not required by
        # by the backend.
        run_config = RunConfig(shots=1024, max_credits=10, memory=False)

    # TODO add a default pass_manager which also sets the transpile_config

    # synthesizing the circuits using the transpiler_config
    new_circuits = transpile(circuits, transpile_config=transpile_config)

    # assembling the circuits into a qobj to be run on the backend
    qobj = assemble_qobj(new_circuits, user_qobj_header=user_qobj_header,
                         run_config=run_config)

    # executing the circuits on the backend and returning the job
    return backend.run(qobj, **kwargs)
