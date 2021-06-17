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

# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

"""
Routines for running Python functions in parallel using process pools
from the multiprocessing library.
"""

import os
from concurrent.futures import ProcessPoolExecutor
import sys

from qiskit.exceptions import QiskitError
from qiskit.utils.multiprocessing import local_hardware_info
from qiskit.tools.events.pubsub import Publisher
from qiskit import user_config

CONFIG = user_config.get_config()

if os.getenv("QISKIT_PARALLEL", None) is not None:
    PARALLEL_DEFAULT = os.getenv("QISKIT_PARALLEL", None).lower() == "true"
else:
    # Default False on Windows
    if sys.platform == "win32":
        PARALLEL_DEFAULT = False
    # On python 3.9 default false to avoid deadlock issues
    elif sys.version_info[0] == 3 and sys.version_info[1] == 9:
        PARALLEL_DEFAULT = False
    # On macOS default false on Python >=3.8
    elif sys.platform == "darwin":
        if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
            PARALLEL_DEFAULT = False
        else:
            PARALLEL_DEFAULT = True
    # On linux (and other OSes) default to True
    else:
        PARALLEL_DEFAULT = True

# Set parallel flag
if os.getenv("QISKIT_IN_PARALLEL") is None:
    os.environ["QISKIT_IN_PARALLEL"] = "FALSE"

if os.getenv("QISKIT_NUM_PROCS") is not None:
    CPU_COUNT = int(os.getenv("QISKIT_NUM_PROCS"))
else:
    CPU_COUNT = CONFIG.get("num_process", local_hardware_info()["cpus"])


def _task_wrapper(param):
    (task, value, task_args, task_kwargs) = param
    return task(value, *task_args, **task_kwargs)


def parallel_map(  # pylint: disable=dangerous-default-value
    task, values, task_args=tuple(), task_kwargs={}, num_processes=CPU_COUNT
):
    """
    Parallel execution of a mapping of `values` to the function `task`. This
    is functionally equivalent to::

        result = [task(value, *task_args, **task_kwargs) for value in values]

    On Windows this function defaults to a serial implementation to avoid the
    overhead from spawning processes in Windows.

    Args:
        task (func): Function that is to be called for each value in ``values``.
        values (array_like): List or array of values for which the ``task``
                            function is to be evaluated.
        task_args (list): Optional additional arguments to the ``task`` function.
        task_kwargs (dict): Optional additional keyword argument to the ``task`` function.
        num_processes (int): Number of processes to spawn.

    Returns:
        result: The result list contains the value of
                ``task(value, *task_args, **task_kwargs)`` for
                    each value in ``values``.

    Raises:
        QiskitError: If user interrupts via keyboard.

    Events:
        terra.parallel.start: The collection of parallel tasks are about to start.
        terra.parallel.update: One of the parallel task has finished.
        terra.parallel.finish: All the parallel tasks have finished.
    """
    if len(values) == 0:
        return []
    if len(values) == 1:
        return [task(values[0], *task_args, **task_kwargs)]

    Publisher().publish("terra.parallel.start", len(values))
    nfinished = [0]

    def _callback(_):
        nfinished[0] += 1
        Publisher().publish("terra.parallel.done", nfinished[0])

    # Run in parallel if not Win and not in parallel already
    if (
        num_processes > 1
        and os.getenv("QISKIT_IN_PARALLEL") == "FALSE"
        and CONFIG.get("parallel_enabled", PARALLEL_DEFAULT)
    ):
        os.environ["QISKIT_IN_PARALLEL"] = "TRUE"
        try:
            results = []
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                param = map(lambda value: (task, value, task_args, task_kwargs), values)
                future = executor.map(_task_wrapper, param)

            results = list(future)
            Publisher().publish("terra.parallel.done", len(results))

        except (KeyboardInterrupt, Exception) as error:
            if isinstance(error, KeyboardInterrupt):
                Publisher().publish("terra.parallel.finish")
                os.environ["QISKIT_IN_PARALLEL"] = "FALSE"
                raise QiskitError("Keyboard interrupt in parallel_map.") from error
            # Otherwise just reset parallel flag and error
            os.environ["QISKIT_IN_PARALLEL"] = "FALSE"
            raise error

        Publisher().publish("terra.parallel.finish")
        os.environ["QISKIT_IN_PARALLEL"] = "FALSE"
        return results

    # Cannot do parallel on Windows , if another parallel_map is running in parallel,
    # or len(values) == 1.
    results = []
    for _, value in enumerate(values):
        result = task(value, *task_args, **task_kwargs)
        results.append(result)
        _callback(0)
    Publisher().publish("terra.parallel.finish")
    return results
