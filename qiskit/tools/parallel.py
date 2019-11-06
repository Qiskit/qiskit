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
import queue
from concurrent import futures
from qiskit.exceptions import QiskitError
from qiskit.tools.events.pubsub import Publisher

# Set parallel core counts
PARALLEL_COUNTS = os.environ.get('QISKIT_PARALLEL_COUNTS', 1)


def parallel_map(  # pylint: disable=dangerous-default-value
        task, values, task_args=tuple(), task_kwargs={}, num_processes=PARALLEL_COUNTS):
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
    if len(values) == 1:
        return [task(values[0], *task_args, **task_kwargs)]

    Publisher().publish("terra.parallel.start", len(values))
    nfinished = [0]

    def _callback(_):
        nfinished[0] += 1
        Publisher().publish("terra.parallel.done", nfinished[0])

    def task_executor(input_queue, output_queue):
        while not input_queue.empty():
            try:
                task_with_args = input_queue.get()
                task, task_args, task_kwargs, callback = task_with_args
                result = task(*task_args, **task_kwargs)
                output_queue.put(result)
                callback(0)
            except queue.Empty:
                pass

    parallel_counts = os.environ.get('QISKIT_PARALLEL_COUNTS', 1)
    input_queue = queue.Queue(maxsize=0)
    output_queue = queue.Queue(maxsize=0)
    for value in values:
        input_queue.put((task, (value,) + task_args, task_kwargs, _callback))

    # start executors
    _executors = futures.ThreadPoolExecutor(max_workers=parallel_counts)
    _futures = [_executors.submit(task_executor, input_queue, output_queue)
                for _ in range(parallel_counts)]

    # wait executors to finish
    [_future.result(timeout=None) for _future in _futures]

    results = []
    while not output_queue.empty():
        results.append(output_queue.get())

    # validate whether all tasks are finished
    if not len(results) == len(values):
        raise QiskitError("Totally %s tasks but only %s finished" % (len(values), len(results)))

    Publisher().publish("terra.parallel.finish")
    return results
