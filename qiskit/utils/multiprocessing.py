# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Multiprocessing utilities"""

import multiprocessing as mp
import platform
import os


def local_hardware_info():
    """Basic hardware information about the local machine.

    Gives actual number of CPU's in the machine, even when hyperthreading is
    turned on. CPU count defaults to 1 when true count can't be determined.

    Returns:
        dict: The hardware information.
    """

    if hasattr(os, "sched_getaffinity"):
        num_cpus = len(os.sched_getaffinity(0))
    else:
        num_cpus = os.cpu_count()
    if num_cpus is None:
        num_cpus = 1
    else:
        num_cpus = int(num_cpus / 2) or 1

    results = {
        "python_compiler": platform.python_compiler(),
        "python_build": ", ".join(platform.python_build()),
        "python_version": platform.python_version(),
        "os": platform.system(),
        "cpus": num_cpus,
    }
    return results


def is_main_process():
    """Checks whether the current process is the main one"""
    if platform.system() == "Windows":
        return not isinstance(mp.current_process(), mp.context.SpawnProcess)
    else:
        return not isinstance(
            mp.current_process(), (mp.context.ForkProcess, mp.context.SpawnProcess)
        )
