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

import psutil


def local_hardware_info():
    """Basic hardware information about the local machine.

    Gives actual number of CPU's in the machine, even when hyperthreading is
    turned on. CPU count defaults to 1 when true count can't be determined.

    Returns:
        dict: The hardware information.
    """
    results = {
        "python_compiler": platform.python_compiler(),
        "python_build": ", ".join(platform.python_build()),
        "python_version": platform.python_version(),
        "os": platform.system(),
        "memory": psutil.virtual_memory().total / (1024**3),
        "cpus": psutil.cpu_count(logical=False) or 1,
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
