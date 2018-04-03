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

# Modified to include Windows system cpu_freq and memory information.
# April 03, 2018
# Paul D. Nation, paul.nation@ibm.com

import os
import sys
import multiprocessing
import numpy as np

"""A Module for getting local hardware information."""

__all__ = ['local_hardware_info', 'verify_qubit_number']


def _mac_hardware_info():
    """Returns system info on OSX.
    """
    info = dict()
    results = dict()
    for line in [l.split(':') for l in os.popen('sysctl hw').readlines()[1:]]:
        info[line[0].strip(' "').replace(' ', '_').lower().strip('hw.')] = \
            line[1].strip('.\n ')
    results.update({'cpus': int(info['physicalcpu'])})
    results.update({'cpu_freq': int(
        float(os.popen('sysctl -n machdep.cpu.brand_string')
              .readlines()[0].split('@')[1][:-4])*1000)})
    results.update({'memsize': int(int(info['memsize']) / (1024 ** 2))})
    # add OS information
    results.update({'os': 'Mac OSX'})
    return results


def _linux_hardware_info():
    """Returns system info on Linux.
    """
    results = {}
    # get cpu number
    sockets = 0
    cores_per_socket = 0
    frequency = 0.0
    for line in [l.split(':') for l in open("/proc/cpuinfo").readlines()]:
        if line[0].strip() == "physical id":
            sockets = max(sockets, int(line[1].strip())+1)
        if line[0].strip() == "cpu cores":
            cores_per_socket = int(line[1].strip())
        if line[0].strip() == "cpu MHz":
            frequency = float(line[1].strip()) / 1000.
    results.update({'cpus': sockets * cores_per_socket})
    # get cpu frequency directly (bypasses freq scaling)
    try:
        file = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
        line = open(file).readlines()[0]
        frequency = float(line.strip('\n')) / 1000000.
    except:
        pass
    results.update({'cpu_freq': frequency})

    # get total amount of memory
    mem_info = dict()
    for line in [l.split(':') for l in open("/proc/meminfo").readlines()]:
        mem_info[line[0]] = line[1].strip('.\n ').strip('kB')
    results.update({'memsize': int(mem_info['MemTotal']) / 1024})
    # add OS information
    results.update({'os': 'Linux'})
    return results


def _win_hardware_info():
    """Returns system info on Windows.
    """
    from comtypes.client import CoGetObject
    results = {'os': 'Windows'}
    try:
        winmgmts_root = CoGetObject("winmgmts:root\\cimv2")
        cpus = winmgmts_root.ExecQuery("Select * from Win32_Processor")
        ncpus = 0
        freq = 0
        for cpu in cpus:
            ncpus += int(cpu.Properties_['NumberOfCores'].Value)
            if not freq:
                freq = int(cpu.Properties_['MaxClockSpeed'].Value)
        results.update({'cpu_freq': freq})
    except:
        ncpus = int(multiprocessing.cpu_count())
    results.update({'cpus': ncpus})
    try:
        mem = winmgmts_root.ExecQuery("Select * from Win32_ComputerSystem")
        tot_mem = 0
        for item in mem:
            tot_mem += int(item.Properties_['TotalPhysicalMemory'].Value)
        tot_mem = int(tot_mem / 1024**2)
        results.update({'memsize': tot_mem})
    except:
        pass
    return results


def local_hardware_info():
    """Returns basic hardware information about the local machine.

    Gives actual number of CPU's in the machine, even when hyperthreading is
    turned on.

    Returns
    -------
    info (dict): Dictionary containing cpu and memory information.

    """
    if sys.platform == 'darwin':
        out = _mac_hardware_info()
    elif sys.platform == 'win32':
        out = _win_hardware_info()
    elif sys.platform in ['linux', 'linux2']:
        out = _linux_hardware_info()
    else:
        out = {}
    return out


def verify_qubit_number(num_qubits):
    """Determines if an user can run a simulation
    with a given number of qubits, as set by their
    system hardware.
    """
    local_hardware = local_hardware_info()
    if 'memsize' in local_hardware.keys():
        # system memory in MB
        sys_mem = local_hardware['memsize']
    else:
        raise Exception('Cannot determine local memory size.')
    max_qubits = np.log2(sys_mem*(1024**2)/128)
    if num_qubits > max_qubits:
        raise Exception("Number of qubits exceeds local memory.")

if __name__ == '__main__':
    print(local_hardware_info())
