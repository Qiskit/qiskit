# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
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

"""
Interface to C++ quantum circuit simulator with realistic noise.
"""

import json
import logging
import numbers
import os
import subprocess
from subprocess import PIPE
import platform

import numpy as np

from qiskit._result import Result
from qiskit.backends import BaseBackend

logger = logging.getLogger(__name__)

EXTENSION = '.exe' if platform.system() == 'Windows' else ''

# Add path to compiled qiskit simulator
DEFAULT_SIMULATOR_PATHS = [
    # This is the path where Makefile creates the simulator by default
    os.path.abspath(os.path.dirname(__file__) + \
                    '../../../out/qiskit_simulator' + EXTENSION),
    # This is the path where PIP installs the simulator
    os.path.abspath(os.path.dirname(__file__) + '/qiskit_simulator' + EXTENSION),
]


class QISKitCppSimulator(BaseBackend):
    """C++ quantum circuit simulator with realistic noise"""

    def __init__(self, configuration=None):
        super(QISKitCppSimulator, self).__init__(configuration)
        self._configuration = configuration

        if not configuration:
            self._configuration = {
                'name': 'local_qiskit_simulator',
                'url': 'https://github.com/QISKit/qiskit-sdk-py/src/cpp-simulator',
                'simulator': True,
                'local': True,
                'description': 'A C++ realistic noise simulator for qobj files',
                'coupling_map': 'all-to-all',
                "basis_gates": 'u1,u2,u3,cx,id,x,y,z,h,s,sdg,t,tdg,wait,noise,save,load,uzz',
            }

        # Try to use the default executable if not specified.
        if self._configuration.get('exe'):
            paths = [self._configuration.get('exe')]
        else:
            paths = DEFAULT_SIMULATOR_PATHS

        # Ensure that the executable is available.
        try:
            self._configuration['exe'] = next(
                path for path in paths if (os.path.exists(path) and
                                           os.path.getsize(path) > 100))
        except StopIteration:
            raise FileNotFoundError('Simulator executable not found (using %s)' %
                                    self._configuration.get('exe', 'default locations'))

    def run(self, q_job):
        qobj = q_job.qobj
        result = run(qobj, self._configuration['exe'])
        return Result(result, qobj)


class CliffordCppSimulator(BaseBackend):
    """"C++ Clifford circuit simulator with realistic noise."""

    def __init__(self, configuration=None):
        super(CliffordCppSimulator, self).__init__(configuration)
        self._configuration = configuration

        if not configuration:
            self._configuration = {
                'name': 'local_clifford_simulator',
                'url': 'https://github.com/QISKit/qiskit-sdk-py/src/cpp-simulator',
                'simulator': True,
                'local': True,
                'description': 'A C++ Clifford simulator with approximate noise',
                'coupling_map': 'all-to-all',
                'basis_gates': 'cx,id,x,y,z,h,s,sdg,wait,noise,save,load'
            }

        # Try to use the default executable if not specified.
        if self._configuration.get('exe'):
            paths = [self._configuration.get('exe')]
        else:
            paths = DEFAULT_SIMULATOR_PATHS

        # Ensure that the executable is available.
        try:
            self._configuration['exe'] = next(
                path for path in paths if (os.path.exists(path) and
                                           os.path.getsize(path) > 100))
        except StopIteration:
            raise FileNotFoundError('Simulator executable not found (using %s)' %
                                    self._configuration.get('exe', 'default locations'))

    def run(self, q_job):
        qobj = q_job.qobj
        # set backend to Clifford simulator
        if 'config' in qobj:
            qobj['config']['simulator'] = 'clifford'
        else:
            qobj['config'] = {'simulator': 'clifford'}

        result = run(qobj, self._configuration['exe'])
        return Result(result, qobj)


def run(qobj, executable):
    """
    Run simulation on C++ simulator inside a subprocess.

    Args:
        qobj (dict): qobj dictionary defining the simulation to run
        executable (string): filename (with path) of the simulator executable
    Returns:
        dict: A dict of simulation results
    """
    if 'config' in qobj:
        qobj['config'] = __to_json_complex(qobj['config'])

    for j in range(len(qobj['circuits'])):
        if 'config' in qobj['circuits'][j]:
            qobj['circuits'][j]['config'] = __to_json_complex(
                qobj['circuits'][j]['config'])

    # Open subprocess and execute external command
    try:
        with subprocess.Popen([executable, '-'],
                              stdin=PIPE, stdout=PIPE, stderr=PIPE) as proc:
            cin = json.dumps(qobj).encode()
            cout, cerr = proc.communicate(cin)
        if cerr:
            logger.error('ERROR: Simulator encountered a runtime error: %s',
                         cerr.decode())

        cresult = json.loads(cout.decode())

        if 'result' in cresult:
            # If not Clifford simulator parse JSON complex numbers in output
            if cresult.get('simulator') != 'clifford':
                for result in cresult['result']:
                    if result['success'] is True:
                        __parse_sim_data(result['data'])
        return cresult

    except FileNotFoundError:
        msg = "ERROR: Simulator exe not found at: %s" % executable
        logger.error(msg)
        return {"status": msg, "success": False}


def __to_json_complex(obj):
    """Converts a numpy array to a nested list.
    This is for exporting to JSON. Complex numbers are converted to
    a length two list z -> [z.real, z.imag].
    """
    if isinstance(obj, complex):
        obj = [obj.real, obj.imag]
        return obj
    elif isinstance(obj, (np.ndarray, list)):
        obj = list(obj)
        for i, _ in enumerate(obj):
            obj[i] = __to_json_complex(obj[i])
        return obj
    elif isinstance(obj, dict):
        for i, j in obj.items():
            obj[i] = __to_json_complex(j)
        return obj

    return obj


def __parse_json_complex_single(val):
    if isinstance(val, list) \
            and len(val) == 2 \
            and isinstance(val[0], numbers.Real) \
            and isinstance(val[1], numbers.Real):
        return val[0] + 1j * val[1]
    elif isinstance(val, numbers.Real):
        return val
    return val


def __parse_json_complex(val):
    if isinstance(val, list):
        return np.array([__parse_json_complex_single(j)
                         for j in val])
    elif isinstance(val, dict):
        return {i: __parse_json_complex_single(j)
                for i, j in val.items()}
    return val


def __parse_sim_data(data):
    if 'quantum_states' in data:
        tmp = [__parse_json_complex(psi)
               for psi in data['quantum_states']]
        data['quantum_states'] = tmp
    if 'density_matrix' in data:
        tmp = np.array([__parse_json_complex(row)
                        for row in data['density_matrix']])
        data['density_matrix'] = tmp
    if 'inner_products' in data:
        tmp = [__parse_json_complex(ips)
               for ips in data['inner_products']]
        data['inner_products'] = tmp
    if 'saved_quantum_states' in data:
        for j in range(len(data['saved_quantum_states'])):
            tmp = {}
            for key, val in data['saved_quantum_states'][j].items():
                val = __parse_json_complex(val)
                tmp[int(key)] = val
            data['saved_quantum_states'][j] = tmp
    if 'saved_density_matrix' in data:
        for j in range(len(data['saved_density_matrix'])):
            tmp = {}
            for key, val in data['saved_density_matrix'][j].items():
                val = np.array([__parse_json_complex(row)
                                for row in val])
                tmp[int(key)] = val
            data['saved_density_matrix'][j] = tmp
