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
                    '../../../out/src/qiskit-simulator/qiskit_simulator' + EXTENSION),
    # This is the path where PIP installs the simulator
    os.path.abspath(os.path.dirname(__file__) + '/qiskit_simulator' + EXTENSION),
]


class QISKitCppSimulator(BaseBackend):
    """C++ quantum circuit simulator with realistic noise"""

    def __init__(self, configuration=None):
        super().__init__(configuration)
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
        super().__init__(configuration)
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
            __generate_coherent_error_matrix(qobj['circuits'][j]['config'])
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
                        if 'noise_params' in result:
                            __parse_noise_params(result['noise_params'])
        return cresult

    except FileNotFoundError:
        msg = "ERROR: Simulator exe not found at: %s" % executable
        logger.error(msg)
        return {"status": msg, "success": False}


def cx_error_matrix(cal_error, zz_error):
    """
    Return the coherent error matrix for CR error model of a CNOT gate.

    Args:
        cal_error (double): calibration error of rotation
        zz_error (double): ZZ interaction term error

    Returns:
        numpy.ndarray: A coherent error matrix U_error for the CNOT gate.

    Details:

    The ideal cross-resonsance (CR) gate corresponds to a 2-qubit rotation
        U_CR_ideal = exp(-1j * (pi/2) * XZ/2)
    where qubit-0 is the control, and qubit-1 is the target. This can be
    converted to a CNOT gate by single-qubit rotations
        U_CX = U_L * U_CR_ideal * U_R.

    The noisy rotation is implemented as
        U_CR_noise = exp(-1j * (pi/2 + cal_error) * (XZ + zz_error ZZ)/2)

    The retured error matrix is given by
        U_error = U_L * U_CR_noise * U_R * U_CX^dagger
    """
    # pylint: disable=invalid-name
    if cal_error == 0 and zz_error == 0:
        return np.eye(4)

    cx_ideal = np.array([[1, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0]])
    b = np.sqrt(1.0 + zz_error * zz_error)
    a = b * (np.pi / 2.0 + cal_error) / 2.0
    sp = (1.0 + 1j * zz_error) * np.sin(a) / b
    sm = (1.0 - 1j * zz_error) * np.sin(a) / b
    c = np.cos(a)
    cx_noise = np.array([[c + sm, 0, -1j * (c - sm), 0],
                         [0, 1j * (c - sm), 0, c + sm],
                         [-1j * (c - sp), 0, c + sp, 0],
                         [0, c + sp, 0, 1j * (c - sp)]]) / np.sqrt(2)
    return cx_noise.dot(cx_ideal.conj().T)


def x90_error_matrix(cal_error, detuning_error):
    """
    Return the coherent error matrix for a X90 rotation gate.

    Args:
        cal_error (double): calibration error of rotation
        detuning_error (double): detuning amount for rotation axis error

    Returns:
        numpy.ndarray: A coherent error matrix U_error for the X90 gate.

    Details:

    The ideal X90 rotation is a pi/2 rotation about the X-axis:
        U_X90_ideal = exp(-1j (pi/2) X/2)
    The noisy rotation is implemented as
        U_X90_noise = exp(-1j (pi/2 + cal_error) (cos(d) X + sin(d) Y)/2)
    where d is the detuning_error.

    The retured error matrix is given by
        U_error = U_X90_noise * U_X90_ideal^dagger
    """
    # pylint: disable=invalid-name
    if cal_error == 0 and detuning_error == 0:
        return np.eye(2)
    else:
        x90_ideal = np.array([[1., -1.j], [-1.j, 1]]) / np.sqrt(2)
        c = np.cos(0.5 * cal_error)
        s = np.sin(0.5 * cal_error)
        gamma = np.exp(-1j * detuning_error)
        x90_noise = np.array([[c - s, -1j * (c + s) * gamma],
                              [-1j * (c + s) * np.conj(gamma), c - s]]) / np.sqrt(2)
    return x90_noise.dot(x90_ideal.conj().T)


def __generate_coherent_error_matrix(config):
    """
    Generate U_error matrix for CX and X90 gates.

    Args:
        config (dict): the config of a qobj circuit

    This parses the config for the following noise parameter keys and returns a
    coherent error matrix for simulation coherent noise.
        'CX' gate: 'calibration_error', 'zz_error'
        'X90' gate: 'calibration_error', 'detuning_error'
    """
    # pylint: disable=invalid-name
    if 'noise_params' in config:
        # Check for CR coherent error parameters
        if 'CX' in config['noise_params']:
            noise_cx = config['noise_params']['CX']
            cal_error = noise_cx.pop('calibration_error', 0)
            zz_error = noise_cx.pop('zz_error', 0)
            # Add to current coherent error matrix
            if not cal_error == 0 or not zz_error == 0:
                u_error = noise_cx.get('U_error', np.eye(4))
                u_error = u_error.dot(cx_error_matrix(cal_error, zz_error))
                config['noise_params']['CX']['U_error'] = u_error
        # Check for X90 coherent error parameters
        if 'X90' in config['noise_params']:
            noise_x90 = config['noise_params']['X90']
            cal_error = noise_x90.pop('calibration_error', 0)
            detuning_error = noise_x90.pop('detuning_error', 0)
            # Add to current coherent error matrix
            if not cal_error == 0 or not detuning_error == 0:
                u_error = noise_x90.get('U_error', np.eye(2))
                u_error = u_error.dot(x90_error_matrix(cal_error,
                                                       detuning_error))
                config['noise_params']['X90']['U_error'] = u_error


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
        return np.array([__parse_json_complex_single(j) for j in val])
    elif isinstance(val, dict):
        return {i: __parse_json_complex_single(j) for i, j in val.items()}
    return val


def __parse_noise_params(noise):
    if isinstance(noise, dict):
        for key, val in noise.items():
            if isinstance(val, dict):
                if 'U_error' in val:
                    tmp = np.array([__parse_json_complex(row)
                                    for row in val['U_error']])
                    noise[key]['U_error'] = tmp


def __parse_sim_data(data):
    if 'quantum_states' in data:
        tmp = [__parse_json_complex(psi) for psi in data['quantum_states']]
        data['quantum_states'] = tmp
    if 'density_matrix' in data:
        tmp = np.array([__parse_json_complex(row)
                        for row in data['density_matrix']])
        data['density_matrix'] = tmp
    if 'inner_products' in data:
        tmp = [__parse_json_complex(ips) for ips in data['inner_products']]
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
                val = np.array([__parse_json_complex(row) for row in val])
                tmp[int(key)] = val
            data['saved_density_matrix'][j] = tmp
