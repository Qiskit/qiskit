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
import os
import subprocess
from subprocess import PIPE
import platform
import warnings

import numpy as np

from qiskit._result import Result
from qiskit.backends import BaseBackend
from qiskit.backends.local.localjob import LocalJob

logger = logging.getLogger(__name__)

EXTENSION = '.exe' if platform.system() == 'Windows' else ''

# Add path to compiled qasm simulator
DEFAULT_SIMULATOR_PATHS = [
    # This is the path where Makefile creates the simulator by default
    os.path.abspath(os.path.join(os.path.dirname(__file__),
                                 '../../../out/src/qasm-simulator-cpp/qasm_simulator_cpp'
                                 + EXTENSION)),
    # This is the path where PIP installs the simulator
    os.path.abspath(os.path.join(os.path.dirname(__file__),
                                 'qasm_simulator_cpp' + EXTENSION)),
]


class QasmSimulatorCpp(BaseBackend):
    """C++ quantum circuit simulator with realistic noise"""

    DEFAULT_CONFIGURATION = {
        'name': 'local_qasm_simulator_cpp',
        'url': 'https://github.com/QISKit/qiskit-sdk-py/src/qasm-simulator-cpp',
        'simulator': True,
        'local': True,
        'description': 'A C++ realistic noise simulator for qobj files',
        'coupling_map': 'all-to-all',
        "basis_gates": 'u0,u1,u2,u3,cx,cz,id,x,y,z,h,s,sdg,t,tdg,rzz,' +
                       'snapshot,wait,noise,save,load'
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy())
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
        """Run a QuantumJob on the the backend."""
        return LocalJob(self._run_job, q_job)

    def _run_job(self, q_job):
        qobj = q_job.qobj
        self._validate(qobj)
        result = run(qobj, self._configuration['exe'])
        return Result(result, qobj)

    def _validate(self, qobj):
        if qobj['config']['shots'] == 1:
            warnings.warn('The behavior of getting statevector from simulators '
                          'by setting shots=1 is deprecated and will be removed. '
                          'Use the local_statevector_simulator instead, or place '
                          'explicit snapshot instructions.',
                          DeprecationWarning)
        for circ in qobj['circuits']:
            if 'measure' not in [op['name'] for
                                 op in circ['compiled_circuit']['operations']]:
                logger.warning("no measurements in circuit '%s', "
                               "classical register will remain all zeros.", circ['name'])
        return


class CliffordSimulatorCpp(BaseBackend):
    """"C++ Clifford circuit simulator with realistic noise."""

    DEFAULT_CONFIGURATION = {
        'name': 'local_clifford_simulator_cpp',
        'url': 'https://github.com/QISKit/qiskit-sdk-py/src/qasm-simulator-cpp',
        'simulator': True,
        'local': True,
        'description': 'A C++ Clifford simulator with approximate noise',
        'coupling_map': 'all-to-all',
        'basis_gates': 'cx,id,x,y,z,h,s,sdg,snapshot,wait,noise,save,load'
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.DEFAULT_CONFIGURATION.copy())

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
        """Run a QuantumJob on the the backend.

        Args:
            q_job (QuantumJob): QuantumJob object

        Returns:
            LocalJob: derived from BaseJob
        """
        return LocalJob(self._run_job, q_job)

    def _run_job(self, q_job):
        qobj = q_job.qobj
        self._validate()
        # set backend to Clifford simulator
        if 'config' in qobj:
            qobj['config']['simulator'] = 'clifford'
        else:
            qobj['config'] = {'simulator': 'clifford'}

        result = run(qobj, self._configuration['exe'])
        return Result(result, qobj)

    def _validate(self):
        return


class QASMSimulatorEncoder(json.JSONEncoder):
    """
    JSON encoder for NumPy arrays and complex numbers.

    This functions as the standard JSON Encoder but adds support
    for encoding:
        complex numbers z as lists [z.real, z.imag]
        ndarrays as nested lists.
    """

    # pylint: disable=method-hidden,arguments-differ
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return [obj.real, obj.imag]
        return json.JSONEncoder.default(self, obj)


class QASMSimulatorDecoder(json.JSONDecoder):
    """
    JSON decoder for the output from C++ qasm_simulator.

    This converts complex vectors and matrices into numpy arrays
    for the following keys.
    """
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    # pylint: disable=method-hidden
    def object_hook(self, obj):
        """Special decoding rules for simulator output."""

        for key in ['U_error', 'density_matrix']:
            # JSON is a complex matrix
            if key in obj and isinstance(obj[key], list):
                tmp = np.array(obj[key])
                obj[key] = tmp[::, ::, 0] + 1j * tmp[::, ::, 1]
        for key in ['statevector', 'inner_products']:
            # JSON is a list of complex vectors
            if key in obj:
                for j in range(len(obj[key])):
                    if isinstance(obj[key][j], list):
                        tmp = np.array(obj[key][j])
                        obj[key][j] = tmp[::, 0] + 1j * tmp[::, 1]
        return obj


def run(qobj, executable):
    """
    Run simulation on C++ simulator inside a subprocess.

    Args:
        qobj (dict): qobj dictionary defining the simulation to run
        executable (string): filename (with path) of the simulator executable
    Returns:
        dict: A dict of simulation results
    """

    # Open subprocess and execute external command
    try:
        with subprocess.Popen([executable, '-'],
                              stdin=PIPE, stdout=PIPE, stderr=PIPE) as proc:
            cin = json.dumps(qobj, cls=QASMSimulatorEncoder).encode()
            cout, cerr = proc.communicate(cin)
        if cerr:
            logger.error('ERROR: Simulator encountered a runtime error: %s',
                         cerr.decode())
        sim_output = cout.decode()
        return json.loads(sim_output, cls=QASMSimulatorDecoder)

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


def _generate_coherent_error_matrix(config):
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
