# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Interface to C++ quantum circuit simulator with realistic noise.
"""


import uuid
import json
import logging
import os
import subprocess
from subprocess import PIPE
import platform

import numpy as np

from qiskit.qobj import Result as QobjResult
from qiskit.qobj import ExperimentResult as QobjExperimentResult
from qiskit.result import Result
from qiskit.result._utils import copy_qasm_from_qobj_into_result
from qiskit.backends import BaseBackend
from qiskit.backends.aer.aerjob import AerJob
from qiskit.qobj import Qobj

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


class QasmSimulator(BaseBackend):
    """C++ quantum circuit simulator with realistic noise"""

    DEFAULT_CONFIGURATION = {
        'backend_name': 'qasm_simulator',
        'backend_version': '1.0',
        'n_qubits': -1,        
        'url': 'https://github.com/QISKit/qiskit-terra/src/qasm-simulator-cpp',
        'simulator': True,
        'local': True,
        'conditional': True,
        'description': 'A C++ realistic noise simulator for qasm experiments',
        'coupling_map': 'all-to-all',
        'basis_gates': 'u0,u1,u2,u3,cx,cz,id,x,y,z,h,s,sdg,t,tdg,rzz,' +
                       'snapshot,wait,noise,save,load'
    }

    def __init__(self, configuration=None, provider=None):
        super().__init__(configuration=configuration or self.DEFAULT_CONFIGURATION.copy(),
                         provider=provider)

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

    def run(self, qobj):
        """Run qobj asynchronously.
        Args:
            qobj (Qobj): payload of the experiments
        Returns:
            AerJob: derived from BaseJob        
        """
        job_id = str(uuid.uuid4())
        aer_job = AerJob(self, job_id, self._run_job, qobj)
        aer_job.submit()
        return aer_job

    def _run_job(self, job_id, qobj):
        """Run experiments in qobj"""
        self._validate(qobj)
        result = launch(qobj, self._configuration['exe'])
        result['job_id'] = job_id
        copy_qasm_from_qobj_into_result(qobj, result)

        experiment_names = [experiment.header.name for experiment in qobj.experiments]
        return Result(QobjResult(**result), experiment_names)

    def _validate(self, qobj):
        for experiment in qobj.experiments:
            if 'measure' not in [op.name for
                                 op in experiment.instructions]:
                logger.warning("no measurements in circuit '%s', "
                               "classical register will remain all zeros.",
                               experiment.header.name)


class CliffordSimulator(BaseBackend):
    """"C++ Clifford circuit simulator with realistic noise."""

    DEFAULT_CONFIGURATION = {
        'backend_name': 'clifford_simulator',
        'backend_version': '1.0',
        'n_qubits': -1,
        'url': 'https://github.com/QISKit/qiskit-terra/src/qasm-simulator-cpp',
        'simulator': True,
        'local': True,
        'conditional': True,
        'description': 'A C++ Clifford simulator with approximate noise',
        'coupling_map': 'all-to-all',
        'basis_gates': 'cx,id,x,y,z,h,s,sdg,snapshot,wait,noise,save,load'
    }

    def __init__(self, configuration=None, provider=None):
        super().__init__(configuration=configuration or self.DEFAULT_CONFIGURATION.copy(),
                         provider=provider)

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

    def run(self, qobj):
        """Run a Qobj on the backend.

        Args:
            qobj (dict): job description

        Returns:
            AerJob: derived from BaseJob
        """
        job_id = str(uuid.uuid4())
        aer_job = AerJob(self, job_id, self._run_job, qobj)
        aer_job.submit()
        return aer_job

    def _run_job(self, job_id, qobj):
        if isinstance(qobj, Qobj):
            qobj_dict = qobj.as_dict()
        else:
            qobj_dict = qobj
        self._validate()
        # set backend to Clifford simulator
        if 'config' in qobj_dict:
            qobj_dict['config']['simulator'] = 'clifford'
        else:
            qobj_dict['config'] = {'simulator': 'clifford'}

        qobj = Qobj.from_dict(qobj_dict)
        result = launch(qobj, self._configuration['exe'])
        result['job_id'] = job_id
        copy_qasm_from_qobj_into_result(qobj, result)

        experiment_names = [experiment.header.name for experiment in qobj.experiments]
        return Result(QobjResult(**result), experiment_names)

    def _validate(self):
        return


class QASMSimulatorEncoder(json.JSONEncoder):
    """
    JSON encoder for NumPy arrays and complex numbers.

    This functions as the standard JSON Encoder but adds support
    for encoding:

        * complex numbers z as lists [z.real, z.imag]
        * ndarrays as nested lists.
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


def launch(qobj, executable):
    """
    Launch a subprocess and run the C++ simulation inside it.

    Args:
        qobj (Qobj): qobj dictionary defining the simulation to run
        executable (string): filename (with path) of the simulator executable

    Returns:
        dict: A dict of simulation results
    """

    # Open subprocess and execute external command
    try:
        with subprocess.Popen([executable, '-'],
                              stdin=PIPE, stdout=PIPE, stderr=PIPE) as proc:
            cin = json.dumps(qobj.as_dict(),
                             cls=QASMSimulatorEncoder).encode()
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
