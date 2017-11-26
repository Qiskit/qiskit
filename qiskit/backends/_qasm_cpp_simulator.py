"""Interface to a fast C++ QASM simulator.

Authors: Erick Winston <ewinston@us.ibm.com>
         Christopher J. Wood <cjwood@us.ibm.com>

"""

import json
import subprocess
from subprocess import PIPE, CalledProcessError
import uuid
import numpy as np
from qiskit.backends._basebackend import BaseBackend
from qiskit._result import Result
from ._simulatorerror import SimulatorError


class QasmCppSimulator(BaseBackend):
    """
    Interface to a fast C++ QASM simulator.
    """

    def __init__(self, configuration=None):
        """
        Args:
            configuration (dict): set configuration

        Raises:
            FileNotFoundError: if executable, 'qasm_simulator', is not on path
        """
        if configuration is None:
            self._configuration = {'name': 'local_qasm_cpp_simulator',
                                   'url': 'https://github.com/IBM/qiskit-sdk-py',
                                   'exe': 'qasm_simulator',
                                   'simulator': True,
                                   'local': True,
                                   'description': 'A c++ simulator for qasm files',
                                   'coupling_map': 'all-to-all',
                                   'basis_gates': 'u1,u2,u3,cx,id'
                                  }
        else:
            self._configuration = configuration
        self._is_simulator = self._configuration['simulator']
        self._is_local = True
        self._exe = self._configuration['exe']
        # This assumes we are getting a quick return help message.
        # so _localsimulator can quickly determine whether the compiled
        # simulator is available.
        try:
            subprocess.check_output([self._exe], stderr=subprocess.STDOUT)
        except CalledProcessError:
            pass
        except FileNotFoundError:
            try:
                subprocess.check_output(
                    ['./' + self._exe], stderr=subprocess.STDOUT)
            except CalledProcessError:
                # simulator with no arguments returns 1
                # so this is the "success" case
                self._exe = './' + self._exe
            except FileNotFoundError:
                cmd = '"{0}" or "{1}" '.format(self._exe, './' + self._exe)
                raise FileNotFoundError(cmd)

    def run(self, q_job):
        """
        Run simulation on C++ simulator.

        Args:
            q_job (QuantumJob): describes job

        Returns:
            Result: result object

        Raises:
            SimulatorError: if executable writes to stderr.
        """
        # Generating a string id for the job
        job_id = str(uuid.uuid4())

        qobj = q_job.qobj
        # TODO: use qobj schema for validation
        if 'config' in qobj:
            self.config = qobj['config']
        else:
            self.config = {}
        # defaults
        if 'shots' in self.config:
            self._default_shots = self.config['shots']
        else:
            self._default_shots = 1024
        if 'seed' in self.config:
            self._default_seed = self.config['seed']
        else:
            self._default_seed = 1
        # Number of threads for simulator
        if 'threads' in self.config:
            self._threads = self.config['threads']
        else:
            self._threads = 1
        # Location of simulator exe
        # if 'exe' in self.config:
        #     self._exe = self.config['exe']
        # else:
        #     self._exe = 'qasm_simulator'
        # C++ simulator backend
        if 'simulator' in self.config:
            self._cpp_backend = self.config['simulator']
        else:
            self._cpp_backend = 'qubit'

        cmd = self._exe + ' - '
        with subprocess.Popen(cmd.split(),
                              stdin=PIPE, stdout=PIPE, stderr=PIPE) as proc:
            cin = json.dumps(qobj).encode()
            cout, cerr = proc.communicate(cin)
        if len(cerr) == 0:
            # no error messages, load std::cout
            cresult = json.loads(cout.decode())
            cresult['job_id'] = job_id
            # convert possible complex valued result fields
            for result in cresult['result']:
                for k in ['state', 'saved_states', 'inner_products']:
                    parse_complex(result['data'], k)

            return Result(cresult, qobj)
        else:
            # custom "backend" or "result" exception handler here?
            raise SimulatorError('local_qasm_cpp_simulator returned: {0}\n{1}'.
                                 format(cout.decode(), cerr.decode()))

    def run_circuit(self, circuit):
        """Run a single circuit on the C++ simulator

        Args:
            circuit (dict): JSON circuit from qobj circuits list
        """
        self.cin_dict = {'qasm': circuit['compiled_circuit'],
                         'config': self.config}
        self.result = {}
        self.result['data'] = {}
        if 'config' in circuit:
            circuit_config = circuit['config']
        else:
            circuit_config = {}
        if 'shots' in circuit_config:
            shots = circuit['config']['shots']
        else:
            shots = self._default_shots
        if 'seed' in circuit_config:
            seed = circuit['config']['seed']
        else:
            seed = self._default_seed
        # Build the command line execution string
        cmd = self._exe + ' -i - -c - -f qiskit'
        cmd += ' -n {shots:d}'.format(shots=shots)
        cmd += ' -t {threads:d}'.format(threads=self._threads)
        cmd += ' -b {backend:s}'.format(backend=self._cpp_backend)
        if seed is not None:
            if seed >= 0:
                if isinstance(seed, float):
                    # _quantumprogram.py usually generates float in [0,1]
                    # try to convert to integer which C++ random expects.
                    seed = hash(seed)
                cmd += ' -s {seed:d}'.format(seed=seed)
            else:
                raise TypeError('seed needs to be an unsigned integer')
        # Open subprocess and execute external command
        with subprocess.Popen(cmd.split(),
                              stdin=PIPE, stdout=PIPE, stderr=PIPE) as proc:
            cin = json.dumps(self.cin_dict).encode()
            cout, cerr = proc.communicate(cin)
        if len(cerr) == 0:
            # no error messages, load std::cout
            cresult = json.loads(cout.decode())
            # convert possible complex valued result fields
            for k in ['state', 'saved_states', 'inner_products']:
                parse_complex(cresult['data'], k)
        else:
            # custom "backend" or "result" exception handler here?
            raise SimulatorError('local_qasm_cpp_simulator returned: {0}\n{1}'.
                                 format(cout.decode(), cerr.decode()))
        # Add simulator data
        self.result['data'] = cresult['data']
        # Add simulation time (in seconds)
        self.result['time_taken'] = cresult['time_taken']
        # Report finished
        self.result['status'] = 'DONE'
        return self.result


def parse_complex(output, key):
    """
    Parse complex entries in C++ simulator output.

    This function converts complex numbers in the C++ simulator output
    into python complex numbers. In JSON c++ output complex entries are
    formatted as::

        z = [re(z), im(z)]
        vec = [re(vec), im(vec)]
        ket = {'00':[re(v[00]), im(v[00])], '01': etc...}

    Args:
        output (dict): simulator output.
        key (str): the output key to search for.
    """
    if key in output:
        ref = output[key]
        if isinstance(ref, list):
            if isinstance(ref[0], list):
                # convert complex vector
                for i, j in enumerate(ref):
                    ref[i] = np.array(j[0]) + 1j * np.array(j[1])
            elif isinstance(ref[0], dict):
                # convert complex ket-form
                for i, j in enumerate(ref):
                    for k in j.keys():
                        ref[i][k] = j[k][0] + 1j * j[k][1]
            elif len(ref) == 2:
                # convert complex scalar
                ref = ref[0] + 1j * ref[1]
