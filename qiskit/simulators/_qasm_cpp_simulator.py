import numpy as np
import subprocess
from subprocess import PIPE, CalledProcessError
import json

__configuration = {"name": "local_qasm_cpp_simulator",
                   "url": "https://github.com/IBM/qiskit-sdk-py",
                   "simulator": True,
                   "description": "A python simulator for qasm files",
                   "nQubits": 10,
                   "couplingMap": "all-to-all",
                   "gateset": "SU2+CNOT"}

class QasmCppSimulator:
    def __init__(self, job, threads=1,
                 config=None, exe='qasm_simulator'):
        self.circuit = {'qasm': job['compiled_circuit']}
        if config:
            self.circuit['config'] = config
        self.siminput = {'qasm': self.circuit}
        self._number_of_qubits = self.circuit['qasm']['header']['number_of_qubits']
        self._number_of_cbits = self.circuit['qasm']['header']['number_of_clbits']
        self.result = {}
        self.result['data'] = {}
        self._quantum_state = 0
        self._classical_state = 0
        self._shots = job['shots']
        self._seed = job['seed']
        self._threads = threads
        self._number_of_operations = len(self.circuit['qasm']['operations'])
        # This assumes we are getting a quick return help message.
        # so _localsimulator can quickly determine whether the compiled
        # simulator is available.
        try:
            output = subprocess.check_output([exe], stderr=subprocess.STDOUT)
        except CalledProcessError:
            pass
        except FileNotFoundError:
            try:
                output = subprocess.check_output(['./'+exe],
                                                 stderr=subprocess.STDOUT)
            except CalledProcessError:
                pass
            except FileNotFoundError:
                cmd = '"{0}" or "{1}" '.format(exe, './'+exe)
                raise FileNotFoundError(cmd)
                
                

    def run(self):
        cmdFmt = self._exe + ' -i - -c - -f qiskit -n {shots:d} -t {threads}'
        cmd = cmdFmt.format(shots = self._shots,
                            threads = self._threads)
        if self._seed:
            if self._seed >= 0:
                if isinstance(self._seed, float):
                    #_quantumprogram.py usually generates float in [0,1]
                    #try to convert to integer which c++ random expects.
                    self._seed = hash(self._seed)
                cmd +=  ' -s {seed:d}'.format(seed=self._seed)
            else:
                raise TypeError('seed needs to be an unsigned integer')
        with subprocess.Popen(cmd.split(),
                              stdin=PIPE,
                              stdout=PIPE,
                              stderr=PIPE) as proc:
            procIn = json.dumps(self.circuit).encode()
            stdOut, errOut = proc.communicate(procIn)
        if len(errOut) == 0:
            # no error messages, load stdOut
            cresult = json.loads(stdOut)
            # convert possible complex valued result fields
            for s in ['state', 'saved_states', 'inner_products']:
                self.__parseComplex(cresult, s)
        else:
            # custom "backend" or "result" exception handler here?
            raise Exception('local_qasm_cpp_simulator returned: {0}\n{1}'.format(
                stdOut.decode(), errOut.decode()))
        # add standard simulator output
        self.result['data']['counts'] = cresult['results']
        # add optional simulator output
        if 'measurements' in cresult:
            # add measurement outcome history for each shot
            self.result['data']['meas_history'] = cresult['measurements']
        if 'state' in cresult:
            # add final states for each shot
            self.result['data']['state'] = cresult['state']
        if 'probs' in cresult:
            # add computational basis final probs for each shot
            self.result['data']['meas_probs'] = cresult['probs']
        if 'saved_states' in cresult:
            # add saved states for each shot
            self.result['data']['saved_states'] = cresult['saved_states']
        if 'inner_products' in cresult:
            # add inner products of final state with targets states for each shot
            self.result['data']['inner_products'] = cresult['inner_products']  
        if 'overlaps' in cresult:
            # add overlap of final state with targets states for each shot
            self.result['data']['overlaps'] = cresult['overlaps']
        # add simulation time (in seconds)
        self.result['time_taken'] = cresult['time_taken']
        self.result['status'] = 'DONE'
        return self.result


    def __parseComplex(self, output, key):
        """
        This function converts complex numbers in the c++ simulator output into python
        complex numbers. In JSON c++ output complex entries are formatted as:
            z = [re(z), im(z)]
            vec = [re(vec), im(vec)]
            ket = {'00':[re(v[00]), im(v[00])], '01': etc...}
        """
        if key in output:
            ref = output[key]
            if isinstance(ref, list):
                if isinstance(ref[0], list):
                    # convert complex vector
                    for x in range(len(ref)):
                        ref[x] = np.array(ref[x][0])+1j*np.array(ref[x][1])
                elif isinstance(ref[0], dict):
                    # convert complex ket-form
                    for x in range(len(ref)):
                        for k in ref[0].keys():
                            ref[x][k] = ref[x][k][0]+1j*ref[x][k][1]
                elif len(ref) == 2:
                    # convert complex scalar
                    ref = ref[0] + 1j*ref[1]
