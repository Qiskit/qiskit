import random
import subprocess
from subprocess import PIPE
import ast
import json
import re

class QasmCppSimulator:
    def __init__(self, compiled_circuit, shots=1024, seed=None, threads=1,
                 exe='qasm_simulator'):
        self.circuit = compiled_circuit
        self._number_of_qubits = self.circuit['header']['number_of_qubits']
        self._number_of_cbits = self.circuit['header']['number_of_clbits']
        self.result = {}
        self.result['data'] = {}
        self._quantum_state = 0
        self._classical_state = 0
        self._shots = shots
        self._seed = seed
        self._threads = threads
        self._number_of_operations = len(self.circuit['operations'])
        self._pattern = re.compile('\{.*\}')
        
    def run(self):
        cmdFmt = 'qasm_simulator -i stdin -f json -n {shots:d} -t {threads}'
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
        match = re.search(self._pattern, stdOut.decode())
        if match:
            cresult = json.loads(match.group(0))
        else:
            # custom "backend" or "result" exception handler here?
            raise Exception('local_qasm_cpp_simulator returned: {0}\n{1}'.format(
                stdOut.decode(), errOut.decode()))
        self.result['data']['counts'] = cresult['results']
        self.result['time_taken'] = cresult['time_taken']
        self.result['status'] = 'DONE'
        return self.result
