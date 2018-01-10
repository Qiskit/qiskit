#!/usr/bin/env python
# coding: utf-8
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

# This tool submits a QASM file to any backend and show the result.
# It requires 'Qconfig.py' to set a token of IBM Quantum Experience.
# It supports the following backends in this version:
#   ibmqx2(5 qubits), ibmqx4(5 qubits), ibmqx5(16 qubits),
#   simulator(20 qubits), ibmqx_hpc_qasm_simulator(32 qubits).
# see https://quantumexperience.ng.bluemix.net/qx/devices for more details of the backends.


import json
import time
from argparse import ArgumentParser

from IBMQuantumExperience import IBMQuantumExperience

try:
    import Qconfig
except ImportError:
    raise RuntimeError('You need "Qconfig.py" with a token in the same directory.')


def options():
    parser = ArgumentParser()
    parser.add_argument('--qasm', action='store', help='QASM file')
    parser.add_argument('--device', action='store', default='sim',
                        help='choose a device to run the input (default: sim, qx2, qx4, qx5, hpc)')
    parser.add_argument('--shots', action='store', default=1000, type=int,
                        help='Number of shots (default: 1000)')
    parser.add_argument('--interval', action='store', default=2, type=int,
                        help='Interval time to poll a result (default: 2)')
    args = parser.parse_args()
    print('options:', args)
    if not args.qasm:
        parser.print_help()
        quit()
    return args


def read_asm(infilename):
    with open(infilename) as infile:
        return ''.join(infile.readlines())


def run_qasm(qasm, device='sim', shots=1000, verbose=True, interval=2):
    api = IBMQuantumExperience(Qconfig.APItoken, Qconfig.config)
    qasms = [{'qasm': qasm}]
    devices = {'sim': 'simulator', 'hpc': 'ibmqx_hpc_qasm_simulator', 'qx2': 'ibmqx2', 'qx4': 'ibmqx4', 'qx5': 'ibmqx5'}
    dev = 'simulator'
    if device in devices:
        dev = devices[device]
    hpc = None
    if dev == 'ibmqx_hpc_qasm_simulator':
        hpc = {'multishot_optimization': True, 'omp_num_threads': 1}
    out = api.run_job(qasms=qasms, backend=dev, shots=shots, max_credits=5, hpc=hpc)
    if 'error' in out:
        print(out['error']['message'])
        return None
    jobids = out['id']
    results = api.get_job(jobids)
    if verbose:
        print(results['status'])
    while results['status'] == 'RUNNING':
        time.sleep(interval)
        results = api.get_job(jobids)
        if verbose:
            print(results['status'])
    return results['qasms'][0]


def main():
    args = options()
    qasm = read_asm(args.qasm)
    interval = max(1, args.interval)
    results = run_qasm(qasm=qasm, device=args.device, shots=args.shots, interval=interval)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
