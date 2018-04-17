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
#
# Examples:
#   $ python run_qasm.py -b              # show backend information
#   $ python run_qasm.py -c              # show remaining credits
#   $ python run_qasm.py -l 10           # show job list (10 jobs)
#   $ python run_qasm.py -j (job id)     # show the result of a job
#   $ python run_qasm.py -q (qasm file)  # submit a qasm file

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
    parser.add_argument('-q', '--qasm', action='store', help='QASM file')
    parser.add_argument('-d', '--device', action='store', default='sim',
                        help='choose a device to run the input (sim [default], qx2, qx4, qx5, hpc)')
    parser.add_argument('-s', '--shots', action='store', default=1000, type=int,
                        help='Number of shots (default: 1000)')
    parser.add_argument('-i', '--interval', action='store', default=2, type=int,
                        help='Interval time to poll a result (default: 2)')
    parser.add_argument('-l', '--job-list', action='store', default=10, type=int,
                        help='Number of jobs to show')
    parser.add_argument('-j', '--jobid', action='store', type=str, help='Get job information')
    parser.add_argument('-b', '--backends', action='store_true', help='Show backends information')
    parser.add_argument('-c', '--credits', action='store_true', help='Show my credits')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')
    args = parser.parse_args()
    if args.verbose:
        print('options:', args)
    return args


class JobManager:
    def __init__(self):
        self._api = IBMQuantumExperience(Qconfig.APItoken, Qconfig.config)

    @staticmethod
    def read_asm(infilename):
        with open(infilename) as infile:
            return ''.join(infile.readlines())

    def run_qasm(self, qasm, device='sim', shots=1000, verbose=True, interval=2):
        qasms = [{'qasm': qasm}]
        devices = {'sim': 'ibmqx_qasm_simulator', 'hpc': 'ibmqx_hpc_qasm_simulator',
                   'qx2': 'ibmqx2', 'qx4': 'ibmqx4', 'qx5': 'ibmqx5'}
        dev = 'simulator'
        if device in devices:
            dev = devices[device]
        hpc = None
        if dev == 'ibmqx_hpc_qasm_simulator':
            hpc = {'multishot_optimization': True, 'omp_num_threads': 1}
        out = self._api.run_job(job=qasms, backend=dev, shots=shots, max_credits=5, hpc=hpc)
        if 'error' in out:
            print(out['error']['message'])
            return None
        jobid = out['id']
        print('job id:', jobid)
        results = self._api.get_job(jobid)
        if verbose:
            print(results['status'])
        while results['status'] == 'RUNNING':
            time.sleep(interval)
            results = self._api.get_job(jobid)
            if verbose:
                print(results['status'])
        return results

    def get_job_list(self, n_jobs):
        jobs = self._api.get_jobs(limit=n_jobs)
        tab = {}
        for v in jobs:
            job_id = v['id']
            status = v['status']
            cdate = v['creationDate']
            tab[cdate] = (status, job_id)
        for cdate, v in sorted(tab.items()):
            print('{}\t{}\t{}'.format(cdate, *v))

    def get_job(self, job_id):
        result = self._api.get_job(job_id)
        print(json.dumps(result, sort_keys=True, indent=2))

    def get_credits(self):
        print('credits :', self._api.get_my_credits())

    def available_backends(self, verbose=False):
        tab = {}
        for e in self._api.available_backends() + self._api.available_backend_simulators():
            status = self._api.backend_status(e['name'])
            try:
                tab[e['name']] = [':', str(e['nQubits']) + ' qubits,', e['description'], status]
            except KeyError:
                tab[e['name']] = [':', status]
            if verbose:
                tab[e['name']].append(e)
        for k, v in sorted(tab.items()):
            print(k, *v)


def main():
    jm = JobManager()
    args = options()
    if args.backends:
        jm.available_backends(args.verbose)
    if args.credits:
        jm.get_credits()
    if args.qasm:
        qasm = jm.read_asm(args.qasm)
        interval = max(1, args.interval)
        results = jm.run_qasm(qasm=qasm, device=args.device, shots=args.shots, interval=interval)
        print(json.dumps(results, indent=2, sort_keys=True))
    elif args.jobid:
        jm.get_job(args.jobid)
    elif args.job_list > 0:
        jm.get_job_list(args.job_list)


if __name__ == '__main__':
    main()
