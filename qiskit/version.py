# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Contains the terra version."""

import os
import subprocess
import sys


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT_DIR, "VERSION.txt"), "r") as version_file:
    __version__ = version_file.read().strip()


def _get_qiskit_versions():
    cmd = [sys.executable, '-m', 'pip', 'freeze']
    reqs = subprocess.check_output(cmd)
    reqs_dict = {}
    for req in reqs.split():
        req_parts = req.decode().split('==')
        if len(req_parts) == 1 and req_parts[0].startswith('git'):
            if 'qiskit' in req_parts[0]:
                package = req_parts[0].split('#egg=')[1]
                sha = req_parts[0].split('@')[-1].split('#')[0]
                reqs_dict[package] = 'dev-' + sha
            continue
        elif len(req_parts) == 1:
            continue
        reqs_dict[req_parts[0]] = req_parts[1]
    out_dict = {}
    # Dev/Egg _ to - conversion
    for package in ['qiskit_terra', 'qiskit_ignis', 'qiskit_aer',
                    'qiskit_ibmq_provider', 'qiskit_aqua']:
        if package in reqs_dict:
            out_dict[package.replace('_', '-')] = reqs_dict[package]

    for package in ['qiskit', 'qiskit-terra', 'qiskit-ignis', 'qiskit-aer',
                    'qiskit-ibmq-provider', 'qiskit-aqua']:
        if package in out_dict:
            continue
        if package in reqs_dict:
            out_dict[package] = reqs_dict[package]
        else:
            out_dict[package] = None
    return out_dict


__qiskit_version__ = _get_qiskit_versions()
