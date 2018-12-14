# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" A module for viewing the details of all available devices.
"""

import math
from qiskit.backends.ibmq import IBMQ

def get_unique_backends():
    """Gets the unique backends that are available.

    Returns:
        list: Unique available backends.
    """
    backends = IBMQ.backends()
    unique_hardware_backends = []
    unique_names = []
    for back in backends:
        if back.name() not in unique_names and not back.configuration().simulator:
            unique_hardware_backends.append(back)
            unique_names.append(back.name())
    return unique_hardware_backends


def backend_overview():
    """Gives overview information on all the IBMQ
    backends that are available.
    """
    unique_hardware_backends = get_unique_backends()
    _backends = []
    # Sort backends by operational or not
    for idx, back in enumerate(unique_hardware_backends):
        if back.status().operational:
            _backends = [back] + _backends
        else:
            _backends = _backends + [back]

    stati = [back.status() for back in _backends]
    idx = list(range(len(_backends)))
    pending = [s.pending_jobs for s in stati]
    _, least_idx = zip(*sorted(zip(pending, idx)))

    # Make sure least pending is operational
    for ind in least_idx:
        if stati[ind].operational:
            least_pending_idx = ind
            break

    num_rows = math.ceil(len(_backends)/3)

    count = 0
    num_backends = len(_backends)
    for _ in range(num_rows):
        max_len = 0
        str_list = ['']*8
        for idx in range(3):
            offset = ' '* 10 if idx else ''
            config = _backends[count].configuration().to_dict()
            props = _backends[count].properties().to_dict()
            n_qubits = config['n_qubits']
            str_list[0] += (' '*(max_len-len(str_list[0]))+offset)
            str_list[0] += _backends[count].name()

            str_list[1] += (' '*(max_len-len(str_list[1]))+offset)
            str_list[1] += '-'*len(_backends[count].name())

            str_list[2] += (' '*(max_len-len(str_list[2]))+offset)
            str_list[2] += 'Num. Qubits:  %s' % config['n_qubits']

            str_list[3] += (' '*(max_len-len(str_list[3]))+offset)
            str_list[3] += 'Pending Jobs: %s' % stati[idx].pending_jobs

            str_list[4] += (' '*(max_len-len(str_list[4]))+offset)
            str_list[4] += 'Least busy:   %s' % (True if idx == least_pending_idx else False)

            str_list[5] += (' '*(max_len-len(str_list[5]))+offset)
            str_list[5] += 'Operational:  %s' % stati[idx].operational

            str_list[6] += (' '*(max_len-len(str_list[6]))+offset)
            str_list[6] += 'Avg. T1:      %s' % round(sum([q[0]['value']
                                                           for q in props['qubits']])/n_qubits, 1)
            str_list[7] += (' '*(max_len-len(str_list[7]))+offset)
            str_list[7] += 'Avg. T2:      %s' % round(sum([q[1]['value']
                                                           for q in props['qubits']])/n_qubits, 1)
            count += 1
            if count == num_backends:
                break
            max_len = max([len(s) for s in str_list])

        print("\n".join(str_list))
        print('\n'*2)
