# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

""" A module for viewing the details of all available devices.
"""

import math
from qiskit.qiskiterror import QISKitError
from qiskit.providers.ibmq import IBMQ
from qiskit.providers.ibmq.ibmqbackend import IBMQBackend


def get_unique_backends():
    """Gets the unique backends that are available.

    Returns:
        list: Unique available backends.

    Raises:
        QISKitError: No backends available.
    """
    backends = IBMQ.backends()
    unique_hardware_backends = []
    unique_names = []
    for back in backends:
        if back.name() not in unique_names and not back.configuration().simulator:
            unique_hardware_backends.append(back)
            unique_names.append(back.name())
    if not unique_hardware_backends:
        raise QISKitError('No backends available.')
    return unique_hardware_backends


def backend_monitor(backend):
    """Monitor a single IBMQ backend.

    Args:
        backend (IBMQBackend): Backend to monitor.
    Raises:
        QISKitError: Input is not a IBMQ backend.
    """
    if not isinstance(backend, IBMQBackend):
        raise QISKitError('Input variable is not of type IBMQBackend.')
    config = backend.configuration().to_dict()
    status = backend.status().to_dict()
    config_dict = {**status, **config}
    if not config['simulator']:
        props = backend.properties().to_dict()

    print(backend.name())
    print('='*len(backend.name()))
    print('Configuration')
    print('-'*13)
    offset = '    '

    upper_list = ['n_qubits', 'operational',
                  'status_msg', 'pending_jobs',
                  'basis_gates', 'local', 'simulator']

    lower_list = list(set(config_dict.keys()).difference(upper_list))
    # Remove gates because they are in a different tab
    lower_list.remove('gates')
    for item in upper_list+lower_list:
        print(offset+item+':', config_dict[item])

    # Stop here if simulator
    if config['simulator']:
        return

    print()
    qubit_header = 'Qubits [Name / Freq / T1 / T2 / U1 err / U2 err / U3 err / Readout err]'
    print(qubit_header)
    print('-'*len(qubit_header))

    sep = ' / '
    for qub in range(len(props['qubits'])):
        name = 'Q%s' % qub
        qubit_data = props['qubits'][qub]
        gate_data = props['gates'][3*qub:3*qub+3]
        t1_info = qubit_data[0]
        t2_info = qubit_data[1]
        freq_info = qubit_data[2]
        readout_info = qubit_data[3]

        freq = str(round(freq_info['value'], 5))+' '+freq_info['unit']
        T1 = str(round(t1_info['value'],  # pylint: disable=invalid-name
                       5))+' ' + t1_info['unit']
        T2 = str(round(t2_info['value'],  # pylint: disable=invalid-name
                       5))+' ' + t2_info['unit']
        # pylint: disable=invalid-name
        U1 = str(round(gate_data[0]['parameters'][0]['value'], 5))
        # pylint: disable=invalid-name
        U2 = str(round(gate_data[1]['parameters'][0]['value'], 5))
        # pylint: disable=invalid-name
        U3 = str(round(gate_data[2]['parameters'][0]['value'], 5))

        readout_error = str(round(readout_info['value'], 5))

        qstr = sep.join([name, freq, T1, T2, U1, U2, U3, readout_error])
        print(offset+qstr)

    print()
    multi_qubit_gates = props['gates'][3*config['n_qubits']:]
    multi_header = 'Multi-Qubit Gates [Name / Type / Gate Error]'
    print(multi_header)
    print('-'*len(multi_header))

    for gate in multi_qubit_gates:
        name = gate['name']
        ttype = gate['gate']
        error = str(round(gate['parameters'][0]['value'], 5))
        mstr = sep.join([name, ttype, error])
        print(offset+mstr)


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
            offset = ' ' * 10 if idx else ''
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
            str_list[3] += 'Pending Jobs: %s' % stati[count].pending_jobs

            str_list[4] += (' '*(max_len-len(str_list[4]))+offset)
            str_list[4] += 'Least busy:   %s' % (True if count == least_pending_idx else False)

            str_list[5] += (' '*(max_len-len(str_list[5]))+offset)
            str_list[5] += 'Operational:  %s' % stati[count].operational

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
