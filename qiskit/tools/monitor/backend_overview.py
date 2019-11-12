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
# pylint: disable=invalid-name

""" A module for viewing the details of all available devices.
"""

import math
from qiskit.exceptions import QiskitError


def get_unique_backends():
    """Gets the unique backends that are available.

    Returns:
        list: Unique available backends.

    Raises:
        QiskitError: No backends available.
        ImportError: If qiskit-ibmq-provider is not installed
    """
    try:
        from qiskit.providers.ibmq import IBMQ
    except ImportError:
        raise ImportError("The IBMQ provider is necessary for this function "
                          " to work. Please ensure it's installed before "
                          "using this function")
    backends = []
    for provider in IBMQ.providers():
        for backend in provider.backends():
            backends.append(backend)
    unique_hardware_backends = []
    unique_names = []
    for back in backends:
        if back.name() not in unique_names and not back.configuration().simulator:
            unique_hardware_backends.append(back)
            unique_names.append(back.name())
    if not unique_hardware_backends:
        raise QiskitError('No backends available.')
    return unique_hardware_backends


def backend_monitor(backend):
    """Monitor a single IBMQ backend.

    Args:
        backend (IBMQBackend): Backend to monitor.
    Raises:
        QiskitError: Input is not a IBMQ backend.
        ImportError: If qiskit-ibmq-provider is not installed
    """
    try:
        # pylint: disable=import-error,no-name-in-module
        from qiskit.providers.ibmq import IBMQBackend
    except ImportError:
        raise ImportError("The IBMQ provider is necessary for this function "
                          " to work. Please ensure it's installed before "
                          "using this function")

    if not isinstance(backend, IBMQBackend):
        raise QiskitError('Input variable is not of type IBMQBackend.')
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
                  'backend_version', 'basis_gates',
                  'local', 'simulator']

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
        gate_data = [g for g in props['gates'] if g['qubits'] == [qub]]
        t1_info = qubit_data[0]
        t2_info = qubit_data[1]
        freq_info = qubit_data[2]
        readout_info = qubit_data[3]

        freq = str(round(freq_info['value'], 5))+' '+freq_info['unit']
        T1 = str(round(t1_info['value'],
                       5))+' ' + t1_info['unit']
        T2 = str(round(t2_info['value'],
                       5))+' ' + t2_info['unit']
        for gd in gate_data:
            if gd['gate'] == 'u1':
                U1 = str(round(gd['parameters'][0]['value'], 5))
                break

        for gd in gate_data:
            if gd['gate'] == 'u2':
                U2 = str(round(gd['parameters'][0]['value'], 5))
                break
        for gd in gate_data:
            if gd['gate'] == 'u3':
                U3 = str(round(gd['parameters'][0]['value'], 5))
                break

        readout_error = str(round(readout_info['value'], 5))

        qstr = sep.join([name, freq, T1, T2, U1, U2, U3, readout_error])
        print(offset+qstr)

    print()
    multi_qubit_gates = [g for g in props['gates'] if len(g['qubits']) > 1]
    multi_header = 'Multi-Qubit Gates [Name / Type / Gate Error]'
    print(multi_header)
    print('-'*len(multi_header))

    for qub, gate in enumerate(multi_qubit_gates):
        gate = multi_qubit_gates[qub]
        qubits = gate['qubits']
        ttype = gate['gate']
        error = round(gate['parameters'][0]['value'], 5)
        mstr = sep.join(["{}{}_{}".format(ttype, qubits[0], qubits[1]), ttype, str(error)])
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
            str_list[4] += 'Least busy:   %s' % (count == least_pending_idx)

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
