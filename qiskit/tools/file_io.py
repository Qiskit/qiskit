# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
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
"""Utilities for File Input/Output."""

import copy
import datetime
import json
import os

import numpy
from qiskit import Result, QISKitError
from qiskit._qobj import Qobj, QobjConfig, QobjCircuit, QobjCircuitConfig
from qiskit._util import random_string


def convert_result_to_json(in_item):
    """Combs recursively through a list/dictionary and finds any non-json
    compatible elements and converts them. E.g. complex ndarray's are converted
    to lists of strings.
    Assume that all such elements are stored in dictionaries.

    Arg:
        in_item: the input dict/list
    """

    key_list = []
    for (item_index, item_iter) in enumerate(in_item):
        if isinstance(in_item, list):
            curkey = item_index
        else:
            curkey = item_iter

        if isinstance(in_item[curkey], (list, dict)):
            # Go recursively through nested list/dictionaries.
            convert_result_to_json(in_item[curkey])
        elif isinstance(in_item[curkey], numpy.ndarray):
            # ndarray's are not json compatible. Save the key.
            key_list.append(curkey)

    # Convert ndarray's to lists.
    # Split complex arrays into two lists because complex values are not json
    # compatible.
    for curkey in key_list:
        if in_item[curkey].dtype == 'complex':
            in_item[curkey + '_ndarray_imag'] = numpy.imag(
                in_item[curkey]).tolist()
            in_item[curkey + '_ndarray_real'] = numpy.real(
                in_item[curkey]).tolist()
            in_item.pop(curkey)
        else:
            in_item[curkey] = in_item[curkey].tolist()


def convert_json_to_result(in_item):
    """Combs recursively through a list/dictionary that was loaded from json
    and finds any lists that were converted from ndarray and converts them back

    Arg:
        in_item: the input dict/list
    """

    key_list = []
    for (item_index, item_iter) in enumerate(in_item):
        if isinstance(in_item, list):
            curkey = item_index
        else:
            curkey = item_iter

            # Flatten the lists so that it can be recombined back into a
            # complex number.
            if '_ndarray_real' in curkey:
                key_list.append(curkey)
                continue

        if isinstance(in_item[curkey], (list, dict)):
            convert_json_to_result(in_item[curkey])

    for curkey in key_list:
        curkey_root = curkey[0:-13]
        in_item[curkey_root] = numpy.array(in_item[curkey])
        in_item.pop(curkey)
        if curkey_root+'_ndarray_imag' in in_item:
            in_item[curkey_root] = in_item[curkey_root] + 1j * numpy.array(
                in_item[curkey_root+'_ndarray_imag'])
            in_item.pop(curkey_root+'_ndarray_imag')


def file_datestr(folder, fileroot):
    """Constructs a filename using the current date-time

    Args:
        folder (str): path to the save folder
        fileroot (str): root string for the file

    Returns:
        str: full file path of the form 'folder/YYYY_MM_DD_HH_MM_fileroot.json'
    """
    # If the fileroot has .json appended strip it off.
    if len(fileroot) > 4 and fileroot[-5:].lower() == '.json':
        fileroot = fileroot[0:-5]

    return os.path.join(folder, '{:%Y_%m_%d_%H_%M_}'.format(
        datetime.datetime.now()) + fileroot + '.json')


def load_result_from_file(filename):
    """Load a results dictionary file (.json) to a Result object.

    Note: The json file may not load properly if it was saved with a previous
    version of the SDK.

    Args:
        filename (str): filename of the dictionary

    Returns:
        (Result, Dict): Tuple of:
            - The new Results object
            - if the metadata exists it will get returned
    """
    if not os.path.exists(filename):
        raise QISKitError('File %s does not exist' % filename)

    with open(filename, 'r') as load_file:
        master_dict = json.load(load_file)

    try:
        qobj = dict_to_qobj(master_dict['qobj'])
        qresult_dict = master_dict['result']
        convert_json_to_result(qresult_dict)
        metadata = master_dict['metadata']
    except KeyError:
        raise QISKitError('File %s does not have the proper dictionary '
                          'structure')

    qresult = Result(qresult_dict['job_id'],
                     qresult_dict['status'],
                     qresult_dict['result'],
                     qobj)

    return qresult, metadata


def save_result_to_file(result, filename, metadata=None):
    """Save a result (qobj + result) and optional metadata
    to a single dictionary file.

    Args:
        result (Result): Result to save
        filename (str): save path (with or without the json extension).
            If the file already exists then numbers will be appended to the
            root to generate a unique filename.
            E.g. if filename=test.json and that file exists then the file will
            be changed to test_1.json
        metadata (dict): Add another dictionary with custom data for the
            result (eg fit results)

    Return:
        str: full file path
    """
    master_dict = {
        'qobj': copy.deepcopy(result._qobj.as_dict()),
        'result': {
            'job_id': result.get_job_id(),
            'status': result.get_status(),
            'result': copy.deepcopy(result._result)
        },
        'metadata': copy.deepcopy(metadata) or {}
    }

    # Need to convert any ndarray variables to lists so that they can be
    # exported to the json file.
    convert_result_to_json(master_dict['result'])

    # If the filename has .json appended strip it off.
    if filename[-5:].lower() == '.json':
        filename = filename[0:-5]

    append_str = ''
    append_num = 0

    while os.path.exists(filename + append_str + '.json'):
        append_num += 1
        append_str = '_%d' % append_num

    with open(filename + append_str + '.json', 'w') as save_file:
        json.dump(master_dict, save_file, indent=1)

    return filename + append_str + '.json'


def dict_to_qobj(qobj_dict):
    """Convert a dict into a Qobj.
    Args:
        qobj_dict (dict): dict version of a Qobj.

    Returns:
        Qobj: Qobj.
    """
    # Create the QobjConfig.
    qobj_config = QobjConfig(
        backend=qobj_dict['config'].get('backend', None),
        max_credits=qobj_dict['config'].get('max_credits', None),
        shots=qobj_dict['config'].get('shots', None)
    )

    # Create the Qobj, with empty circuits.
    qobj = Qobj(id_=qobj_dict.get('id', random_string(30)),
                config=qobj_config,
                circuits=[])

    for circuit_dict in qobj_dict['circuits']:
        # Create the QobjCircuitConfig.
        qobj_circuit_config = QobjCircuitConfig(**circuit_dict['config'])

        # Create the QobjCircuit.
        qobj_circuit = QobjCircuit(
            name=circuit_dict.get('name', random_string(30)),
            config=qobj_circuit_config,
            compiled_circuit=circuit_dict.get('compiled_circuit', None),
            circuit=circuit_dict.get('circuit', None),
            compiled_circuit_qasm=circuit_dict.get('compiled_circuit_qasm',
                                                   None)
        )

        qobj.circuits.append(qobj_circuit)

    return qobj
