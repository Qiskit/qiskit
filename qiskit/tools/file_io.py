# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Utilities for File Input/Output."""

import copy
import datetime
import json
import os

import numpy
from sympy import Basic

from qiskit.result._utils import result_from_old_style_dict
from qiskit._qiskiterror import QISKitError
from qiskit.backends import BaseBackend


def convert_qobj_to_json(in_item):
    """
    Combs recursively through a list/dictionary and finds any non-json
    compatible elements and converts them. E.g. complex ndarray's are
    converted to lists of strings. Assume that all such elements are
    stored in dictionaries!

    Arg:
        in_item (dict or list): the input dict/list
    """

    key_list = []
    for (item_index, item_iter) in enumerate(in_item):
        if isinstance(in_item, list):
            curkey = item_index
        else:
            curkey = item_iter

        if isinstance(in_item[curkey], (list, dict)):
            # go recursively through nested list/dictionaries
            convert_qobj_to_json(in_item[curkey])
        elif isinstance(in_item[curkey], numpy.ndarray):
            # ndarray's are not json compatible. Save the key.
            key_list.append(curkey)

    # convert ndarray's to lists
    # split complex arrays into two lists because complex values are not
    # json compatible
    for curkey in key_list:
        if in_item[curkey].dtype == 'complex':
            in_item[curkey + '_ndarray_imag'] = numpy.imag(
                in_item[curkey]).tolist()
            in_item[curkey + '_ndarray_real'] = numpy.real(
                in_item[curkey]).tolist()
            in_item.pop(curkey)
        else:
            in_item[curkey] = in_item[curkey].tolist()


def convert_json_to_qobj(in_item):
    """Combs recursively through a list/dictionary that was loaded from json
    and finds any lists that were converted from ndarray and converts them back

    Arg:
        in_item (dict or list): the input dict/list
    """

    key_list = []
    for (item_index, item_iter) in enumerate(in_item):
        if isinstance(in_item, list):
            curkey = item_index
        else:
            curkey = item_iter

            # flat these lists so that we can recombine back into a complex
            # number
            if '_ndarray_real' in curkey:
                key_list.append(curkey)
                continue

        if isinstance(in_item[curkey], (list, dict)):
            convert_json_to_qobj(in_item[curkey])

    for curkey in key_list:
        curkey_root = curkey[0:-13]
        in_item[curkey_root] = numpy.array(in_item[curkey])
        in_item.pop(curkey)
        if curkey_root + '_ndarray_imag' in in_item:
            in_item[curkey_root] = in_item[curkey_root] + 1j * numpy.array(
                in_item[curkey_root + '_ndarray_imag'])
            in_item.pop(curkey_root + '_ndarray_imag')


def file_datestr(folder, fileroot):
    """Constructs a filename using the current date-time

    Args:
        folder (str): path to the save folder
        fileroot (str): root string for the file

    Returns:
        String: full file path of the form
            'folder/YYYY_MM_DD_HH_MM_fileroot.json'
    """

    # if the fileroot has .json appended strip it off
    if len(fileroot) > 4 and fileroot[-5:].lower() == '.json':
        fileroot = fileroot[0:-5]

    return os.path.join(
        folder, ('{:%Y_%m_%d_%H_%M_}'.format(datetime.datetime.now()) +
                 fileroot + '.json'))


def load_result_from_file(filename):
    """Load a results dictionary file (.json) to a Result object.
    Note: The json file may not load properly if it was saved with a previous
    version of the SDK.

    Args:
        filename (str): filename of the dictionary

    Returns:
        tuple(Result, dict):
            The new Results object
            if the metadata exists it will get returned
    Raises:
        QISKitError: if the file does not exist or does not have the proper
            dictionary structure.
    """

    if not os.path.exists(filename):
        raise QISKitError('File %s does not exist' % filename)

    with open(filename, 'r') as load_file:
        master_dict = json.load(load_file)

    try:
        qresult_dict = master_dict['result']
        convert_json_to_qobj(qresult_dict)
        metadata = master_dict['metadata']
    except KeyError:
        raise QISKitError('File %s does not have the proper dictionary '
                          'structure')

    # TODO: To keep backwards compatibility with previous saved versions,
    # the method adapts the recovered JSON to match the new format. Since
    # the save function takes a Result, not all the fields required by
    # the new Qobj are saved so they are marked with 'TODO'.
    qresult_dict['id'] = qresult_dict.get('id', 'TODO')
    for experiment in qresult_dict['result']:
        is_done = experiment['status'] == 'DONE'
        experiment['success'] = experiment.get('success', is_done)
        experiment['shots'] = experiment.get('shots', 'TODO')

    qresult = result_from_old_style_dict(
        qresult_dict,
        [circuit_data['name'] for circuit_data in qresult_dict['result']]
    )

    return qresult, metadata


class ResultEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for sympy types.
    """
    def default(self, o):
        # pylint: disable=method-hidden
        if isinstance(o, Basic):  # The element to serialize is a Symbolic type
            if o.is_Integer:
                return int(o)
            if o.is_Float:
                return float(o)
            return str(o)
        elif isinstance(o, BaseBackend):
            # TODO: replace when the deprecation is completed (see also note in
            # Result.__iadd__).
            return o.configuration()['name']

        return json.JSONEncoder.default(self, o)


def save_result_to_file(resultobj, filename, metadata=None):
    """Save a result and optional metatdata to a single dictionary file.

    Args:
        resultobj (Result): Result to save
        filename (str): save path (with or without the json extension). If the
            file already exists then numbers will be appended to the root to
            generate a unique  filename.
            E.g. if filename=test.json and that file exists then the file will
            be changed to test_1.json
        metadata (dict): Add another dictionary with custom data for the
            result (eg fit results)

    Return:
        String: full file path
    """
    master_dict = {
        'result': _old_style_dict_from_result(resultobj)
    }
    if metadata is None:
        master_dict['metadata'] = {}
    else:
        master_dict['metadata'] = copy.deepcopy(metadata)

    # need to convert any ndarray variables to lists so that they can be
    # exported to the json file
    convert_qobj_to_json(master_dict['result'])

    # if the filename has .json appended strip it off
    if filename[-5:].lower() == '.json':
        filename = filename[0:-5]

    append_str = ''
    append_num = 0

    while os.path.exists(filename + append_str + '.json'):
        append_num += 1
        append_str = '_%d' % append_num

    with open(filename + append_str + '.json', 'w') as save_file:
        json.dump(master_dict, save_file, indent=1, cls=ResultEncoder)

    return filename + append_str + '.json'


def _old_style_dict_from_result(result):
    """Convert a ``qiskit.Result`` instance into the old style dict format
    expected by ``save_result_to_file``.

    Args:
        result (qiskit.Result): a ``qiskit.Result`` instance.

    Returns:
        dict: a dictionary with the format previous to Qobj's Result.
    """
    return {
        'job_id': result.job_id,
        'status': result.status,
        'backend': result.backend_name,
        'result': [{
            'name': name,
            'compiled_circuit_qasm': experiment.compiled_circuit_qasm,
            'status': experiment.status,
            'data': experiment.data
        } for name, experiment in result.results.items()]
    }
