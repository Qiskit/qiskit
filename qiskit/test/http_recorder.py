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

"""Utilities (based on VCRpy) to record remote requests and allow testing offline/cached."""

import json
from contextlib import suppress
from vcr.persisters.filesystem import FilesystemPersister
from vcr import VCR


class IdRemoverPersister(FilesystemPersister):
    """VCR Persister for Qiskit.

    IdRemoverPersister is a VCR persister. This is, it implements a way to save and load cassettes.
    This persister in particular inherits load_cassette from FilesystemPersister (basically, it
    loads a standard cassette in the standard way from the file system). On the saving side, it
    replaces some fields in the JSON content of the responses with dummy values.
    """

    @staticmethod
    def get_responses_with(string_to_find, cassette_dict):
        """Filters the requests from cassette_dict

        Args:
            string_to_find (str): request path
            cassette_dict (dict): a VCR cassette dictionary

        Returns:
            Request: VCR's representation of a request.
        """
        return [response for response, request in
                zip(cassette_dict['responses'], cassette_dict['requests'])
                if string_to_find in request.path]

    @staticmethod
    def get_new_id(field, path, id_tracker, type_=str):
        """Creates a new dummy id (or value) for replacing an existing id (or value).

        Args:
            field (str): field name is used, in same cases, to create a dummy value.
            path (str): path of the request is used, in same cases, to create a dummy value.
            id_tracker (dict): a map of already assigned ids and generated ids.
            type_ (type): type of the value.

        Returns:
            str: that is used to replace a value.
        """

        if type_ == float:
            return 0.42
        if type_ == int:
            return 42
        dummy_name = 'dummy%s%s' % (path.replace('/', ''), field)
        count = len(list(filter(lambda x: str(x).startswith(dummy_name), id_tracker.values())))
        return "%s%02d" % (dummy_name, count + 1)

    @staticmethod
    def get_matching_dicts(data_dict, map_list):
        """Find subdicts that are described in map_list.

        Args:
            data_dict (dict): in which the map_list is going to be searched.
            map_list (list): the list of nested keys to find in the data_dict

        Returns:
            list: a list of dictionaries, each of them matches map_list.
        """
        ret = []
        if not map_list:
            return ret
        if isinstance(data_dict, list):
            for sub_data_dict in data_dict:
                ret.extend(IdRemoverPersister.get_matching_dicts(sub_data_dict, map_list))
        if isinstance(data_dict, dict):
            if map_list[0] in data_dict.keys():
                if len(map_list) == 1:
                    return [data_dict]
                else:
                    ret.extend(
                        IdRemoverPersister.get_matching_dicts(data_dict[map_list[0]], map_list[1:]))
        return ret

    @staticmethod
    def remove_id_in_a_json(jsonobj, field, path, id_tracker):
        """Replaces ids with dummy values in a json.

        Replaces in jsonobj (in-place) the field with dummy value (which is constructed with
        id_tracker, if it was already replaced, or path, if it needs to be created).

        Args:
            jsonobj (dict): json dictionary from the response body
            field (str): string with the field in the response to by replaced
            path (str): request path
            id_tracker (dict): a dictionary of the ids already assigned.
        """

        map_list = field.split('.')
        for matching_dict in IdRemoverPersister.get_matching_dicts(jsonobj, map_list):
            with suppress(KeyError):
                old_id = matching_dict[map_list[-1]]
                if old_id not in id_tracker:
                    new_id = IdRemoverPersister.get_new_id(field, path, id_tracker, type(old_id))
                    id_tracker[old_id] = new_id
                matching_dict[map_list[-1]] = id_tracker[old_id]

    @staticmethod
    def remove_ids_in_a_response(response, fields, path, id_tracker):
        """Replaces ids with dummy values in a response.

        Replaces in response (in-place) the fields with dummy values (which is constructed with
        id_tracker, if it was already replaced, or path, if it needs to be created).

        Args:
            response (dict): dictionary of the response body
            fields (list): list of fields in the response to by replaced
            path (str): request path
            id_tracker (dict): a dictionary of the ids already assigned.
        """
        body = json.loads(response['body']['string'].decode('utf-8'))
        for field in fields:
            IdRemoverPersister.remove_id_in_a_json(body, field, path, id_tracker)
        response['body']['string'] = json.dumps(body).encode('utf-8')

    @staticmethod
    def remove_ids(ids2remove, cassette_dict):
        """Replaces ids with dummy values in a cassette.

        Replaces in cassette_dict (in-place) the fields defined by ids2remove with dummy values.
        Internally, it used a map (id_tracker) between real values and dummy values to keep
        consistency during the renaming.

        Args:
            ids2remove (dict): {request_path: [json_fields]}
            cassette_dict (dict): a VCR cassette dictionary.
        """

        id_tracker = {}  # {old_id: new_id}
        for path, fields in ids2remove.items():
            responses = IdRemoverPersister.get_responses_with(path, cassette_dict)
            for response in responses:
                IdRemoverPersister.remove_ids_in_a_response(response, fields, path, id_tracker)
        for old_id, new_id in id_tracker.items():
            if isinstance(old_id, str):
                for request in cassette_dict['requests']:
                    request.uri = request.uri.replace(old_id, new_id)

    @staticmethod
    def save_cassette(cassette_path, cassette_dict, serializer):
        """Extends FilesystemPersister.save_cassette

        Extends FilesystemPersister.save_cassette. Replaces particular values (defined by
        ids2remove) which are replaced by a dummy value. The full manipulation is in
        cassette_dict, before saving it using FilesystemPersister.save_cassette

        Args:
            cassette_path (str): the file location where the cassette will be saved.
            cassette_dict (dict): a VCR cassette dictionary. This is the information that will
            be dump in cassette_path, using serializer.
            serializer (callable): the serializer for dumping cassette_dict in cassette_path.
        """
        ids2remove = {'/api/users/loginWithToken': ['id',
                                                    'userId',
                                                    'created'],
                      '/api/Jobs': ['id',
                                    'userId',
                                    'creationDate',
                                    'qasms.executionId',
                                    'qasms.result.date',
                                    'qasms.result.data.time',
                                    'qasms.result.data.additionalData.seed'],
                      '/api/Backends': ['internalId',
                                        'topologyId'],
                      '/api/Backends/ibmqx5/queue/status': ['lengthQueue'],
                      '/api/Backends/ibmqx4/queue/status': ['lengthQueue']}
        IdRemoverPersister.remove_ids(ids2remove, cassette_dict)
        super(IdRemoverPersister, IdRemoverPersister).save_cassette(cassette_path,
                                                                    cassette_dict,
                                                                    serializer)


def http_recorder(vcr_mode, cassette_dir):
    """Creates a VCR object in vcr_mode mode.

    Args:
        vcr_mode (string): the parameter for record_mode.
        cassette_dir (string): path to the cassettes.

    Returns:
        VCR: a VCR object.
    """
    my_vcr = VCR(
        cassette_library_dir=cassette_dir,
        record_mode=vcr_mode,
        match_on=['method', 'scheme', 'host', 'port', 'path', 'unordered_query'],
        filter_headers=['x-qx-client-application', 'User-Agent'],
        filter_query_parameters=[('access_token', 'dummyapiusersloginWithTokenid01')],
        filter_post_data_parameters=[('apiToken', 'apiToken_dummy')],
        decode_compressed_response=True,
        before_record_response=_purge_headers_cb(['Date',
                                                  ('Set-Cookie', 'dummy_cookie'),
                                                  'X-Global-Transaction-ID',
                                                  'Etag',
                                                  'Content-Security-Policy',
                                                  'X-Content-Security-Policy',
                                                  'X-Webkit-Csp',
                                                  'content-length']))
    my_vcr.register_matcher('unordered_query', _unordered_query_matcher)
    my_vcr.register_persister(IdRemoverPersister)
    return my_vcr


def _purge_headers_cb(headers):
    """Remove headers from the response.

    Args:
        headers (list): headers to remove from the response

    Returns:
        callable: for been used in before_record_response VCR constructor.
    """
    header_list = []
    for item in headers:
        if not isinstance(item, tuple):
            item = (item, None)
        header_list.append(item[0:2])  # ensure the tuple is a pair

    def before_record_response_cb(response):
        """Purge headers from response.

        Args:
            response (dict): a VCR response

        Returns:
            dict: a VCR response
        """
        for (header, value) in header_list:
            with suppress(KeyError):
                if value:
                    response['headers'][header] = value
                else:
                    del response['headers'][header]
        return response

    return before_record_response_cb


def _unordered_query_matcher(request1, request2):
    """A VCR matcher that ignores the order of values in the query string.

    A VCR matcher (a la VCR.matcher) that ignores the order of the values in the query string.
    Useful for filter params, for example.

    Args:
        request1 (Request): a VCR request
        request2 (Request): a VCR request

    Returns:
        bool: True if they match.
    """
    if request1.query == request2.query:
        return True

    dict1 = dict(request1.query)
    dict2 = dict(request2.query)

    if dict1 == dict2:
        return True

    if dict1.keys() != dict2.keys():
        return False

    for key, value in dict1.items():
        with suppress(ValueError):
            dict1[key] = json.loads(value)
            dict2[key] = json.loads(dict2[key])

    return dict1 == dict2
