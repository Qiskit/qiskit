# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Utilities (based on VCRpy) to record remote requets and allow testing offline/cached."""

import json
from vcr.persisters.filesystem import FilesystemPersister
from vcr import VCR

class IdRemoverPersister(FilesystemPersister):
    '''
    IdRemoverPersister is a VCR persister. This is, it implements a way to save and load cassettes.
    This persister in particular inherits load_cassette from FilesystemPersister (basically, it
    loads a standard cassette in the standard way from the FS). On the saving side, it removes some
    fields in the JSON content of the responses for dummy values.
    '''

    @staticmethod
    def get_responses_with(string2find, cassette_dict):
        '''
        Filters the requests from cassette_dict
        :param string2find (str): request path
        :param cassette_dict (dict): a VCR cassette dictionary
        :return (Request): VCR's representation of a request.
        '''
        requests_indeces = [i for i, x in enumerate(cassette_dict['requests']) if
                            string2find in x.path]
        return [cassette_dict['responses'][i] for i in requests_indeces]

    @staticmethod
    def get_new_id(field, path, id_tracker, _type=str):
        '''
        Creates a new dummy id (or value) for replacing an existing id (or value).
        :param field (str): field name is used, in same cases, to create a dummy value.
        :param path (str): path of the request is used, in same cases, to create a dummy value.
        :param id_tracker (dict): a map of already assigned ids and generated ids.
        :param _type (type): type of the value.
        :return (str): that is used to replace a value.
        '''
        if _type == float:
            return 0.42
        if _type == int:
            return 42
        dummy_name = 'dummy%s%s' % (path.replace('/', ''), field)
        count = len(list(filter(lambda x: str(x).startswith(dummy_name), id_tracker.values())))
        return "%s%02d" % (dummy_name, count + 1)

    @staticmethod
    def get_maching_dicts(data_dict, map_list):
        '''
        :param data_dict (dict): in which the map_list is going to be searched.
        :param map_list (list): the list of nested keys to find in the data_dict
        :return (dict): the dictionary in which matches map_list.
        '''
        ret = []
        if map_list:
            return ret
        if isinstance(data_dict, list):
            _ = [ret.extend(IdRemoverPersister.get_maching_dicts(i, map_list)) for i in data_dict]
        if isinstance(data_dict, dict):
            if map_list[0] in data_dict.keys():
                if len(map_list) == 1:
                    return [data_dict]
                else:
                    ret.extend(
                        IdRemoverPersister.get_maching_dicts(data_dict[map_list[0]], map_list[1:]))
        return ret

    @staticmethod
    def remove_id_in_a_json(jsonobj, field, path, id_tracker):
        '''
        Replaces in jsonobj (in-place) the field with dummy value (which is constructed with
        id_tracker, if it was already reaplced, or path, if it needs to be created).
        :param jsonobj (dict): json dictionary from the response body
        :param field (str): string with the field in the response to by replaced
        :param path (str): request path
        :param id_tracker (dict): a dictionary of the ids already assigned.
        '''
        map_list = field.split('.')
        for maching_dict in IdRemoverPersister.get_maching_dicts(jsonobj, map_list):
            try:
                old_id = maching_dict[map_list[-1]]
                if old_id not in id_tracker:
                    new_id = IdRemoverPersister.get_new_id(field, path, id_tracker, type(old_id))
                    id_tracker[old_id] = new_id
                maching_dict[map_list[-1]] = id_tracker[old_id]
            except KeyError:
                pass

    @staticmethod
    def remove_ids_in_a_response(response, fields, path, id_tracker):
        '''
        Replaces in response (in-place) the fields with dummy values (which is constructed with
        id_tracker, if it was already reaplced, or path, if it needs to be created).
        :param response (dict): dictionary of the response body
        :param fields (list): list of fields in the response to by replaced
        :param path (str): request path
        :param id_tracker (dict): a dictionary of the ids already assigned.
        '''
        body = json.loads(response['body']['string'].decode('utf-8'))
        for field in fields:
            IdRemoverPersister.remove_id_in_a_json(body, field, path, id_tracker)
        response['body']['string'] = json.dumps(body).encode('utf-8')

    @staticmethod
    def remove_ids(ids2remove, cassette_dict):
        '''
        Replaces in cassette_dict (in-place) the fields defined by ids2remove with dummy values.
        :param ids2remove (dict): {request_path: [json_fields]}
        :param cassette_dict (dict): a VCR cassette dictionary.
        '''
        id_tracker = {}  # {old_id: new_id}
        for path, fields in ids2remove.items():
            responses = IdRemoverPersister.get_responses_with(path, cassette_dict)
            for response in responses:
                IdRemoverPersister.remove_ids_in_a_response(response, fields, path, id_tracker)
        for old_id, new_id in id_tracker.items():
            if not isinstance(old_id, str):
                continue
            for request in cassette_dict['requests']:
                request.uri = request.uri.replace(old_id, new_id)

    @staticmethod
    def save_cassette(cassette_path, cassette_dict, serializer):
        '''
        Extendeds FilesystemPersister.save_cassette. Replaces particular values (defined by
        ids2remove) which are replaced by a dummy value. The full manipulation is in
        cassette_dict, before saving it using FilesystemPersister.save_cassette
        :param cassette_path (str): the file location where the cassette will be saved.
        :param cassette_dict (dict): a VCR cassette dictionary. This is the information that will
        be dump in cassette_path, using serializer.
        :param serializer (func): the serializer for dumping cassette_dict in cassette_path.
        '''
        ids2remove = {'api/users/loginWithToken': ['id',
                                                   'userId',
                                                   'created'],
                      'api/Jobs': ['id',
                                   'userId',
                                   'creationDate',
                                   'qasms.executionId',
                                   'qasms.result.date',
                                   'qasms.result.data.time',
                                   'qasms.result.data.additionalData.seed'],
                      'api/Backends/ibmqx5/queue/status': ['lengthQueue'],
                      'api/Backends/ibmqx4/queue/status': ['lengthQueue']}
        IdRemoverPersister.remove_ids(ids2remove, cassette_dict)
        super(IdRemoverPersister, IdRemoverPersister).save_cassette(cassette_path,
                                                                    cassette_dict,
                                                                    serializer)


def purge_headers(headers):
    '''
    :param headers (list): headers to remove from the response
    :return func: before_record_response
    '''
    header_list = list()
    for item in headers:
        if not isinstance(item, tuple):
            item = (item, None)
        header_list.append((item[0], item[1]))

    def before_record_response(response):
        '''
        :param response (dict): a VCR response
        :return (dict): a VCR response
        '''
        for (header, value) in header_list:
            try:
                if value:
                    response['headers'][header] = value
                else:
                    del response['headers'][header]
            except KeyError:
                pass
        return response

    return before_record_response

def http_recorder(VCR_MODE):
    recorder = VCR(
    cassette_library_dir='test/cassettes',
    record_mode=VCR_MODE,
    match_on=['uri', 'method'],
    filter_headers=['x-qx-client-application', 'User-Agent'],
    filter_query_parameters=[('access_token', 'dummyapiusersloginWithTokenid01')],
    filter_post_data_parameters=[('apiToken', 'apiToken_dummy')],
    decode_compressed_response=True,
    before_record_response=purge_headers(['Date',
                                          ('Set-Cookie', 'dummy_cookie'),
                                          'X-Global-Transaction-ID',
                                          'Etag',
                                          'Content-Security-Policy',
                                          'X-Content-Security-Policy',
                                          'X-Webkit-Csp',
                                          'content-length']))
    recorder.register_persister(IdRemoverPersister)
    return recorder