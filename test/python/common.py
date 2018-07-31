# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Shared functionality and helpers for the unit tests."""

from enum import Enum
import functools
import inspect
import logging
import os
import unittest
from unittest.util import safe_repr
import json
from vcr.persisters.filesystem import FilesystemPersister
from vcr import VCR as vcr
from qiskit import __path__ as qiskit_path
from qiskit.backends.ibmq import IBMQProvider
from qiskit.wrapper.credentials import discover_credentials, get_account_name
from qiskit.wrapper.defaultqiskitprovider import DefaultQISKitProvider


class Path(Enum):
    """Helper with paths commonly used during the tests."""
    # Main SDK path:    qiskit/
    SDK = qiskit_path[0]
    # test.python path: qiskit/test/python/
    TEST = os.path.dirname(__file__)
    # Examples path:    examples/
    EXAMPLES = os.path.join(SDK, '../examples')
    # Schemas path:     qiskit/schemas
    SCHEMAS = os.path.join(SDK, 'schemas')


class QiskitTestCase(unittest.TestCase):
    """Helper class that contains common functionality."""

    @classmethod
    def setUpClass(cls):
        cls.moduleName = os.path.splitext(inspect.getfile(cls))[0]
        cls.log = logging.getLogger(cls.__name__)

        # Set logging to file and stdout if the LOG_LEVEL environment variable
        # is set.
        if os.getenv('LOG_LEVEL'):
            # Set up formatter.
            log_fmt = ('{}.%(funcName)s:%(levelname)s:%(asctime)s:'
                       ' %(message)s'.format(cls.__name__))
            formatter = logging.Formatter(log_fmt)

            # Set up the file handler.
            log_file_name = '%s.log' % cls.moduleName
            file_handler = logging.FileHandler(log_file_name)
            file_handler.setFormatter(formatter)
            cls.log.addHandler(file_handler)

            # Set the logging level from the environment variable, defaulting
            # to INFO if it is not a valid level.
            level = logging._nameToLevel.get(os.getenv('LOG_LEVEL'),
                                             logging.INFO)
            cls.log.setLevel(level)

    def tearDown(self):
        # Reset the default provider, as in practice it acts as a singleton
        # due to importing the wrapper from qiskit.
        from qiskit.wrapper import _wrapper
        _wrapper._DEFAULT_PROVIDER = DefaultQISKitProvider()

    @staticmethod
    def _get_resource_path(filename, path=Path.TEST):
        """ Get the absolute path to a resource.

        Args:
            filename (string): filename or relative path to the resource.
            path (Path): path used as relative to the filename.
        Returns:
            str: the absolute path to the resource.
        """
        return os.path.normpath(os.path.join(path.value, filename))

    def assertNoLogs(self, logger=None, level=None):
        """
        Context manager to test that no message is sent to the specified
        logger and level (the opposite of TestCase.assertLogs()).
        """
        # pylint: disable=invalid-name
        return _AssertNoLogsContext(self, logger, level)

    def assertDictAlmostEqual(self, dict1, dict2, delta=None, msg=None,
                              places=None, default_value=0):
        """
        Assert two dictionaries with numeric values are almost equal.

        Fail if the two dictionaries are unequal as determined by
        comparing that the difference between values with the same key are
        not greater than delta (default 1e-8), or that difference rounded
        to the given number of decimal places is not zero. If a key in one
        dictionary is not in the other the default_value keyword argument
        will be used for the missing value (default 0). If the two objects
        compare equal then they will automatically compare almost equal.

        Args:
            dict1 (dict): a dictionary.
            dict2 (dict): a dictionary.
            delta (number): threshold for comparison (defaults to 1e-8).
            msg (str): return a custom message on failure.
            places (int): number of decimal places for comparison.
            default_value (number): default value for missing keys.

        Raises:
            TypeError: raises TestCase failureException if the test fails.
        """
        # pylint: disable=invalid-name
        if dict1 == dict2:
            # Shortcut
            return
        if delta is not None and places is not None:
            raise TypeError("specify delta or places not both")

        if places is not None:
            success = True
            standard_msg = ''
            # check value for keys in target
            keys1 = set(dict1.keys())
            for key in keys1:
                val1 = dict1.get(key, default_value)
                val2 = dict2.get(key, default_value)
                if round(abs(val1 - val2), places) != 0:
                    success = False
                    standard_msg += '(%s: %s != %s), ' % (safe_repr(key),
                                                          safe_repr(val1),
                                                          safe_repr(val2))
            # check values for keys in counts, not in target
            keys2 = set(dict2.keys()) - keys1
            for key in keys2:
                val1 = dict1.get(key, default_value)
                val2 = dict2.get(key, default_value)
                if round(abs(val1 - val2), places) != 0:
                    success = False
                    standard_msg += '(%s: %s != %s), ' % (safe_repr(key),
                                                          safe_repr(val1),
                                                          safe_repr(val2))
            if success is True:
                return
            standard_msg = standard_msg[:-2] + ' within %s places' % places

        else:
            if delta is None:
                delta = 1e-8  # default delta value
            success = True
            standard_msg = ''
            # check value for keys in target
            keys1 = set(dict1.keys())
            for key in keys1:
                val1 = dict1.get(key, default_value)
                val2 = dict2.get(key, default_value)
                if abs(val1 - val2) > delta:
                    success = False
                    standard_msg += '(%s: %s != %s), ' % (safe_repr(key),
                                                          safe_repr(val1),
                                                          safe_repr(val2))
            # check values for keys in counts, not in target
            keys2 = set(dict2.keys()) - keys1
            for key in keys2:
                val1 = dict1.get(key, default_value)
                val2 = dict2.get(key, default_value)
                if abs(val1 - val2) > delta:
                    success = False
                    standard_msg += '(%s: %s != %s), ' % (safe_repr(key),
                                                          safe_repr(val1),
                                                          safe_repr(val2))
            if success is True:
                return
            standard_msg = standard_msg[:-2] + ' within %s delta' % delta

        msg = self._formatMessage(msg, standard_msg)
        raise self.failureException(msg)


class _AssertNoLogsContext(unittest.case._AssertLogsContext):
    """A context manager used to implement TestCase.assertNoLogs()."""

    # pylint: disable=inconsistent-return-statements
    def __exit__(self, exc_type, exc_value, tb):
        """
        This is a modified version of TestCase._AssertLogsContext.__exit__(...)
        """
        self.logger.handlers = self.old_handlers
        self.logger.propagate = self.old_propagate
        self.logger.setLevel(self.old_level)
        if exc_type is not None:
            # let unexpected exceptions pass through
            return False

        if self.watcher.records:
            msg = 'logs of level {} or higher triggered on {}:\n'.format(
                logging.getLevelName(self.level), self.logger.name)
            for record in self.watcher.records:
                msg += 'logger %s %s:%i: %s\n' % (record.name, record.pathname,
                                                  record.lineno,
                                                  record.getMessage())

            self._raiseFailure(msg)


def slow_test(func):
    """
    Decorator that signals that the test takes minutes to run.

    Args:
        func (callable): test function to be decorated.

    Returns:
        callable: the decorated function.
    """

    @functools.wraps(func)
    def _(*args, **kwargs):
        if SKIP_SLOW_TESTS:
            raise unittest.SkipTest('Skipping slow tests')
        return func(*args, **kwargs)

    return _


def requires_qe_access(func):
    """
    Decorator that signals that the test uses the online API:
        * determines if the test should be skipped by checking environment
            variables.
        * if the test is not skipped, it reads `QE_TOKEN` and `QE_URL` from
            `Qconfig.py`, environment variables or qiskitrc.
        * if the test is not skipped, it appends `QE_TOKEN` and `QE_URL` as
            arguments to the test function.
    Args:
        func (callable): test function to be decorated.

    Returns:
        callable: the decorated function.
    """
    func = VCR.use_cassette()(func)

    @functools.wraps(func)
    def _(*args, **kwargs):
        # Cleanup the credentials, as this file is shared by the tests.
        from qiskit.wrapper import _wrapper
        _wrapper._DEFAULT_PROVIDER = DefaultQISKitProvider()

        # Attempt to read the standard credentials.
        account_name = get_account_name(IBMQProvider)
        discovered_credentials = discover_credentials()
        if account_name in discovered_credentials.keys():
            credentials = discovered_credentials[account_name]
            if RECORD_TEST_RESPONSE:
                qe_token = credentials.get('token')
            else:
                qe_token = 'dummyapiusersloginWithTokenid01'
            kwargs.update({
                'QE_TOKEN': qe_token,
                'QE_URL': credentials.get('url'),
                'hub': credentials.get('hub'),
                'group': credentials.get('group'),
                'project': credentials.get('project'),
            })
        else:
            raise Exception('Could not locate valid credentials')

        return func(*args, **kwargs)

    return _


def _is_ci_fork_pull_request():
    """
    Check if the tests are being run in a CI environment and if it is a pull
    request.

    Returns:
        bool: True if the tests are executed inside a CI tool, and the changes
            are not against the "master" branch.
    """
    if os.getenv('TRAVIS'):
        # Using Travis CI.
        if os.getenv('TRAVIS_PULL_REQUEST_BRANCH'):
            return True
    elif os.getenv('APPVEYOR'):
        # Using AppVeyor CI.
        if os.getenv('APPVEYOR_PULL_REQUEST_NUMBER'):
            return True
    return False


SKIP_SLOW_TESTS = os.getenv('SKIP_SLOW_TESTS', True) not in ['false', 'False', '-1']
RECORD_TEST_RESPONSE = os.getenv('RECORD_TEST_RESPONSE', False) is not False
VCR_MODE = 'none'
if RECORD_TEST_RESPONSE:
    SKIP_SLOW_TESTS = True  # TODO Activate later
    VCR_MODE = 'all'


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


VCR = vcr(
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

VCR.register_persister(IdRemoverPersister)
