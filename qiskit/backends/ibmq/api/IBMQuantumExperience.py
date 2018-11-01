"""
    IBM Quantum Experience Python API Client
"""
try:
    import simplejson as json
except ImportError:
    import json
import time
import logging
from datetime import datetime
import sys
import traceback
import requests
import re
from requests_ntlm import HttpNtlmAuth
# from .HTTPProxyDigestAuth import HTTPProxyDigestAuth

log = logging.getLogger(__name__)
CLIENT_APPLICATION = 'qiskit-api-py'


def get_job_url(config, hub, group, project):
    """
    Util method to get job url
    """
    if ((config is not None) and ('hub' in config) and (hub is None)):
        hub = config["hub"]
    if ((config is not None) and ('group' in config) and (group is None)):
        group = config["group"]
    if ((config is not None) and ('project' in config) and (project is None)):
        project = config["project"]
    if ((hub is not None) and (group is not None) and (project is not None)):
        return '/Network/{}/Groups/{}/Projects/{}/jobs'.format(hub, group, project)
    return '/Jobs'

def get_backend_stats_url(config, hub, backend_type):
    """
    Util method to get backend stats url
    """
    if ((config is not None) and ('hub' in config) and (hub is None)):
        hub = config["hub"]
    if (hub is not None):
        return '/Network/{}/devices/{}'.format(hub, backend_type)
    return '/Backends/{}'.format(backend_type)


def get_backend_url(config, hub, group, project):
    """
    Util method to get backend url
    """
    if ((config is not None) and ('hub' in config) and (hub is None)):
        hub = config["hub"]
    if ((config is not None) and ('group' in config) and (group is None)):
        group = config["group"]
    if ((config is not None) and ('project' in config) and (project is None)):
        project = config["project"]
    if ((hub is not None) and (group is not None) and (project is not None)):
        return '/Network/{}/Groups/{}/Projects/{}/devices'.format(hub, group, project)
    return '/Backends'

class _Credentials(object):
    """
    The Credential class to manage the tokens
    """
    config_base = {'url': 'https://quantumexperience.ng.bluemix.net/api'}

    def __init__(self, token, config=None, verify=True, proxy_urls=None,
                 ntlm_credentials=None):
        self.token_unique = token
        self.verify = verify
        self.config = config
        self.proxy_urls = proxy_urls
        self.ntlm_credentials = ntlm_credentials

        # Set the extra arguments to requests (proxy and auth).
        self.extra_args = {}
        if self.proxy_urls:
            self.extra_args['proxies'] = self.proxy_urls
        if self.ntlm_credentials:
            self.extra_args['auth'] = HttpNtlmAuth(
                self.ntlm_credentials['username'],
                self.ntlm_credentials['password'])

        if not verify:
            import requests.packages.urllib3 as urllib3
            urllib3.disable_warnings()
            print('-- Ignoring SSL errors.  This is not recommended --')
        if self.config and ("url" not in self.config):
            self.config["url"] = self.config_base["url"]
        elif not self.config:
            self.config = self.config_base

        self.data_credentials = {}
        if token:
            self.obtain_token(config=self.config)
        else:
            access_token = self.config.get('access_token', None)
            if access_token:
                user_id = self.config.get('user_id', None)
                if access_token:
                    self.set_token(access_token)
                if user_id:
                    self.set_user_id(user_id)
            else:
                self.obtain_token(config=self.config)

    def obtain_token(self, config=None):
        """Obtain the token to access to QX Platform.

        Raises:
            CredentialsError: when token is invalid or the user has not
                accepted the license.
            ApiError: when the response from the server couldn't be parsed.
        """
        client_application = CLIENT_APPLICATION
        if self.config and ("client_application" in self.config):
            client_application += ':' + self.config["client_application"]
        headers = {'x-qx-client-application': client_application}

        if self.token_unique:
            try:
                response = requests.post(str(self.config.get('url') +
                                             "/users/loginWithToken"),
                                         data={'apiToken': self.token_unique},
                                         verify=self.verify,
                                         headers=headers,
                                         **self.extra_args)
            except requests.RequestException as e:
                raise ApiError('error during login: %s' % str(e))
        elif config and ("email" in config) and ("password" in config):
            email = config.get('email', None)
            password = config.get('password', None)
            credentials = {
                'email': email,
                'password': password
            }
            try:
                response = requests.post(str(self.config.get('url') +
                                             "/users/login"),
                                         data=credentials,
                                         verify=self.verify,
                                         headers=headers,
                                         **self.extra_args)
            except requests.RequestException as e:
                raise ApiError('error during login: %s' % str(e))
        else:
            raise CredentialsError('invalid token')

        if response.status_code == 401:
            error_message = None
            try:
                # For 401: ACCEPT_LICENSE_REQUIRED, a detailed message is
                # present in the response and passed to the exception.
                error_message = response.json()['error']['message']
            except:
                pass

            if error_message:
                raise CredentialsError('error during login: %s' % error_message)
            else:
                raise CredentialsError('invalid token')
        try:
            response.raise_for_status()
            self.data_credentials = response.json()
        except (requests.HTTPError, ValueError) as e:
            raise ApiError('error during login: %s' % str(e))

        if self.get_token() is None:
            raise CredentialsError('invalid token')

    def get_token(self):
        """
        Get Authenticated Token to connect with QX Platform
        """
        return self.data_credentials.get('id', None)

    def get_user_id(self):
        """
        Get User Id in QX Platform
        """
        return self.data_credentials.get('userId', None)

    def get_config(self):
        """
        Get Configuration setted to connect with QX Platform
        """
        return self.config

    def set_token(self, access_token):
        """
        Set Access Token to connect with QX Platform API
        """
        self.data_credentials['id'] = access_token

    def set_user_id(self, user_id):
        """
        Set Access Token to connect with QX Platform API
        """
        self.data_credentials['userId'] = user_id


class _Request(object):
    """
    The Request class to manage the methods
    """
    def __init__(self, token, config=None, verify=True, retries=5,
                 timeout_interval=1.0):
        self.verify = verify
        self.client_application = CLIENT_APPLICATION
        self.config = config
        self.errorsNotRetry = [401, 403, 413]

        # Set the proxy information, if present, from the configuration,
        # with the following format:
        # config = {
        #     'proxies': {
        #         # If using 'urls', assume basic auth or no auth.
        #         'urls': {
        #             'http': 'http://user:password@1.2.3.4:5678',
        #             'https': 'http://user:password@1.2.3.4:5678',
        #         }
        #         # If using 'ntlm', assume NTLM authentication.
        #         'username_ntlm': 'domain\\username',
        #         'password_ntlm': 'password'
        #     }
        # }

        # Set the basic proxy settings, if present.
        self.proxy_urls = None
        self.ntlm_credentials = None
        if config and 'proxies' in config:
            if 'urls' in config['proxies']:
                self.proxy_urls = self.config['proxies']['urls']
            if 'username_ntlm' and 'password_ntlm' in config['proxies']:
                self.ntlm_credentials = {
                    'username': self.config['proxies']['username_ntlm'],
                    'password': self.config['proxies']['password_ntlm']
                }

        # Set the extra arguments to requests (proxy and auth).
        self.extra_args = {}
        if self.proxy_urls:
            self.extra_args['proxies'] = self.proxy_urls
        if self.ntlm_credentials:
            self.extra_args['auth'] = HttpNtlmAuth(
                self.ntlm_credentials['username'],
                self.ntlm_credentials['password'])

        if self.config and ("client_application" in self.config):
            self.client_application += ':' + self.config["client_application"]
        self.credential = _Credentials(token, self.config, verify,
                                       proxy_urls=self.proxy_urls,
                                       ntlm_credentials=self.ntlm_credentials)

        if not isinstance(retries, int):
            raise TypeError('post retries must be positive integer')
        self.retries = retries
        self.timeout_interval = timeout_interval
        self.result = None
        self._max_qubit_error_re = re.compile(
            r".*registers exceed the number of qubits, "
            r"it can\'t be greater than (\d+).*")

    def check_token(self, respond):
        """
        Check is the user's token is valid
        """
        if respond.status_code == 401:
            self.credential.obtain_token(config=self.config)
            return False
        return True

    def post(self, path, params='', data=None):
        """
        POST Method Wrapper of the REST API
        """
        self.result = None
        data = data or {}
        headers = {'Content-Type': 'application/json',
                   'x-qx-client-application': self.client_application}
        url = str(self.credential.config['url'] + path + '?access_token=' +
                  self.credential.get_token() + params)
        retries = self.retries
        while retries > 0:
            respond = requests.post(url, data=data, headers=headers,
                                    verify=self.verify, **self.extra_args)
            if not self.check_token(respond):
                respond = requests.post(url, data=data, headers=headers,
                                        verify=self.verify,
                                        **self.extra_args)

            if self._response_good(respond):
                if self.result:
                    return self.result
                elif retries < 2:
                    return respond.json()
                else:
                    retries -= 1
            else:
                retries -= 1
                time.sleep(self.timeout_interval)

        # timed out
        raise ApiError(usr_msg='Failed to get proper ' +
                       'response from backend.')

    def put(self, path, params='', data=None):
        """
        PUT Method Wrapper of the REST API
        """
        self.result = None
        data = data or {}
        headers = {'Content-Type': 'application/json',
                   'x-qx-client-application': self.client_application}
        url = str(self.credential.config['url'] + path + '?access_token=' +
                  self.credential.get_token() + params)
        retries = self.retries
        while retries > 0:
            respond = requests.put(url, data=data, headers=headers,
                                   verify=self.verify, **self.extra_args)
            if not self.check_token(respond):
                respond = requests.put(url, data=data, headers=headers,
                                       verify=self.verify,
                                       **self.extra_args)
            if self._response_good(respond):
                if self.result:
                    return self.result
                elif retries < 2:
                    return respond.json()
                else:
                    retries -= 1
            else:
                retries -= 1
                time.sleep(self.timeout_interval)
        # timed out
        raise ApiError(usr_msg='Failed to get proper ' +
                       'response from backend.')

    def get(self, path, params='', with_token=True):
        """
        GET Method Wrapper of the REST API
        """
        self.result = None
        access_token = ''
        if with_token:
            access_token = self.credential.get_token() or ''
            if access_token:
                access_token = '?access_token=' + str(access_token)
        url = self.credential.config['url'] + path + access_token + params
        retries = self.retries
        headers = {'x-qx-client-application': self.client_application}
        while retries > 0:  # Repeat until no error
            respond = requests.get(url, verify=self.verify, headers=headers,
                                   **self.extra_args)
            if not self.check_token(respond):
                respond = requests.get(url, verify=self.verify,
                                       headers=headers, **self.extra_args)
            if self._response_good(respond):
                if self.result:
                    return self.result
                elif retries < 2:
                    return respond.json()
                else:
                    retries -= 1
            else:
                retries -= 1
                time.sleep(self.timeout_interval)
        # timed out
        raise ApiError(usr_msg='Failed to get proper ' +
                       'response from backend.')

    def _response_good(self, respond):
        """check response

        Args:
            respond (str): HTTP response.

        Returns:
            bool: True if the response is good, else False.

        Raises:
            ApiError: response isn't formatted properly.
        """
        if respond.status_code != requests.codes.ok:
            log.warning('Got a {} code response to {}: {}'.format(
                respond.status_code,
                respond.url,
                respond.text))
            if respond.status_code in self.errorsNotRetry:
              raise ApiError(usr_msg='Got a {} code response to {}: {}'.format(
                respond.status_code,
                respond.url,
                respond.text))
            else:
              return self._parse_response(respond)
        try:
            if (str(respond.headers['content-type']).startswith("text/html;")):
                self.result = respond.text
                return True
            else:
                self.result = respond.json()
        except (json.JSONDecodeError, ValueError):
            usr_msg = 'device server returned unexpected http response'
            dev_msg = usr_msg + ': ' + respond.text
            raise ApiError(usr_msg=usr_msg, dev_msg=dev_msg)
        if not isinstance(self.result, (list, dict)):
            msg = ('JSON not a list or dict: url: {0},'
                   'status: {1}, reason: {2}, text: {3}')
            raise ApiError(
                usr_msg=msg.format(respond.url,
                                   respond.status_code,
                                   respond.reason, respond.text))
        if ('error' not in self.result or
                ('status' not in self.result['error'] or
                 self.result['error']['status'] != 400)):
            return True
        else:
            log.warning("Got a 400 code JSON response to %s", respond.url)
            return False

    def _parse_response(self, respond):
        """parse text of response for HTTP errors

        This parses the text of the response to decide whether to
        retry request or raise exception. At the moment this only
        detects an exception condition.

        Args:
            respond (Response): requests.Response object

        Returns:
            bool: False if the request should be retried, True
                if not.

        Raises:
            RegisterSizeError
        """
        # convert error messages into exceptions
        mobj = self._max_qubit_error_re.match(respond.text)
        if mobj:
            raise RegisterSizeError(
                'device register size must be <= {}'.format(mobj.group(1)))
        return True


class IBMQuantumExperience(object):
    """
    The Connector Class to do request to QX Platform
    """
    __names_backend_ibmqxv2 = ['ibmqx5qv2', 'ibmqx2', 'qx5qv2', 'qx5q', 'real']
    __names_backend_ibmqxv3 = ['ibmqx3']
    __names_backend_simulator = ['simulator', 'sim_trivial_2',
                                 'ibmqx_qasm_simulator', 'ibmq_qasm_simulator']

    def __init__(self, token=None, config=None, verify=True):
        """ If verify is set to false, ignore SSL certificate errors """
        self.config = config

        if self.config and ('url' in self.config):
          url_parsed = self.config['url'].split('/api')
          if len(url_parsed) == 2:
            hub = group = project = None
            project_parse = url_parsed[1].split('/Projects/')
            if len(project_parse) == 2:
              project = project_parse[1]
              group_parse = project_parse[0].split('/Groups/')
              if len(group_parse) == 2:
                group = group_parse[1]
                hub_parse = group_parse[0].split('/Hubs/')
                if len(hub_parse) == 2:
                  hub = hub_parse[1]
            if (hub and group and project):
              self.config['project'] = project
              self.config['group'] = group
              self.config['hub'] = hub
              self.config['url'] = url_parsed[0] + '/api'

        self.req = _Request(token, config=config, verify=verify)

    def _check_backend(self, backend, endpoint):
        """
        Check if the name of a backend is valid to run in QX Platform
        """
        # First check against hacks for old backend names
        original_backend = backend
        backend = backend.lower()
        if endpoint == 'experiment':
            if backend in self.__names_backend_ibmqxv2:
                return 'real'
            elif backend in self.__names_backend_ibmqxv3:
                return 'ibmqx3'
            elif backend in self.__names_backend_simulator:
                return 'sim_trivial_2'

        # Check for new-style backends
        backends = self.available_backends()
        for backend in backends:
            if backend['name'] == original_backend:
              return original_backend
        # backend unrecognized
        return None

    def check_credentials(self):
        """
        Check if the user has permission in QX platform
        """
        return bool(self.req.credential.get_token())

    def get_execution(self, id_execution, access_token=None, user_id=None):
        """
        Get a execution, by its id
        """
        if access_token:
            self.req.credential.set_token(access_token)
        if user_id:
            self.req.credential.set_user_id(user_id)
        if not self.check_credentials():
            raise CredentialsError('credentials invalid')
        execution = self.req.get('/Executions/' + id_execution)
        if "codeId" in execution:
            execution['code'] = self.get_code(execution["codeId"])
        return execution

    def get_result_from_execution(self, id_execution, access_token=None, user_id=None):
        """
        Get the result of a execution, by the execution id
        """
        if access_token:
            self.req.credential.set_token(access_token)
        if user_id:
            self.req.credential.set_user_id(user_id)
        if not self.check_credentials():
            raise CredentialsError('credentials invalid')
        execution = self.req.get('/Executions/' + id_execution)
        result = {}
        if "result" in execution and "data" in execution["result"]:
            if execution["result"]["data"].get('p', None):
                result["measure"] = execution["result"]["data"]["p"]
            if execution["result"]["data"].get('valsxyz', None):
                result["bloch"] = execution["result"]["data"]["valsxyz"]
            if "additionalData" in execution["result"]["data"]:
                ad_aux = execution["result"]["data"]["additionalData"]
                result["extraInfo"] = ad_aux
            if "calibration" in execution:
                result["calibration"] = execution["calibration"]
            if execution["result"]["data"].get('cregLabels', None):
                result["creg_labels"] = execution["result"]["data"]["cregLabels"]
            if execution["result"]["data"].get('time', None):
                result["time_taken"] = execution["result"]["data"]["time"]

        return result

    def get_code(self, id_code, access_token=None, user_id=None):
        """
        Get a code, by its id
        """
        if access_token:
            self.req.credential.set_token(access_token)
        if user_id:
            self.req.credential.set_user_id(user_id)
        if not self.check_credentials():
            raise CredentialsError('credentials invalid')
        code = self.req.get('/Codes/' + id_code)
        executions = self.req.get('/Codes/' + id_code + '/executions',
                                  '&filter={"limit":3}')
        if isinstance(executions, list):
            code["executions"] = executions
        return code

    def get_image_code(self, id_code, access_token=None, user_id=None):
        """
        Get the image of a code, by its id
        """
        if access_token:
            self.req.credential.set_token(access_token)
        if user_id:
            self.req.credential.set_user_id(user_id)
        if not self.check_credentials():
            raise CredentialsError('credentials invalid')
        return self.req.get('/Codes/' + id_code + '/export/png/url')

    def get_last_codes(self, access_token=None, user_id=None):
        """
        Get the last codes of the user
        """
        if access_token:
            self.req.credential.set_token(access_token)
        if user_id:
            self.req.credential.set_user_id(user_id)
        if not self.check_credentials():
            raise CredentialsError('credentials invalid')
        last = '/users/' + self.req.credential.get_user_id() + '/codes/lastest'
        return self.req.get(last, '&includeExecutions=true')['codes']

    def run_experiment(self, qasm, backend='simulator', shots=1, name=None,
                       seed=None, timeout=60, access_token=None, user_id=None):
        """
        Execute an experiment
        """
        if access_token:
            self.req.credential.set_token(access_token)
        if user_id:
            self.req.credential.set_user_id(user_id)
        if not self.check_credentials():
            raise CredentialsError('credentials invalid')

        backend_type = self._check_backend(backend, 'experiment')
        if not backend_type:
            raise BadBackendError(backend)

        if backend not in self.__names_backend_simulator and seed:
            raise ApiError('seed not allowed for'
                           ' non-simulator backend "{}"'.format(backend))

        name = name or 'Experiment #{:%Y%m%d%H%M%S}'.format(datetime.now())
        qasm = qasm.replace('IBMQASM 2.0;', '').replace('OPENQASM 2.0;', '')
        data = json.dumps({'qasm': qasm, 'codeType': 'QASM2', 'name': name})

        if seed and len(str(seed)) < 11 and str(seed).isdigit():
            params = '&shots={}&seed={}&deviceRunType={}'.format(shots, seed,
                                                                 backend_type)
            execution = self.req.post('/codes/execute', params, data)
        elif seed:
            raise ApiError('invalid seed ({}), seeds can have'
                           ' a maximum length of 10 digits'.format(seed))
        else:
            params = '&shots={}&deviceRunType={}'.format(shots, backend_type)
            execution = self.req.post('/codes/execute', params, data)
        respond = {}
        try:
            status = execution["status"]["id"]
            id_execution = execution["id"]
            result = {}
            respond["status"] = status
            respond["idExecution"] = id_execution
            respond["idCode"] = execution["codeId"]

            if 'infoQueue' in execution:
                respond['infoQueue'] = execution['infoQueue']

            if status == "DONE":
                if "result" in execution and "data" in execution["result"]:
                    if "additionalData" in execution["result"]["data"]:
                        ad_aux = execution["result"]["data"]["additionalData"]
                        result["extraInfo"] = ad_aux
                    if execution["result"]["data"].get('p', None):
                        result["measure"] = execution["result"]["data"]["p"]
                    if execution["result"]["data"].get('valsxyz', None):
                        valsxyz = execution["result"]["data"]["valsxyz"]
                        result["bloch"] = valsxyz
                    respond["result"] = result
                    respond.pop('infoQueue', None)

                    return respond
            elif status == "ERROR":
                respond.pop('infoQueue', None)
                return respond
            else:
                if timeout:
                    for _ in range(1, timeout):
                        print("Waiting for results...")
                        result = self.get_result_from_execution(id_execution)
                        if result:
                            respond["status"] = 'DONE'
                            respond["result"] = result
                            respond["calibration"] = result["calibration"]
                            del result["calibration"]
                            respond.pop('infoQueue', None)
                            return respond
                        else:
                            time.sleep(2)
                    return respond
                else:
                    return respond
        except Exception:
            respond["error"] = execution
            return respond

    def run_job(self, job, backend='simulator', shots=1,
                max_credits=None, seed=None, hub=None, group=None,
                project=None, hpc=None, access_token=None, user_id=None):
        """
        Execute a job
        """
        if access_token:
            self.req.credential.set_token(access_token)
        if user_id:
            self.req.credential.set_user_id(user_id)
        if not self.check_credentials():
            return {"error": "Not credentials valid"}

        backend_type = self._check_backend(backend, 'job')

        if not backend_type:
            raise BadBackendError(backend)

        if isinstance(job, (list, tuple)):
          qasms = job
          for qasm in qasms:
              qasm['qasm'] = qasm['qasm'].replace('IBMQASM 2.0;', '')
              qasm['qasm'] = qasm['qasm'].replace('OPENQASM 2.0;', '')

          data = {'qasms': qasms,
                  'shots': shots,
                  'backend': {}}

          if max_credits:
            data['maxCredits'] = max_credits

          if seed and len(str(seed)) < 11 and str(seed).isdigit():
              data['seed'] = seed
          elif seed:
              return {"error": "Not seed allowed. Max 10 digits."}

          data['backend']['name'] = backend_type
        elif isinstance(job, dict):
          q_obj = job
          data = {'qObject': q_obj,
                  'backend': {}}

          data['backend']['name'] = backend_type
        else:
          return {"error": "Not a valid data to send"}

        if hpc:
          data['hpc'] = hpc

        url = get_job_url(self.config, hub, group, project)
        
        job = self.req.post(url, data=json.dumps(data))

        return job

    def get_job(self, id_job, hub=None, group=None, project=None,
                access_token=None, user_id=None):
        """
        Get the information about a job, by its id
        """
        if access_token:
            self.req.credential.set_token(access_token)
        if user_id:
            self.req.credential.set_user_id(user_id)
        if not self.check_credentials():
            respond = {}
            respond["status"] = 'Error'
            respond["error"] = "Not credentials valid"
            return respond
        if not id_job:
            respond = {}
            respond["status"] = 'Error'
            respond["error"] = "Job ID not specified"
            return respond

        url = get_job_url(self.config, hub, group, project)

        url += '/' + id_job

        job = self.req.get(url)

        if 'qasms' in job: 
            for qasm in job['qasms']:
                if ('result' in qasm) and ('data' in qasm['result']):
                    qasm['data'] = qasm['result']['data']
                    del qasm['result']['data']
                    for key in qasm['result']:
                        qasm['data'][key] = qasm['result'][key]
                    del qasm['result']

        return job

    def get_jobs(self, limit=10, skip=0, backend=None, only_completed=False, filter=None, hub=None, group=None, project=None, access_token=None, user_id=None):
        """
        Get the information about the user jobs
        """
        if access_token:
            self.req.credential.set_token(access_token)
        if user_id:
            self.req.credential.set_user_id(user_id)
        if not self.check_credentials():
            return {"error": "Not credentials valid"}

        url = get_job_url(self.config, hub, group, project)
        url_filter = '&filter='
        query = {
          "order": "creationDate DESC",
          "limit": limit,
          "skip": skip,
          "where" : {}
        }
        if filter is not None:
          query['where'] = filter
        else:
          if backend is not None:
            query['where']['backend.name'] = backend
          if only_completed:
            query['where']['status'] = 'COMPLETED'
  
        url_filter = url_filter + json.dumps(query)
        jobs = self.req.get(url, url_filter)
        return jobs

    def get_status_job(self, id_job, hub=None, group=None, project=None,
                       access_token=None, user_id=None):
        """
        Get the status about a job, by its id
        """
        if access_token:
            self.req.credential.set_token(access_token)
        if user_id:
            self.req.credential.set_user_id(user_id)
        if not self.check_credentials():
            respond = {}
            respond["status"] = 'Error'
            respond["error"] = "Not credentials valid"
            return respond
        if not id_job:
            respond = {}
            respond["status"] = 'Error'
            respond["error"] = "Job ID not specified"
            return respond

        url = get_job_url(self.config, hub, group, project)

        url += '/' + id_job + '/status'

        status = self.req.get(url)

        return status

    def get_status_jobs(self, limit=10, skip=0, backend=None, filter=None, hub=None, group=None, project=None, access_token=None, user_id=None):
        """
        Get the information about the user jobs
        """
        if access_token:
            self.req.credential.set_token(access_token)
        if user_id:
            self.req.credential.set_user_id(user_id)
        if not self.check_credentials():
            return {"error": "Not credentials valid"}

        url = get_job_url(self.config, hub, group, project)
        url_filter = '&filter='
        query = {
          "order": "creationDate DESC",
          "limit": limit,
          "skip": skip,
          "where" : {}
        }
        if filter is not None:
          query['where'] = filter
        else:
          if backend is not None:
            query['where']['backend.name'] = backend
  
        url += '/status'

        url_filter = url_filter + json.dumps(query)
        
        jobs = self.req.get(url, url_filter)

        return jobs

    def cancel_job(self, id_job, hub=None, group=None, project=None,
                   access_token=None, user_id=None):
        """
        Cancel the information about a job, by its id
        """
        if access_token:
            self.req.credential.set_token(access_token)
        if user_id:
            self.req.credential.set_user_id(user_id)
        if not self.check_credentials():
            respond = {}
            respond["status"] = 'Error'
            respond["error"] = "Not credentials valid"
            return respond
        if not id_job:
            respond = {}
            respond["status"] = 'Error'
            respond["error"] = "Job ID not specified"
            return respond
        
        url = get_job_url(self.config, hub, group, project)

        url += '/{}/cancel'.format(id_job)

        res = self.req.post(url)

        return res

    def backend_status(self, backend='ibmqx4', access_token=None, user_id=None):
        """
        Get the status of a chip
        """
        if access_token:
            self.req.credential.set_token(access_token)
        if user_id:
            self.req.credential.set_user_id(user_id)
        backend_type = self._check_backend(backend, 'status')
        if not backend_type:
            raise BadBackendError(backend)

        status = self.req.get('/Backends/' + backend_type + '/queue/status',
                              with_token=False)

        ret = {}
        if 'state' in status:
            ret['available'] = bool(status['state'])
        if 'busy' in status:
            ret['busy'] = bool(status['busy'])
        if 'lengthQueue' in status:
            ret['pending_jobs'] = status['lengthQueue']
        
        ret['backend'] = backend_type

        return ret

    def backend_calibration(self, backend='ibmqx4', hub=None, access_token=None, user_id=None):
        """
        Get the calibration of a real chip
        """
        if access_token:
            self.req.credential.set_token(access_token)
        if user_id:
            self.req.credential.set_user_id(user_id)
        if not self.check_credentials():
            raise CredentialsError('credentials invalid')

        backend_type = self._check_backend(backend, 'calibration')

        if not backend_type:
            raise BadBackendError(backend)

        if backend_type in self.__names_backend_simulator:
            ret = {}
            return ret

        url = get_backend_stats_url(self.config, hub, backend_type)

        ret = self.req.get(url + '/calibration')
        if not bool(ret):
          ret = {}
        else:
          ret["backend"] = backend_type
        return ret

    def backend_parameters(self, backend='ibmqx4', hub=None, access_token=None, user_id=None):
        """
        Get the parameters of calibration of a real chip
        """
        if access_token:
            self.req.credential.set_token(access_token)
        if user_id:
            self.req.credential.set_user_id(user_id)
        if not self.check_credentials():
            raise CredentialsError('credentials invalid')

        backend_type = self._check_backend(backend, 'calibration')

        if not backend_type:
            raise BadBackendError(backend)

        if backend_type in self.__names_backend_simulator:
            ret = {}
            return ret

        url = get_backend_stats_url(self.config, hub, backend_type)

        ret = self.req.get(url + '/parameters')
        if not bool(ret):
          ret = {}
        else:
          ret["backend"] = backend_type
        return ret

    def available_backends(self, hub=None, group=None, project=None, access_token=None, user_id=None):
        """
        Get the backends available to use in the QX Platform
        """
        if access_token:
            self.req.credential.set_token(access_token)
        if user_id:
            self.req.credential.set_user_id(user_id)
        if not self.check_credentials():
            raise CredentialsError('credentials invalid')
        else:

            url = get_backend_url(self.config, hub, group, project)

            ret = self.req.get(url)
            if (ret is not None) and (isinstance(ret, dict)):
                return []
            return [backend for backend in ret
                    if backend.get('status') == 'on']

    def available_backend_simulators(self, access_token=None, user_id=None):
        """
        Get the backend simulators available to use in the QX Platform
        """
        if access_token:
            self.req.credential.set_token(access_token)
        if user_id:
            self.req.credential.set_user_id(user_id)
        if not self.check_credentials():
            raise CredentialsError('credentials invalid')
        else:
            ret = self.req.get('/Backends')
            if (ret is not None) and (isinstance(ret, dict)):
                return []
            return [backend for backend in ret
                    if backend.get('status') == 'on' and
                    backend.get('simulator') is True]

    def get_my_credits(self, access_token=None, user_id=None):
        """
        Get the credits by user to use in the QX Platform
        """
        if access_token:
            self.req.credential.set_token(access_token)
        if user_id:
            self.req.credential.set_user_id(user_id)
        if not self.check_credentials():
            raise CredentialsError('credentials invalid')
        else:
            user_data_url = '/users/' + self.req.credential.get_user_id()
            user_data = self.req.get(user_data_url)
            if "credit" in user_data:
                if "promotionalCodesUsed" in user_data["credit"]:
                    del user_data["credit"]["promotionalCodesUsed"]
                if "lastRefill" in user_data["credit"]:
                    del user_data["credit"]["lastRefill"]
                return user_data["credit"]
            return {}

    def api_version(self):
        """
        Get the API Version of the QX Platform
        """
        return self.req.get('/version')


class ApiError(Exception):
    """
    IBMQuantumExperience API error handling base class.
    """
    def __init__(self, usr_msg=None, dev_msg=None):
        """
        Args:
            usr_msg (str): Short user facing message describing error.
            dev_msg (str or None, optional): More detailed message to assist
                developer with resolving issue.
        """
        Exception.__init__(self, usr_msg)
        self.usr_msg = usr_msg
        self.dev_msg = dev_msg

    def __repr__(self):
        return repr(self.dev_msg)

    def __str__(self):
        return str(self.usr_msg)


class BadBackendError(ApiError):
    """
    Unavailable backend error.
    """
    def __init__(self, backend):
        """
        Parameters
        ----------
        backend : str
           Name of backend.
        """
        usr_msg = ('Could not find backend "{0}" available.').format(backend)
        dev_msg = ('Backend "{0}" does not exist. Please use '
                   'available_backends to see options').format(backend)
        ApiError.__init__(self, usr_msg=usr_msg,
                          dev_msg=dev_msg)


class CredentialsError(ApiError):
    """Exception associated with bad server credentials."""
    pass


class RegisterSizeError(ApiError):
    """Exception due to exceeding the maximum number of allowed qubits."""
    pass
