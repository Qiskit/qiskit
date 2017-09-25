import subprocess
from subprocess import Popen, PIPE, CalledProcessError

from qiskit.backends._basebackend import BaseBackend

class QeLocal(BaseBackend):

    def __init__(self, qobj):
        self._qobj = qobj
        # TODO should get this from json file for backend
        self._configuration = {
            'basis_gates': 'u1,u2,u3,cx,id',
            'chip_name': 'Sparrow',
            'coupling_map': {0: [1, 2], 1: [2], 3: [2, 4], 4: [2]},
            'description': '5 transmon bowtie',
            'n_qubits': 5,
            'name': 'ibmqx2',
            'online_date': '2017-01-10T12:00:00.000Z',
            'simulator': False,
            'local': True,
            'url': 'https://ibm.biz/qiskit-ibmqx2',
            'version': '1'
        }
        if 'config' in qobj:
            self._qobj_config = qobj['config']
        else:
            self._qobj_config = {}
        if 'timeout' in self._qobj_config:
            self._timeout = self._qobj_config['timeout']
        else:
            self._timeout = 300
        if 'exe' in self._qobj_config:
            self._exe = self._qobj_config['exe']
        else:
            self._exe = 'run_sqore'
        try:
            subprocess.check_call([self._exe, '--help'], stderr=subprocess.STDOUT)
        except CalledProcessError as cperr:
            pass
        except FileNotFoundError:
            raise

    def run(self):
        command = [self._exe, '--help']
        pipe_cl = Popen(command, stdout=PIPE, stderr=PIPE, shell=True, preexec_fn=os.setsid)
        try:
            output, err = pipe_cl.communicate(timeout=timeout)
            return output,err
        except TimeoutExpired:
            return (None, '[QX: Timeout %d seconds spent.]' % (timeout))
        except Exception as e:  # pylint: disable bare-exception
            return (None, 'An exception occurred')
        finally:
            try:
                os.killpg(os.getpgid(pipe_cl.pid), signal.SIGKILL)
            except:
                pass

    @property
    def configuration(self):
        return self._configuration

    @configuration.setter
    def configuration(self, configuration):
        self._configuration = configuration
