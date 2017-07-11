from ._qasmsimulator import QasmSimulator
from ._unitarysimulator import UnitarySimulator
from ._qasm_cpp_simulator import QasmCppSimulator

class LocalSimulator:
    def __init__(self, backend, job):
        self._backend = backend
        self._job = job
        self._result = {'result': None, 'status': "Error"}

    def run(self):
        if self._backend is 'local_qasm_simulator':
            sim = QasmSimulator(self._job["compiled_circuit"],
                                self._job["shots"],
                                self._job["seed"])
        elif self._backend is 'local_unitary_simulator':
            sim = UnitarySimulator(self._job["compiled_circuit"])
        elif self._backend is 'local_qasm_cpp_simulator':
            sim = QasmCppSimulator(self._job["compiled_circuit"],
                                   self._job["shots"],
                                   self._job["seed"])
        else:
            raise ValueError('Unrecognized backend: {0}'.format(self._backend))
        simOutput = sim.run()
        self._result["result"] = {}
        self._result["result"]["data"] = simOutput["data"]
        self._result["status"] = simOutput["status"]

    def result(self):
        return self._result
