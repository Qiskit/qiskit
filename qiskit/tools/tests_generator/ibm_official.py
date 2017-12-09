import sys
import time

if sys.version_info.major > 2:  # Python 3
    from IBMQuantumExperience.IBMQuantumExperience import IBMQuantumExperience
else:                           # Python 2
    from IBMQuantumExperience import IBMQuantumExperience

ibmqx_token = "aec4a650bb3ab88325ab6a662871202e0d582fbd8e3bbf71e476963b2c1931012654756ff1398495968663438708011dfdf2dd7f85b1499380fc36b7df283fe1"
ibmqx_config = {"url": 'https://quantumexperience.ng.bluemix.net/api'}


# this way of getting img from ibmqx requires at least one execution: slow and weird, do not use; self generate instead
def get_code_image(qasm_code, ibmqx_shots=128, backend="ibmqx2"):
    global ibmqx_token, ibmqx_config

    api = IBMQuantumExperience(ibmqx_token, ibmqx_config)
    resp = api.run_experiment(qasm_code, shots=ibmqx_shots, device=backend, timeout=300)
    return api.get_image_code(resp['idCode'])['url']

#Only two option of devices are possibles: simulator or ibmqx2
# honestly, I am not interested in simulator at all...
def get_execution_result(qasm_code, ibmqx_shots=128, backend="ibmqx2"):
    global ibmqx_token, ibmqx_config
    api = IBMQuantumExperience(ibmqx_token, ibmqx_config)
    resp = api.run_experiment(qasm_code, shots=ibmqx_shots, device=backend, timeout=300) # max timeout=300
    return api.get_result_from_execution(resp['idExecution'])['measure']


def get_image_and_result(qasm_code, ibmqx_shots=128, backend="ibmqx2"):
    global ibmqx_token, ibmqx_config
    try:
        api = IBMQuantumExperience(ibmqx_token, ibmqx_config)
        resp = api.run_experiment(qasm_code, shots=ibmqx_shots, device=backend, timeout=300) # max timeout=300
        return api.get_image_code(resp['idCode'])['url'], api.get_result_from_execution(resp['idExecution'])['measure']
    except:
        try:
            time.sleep(5)
            api = IBMQuantumExperience(ibmqx_token, ibmqx_config)
            resp = api.run_experiment(qasm_code, shots=ibmqx_shots, device=backend, timeout=300) # max timeout=300
            return api.get_image_code(resp['idCode'])['url'], api.get_result_from_execution(resp['idExecution'])['measure']
        except:
            print "Unexpected error:", sys.exc_info()[0]
            return None, None


