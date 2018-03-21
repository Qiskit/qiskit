#!/usr/bin/python3

'''
This test generates the files test_rcg_seed_1.json and test_rcg_seed_1.ref
'''

import importlib
tst = importlib.import_module('src.qiskit-simulator.test.random.system.test_rcg', None)
import json
import subprocess as sp
serv = importlib.import_module('src.qiskit-simulator.test.service_functions_for_tests', None)

input_file = open('test_rcg_seed_1.json', 'w+')
ref_file = open('test_rcg_seed_1.ref', 'w+')

qobj = tst.create_qobj(seed = 1)
json.dump(qobj, input_file, indent = 4)

# The following commented lines are not good:
# The reference file will be different in each execution,
# because the Python dictionary order is not preserved.
#result = tst.run_simulator(qobj)
#json.dump(result, ref_file, indent = 4)

proc = sp.Popen([serv.default_executable(), '-'],
                stdin = sp.PIPE,
                stdout = ref_file,
                stderr = sp.PIPE)
cin = json.dumps(qobj).encode()
cout = proc.communicate(cin)




