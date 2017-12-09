# this python script is needed by the js website, we assume this python always runs in the same machine as the js website.
# since they run on the same machine, we use the external file system to share images, jsons, or other large data.
# this python script needs to know where the js website folder is since it will copy data there (same origin constraint)
import os

import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')


from flask import Flask
from flask_socketio import SocketIO, emit

import generate_svg_circuit
from circuit_builder import buildAST, buildCircuit
from equivalence_checker import check_equivalent_between_two_results
from extensible_gate_domain import _basic_gates_string_IBM_advanced
from math_executor import math_execute
from qiskit.qanalyzer.qchecker import makeArgs

from qfuzzer import makeFuzzArgs, qfuzzmain



app = Flask(__name__)
socketio = SocketIO(app, async_handlers=True)

@socketio.on('message')
def handle_message(message):
    print(message)


@socketio.on('invoke-debugger')
def invoke_debugger(qasm):
    print('received qasm code to visualize: ' + qasm)
    try:
        with open("./tmp/tmp.qasm", 'w') as qasmF:
            qasmF.write(qasm)
            qasmpath = os.path.realpath(qasmF.name)

        # svg:
        ast = buildAST(qasmpath)
        circuit = buildCircuit(ast, basis_gates=_basic_gates_string_IBM_advanced.split(",")) # as shown in IBM qunatum computing # you can add more
        qasm_svg = generate_svg_circuit.SVGenWithCircuit(circuit).generate_svg()

        # math execution:
        args = makeArgs()
        html_states = {}
        math_execute(args, circuit, states_per_step=html_states)
        emit('debugger-return', {'svg':qasm_svg, 'states': html_states})
    except Exception as err:
        emit('debugger-error', 'Error occurred: ' + str(err))




#TODO: (1) python side, (2) js side.
@socketio.on('invoke-checker')
def invoke_checker(mydict):
    try:
        print "invoke_checker request comes"
        args = makeArgs()

        qasm1 = mydict['qasm1']
        qasm2 = mydict['qasm2']
        with open("./tmp/tmp1.qasm", 'w') as qasmF1:
            qasmF1.write(qasm1)
            qasm_file1 = os.path.realpath(qasmF1.name)

        with open("./tmp/tmp2.qasm", 'w') as qasmF2:
            qasmF2.write(qasm2)
            qasm_file2 = os.path.realpath(qasmF2.name)

        ast1 = buildAST(qasm_file1)
        circuit1 = buildCircuit(ast1, basis_gates=_basic_gates_string_IBM_advanced.split(",")) # as shown in IBM qunatum computing # you can add more
        represent1, label1, stringrep1 = math_execute(args, circuit1)

        ast2 = buildAST(qasm_file2)
        circuit2 = buildCircuit(ast2, basis_gates=_basic_gates_string_IBM_advanced.split(",")) # as shown in IBM qunatum computing # you can add more
        represent2, label2, stringrep2 = math_execute(args, circuit2)

        if label1 == label2:
            isEquivalent, solution = check_equivalent_between_two_results(represent1, represent2)
            if isEquivalent:
                print "equivalent:", qasm_file1, qasm_file2
                print "solution is: E^(I*x), where x is ", solution

            emit('checker-return', {'equivalent':isEquivalent, 'final1': stringrep1, 'final2': stringrep2, 'x': str(solution)})
        else:
            print "the label does not match!"

            emit('checker-return', {'equivalent':False, 'final1': stringrep1, 'final2': stringrep2, 'x': None})
    except Exception as err:
        emit('checker-error', 'Error occurred: ' + str(err))



@socketio.on('invoke-fuzzer')
def invoke_fuzzer(mydict):
    try:
        qasm = mydict['qasm']

        with open("./tmp/tmp.qasm", 'w') as qasmF:
            qasmF.write(qasm)
            qasmpath = os.path.realpath(qasmF.name)

        # math execution:
        args = makeFuzzArgs()
        args.qasm_files = qasmpath
        if 'mode' in mydict:
            args.mode = mydict['mode']
        if 'batch_count' in mydict:
            args.batch_count = mydict['batch_count']
        if 'mutate_option' in mydict:
            args.mutate_option = mydict['mutate_option']

        ret = qfuzzmain(args)  # list of dictionaries, each for a testcase
        emit('fuzzer-return', ret)
    except Exception as err:
        emit('fuzzer-error', 'Error occurred: ' + str(err))


@socketio.on('invoke-batch-fuzzer')
def invoke_fuzzer(mydict):
    try:
        qasm = mydict['qasm']

        with open("./tmp/tmp.qasm", 'w') as qasmF:
            qasmF.write(qasm)
            qasmpath = os.path.realpath(qasmF.name)

        # math execution:
        args = makeFuzzArgs()
        args.qasm_files = qasmpath
        if 'mode' in mydict:
            args.mode = mydict['mode']
        if 'batch_count' in mydict:
            args.batch_count = mydict['batch_count']
        print args.batch_count
        if 'mutate_option' in mydict:
            args.mutate_option = mydict['mutate_option']

        ret = qfuzzmain(args)  # list of dictionaries, each for a testcase
        emit('batch-fuzzer-return', ret)
    except Exception as err:
        emit('batch-fuzzer-error', 'Error occurred: ' + str(err))




if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', debug=True, port=12347)

# with open("./tmp/tmp1.qasm", 'w') as qasmF:
#     qasmpath = os.path.realpath(qasmF.name)
#     ast = buildAST(qasmpath)
#     circuit = buildCircuit(ast, basis_gates=_basic_gates_string_IBM_advanced.split(",")) # as shown in IBM qunatum computing # you can add more
#     qasm_svg = generate_svg_circuit.SVGenWithCircuit(circuit).generate_svg()
#     print qasm_svg
