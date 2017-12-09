from math_executor import math_execute
from circuit_builder import buildAST, buildCircuit
from circuit_layer_analyzer import gate_qubit_tuples_of_circuit_as_layers

class CacheExecutor:

    def __init__(self, _basic_gates_string_para):
        self.cache = {}
        self._basic_gates_string = _basic_gates_string_para

    def cache_execute(self, args, current_file):
        if current_file not in self.cache:
            orig_ast = buildAST(current_file)
            orig_circuit = buildCircuit(orig_ast, basis_gates=self._basic_gates_string.split(","))
            orig_schedule_list_of_layers, global_qubit_inits, global_cbit_inits = gate_qubit_tuples_of_circuit_as_layers(orig_circuit)
            orig_matrix, _, _ = math_execute(args, orig_circuit)
            bitDomain = global_qubit_inits.keys()
            self.cache[current_file] = {'ast': orig_ast, 'circuit': orig_circuit, 'schedule': orig_schedule_list_of_layers, 'matrix': orig_matrix, 'bitDomain':bitDomain}
        else:
            #print "this file has been cached"
            pass


    def cache_get(self, current_file, property_name):
        if current_file in self.cache:
            perItemDict = self.cache[current_file]
            if perItemDict != None:
                return perItemDict[property_name]
            else:
                return None
        else:
            return None


    def cache_purge(self, current_file):
        perItemDict = self.cache[current_file]
        if perItemDict != None:
            perItemDict.clear()

        if current_file in self.cache:
            del self.cache[current_file]






