
#from qiskit.transpiler.exceptions import 


from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import Qubit
from qiskit.transpiler.passes.paulihedral.block_ordering import *
from qiskit.transpiler.passes.paulihedral.block_compilation import *
from qiskit.transpiler.passes.layout.noise_adaptive_layout import *

    


class Paulihedral:
    
    def __init__(self, inputprogram=None, input_format='pauliIR', evolution_time_step = 1, var_form_parameters = None, ordering_method=None, backend=None, coupling_map=None,props=None):
        if input_format == 'qubit-op':
            self.pauliIRprogram = self.load_from_qubit_op(inputprogram, evolution_time_step)
        elif input_format == 'UCCSD':
            self.pauliIRprogram = self.load_from_UCCSD(inputprogram, var_form_parameters)
        else:
            if inputprogram == None:
                '''need to raise some error'''
                print("raise some error")
            self.pauliIRprogram = inputprogram
        
        if ordering_method == 'gco' or ordering_method == 'gate-count-oriented':
            self.ordering_method = gco_ordering
        elif ordering_method == 'do' or ordering_method == 'depth-oriented':
            self.ordering_method = do_ordering
        else:
            '''need to raise some error'''
            print('unknown ordering method')
            
        if backend == 'ft' or backend == 'fault-tolerant':
            self.block_opt_method = opt_ft_backend
        elif backend == 'sc' or backend == 'superconducting':
            self.block_opt_method = opt_sc_backend
            if props==None:
                print("raise error: there is no information about backend properties which is required in superconducting backend.")
            if coupling_map == None:
                '''need to raise some error'''
                print('coupling_map missing')
            else:
                self.coupling_map = coupling_map
        else:
            '''need to raise some error'''
            print('backend not support')
        self.output_circuit = None
        self.adj_mat=None
        self.dist_mat=None
        if props!=None:
            noise_res = NoiseAdaptiveLayout(props)
            noise_res._initialize_backend_prop()
            gate_reliability=noise_res.gate_reliability
            swap_reliabs=noise_res.swap_reliabs
            qubit_num=len(swap_reliabs)
            #self.adj_mat=[0]*qubit_num
            #self.adj_mat=[self.adj_mat]*qubit_num
            self.adj_mat=[]
            for i in range(qubit_num):
                self.adj_mat.append([])
                for j in range(qubit_num):
                    self.adj_mat[i].append(0)
            self.dist_mat = []
            for i in range(qubit_num):
                self.dist_mat.append([])
                for j in range(qubit_num):
                    self.dist_mat[i].append(0)
            #print(self.adj_mat)
            for cx in gate_reliability.keys():
                i=int(cx[0])
                j=int(cx[1])
                (self.adj_mat)[i][j]=1
                (self.adj_mat)[j][i]=1
            for key1,value1 in swap_reliabs.items():
                for key2,value2 in value1.items():
                    self.dist_mat[key1][key2]=1.0-value2
        
    def load_from_qubit_op(self, qubit_op_list, evolution_time_step):
        pauliIRprogram = []
        for op in qubit_op_list:
            pauliIRprogram.append([[op.primitive.to_label(), op.coeff], evolution_time_step])
        return pauliIRprogram
    
    def load_from_UCCSD(self, UCCSD_instance, parameters):
        pauliIRprogram = []
        for i, parameter_block in enumerate(UCCSD_instance._hopping_ops):
            pauli_block = []
            parameter_block = parameter_block.to_dict()['paulis']
            for j in parameter_block:
                pauli_block.append([j['label'], j['coeff']['imag']])
            pauli_block.append(parameters[i])
            pauliIRprogram.append(pauli_block)
        print(pauliIRprogram)
        return pauliIRprogram 
                
            
    def opt(self, do_max_iteration = 30):
        self.pauli_layers = self.ordering_method(self.pauliIRprogram, do_max_iteration)
        self.output_circuit = self.block_opt_method(self.pauli_layers,self.adj_mat,self.dist_mat)
        return self.output_circuit
            
    def to_circuit(self):
        if self.output_circuit == None:
            self.output_circuit = self.opt()
        return self.output_circuit
        
        


    