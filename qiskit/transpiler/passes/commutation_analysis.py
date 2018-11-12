"""
    Pass for detecting commutativity in a circuit.  property_set['commutative_set']  is a dictionary that map each gate to neighbors that commute with itself. 
"""

from qiskit.transpiler import AnalysisPass
import numpy as np
import networkx as nx
from collections import defaultdict

class CommutationAnalysis(AnalysisPass):
    def __init__(self):
        super().__init__()

    def run(self, dag):

        if self.property_set['commutative_set'] is None:
            self.property_set['commutative_set'] = defaultdict(list)

        def _GateMasterDef(name = '', para = 0.0) :
            if name == 'h':
                return 1./np.sqrt(2) * np.array([[1.0,1.0],[1.0,-1.0]], dtype = np.complex)
            if name == 'x':
                return np.array([[0.0, 1.0],[1.0,0.0]], dtype = np.complex)
            if name == 'cx': 
                return np.array([[1.0,0.0,0.0, 0.0],[0.0,1.0,0.0, 0.0],[0.0,0.0,0.0, 1.0],[0.0,0.0,1.0, 0.0]], dtype = np.complex)
            if name == 'u1':
                rtevl = np.array([[1.0,0.0],[0.0, np.exp(1.0j * np.pi)/8.0]], dtype = np.complex)
                rtevl[1,1] = np.exp(1j*float(para[0]))
                return rtevl
            if name == 'cz':
                return np.array([[1.0,0.0,0.0, 0.0],[0.0,1.0,0.0, 0.0],[0.0,0.0,1.0, 0.0],[0.0,0.0,0.0, -1.0]], dtype = np.complex)
        
        # change the 2-qubit unitary representation from qubit_A tensor qubit_B to qubit_B tensor
        def _swap_2_unitary_tensor(node):

            temp_matrix = np.copy(_GateMasterDef(name = node["name"], para = node['params']))
            temp_matrix[[1,2]] = temp_matrix[[2,1]]
            temp_matrix[:, [1,2]] = temp_matrix[:, [2,1]]

            return temp_matrix

        # The product of the two input unitaries(of 1 or 2 qubits)
        def _simple_product(node1, node2):

            q1_list = node1["qargs"]
            q2_list = node2["qargs"]

            nargs1 = len(node1["qargs"])
            nargs2 = len(node2["qargs"])

            unitary_1 = _GateMasterDef(name = node1["name"], para = node1['params'])
            unitary_2 = _GateMasterDef(name = node2["name"], para = node2['params'])

            if nargs1 == nargs2 == 1:
                return  np.matmul(unitary_1, unitary_2) 

            if  nargs1 == nargs2 == 2:
                if q1_list[0] == q2_list[0] and q1_list[1] == q2_list[1]:

                    return np.matmul(unitary_1,unitary_2)

                elif q1_list[0] == q2_list[1] and q1_list[1] == q2_list[0]:
                    return np.matmul(unitary_1, _swap_2_unitary_tensor(node2))

                elif q1_list[0] == q2_list[1]:
                    return np.matmul(np.kron(unitary_1, np.identity(2)), np.kron(np.identity(2),unitary_2))

                elif q1_list[1] == q2_list[0]:
                    return np.matmul(np.kron(np.identity(2), unitary_1), np.kron(unitary_2, np.identity(2)))

                elif q1_list[0] == q2_list[0]:
                    return np.matmul(np.kron(_swap_2_unitary_tensor(node1), np.identity(2)), np.kron(np.identity(2), unitary_2))

                elif q1_list[1] == q2_list[1]:
                    return np.matmul(np.kron(np.identity(2), _swap_2_unitary_tensor(node1)), np.kron( unitary_2, np.identity(2),))

            if nargs1 == 2 and q1_list[0] == q2_list[0]:
                return np.matmul(unitary_1,np.kron(unitary_2, np.identity(2))) 

            if nargs1 == 2 and q1_list[1] == q2_list[0]:
                return np.matmul(unitary_1,np.kron(np.identity(2), unitary_2)) 

            if nargs2 == 2 and q1_list[0] == q2_list[0]: 
                return np.matmul(np.kron(unitary_1, np.identity(2)), unitary_2) 

            if nargs2 == 2 and q1_list[0] == q2_list[1]:
                return np.matmul(np.kron(np.identity(2), unitary_1), unitary_2) 


        def _matrix_commute(node1, node2):
            # Good for composite gates or any future user defined gate of equal or less than 2 qubits.
            
            if len(set(node1["qargs"]) | set(node2["qargs"])) == len(set(node1["qargs"])) + len(set(node2["qargs"])):
                return True
            
            return np.array_equal(_simple_product(node1, node2), _simple_product(node2, node1))
             
        def commute(node1, node2):

            if node1["type"] != "op" or node2["type"] != "op":
                return False
            
            return _matrix_commute(node1, node2)
            #return _rule_commute(node1, node2)

        def overlap(node1, node2):
            return len(set(node1["qargs"]) | set(node2["qargs"])) != len(set(node1["qargs"])) + len(set(node2["qargs"]))
    
        def search_commutative_successors(node, ori_node):
            
            nd = dag.multi_graph.node[node]
            ori_nd = dag.multi_graph.node[ori_node]
            
            for suc in dag.multi_graph.successors(node):
                suc_nd = dag.multi_graph.node[suc]

                if ori_node in self.property_set['commutative_set'][suc] and suc not in self.property_set['commutative_set'][ori_node]:
                    self.property_set['commutative_set'][ori_node].append(suc)
                    search_commutative_successors(suc, ori_node)

                if commute(suc_nd, ori_nd) and overlap(suc_nd, ori_nd):
                    self.property_set['commutative_set'][ori_node].append(suc)
                    search_commutative_successors(suc, ori_node)
            
        def search_commutative_predecessors(node, ori_node):

            nd = dag.multi_graph.node[node]
            ori_nd = dag.multi_graph.node[ori_node]

            for pred in dag.multi_graph.predecessors(node):
                pred_nd = dag.multi_graph.node[pred]

                if ori_node in self.property_set['commutative_set'][pred] and pred not in self.property_set['commutative_set'][ori_node]:
                    self.property_set['commutative_set'][ori_node].append(pred)
                    search_commutative_predecessors(pred, ori_node)

                if commute(pred_nd, ori_nd) and overlap(pred_nd, ori_nd):
                    self.property_set['commutative_set'][ori_node].append(pred)
                    search_commutative_predecessors(pred, ori_node)

        ts = list(nx.topological_sort(dag.multi_graph))
        nodes_seen = dict(zip(ts, [False] * len(ts)))

        for node in ts:

            nd = dag.multi_graph.node[node]
            self.property_set['commutative_set'][node] = []

            if nd["type"] == "op" and not nodes_seen[node]:
                search_commutative_successors(node, node)
                search_commutative_predecessors(node, node)

                nodes_seen[node] = True
