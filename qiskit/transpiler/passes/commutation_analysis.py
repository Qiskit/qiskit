"""
    - Pass for detecting commutativity in a circuit.  property_set['commutation_set']  is a dictionary that describes the commutation relations on a given wire, all the gates on a wire are grouped into a set of gates that commute. This pass also provides useful methods to determine if two gates can commute in the circuit.
"""

from qiskit.transpiler import AnalysisPass
import numpy as np
import networkx as nx
from collections import defaultdict

class CommutationAnalysis(AnalysisPass):
    def __init__(self, max_depth = 100):
        super().__init__()
        self.max_depth = max_depth
        self.wire_op = {}
        self.node_order = {}
        self.node_commute_group = {}

    def run(self, dag):
        
        # get the order of the edge on a wire
        def _get_node_order(node):
            return self.node_order[node]

        def _GateMasterDef(name = '', para = None):

            if name == 'h':
                return 1./np.sqrt(2) * np.array([[1.0,1.0],[1.0,-1.0]], dtype = np.complex)
            if name == 'x':
                return np.array([[0.0, 1.0],[1.0,0.0]], dtype = np.complex)
            if name == 'y':
                return np.array([[0.0, -1.0j],[1.0j,0.0]], dtype = np.complex)
            if name == 'cx': 
                return np.array([[1.0,0.0,0.0, 0.0],[0.0,1.0,0.0, 0.0],[0.0,0.0,0.0, 1.0],[0.0,0.0,1.0, 0.0]], dtype = np.complex)
            if name == 'cz': 
                return np.array([[1.0,0.0,0.0, 0.0],[0.0,1.0,0.0, 0.0],[0.0,0.0,1.0, 0.0],[0.0,0.0,0.0, -1.0]], dtype = np.complex)
            if name == 'cy': 
                return np.array([[1.0,0.0,0.0, 0.0],[0.0,1.0,0.0, 0.0],[0.0,0.0,0.0, 1.0j],[0.0,0.0, -1.0j, 0.0]], dtype = np.complex)
            if name == 'z':
                return np.array([[1.0, 0.0],[0.0,-1.0]], dtype = np.complex)
            if name == 't':
                return np.array([[1.0, 0.0],[0.0,np.exp(1j*np.pi/4.0)]], dtype = np.complex)
            if name == 's':
                return np.array([[1.0, 0.0],[0.0,np.exp(1j*np.pi/2.0)]], dtype = np.complex)
            if name == 'sdag':
                return np.array([[1.0, 0.0],[0.0,-np.exp(1j*np.pi/2.0)]], dtype = np.complex)
            if name == 'tdag':
                return np.array([[1.0, 0.0],[0.0,-np.exp(1j*np.pi/4.0)]], dtype = np.complex)
            if name == 'rz' or name == 'u1':
                return np.array([[np.exp(-1j * float(para[0]) / 2), 0],[0, np.exp(1j * float(para[0]) / 2)]], dtype = np.complex)
            if name == 'rx':
                return np.array([[np.cos(float(para[0])/2), -1j * np.sin(float(para[0]) / 2)], [-1j * np.sin(float(para[0]) / 2), np.cos(float(para[0]) / 2)]], dtype = np.complex)
            if name == 'ry':
                return np.array([[np.cos(float(para[0]) / 2), - np.sin(float(para[0]) / 2)], [np.sin(float(para[0]) / 2), np.cos(float(para[0]) / 2)]], dtype = np.complex)
            if name == 'u2':
                return 1./np.sqrt(2) * np.array([[1, -np.exp(1j*float(para[1]))],[np.exp(1j*float(para[0])), np.exp(1j * (float(para[0]) + float(para[1])))]], dtype = np.complex)
            if name == 'u3':
                return 1./np.sqrt(2) * np.array([[np.cos(float(para[0]) /2.), -np.exp(1j*float(para[2]))*np.sin(float(para[0]) /2.)],[np.exp(1j*float(para[1]))* np.sin(float(para[0]) /2.), np.cos(float(para[0]) /2.) * np.exp(1j * (float(para[2]) + float(para[1])))]], dtype = np.complex)

            return None

            """ # Doesn't consider directly implemented gates now. If enabled, then parameter 'para' would be passed for the rotation angles
            
            # Example implementation of directly implemented gates:
            if name == 'u1':
                rtevl = np.array([[1.0,0.0],[0.0, np.exp(1.0j * np.pi)/8.0]], dtype = np.complex)
                rtevl[1,1] = np.exp(1j*float(para[0]))
                return rtevl
            if name == 'cz':
                return np.array([[1.0,0.0,0.0, 0.0],[0.0,1.0,0.0, 0.0],[0.0,0.0,1.0, 0.0],[0.0,0.0,0.0, -1.0]], dtype = np.complex)
            """
        
        # change the 2-qubit unitary representation from qubit_A tensor qubit_B to qubit_B tensor
        def _swap_2_unitary_tensor(node):

            temp_matrix = np.copy(_GateMasterDef(name = node["name"]))
            temp_matrix[[1,2]] = temp_matrix[[2,1]]
            temp_matrix[:, [1,2]] = temp_matrix[:, [2,1]]

            return temp_matrix

        # The product of the two input unitaries(of 1 or 2 qubits)
        def _simple_product(node1, node2):

            q1_list = node1["qargs"]
            q2_list = node2["qargs"]

            nargs1 = len(node1["qargs"])
            nargs2 = len(node2["qargs"])

            unitary_1 = _GateMasterDef(name = node1["name"], para = node1["op"].param)
            unitary_2 = _GateMasterDef(name = node2["name"], para = node2["op"].param)

            if unitary_1 is None or unitary_2 is None:
                return None

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
            
            if _simple_product(node1, node2) is not None:
                return np.array_equal(_simple_product(node1, node2), _simple_product(node2, node1))
            else:
                return False
        
        def commute(node1, node2):

            if node1["type"] != "op" or node2["type"] != "op":
                return False
            
            return _matrix_commute(node1, node2)
            # return _rule_commute(node1, node2) # Faster, but limited

        def overlap(node1, node2):
            return len(set(node1["qargs"]) | set(node2["qargs"])) != len(set(node1["qargs"])) + len(set(node2["qargs"]))

        ts = list(nx.topological_sort(dag.multi_graph))

        # Initiation of the node_order
        for num, node in enumerate(ts):
            self.node_order[node] = num

        # Initiate the commutation set
        if self.property_set['commutation_set'] is None:
            self.property_set['commutation_set'] = defaultdict(list)

        # Build a dictionary to keep track of the gates on each qubit
        for wire in dag.wires:
            wire_name = "{0}[{1}]".format(str(wire[0].name), str(wire[1]))
            self.wire_op[wire_name] = []
            self.property_set['commutation_set'][wire_name] = []

        # Add edges to the dictionary for each qubit
        for node in ts:
            for edge in dag.multi_graph.edges([node], data = True):

                edge_name = edge[2]['name']

                if edge[0] == node:
                    self.wire_op[edge_name].append(edge[0])

                    self.property_set['commutation_set'][(node, edge_name)] = -1

                if dag.multi_graph.node[edge[1]]['type'] == "out":
                    self.wire_op[edge_name].append(edge[1])

        """    
        # With traversing the circuit in topological order, the list of gates on a qubit doesn't have to be sorted
        for key in self.wire_op:
            self.wire_op[key].sort(key = _get_node_order)
        """  
        
        for wire in dag.wires:
            wire_name = "{0}[{1}]".format(str(wire[0].name), str(wire[1]))
            for node in self.wire_op[wire_name]:

                if len(self.property_set['commutation_set'][wire_name]) == 0:
                    self.property_set['commutation_set'][wire_name].append([node])

                if node not in self.property_set['commutation_set'][wire_name][-1]:
                    if commute(dag.multi_graph.node[node], dag.multi_graph.node[self.property_set['commutation_set'][wire_name][-1][-1]]):
                        self.property_set['commutation_set'][wire_name][-1].append(node)

                    else:
                        self.property_set['commutation_set'][wire_name].append([node])

                self.property_set['commutation_set'][(node, wire_name)] = len(self.property_set['commutation_set'][wire_name]) - 1

        """
        # Output the grouped set for testing
        for wire in dag.wires:
            wire_name = "{0}[{1}]".format(str(wire[0].name), str(wire[1]))
            print(wire_name)
            for group in self.property_set['commutation_set'][wire_name]:
                print(group)
        """

