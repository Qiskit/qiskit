""" Pass for constructing commutativity aware DAGCircuit from basic DAGCircuit. The generated DAGCircuit is not ready for simple scheduling.
"""

from qiskit.transpiler._basepasses import TransformationPass
from qiskit.transpiler.passes.commutation_analysis import CommutationAnalysis
from qiskit.transpiler import AnalysisPass
import numpy as np
import networkx as nx

class CommutationTransformation(TransformationPass):
    def __init__(self):
        super().__init__()
        self.requires.append(CommutationAnalysis())
        self.preserves.append(CommutationAnalysis())

    def run(self, dag):
        """
        Construct a new DAG that is commutativity aware. The new DAG is:
        - not friendly to simple scheduling(conflicts might arise), but leave more room for optimization. 
        - The depth() method will not be accurate anymore.
        - Preserves the gate count but not edge count in the MultiDiGraph

        Args:
            dag (DAGCircuit): the directed acyclic graph
        Return: 
            DAGCircuit: Transformed DAG.
        """
        def _first_pred(node, qarg):
            for pred in dag.multi_graph.predecessors(node):

                pred_nd = dag.multi_graph.node[pred]
                pred_qarg = []

                if pred_nd["type"] == "op": pred_qarg = pred_nd["qargs"]
                else: pred_qarg = [pred_nd["name"]]
                if pred not in self.property_set['commutative_set'][node] and qarg in pred_qarg:
                    return pred

                elif qarg not in pred_qarg:
                    continue

                else:
                    return _first_pred(pred, qarg)

        def _last_suc(node, qarg):


            for suc in dag.multi_graph.successors(node):

                suc_nd = dag.multi_graph.node[suc]
                suc_qarg = []

                if suc_nd["type"] == "op": suc_qarg = suc_nd["qargs"]
                else: suc_qarg = [suc_nd["name"]]

                if suc not in self.property_set['commutative_set'][node] and qarg in suc_qarg:
                    return suc

                elif qarg not in suc_qarg:
                    continue

                else:
                    return _last_suc(suc, qarg)

        ts = list(nx.topological_sort(dag.multi_graph))
        nodes_seen = dict(zip(ts, [False] * len(ts)))

        for node in ts:

            nd = dag.multi_graph.node[node]

            if nd["type"] == "op" and not nodes_seen[node]:
                
                for com_node in self.property_set['commutative_set'][node]:

                    com_nd = dag.multi_graph.node[com_node]

                    if com_node in dag.multi_graph.successors(node):

                        inter = set(nd['qargs']).intersection(com_nd['qargs'])

                        for i in range(len(inter)):
                            dag.multi_graph.remove_edge(node, com_node)

                        for qarg in inter:
                            pred = _first_pred(node, qarg)
                            dag.multi_graph.add_edge(pred, com_node, name = qarg)

                            suc = _last_suc(com_node, qarg)
                            dag.multi_graph.add_edge(node, suc, name = qarg)
                            

            nodes_seen[node] = True


        return dag
