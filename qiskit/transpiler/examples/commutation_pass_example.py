# coding: utf-8
# Converted from Jupyter notebook
# An example to demonstrate the commutation pass

# In[1]:

from qiskit import *
from qiskit.dagcircuit import DAGCircuit
from qiskit.tools.visualization import *
import networkx as nx

from qiskit.transpiler import PassManager, transpile
from qiskit.transpiler.passes import CXCancellation, CommutationTransformation, CommutationAnalysis

# In[2]:

q = QuantumRegister(7)
c = ClassicalRegister(2)
circ = QuantumCircuit(q, c)
circ.x(q[0])
circ.s(q[1])
circ.cz(q[1], q[2])
circ.cz(q[2], q[1])
circ.cx(q[2], q[3])
circ.cx(q[4], q[5])
circ.cx(q[1], q[3])
circ.cx(q[0], q[3])
circ.cx(q[0], q[5])
circ.cx(q[4], q[6])
circ.cx(q[6], q[3])
circuit_drawer(circ)

# In[3]:

qdag = DAGCircuit.fromQuantumCircuit(circ)
dag_drawer(qdag)

# In[4]:

pm = PassManager()
pm.add_passes(CommutationAnalysis())
pm.add_passes(CommutationTransformation())

# In[5]:

qdag = transpile(qdag, pass_manager=pm)
dag_drawer(qdag)
