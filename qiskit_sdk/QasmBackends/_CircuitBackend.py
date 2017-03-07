from ._BaseBackend import BaseBackend, BackendException
from .. import CircuitGraph
# Author: Andrew Cross

class CircuitBackend(BaseBackend):

  def __init__(self,basis=[]):
    self.prec = 15
    self.creg = None
    self.cval = None
    self.cg = CircuitGraph.CircuitGraph()
    self.basis = basis
    self.listen = True
    self.in_gate = ""
    self.gates = {}

  def set_basis(self,basis):
    """Declare the set of user-defined gates to emit"""
    self.basis = basis

  def define_gate(self,name,gatedata):
    """Record and pass down the data for this gate"""
    self.gates[name] = gatedata
    self.cg.add_gate_data(name,gatedata)

  def version(self,v):
    pass

  def new_qreg(self,name,sz):
    self.cg.add_qreg(name,sz)

  def new_creg(self,name,sz):
    self.cg.add_creg(name,sz)

  def U(self,arg,qubit):
    if self.listen:
      if self.creg is not None:
        condition = (self.creg,self.cval)
      else:
        condition = None
      if "U" not in self.basis:
        self.basis.append("U")
        self.cg.add_basis_element("U",1,0,3)
      self.cg.apply_operation_back("U",[qubit],[],\
        list(arg),condition)

  def CX(self,qubit0,qubit1):
    if self.listen:
      if self.creg is not None:
        condition = (self.creg,self.cval)
      else:
        condition = None
      if "CX" not in self.basis:
        self.basis.append("CX")
        self.cg.add_basis_element("CX",2)
      self.cg.apply_operation_back("CX",[qubit0,qubit1],\
        [],[],condition)

  def measure(self,qubit,bit):
    if self.creg is not None:
      condition = (self.creg,self.cval)
    else:
      condition = None
    if "measure" not in self.basis:
      self.basis.append("measure")
      self.cg.add_basis_element("measure",1,1)
    self.cg.apply_operation_back("measure",[qubit],\
      [bit],[],condition)

  def barrier(self,qubitlists):
    if self.listen:
      names = []
      for x in qubitlists:
        for j in range(len(x)):
          names.append(x[j])
      if "barrier" not in self.basis:
        self.basis.append("barrier")
        self.cg.add_basis_element("barrier",-1)
      self.cg.apply_operation_back("barrier",names)

  def reset(self,qubit):
    if self.creg is not None:
      condition = (self.creg,self.cval)
    else:
      condition = None
    if "reset" not in self.basis:
      self.basis.append("reset")
      self.cg.add_basis_element("reset",1)
    self.cg.apply_operation_back("reset",[qubit],\
      [],[],condition)

  def set_condition(self,creg,cval):
    self.creg = creg
    self.cval = cval

  def drop_condition(self):
    self.creg = None
    self.cval = None

  def start_gate(self,name,args,qubits):
    if self.listen and name not in self.basis and self.gates[name]["opaque"]:
      raise BackendException("opaque gate %s not in basis"%name)
    if self.listen and name in self.basis:
      if self.creg is not None:
        condition = (self.creg,self.cval)
      else:
        condition = None
      self.in_gate = name
      self.listen = False
      self.cg.add_basis_element(name,len(qubits),0,len(args))
      self.cg.apply_operation_back(name,qubits,[],args,condition)

  def end_gate(self,name,args,qubits):
    if name == self.in_gate:
      self.in_gate = ""
      self.listen = True

