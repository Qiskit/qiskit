# Author: Andrew Cross

class BackendException(Exception):
  def __init__(self, *msg):
    self.msg = ' '.join(msg)

  def __str__(self):
    return repr(self.msg)

class BaseBackend(object):

  def __init__(self, basis=[]):
    self.prec = 15
    self.creg = None
    self.cval = None
    self.gates = {}
    self.comments = False
    self.basis = basis
    self.listen = True
    self.in_gate = ""
    self.printed_gates = []

  def set_comments(self,comments):
    self.comments = comments

  def set_basis(self,basis):
    """Declare the set of user-defined gates to emit"""
    self.basis = basis

  def fs(self,f):
    fmt = "{0:0.%sf}"%self.prec
    return fmt.format(f)

  def version(self,v):
    print("IBMQASM %s;"%v)

  def new_qreg(self,name,sz):
    print("qreg %s[%d];"%(name,sz))

  def new_creg(self,name,sz):
    print("creg %s[%d];"%(name,sz))

  def _gate_string(self,name):
    out = ""
    if self.gates[name]["opaque"]:
      out = "opaque " + name
    else:
      out = "gate " + name 
    if self.gates[name]["n_args"] > 0:
      out += "(" + ",".join(self.gates[name]["args"]) + ")"
    out += " " + ",".join(self.gates[name]["bits"]) 
    if self.gates[name]["opaque"]:
      out += ";"
    else:
      out += "\n{\n" + self.gates[name]["body"].qasm() + "}"
    return out

  def define_gate(self,name,gatedata):
    self.gates[name] = gatedata
    # Print out the gate definition if it is in self.basis
    if name in self.basis and \
       not name in ["U","CX","measure","reset","barrier"]:
      if not self.gates[name]["opaque"]:
        calls = self.gates[name]["body"].calls()
        for c in calls:
          if c not in self.printed_gates:
            print(self._gate_string(c))
            self.printed_gates.append(c)
      if name not in self.printed_gates:
        print(self._gate_string(name))
        self.printed_gates.append(name)

  def U(self,arg,qubit):
    if self.listen:
      if "U" not in self.basis:
        self.basis.append("U")
      if self.creg is not None:
        print("if(%s==%d) "%(self.creg,self.cval),end="")
      print("U(%s,%s,%s) %s[%d];"%(self.fs(arg[0]),self.fs(arg[1]),\
        self.fs(arg[2]),qubit[0],qubit[1]))

  def CX(self,qubit0,qubit1):
    if self.listen:
      if "CX" not in self.basis:
        self.basis.append("CX")
      if self.creg is not None:
        print("if(%s==%d) "%(self.creg,self.cval),end="")
      print("CX %s[%d],%s[%d];"%(qubit0[0],qubit0[1],qubit1[0],qubit1[1]))

  def measure(self,qubit,bit):
    if "measure" not in self.basis:
      self.basis.append("measure")
    if self.creg is not None:
      print("if(%s==%d) "%(self.creg,self.cval),end="")
    print("measure %s[%d] -> %s[%d];"%(qubit[0],qubit[1],bit[0],bit[1]))

  def barrier(self,qubitlists):
    if self.listen:
      if "barrier" not in self.basis:
        self.basis.append("barrier")
      names = []
      for x in qubitlists:
        if len(x) == 1:
          names.append("%s[%d]"%(x[0][0],x[0][1]))
        else:
          names.append("%s"%x[0][0])
      print("barrier %s;"%",".join(names))

  def reset(self,qubit):
    if "reset" not in self.basis:
      self.basis.append("reset")
    if self.creg is not None:
      print("if(%s==%d) "%(self.creg,self.cval),end="")
    print("reset %s[%d];"%(qubit[0],qubit[1]))

  def set_condition(self,creg,cval):
    self.creg = creg
    self.cval = cval
    if self.comments:
      print("// set condition %s, %s"%(creg,cval))

  def drop_condition(self):
    self.creg = None
    self.cval = None
    if self.comments:
      print("// drop condition")

  def start_gate(self,name,args,qubits):
    if self.listen and self.comments:
      print("// start %s, %s, %s"%(name,list(map(self.fs,args)),qubits))
    if self.listen and name not in self.basis and self.gates[name]["opaque"]:
      raise BackendException("opaque gate %s not in basis"%name)
    if self.listen and name in self.basis:

      if self.creg is not None:
        condition = (self.creg,self.cval)
      else:
        condition = None
      self.in_gate = name
      self.listen = False

      squbits = []
      for x in qubits:
        squbits.append("%s[%d]"%(x[0],x[1]))

      print(name,end="")
      if len(args)>0:
        print("(%s)"%",".join(map(self.fs,args)),end="")
      print(" %s;"%",".join(squbits))

  def end_gate(self,name,args,qubits):
    if name == self.in_gate:
      self.in_gate = ""
      self.listen = True
    if self.listen and self.comments:
      print("// end %s, %s, %s"%(name,list(map(self.fs,args)),qubits))
