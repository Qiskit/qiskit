import math
import copy
# Author: Andrew Cross

class InterpreterException(Exception):
    def __init__(self, *msg):
        self.msg = ' '.join(msg)

    def __str__(self):
        return repr(self.msg)

class Unroller(object):

  def __init__(self, ast, be=None):
    """Initialize interpreter's data"""
    # Abstract syntax tree from parser
    self.ast = ast
    # Backend object
    self.be = be
    # IBMQASM version number
    self.version = 0.0
    # Dict of qreg names and sizes
    self.qregs = {}
    # Dict of creg names and sizes
    self.cregs = {}
    # Dict of gates names and properties
    self.gates = {}
    # List of dictionaries mapping local parameter ids to real values
    self.arg_stack = [{}]
    # List of dictionaries mapping local bit ids to global ids (name,idx)
    self.bit_stack = [{}]

  def process_bit_id(self,n):
    """Process Id or IndexedId node as a bit or register type
       Return a list of tuples (name,index)
    """
    if n.type == "indexed_id":
      # an indexed bit or qubit
      return [(n.name,n.index)]
    elif n.type == "id":
      # a qubit or qreg or creg
      if len(self.bit_stack[-1]) == 0:
        # global scope
        if n.name in self.qregs:
          return [(n.name,j) for j in range(self.qregs[n.name])]
        elif n.name in self.cregs:
          return [(n.name,j) for j in range(self.cregs[n.name])]
        else:
          raise InterpreterException("expected qreg or creg name:",\
                "line=%s"%n.line,"file=%s"%n.file)
      else:
        # local scope
        if n.name in self.bit_stack[-1]:
          return [self.bit_stack[-1][n.name]]
        else:
          raise InterpreterException("excepted local bit name:",\
                "line=%s"%n.line,"file=%s"%n.file)

  def process_children(self,n):
    """Call process_node for all children of n"""
    for c in n.children:
      self.process_node(c)

  def process_node(self,n):
    """Carry out action associated with node n"""

    if n.type == "program":
      self.process_children(n)

    elif n.type == "qreg":
      self.qregs[n.name] = int(n.index)
      self.be.new_qreg(n.name,int(n.index))

    elif n.type == "creg":
      self.cregs[n.name] = int(n.index)
      self.be.new_creg(n.name,int(n.index))

    elif n.type == "id":
      # If we process_node on an id, it must be in arg_stack
      # i.e. the id is inside a gate_body
      id_dict = self.arg_stack[-1]
      if n.name in id_dict:
        return float(id_dict[n.name])
      else:
        raise InterpreterException("excepted local parameter name:",\
              "line=%s"%n.line,"file=%s"%n.file)

    elif n.type == "int":
      # We process int nodes when they are leaves of expressions
      # and cast to float to avoid, for example, 3/2 = 1
      return float(n.value)

    elif n.type == "real":
      return float(n.value)

    elif n.type == "indexed_id":
      raise InterpreterException("internal error n.type==indexed_id:",\
            "line=%s"%n.line,"file=%s"%n.file)

    elif n.type == "id_list":
      # We process id_list nodes when they are leaves of barriers
      return [self.process_bit_id(m) for m in n.children]

    elif n.type == "primary_list":
      # Should only be called for barrier
      return [self.process_bit_id(m) for m in n.children]

    elif n.type == "gate":
      self.gates[n.name] = {}
      de = self.gates[n.name]
      de["opaque"] = False
      de["n_args"] = n.n_args()
      de["n_bits"] = n.n_bits()
      if n.n_args() > 0:
        de["args"] = [c.name for c in n.arguments.children]
      else:
        de["args"] = []
      de["bits"] = [c.name for c in n.bitlist.children]
      de["body"] = n.body
      self.be.define_gate(n.name,copy.deepcopy(de))

    elif n.type == "custom_unitary":
      name = n.name
      if n.arguments is not None:
        args = self.process_node(n.arguments)
      else:
        args = []
      bits = [self.process_bit_id(m) for m in n.bitlist.children]
      if name in self.gates:
        gargs = self.gates[name]["args"]
        gbits = self.gates[name]["bits"]
        gbody = self.gates[name]["body"]
        # loop over register arguments, if any
        maxidx = max(map(len,bits))
        for idx in range(maxidx):
          self.arg_stack.append({gargs[j]: args[j] \
            for j in range(len(gargs))})
          # only index into register arguments
          f = list(map(lambda x:idx*x,\
                [len(bits[j])>1 for j in range(len(bits))]))
          self.bit_stack.append({gbits[j]: bits[j][f[j]] \
            for j in range(len(gbits))})
          self.be.start_gate(name,\
            [self.arg_stack[-1][s] for s in gargs],\
            [self.bit_stack[-1][s] for s in gbits])
          if not self.gates[name]["opaque"]:
            self.process_children(gbody)
          self.be.end_gate(name,\
            [self.arg_stack[-1][s] for s in gargs],\
            [self.bit_stack[-1][s] for s in gbits])
          self.arg_stack.pop()
          self.bit_stack.pop()
      else:
        raise InterpreterException("internal error undefined gate:",\
              "line=%s"%n.line,"file=%s"%n.file)

    elif n.type == "universal_unitary":
      args = tuple(self.process_node(n.children[0]))
      qid = self.process_bit_id(n.children[1])
      for idx in range(len(qid)):
        self.be.U(args,qid[idx])

    elif n.type == "cnot":
      id0 = self.process_bit_id(n.children[0])
      id1 = self.process_bit_id(n.children[1])
      if not( len(id0) == len(id1) or len(id0) == 1 or len(id1) == 1 ):
        raise InterpreterException("internal error: qreg size mismatch",\
              "line=%s"%n.line,"file=%s"%n.file)
      maxidx = max([len(id0), len(id1)])
      for idx in range(maxidx):
        if len(id0) > 1 and len(id1) > 1:
          self.be.CX(id0[idx],id1[idx])
        elif len(id0) > 1:
          self.be.CX(id0[idx],id1[0])
        else:
          self.be.CX(id0[0],id1[idx])

    elif n.type == "expression_list":
      return [self.process_node(m) for m in n.children]

    elif n.type == "binop":
      op = n.children[0]
      lexpr = n.children[1]
      rexpr = n.children[2]
      if op == '+':
        return self.process_node(lexpr) + self.process_node(rexpr)
      elif op == '-':
        return self.process_node(lexpr) - self.process_node(rexpr)
      elif op == '*':
        return self.process_node(lexpr) * self.process_node(rexpr)
      elif op == '/':
        # TODO: check divide by zero? Python will raise
        return self.process_node(lexpr) / self.process_node(rexpr)
      elif op == '^':
        # TODO: check 0 ** 0? Python will raise
        return self.process_node(lexpr) ** self.process_node(rexpr)
      else:
        raise InterpreterException("internal error: undefined binop",\
              "line=%s"%n.line,"file=%s"%n.file)

    elif n.type == "prefix":
      op = n.children[0]
      expr = n.children[1]
      if op == '+':
        return self.process_node(expr)
      elif op == '-':
        return -self.process_node(expr)
      else:
        raise InterpreterException("internal error: undefined prefix",\
              "line=%s"%n.line,"file=%s"%n.file)

    elif n.type == "measure":
      id0 = self.process_bit_id(n.children[0])
      id1 = self.process_bit_id(n.children[1])
      if len(id0) != len(id1):
        raise InterpreterException("internal error: reg size mismatch",\
              "line=%s"%n.line,"file=%s"%n.file)
      for idx in range(len(id0)):
        self.be.measure(id0[idx],id1[idx])

    elif n.type == "magic":
      self.version = float(n.children[0])
      self.be.version(n.children[0])

    elif n.type == "barrier":
      ids = self.process_node(n.children[0])
      self.be.barrier(ids)

    elif n.type == "reset":
      id0 = self.process_bit_id(n.children[0])
      for idx in range(len(id0)):
        self.be.reset(id0[idx])

    elif n.type == "if":
      creg = n.children[0].name
      cval = n.children[1]
      self.be.set_condition(creg,cval)
      self.process_node(n.children[2])
      self.be.drop_condition()
      pass

    elif n.type == "opaque":
      self.gates[n.name] = {}
      de = self.gates[n.name]
      de["opaque"] = True
      de["n_args"] = n.n_args()
      de["n_bits"] = n.n_bits()
      if n.n_args() > 0:
        de["args"] = [c.name for c in n.arguments.children]
      else:
        de["args"] = []
      de["bits"] = [c.name for c in n.bitlist.children]
      de["body"] = None
      self.be.define_gate(n.name,copy.deepcopy(de))

    elif n.type == "external":
      op = n.children[0].name
      expr = n.children[1]
      if op == 'sin':
        return math.sin(self.process_node(expr))
      elif op == 'cos':
        return math.cos(self.process_node(expr))
      elif op == 'tan':
        return math.tan(self.process_node(expr))
      elif op == 'exp':
        return math.exp(self.process_node(expr))
      elif op == 'ln':
        # TODO: check domain? Python will raise
        return math.ln(self.process_node(expr))
      elif op == 'sqrt':
        # TODO: check domain? Python will raise
        return math.sqrt(self.process_node(expr))
      else:
        raise InterpreterException("internal error: undefined external",\
              "line=%s"%n.line,"file=%s"%n.file)

    else:
      raise InterpreterException("internal error: undefined node type",\
            n.type,"line=%s"%n.line,"file=%s"%n.file)

  def set_backend(self,be):
    self.be = be

  def execute(self):
    if self.be is not None:
      self.process_node(self.ast)
    else:
      raise InterpreterException("backend not attached")
