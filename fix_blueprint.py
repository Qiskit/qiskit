import re

with open('qiskit/circuit/library/blueprintcircuit.py', 'r') as f:
    content = f.read()

# 1. Replace all simple guards with self._trigger_build()
# Only the public methods and property overrides have EXACTLY this 2-line guard.
guard_pattern = re.compile(r'If not self\._is_built:\n\s+self\._build\(\)', re.IGNORECASE)
content = re.sub(r'if not self\._is_built:\n(\s+)self\._build\(\)', r'self._trigger_build()', content)

# 2. Add _trigger_build
trigger_build_code = """
    def _trigger_build(self):
        \"\"\"Helper to safely trigger and wrap the lazy build.\"\"\"
        if getattr(self, "_is_building", False) or self._is_built:
            return
            
        self._is_building = True
        try:
            self._build()
            self._is_built = True
        finally:
            self._is_building = False
"""
# Insert before _invalidate
content = content.replace("    def _invalidate(self) -> None:", trigger_build_code + "\n    def _invalidate(self) -> None:")

# 3. Add _data property and setter
data_property_code = """
    @property
    def _data(self):
        \"\"\"The underlying circuit data. Property triggers build if accessed.\"\"\"
        if not hasattr(self, "_BlueprintCircuit__data"):
            return self.__dict__.get("_data")
            
        self._trigger_build()
        return self.__data

    @_data.setter
    def _data(self, value):
        if not hasattr(self, "_BlueprintCircuit__data"):
            self.__dict__["_data"] = value
        else:
            self.__data = value
"""
content = content.replace("    def _invalidate(self) -> None:", data_property_code + "\n    def _invalidate(self) -> None:")

# 4. _init__
init_replace = """
        self._is_initialized = False
        super().__init__(*regs, name=name)
        if "_data" in self.__dict__:
            self.__data = self.__dict__.pop("_data")
        self._qregs: list[QuantumRegister] = []
        self._cregs: list[ClassicalRegister] = []
        self._is_built = False
        self._is_building = False
        self._is_initialized = True
"""
content = re.sub(
    r'self\._is_initialized = False\n\s+super\(\)\.__init__\(\*regs, name=name\)\n\s+self\._qregs: list\[QuantumRegister\] \= \[\]\n\s+self\._cregs: list\[ClassicalRegister\] \= \[\]\n\s+self\._is_built \= False\n\s+self\._is_initialized \= True',
    init_replace.strip('\n'),
    content
)

# 5. Fix _invalidate
invalidate_fix = """
    def _invalidate(self) -> None:
        \"\"\"Invalidate the current circuit build.\"\"\"
        raw = self.__data if hasattr(self, "_BlueprintCircuit__data") else self.__dict__.get("_data")
        qregs = raw.qregs
        cregs = raw.cregs
        new_data = CircuitData(raw.qubits, raw.clbits)
        for qreg in qregs:
            new_data.add_qreg(qreg)
        for creg in cregs:
            new_data.add_creg(creg)
            
        if hasattr(self, "_BlueprintCircuit__data"):
            self.__data = new_data
        else:
            self.__dict__["_data"] = new_data
            
        self._is_built = False
        new_data.global_phase = 0
"""
content = re.sub(
    r'def _invalidate\(self\) -> None:\n.*?self\._is_built = False',
    invalidate_fix.strip(),
    content,
    flags=re.DOTALL
)

# 6. Fix qregs.setter
qregs_setter_fix = """
    @qregs.setter
    def qregs(self, qregs):
        \"\"\"Set the quantum registers associated with the circuit.\"\"\"
        if not self._is_initialized:
            return
        self._qregs = []
        self._ancillas = []
        raw = self.__data if hasattr(self, "_BlueprintCircuit__data") else self.__dict__.get("_data")
        new_data = CircuitData(clbits=raw.clbits)
        if hasattr(self, "_BlueprintCircuit__data"):
            self.__data = new_data
        else:
            self.__dict__["_data"] = new_data
            
        self._is_built = False
        new_data.global_phase = 0

        self.add_register(*qregs)
"""
content = re.sub(
    r'@qregs\.setter\n\s+def qregs\(self, qregs\):\n.*?self\.add_register\(\*qregs\)',
    qregs_setter_fix.strip('\n'),
    content,
    flags=re.DOTALL
)

with open('qiskit/circuit/library/blueprintcircuit.py', 'w') as f:
    f.write(content)

print("Applied Jake's fix directly in BlueprintCircuit.")
