// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::ast::{
    Expression, Identifier, Parameter, QuantumBlock, QuantumGateDefinition, QuantumGateSignature,
};
use crate::error::QASM3ExporterError;
use crate::exporter::{
    BitType, Counter, RegisterType, GATES_DEFINED_BY_STDGATES, RESERVED_KEYWORDS, VALID_IDENTIFIER,
    _BAD_IDENTIFIER_CHARACTERS,
};
use hashbrown::{HashMap, HashSet};
use indexmap::IndexMap;

type ExporterResult<T> = Result<T, QASM3ExporterError>;

#[derive(Debug)]
pub struct LocalSymbols {
    pub symbols: HashMap<String, Identifier>,
    pub bitinfo: HashMap<BitType, Expression>,
    pub reginfo: HashMap<RegisterType, Expression>,
}

impl LocalSymbols {
    pub fn new() -> Self {
        Self {
            symbols: HashMap::new(),
            bitinfo: HashMap::new(),
            reginfo: HashMap::new(),
        }
    }
}

pub struct SymbolTable {
    scopes: Vec<LocalSymbols>,
    pub gates: IndexMap<String, QuantumGateDefinition>,
    pub stdgates: HashSet<String>,
    _counter: Counter,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            scopes: vec![LocalSymbols::new()],
            gates: IndexMap::new(),
            stdgates: HashSet::new(),
            _counter: Counter::new(),
        }
    }

    pub fn bind(&mut self, name: &str) -> ExporterResult<()> {
        if let Some(scope) = self.scopes.last() {
            if !scope.symbols.contains_key(name) {
                self.bind_no_check(name);
            }
        } else {
            return Err(QASM3ExporterError::Error(format!(
                "Symbol table is empty, cannot bind '{}'",
                name
            )));
        }

        Ok(())
    }

    pub fn bind_no_check(&mut self, name: &str) {
        let id = Identifier {
            string: name.to_string(),
        };
        if let Some(scope) = self.scopes.last_mut() {
            scope.symbols.insert(name.to_string(), id);
        }
    }

    pub fn contains_name(&self, name: &str) -> bool {
        if let Some(scope) = self.scopes.last() {
            scope.symbols.contains_key(name)
        } else {
            false
        }
    }

    pub fn symbol_defined(&self, name: &str) -> bool {
        RESERVED_KEYWORDS.contains(&name)
            || self.gates.contains_key(name)
            || self
                .scopes
                .iter()
                .rev()
                .any(|scope| scope.symbols.contains_key(name))
    }

    pub fn can_shadow_symbol(&self, name: &str) -> bool {
        self.scopes
            .last()
            .map(|scope| !scope.symbols.contains_key(name))
            .unwrap_or(true)
            && !self.gates.contains_key(name)
            && !RESERVED_KEYWORDS.contains(&name)
    }

    pub fn escaped_declarable_name(
        &mut self,
        mut name: String,
        allow_rename: bool,
        unique: bool,
    ) -> ExporterResult<String> {
        let name_allowed = if unique {
            |n: &str, this: &SymbolTable| !this.symbol_defined(n)
        } else {
            |n: &str, this: &SymbolTable| this.can_shadow_symbol(n)
        };
        if allow_rename {
            if !VALID_IDENTIFIER.is_match(&name) {
                name = format!("_{}", _BAD_IDENTIFIER_CHARACTERS.replace_all(&name, "_"));
            }
            while !name_allowed(&name, self) {
                name = format!(
                    "{}{}",
                    name,
                    self._counter.next().expect("Counter should never fail")
                );
            }
            return Ok(name);
        }
        if !VALID_IDENTIFIER.is_match(&name) {
            return Err(QASM3ExporterError::Error(format!(
                "cannot use '{}' as a name; it is not a valid identifier",
                name
            )));
        }

        if RESERVED_KEYWORDS.contains(&name.as_str()) {
            return Err(QASM3ExporterError::Error(format!(
                "cannot use the keyword '{}' as a variable name",
                name
            )));
        }

        if !name_allowed(&name, self) {
            if self.gates.contains_key(name.as_str()) {
                return Err(QASM3ExporterError::Error(format!(
                    "cannot shadow variable '{}', as it is already defined as a gate",
                    name
                )));
            }

            for scope in self.scopes.iter().rev() {
                if let Some(other) = scope.symbols.get(&name) {
                    return Err(QASM3ExporterError::Error(format!(
                        "cannot shadow variable '{}', as it is already defined as '{:?}'",
                        name, other
                    )));
                }
            }

            return Err(QASM3ExporterError::Error(format!(
                "internal error: could not locate unshadowable '{}'",
                name
            )));
        }

        Ok(name.to_string())
    }

    pub fn add_standard_library_gates(&mut self) {
        for gate in GATES_DEFINED_BY_STDGATES.iter() {
            self.stdgates.insert(gate.to_string());
        }
    }

    pub fn set_bitinfo(&mut self, id: Expression, bit: BitType) {
        if self.scopes.is_empty() {
            self.scopes.push(LocalSymbols::new());
        }
        if let Some(scope) = self.scopes.last_mut() {
            scope.bitinfo.insert(bit, id);
        }
    }

    pub fn get_bitinfo(&self, bit: &BitType) -> Option<&Expression> {
        for scope in self.scopes.iter().rev() {
            if let Some(id) = scope.bitinfo.get(bit) {
                return Some(id);
            }
        }
        None
    }

    pub fn set_reginfo(&mut self, id: Expression, reg: RegisterType) {
        if self.scopes.is_empty() {
            self.scopes.push(LocalSymbols::new());
        }
        if let Some(scope) = self.scopes.last_mut() {
            scope.reginfo.insert(reg, id);
        }
    }

    pub fn register_gate(
        &mut self,
        op_name: String,
        params_def: Vec<Identifier>,
        qubits: Vec<Identifier>,
        body: QuantumBlock,
    ) -> ExporterResult<()> {
        // Changing the name is not allowed when defining new gates.
        let name = self.escaped_declarable_name(op_name.clone(), false, false)?;
        let _ = self.bind(&name);
        self.gates.insert(
            op_name,
            QuantumGateDefinition {
                quantum_gate_signature: QuantumGateSignature {
                    name: Identifier { string: name },
                    qarg_list: qubits,
                    params: Some(
                        params_def
                            .into_iter()
                            .map(|id| {
                                Expression::Parameter(Parameter {
                                    obj: id.string.clone(),
                                })
                            })
                            .collect(),
                    ),
                },
                quantum_block: body,
            },
        );
        Ok(())
    }

    pub fn register_bits(
        &mut self,
        name: String,
        bit: &BitType,
        allow_rename: bool,
        allow_hardware_qubit: bool,
    ) -> ExporterResult<Identifier> {
        use crate::exporter::_VALID_HARDWARE_QUBIT;

        if allow_hardware_qubit && _VALID_HARDWARE_QUBIT.is_match(&name) {
            if self.symbol_defined(&name) {
                return Err(QASM3ExporterError::Error(format!(
                    "internal error: cannot redeclare hardware qubit {}",
                    name
                )));
            }
        } else {
            self.escaped_declarable_name(name.clone(), allow_rename, false)?;
        }
        let identifier = Identifier {
            string: name.clone(),
        };
        let _ = self.bind(&name);
        self.set_bitinfo(
            Expression::Parameter(Parameter {
                obj: identifier.string.clone(),
            }),
            (*bit).to_owned(),
        );
        Ok(identifier)
    }

    pub fn register_registers(
        &mut self,
        name: String,
        register: &RegisterType,
    ) -> ExporterResult<Identifier> {
        let name = self.escaped_declarable_name(name.clone(), true, false)?;
        let identifier = Identifier {
            string: name.clone(),
        };
        let _ = self.bind(&name);
        self.set_reginfo(
            Expression::Parameter(Parameter {
                obj: identifier.string.clone(),
            }),
            (*register).to_owned(),
        );
        Ok(identifier)
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(LocalSymbols::new());
    }

    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn new_context(&mut self) -> Self {
        let mut new_table = SymbolTable::new();
        new_table.gates.clone_from(&self.gates);
        new_table.stdgates.clone_from(&self.stdgates);
        new_table
    }
}
