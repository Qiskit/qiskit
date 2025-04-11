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
    Barrier, BitArray, Break, ClassicalDeclaration, ClassicalType, Constant, Continue, Delay, Designator, DurationLiteral, DurationUnit, Expression, Float, GateCall, Header, IODeclaration, IOModifier, Identifier, IdentifierOrSubscripted, Include, IntegerLiteral, Node, Parameter, Program, QuantumBlock, QuantumDeclaration, QuantumGateDefinition, QuantumGateModifier, QuantumGateModifierName, QuantumGateSignature, QuantumInstruction, QuantumMeasurement, QuantumMeasurementAssignment, Reset, Statement, SubscriptedIdentifier, Version
};

use crate::printer::BasicPrinter;
use hashbrown::{HashMap, HashSet};
use oq3_syntax::ted::replace;
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3::PyRef;
use pyo3::Python;
use qiskit_circuit::bit::{Register, ShareableBit, ShareableClbit, ShareableQubit};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::duration;
use qiskit_circuit::operations::{DelayUnit, StandardInstruction};
use qiskit_circuit::operations::{Operation, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedInstruction;
use std::result;
use std::sync::Mutex;
use thiserror::Error;

use lazy_static::lazy_static;
use regex::Regex;

type ExporterResult<T> = Result<T, QASM3ExporterError>;

struct BuildScope {
    circuit_data: CircuitData,
    qubit_map: HashMap<ShareableQubit, ShareableQubit>,
    clbit_map: HashMap<ShareableClbit, ShareableClbit>,
}

impl BuildScope {
    fn new(circuit_data: &CircuitData, qubits: &Vec<ShareableQubit>, clbits: &Vec<ShareableClbit>) -> Self {
        let qubit_map = qubits
            .iter()
            .map(|q| (q.clone(), q.clone()))
            .collect();

        let clbit_map = clbits
            .iter()
            .map(|c| (c.clone(), c.clone()))
            .collect();

        Self {
            circuit_data: circuit_data.clone(),
            qubit_map,
            clbit_map,
        }
    }

    fn with_mappings(
        circuit_data: CircuitData,
        qubit_map: HashMap<ShareableQubit, ShareableQubit>,
        clbit_map: HashMap<ShareableClbit, ShareableClbit>,
    ) -> Self {
        Self {
            circuit_data,
            qubit_map,
            clbit_map,
        }
    }
}

struct SymbolTable {
    symbols: Vec<HashMap<String, Identifier>>,
    clbitinfo: Vec<HashMap<ShareableClbit, IdentifierOrSubscripted>>,
    qubitinfo: Vec<HashMap<ShareableQubit, IdentifierOrSubscripted>>,
    gates: HashMap<String, QuantumGateDefinition>,
    stdgates: HashSet<String>,
    _counter: Counter,
}

impl SymbolTable {
    fn new() -> Self {
        let symbols = vec![HashMap::new()];
        let clbitinfo = vec![HashMap::new()];
        let qubitinfo = vec![HashMap::new()];
        Self {
            symbols,
            clbitinfo,
            qubitinfo,
            gates: HashMap::new(),
            stdgates: HashSet::new(),
            _counter: Counter::new(),
        }
    }

    fn bind(&mut self, name: &str) -> ExporterResult<()> {
        if let Some(symbols) = self.symbols.last() {
            if !symbols.contains_key(name) {
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

    fn bind_no_check(&mut self, name: &str) {
        let id = Identifier {
            string: name.to_string(),
        };
        if let Some(last) = self.symbols.last_mut() {
            last.insert(name.to_string(), id);
        }
    }

    fn contains_name(&self, name: &str) -> bool {
        if let Some(symbols) = self.symbols.last() {
            symbols.contains_key(name)
        } else {
            false
        }
    }

    fn symbol_defined(&self, name: &str) -> bool {
        RESERVED_KEYWORDS.contains(name) || self.gates.contains_key(name) || self.symbols.iter().rev().any(|symbol| symbol.contains_key(name))
    }

    fn can_shadow_symbol(&self, name: &str) -> bool {
        !self.symbols.last().unwrap().contains_key(name) && !self.gates.contains_key(name) && !RESERVED_KEYWORDS.contains(name)
    }

    fn escaped_declarable_name(&mut self, name: &str, allow_rename: bool, unique: bool) -> ExporterResult<String> {
        let name_allowed = if unique {
            |n: &str, this: &SymbolTable| !this.symbol_defined(n)
        } else {
            |n: &str, this: &SymbolTable| this.can_shadow_symbol(n)
        };
        if allow_rename {
            let mut name = name.to_string();
            if !VALID_IDENTIFIER.is_match(&name) {
                name = format!(
                    "_{}",
                    _BAD_IDENTIFIER_CHARACTERS.replace_all(&name, "_")
                );
            }
            let base = name.clone();
            while !name_allowed(&name, self) {
                name = format!("{}{}", base, self._counter.next().unwrap());
            }
            return Ok(name);
        }
        if !VALID_IDENTIFIER.is_match(name) {
            return Err(QASM3ExporterError::Error(format!(
                "cannot use '{}' as a name; it is not a valid identifier",
                name
            )));
        }

        if RESERVED_KEYWORDS.contains(name) {
            return Err(QASM3ExporterError::Error(format!(
                "cannot use the keyword '{}' as a variable name",
                name
            )));
        }

        if !name_allowed(name, self) {
            if self.gates.contains_key(name) {
                return Err(QASM3ExporterError::Error(format!(
                    "cannot shadow variable '{}', as it is already defined as a gate",
                    name
                )));
            }

            for scope in self.symbols.iter().rev() {
                if let Some(other) = scope.get(name) {
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

    fn add_standard_library_gates(&mut self) {
        for gate in GATES_DEFINED_BY_STDGATES.iter() {
            self.stdgates.insert(gate.to_string());
        }
    }

    fn set_clbitinfo(&mut self, id: IdentifierOrSubscripted, clbit: ShareableClbit) {
        if self.clbitinfo.is_empty() {
            self.clbitinfo.push(HashMap::new());
        }
        if let Some(last) = self.clbitinfo.last_mut() {
            last.insert(clbit, id);
        }
    }

    fn set_qubitinfo(&mut self, id: IdentifierOrSubscripted, qubit: ShareableQubit) {
        if self.qubitinfo.is_empty() {
            self.qubitinfo.push(HashMap::new());
        }
        if let Some(last) = self.qubitinfo.last_mut() {
            last.insert(qubit, id);
        }
    }

    fn get_qubitinfo(&self, qubit: &ShareableQubit) -> Option<&IdentifierOrSubscripted> {
        for info in self.qubitinfo.iter().rev() {
            if let Some(id) = info.get(qubit) {
                return Some(id);
            }
        }
        None
    }
    fn get_clbitinfo(&self, clbit: &ShareableClbit) -> Option<&IdentifierOrSubscripted> {
        for info in self.clbitinfo.iter().rev() {
            if let Some(id) = info.get(clbit) {
                return Some(id);
            }
        }
        None
    }

    fn register_gate(
        &mut self,
        stdgate: StandardGate,
        params_def: Vec<Identifier>,
        qubits: Vec<Identifier>,
        body: QuantumBlock,
    ) -> ExporterResult<()> {
        let name = self.escaped_declarable_name(&stdgate.name(), true, false)?;
        if !self.contains_name(&name) {
            let _ = self.bind(&name);
        }
        self.gates.insert(
            stdgate.name().to_string(),
            QuantumGateDefinition {
                quantum_gate_signature: QuantumGateSignature {
                    name: Identifier { string: name },
                    qarg_list: qubits,
                    params: Some(params_def.into_iter().map(Expression::Identifier).collect()),
                },
                quantum_block: body,
            },
        );
        Ok(())
    }

    fn push_scope(&mut self) {
        self.symbols.push(HashMap::new());
        self.qubitinfo.push(HashMap::new());
        self.clbitinfo.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.symbols.pop();
        self.qubitinfo.pop();
        self.clbitinfo.pop();
    }

    fn new_context(&mut self) -> Self {
        let mut new_table = SymbolTable::new();
        new_table.gates = self.gates.clone();
        new_table.stdgates = self.stdgates.clone();
        new_table
    }
}

#[derive(Error, Debug)]
pub enum QASM3ExporterError {
    #[error("Error: {0}")]
    Error(String),
    #[error("Symbol '{0}' is not found in the table")]
    SymbolNotFound(String),
    #[error("PyError: {0}")]
    PyErr(PyErr),
}

impl From<PyErr> for QASM3ExporterError {
    fn from(err: PyErr) -> Self {
        QASM3ExporterError::PyErr(err)
    }
}

lazy_static! {
    static ref BIT_PREFIX:&'static str =  "_bit";
}
lazy_static! {
    static ref QUBIT_PREFIX:&'static str =  "_qubit";
}
lazy_static! {
    static ref GATE_PARAM_PREFIX:&'static str =  "_gate_p";
}
lazy_static! {
    static ref GATE_QUBIT_PREFIX:&'static str =  "_gate_q";
}
lazy_static! {
    static ref GATES_DEFINED_BY_STDGATES: HashSet<&'static str> = [
        "p", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "rx", "ry", "rz", "cx", "cy", "cz",
        "rzz", "cp", "crx", "cry", "crz", "ch", "swap", "ccx", "cswap", "cu", "CX", "phase",
        "cphase", "id", "u1", "u2", "u3",
    ]
    .into_iter()
    .collect();
}

lazy_static! {
    static ref RESERVED_KEYWORDS: HashSet<&'static str> = [
        "OPENQASM",
        "angle",
        "array",
        "barrier",
        "bit",
        "bool",
        "box",
        "break",
        "cal",
        "complex",
        "const",
        "continue",
        "creg",
        "ctrl",
        "def",
        "defcal",
        "defcalgrammar",
        "delay",
        "duration",
        "durationof",
        "else",
        "end",
        "extern",
        "float",
        "for",
        "gate",
        "gphase",
        "if",
        "in",
        "include",
        "input",
        "int",
        "inv",
        "let",
        "measure",
        "mutable",
        "negctrl",
        "output",
        "pow",
        "qreg",
        "qubit",
        "reset",
        "return",
        "sizeof",
        "stretch",
        "uint",
        "while"
    ]
    .into_iter()
    .collect();
}

lazy_static! {
    static ref VALID_IDENTIFIER: Regex = Regex::new(r"(^[\w][\w\d]*$|^\$\d+$)").unwrap();
}

lazy_static! {
    static ref _BAD_IDENTIFIER_CHARACTERS: Regex = Regex::new(r"[^\w\d]").unwrap();
}

pub struct Exporter {
    includes: Vec<&'static str>,
    basis_gates: Vec<&'static str>,
    disable_constants: bool,
    _alias_classical_registers: bool,
    allow_aliasing: bool,
    indent: &'static str,
}

impl Exporter {
    pub fn new(
        includes: Vec<&'static str>,
        basis_gates: Vec<&'static str>,
        disable_constants: bool,
        alias_classical_registers: bool,
        allow_aliasing: bool,
        indent: &'static str,
    ) -> Self {
        Self {
            includes,
            basis_gates,
            disable_constants,
            _alias_classical_registers: alias_classical_registers,
            allow_aliasing,
            indent,
        }
    }

    pub fn dumps(&self, circuit_data: &CircuitData, islayout: bool) -> ExporterResult<String> {
        let mut builder = QASM3Builder::new(
            circuit_data,
            islayout,
            self.includes.clone(),
            self.basis_gates.clone(),
            self.disable_constants,
            self.allow_aliasing,
        );
        match builder.build_program() {
            Ok(program) => {
                let mut output = String::new();
                BasicPrinter::new(&mut output, self.indent.to_string(), false)
                    .visit(&Node::Program(&program));
                Ok(output)
            }
            Err(e) => Err(QASM3ExporterError::Error(e.to_string())),
        }
    }
}


#[derive(Debug, Clone)]
struct Counter {
    current: usize,
}

impl Counter {
    fn new() -> Self {
        Self { current: 0 }
    }
}

impl Iterator for Counter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.current;
        self.current = self.current.wrapping_add(1);
        Some(n)
    }
}

pub struct QASM3Builder {
    _builtin_instr: HashSet<&'static str>,
    loose_bit_prefix: &'static str,
    loose_qubit_prefix: &'static str,
    _gate_param_prefix: &'static str,
    _gate_qubit_prefix: &'static str,
    circuit_scope: BuildScope,
    is_layout: bool,
    symbol_table: SymbolTable,
    global_io_decls: Vec<IODeclaration>,
    global_classical_decls: Vec<ClassicalDeclaration>,
    includes: Vec<&'static str>,
    basis_gates: Vec<&'static str>,
    disable_constants: bool,
    _allow_aliasing: bool,
}

impl<'a> QASM3Builder {
    pub fn new(
        circuit_data: &'a CircuitData,
        is_layout: bool,
        includes: Vec<&'static str>,
        basis_gates: Vec<&'static str>,
        disable_constants: bool,
        allow_aliasing: bool,
    ) -> Self {
        Self {
            _builtin_instr: [
                "barrier",
                "measure",
                "reset",
                "delay",
                "break_loop",
                "continue_loop",
                "store",
            ]
            .into_iter()
            .collect(),
            loose_bit_prefix: &BIT_PREFIX,
            loose_qubit_prefix: &QUBIT_PREFIX,
            _gate_param_prefix: &GATE_PARAM_PREFIX,
            _gate_qubit_prefix: &GATE_QUBIT_PREFIX,
            circuit_scope: BuildScope::new(circuit_data, circuit_data.qubits().objects(), circuit_data.clbits().objects()),
            is_layout,
            symbol_table: SymbolTable::new(),
            global_io_decls: Vec::new(),
            global_classical_decls: Vec::new(),
            includes,
            basis_gates,
            disable_constants,
            _allow_aliasing: allow_aliasing,
        }
    }

    fn new_scope<F>(
        &mut self,
        circuit_data: &CircuitData,
        qubits: Vec<ShareableQubit>,
        clbits: Vec<ShareableClbit>,
        f: F
    ) -> ExporterResult<()>
    where
        F: FnOnce(&mut QASM3Builder) -> ExporterResult<()>,
    {
        let current_qubitmap = &self.circuit_scope.qubit_map;
        let current_clbitmap = &self.circuit_scope.clbit_map;
        let new_qubits: Vec<ShareableQubit> = qubits.iter().map(|q| current_qubitmap[q].clone()).collect();
        let new_clbits: Vec<ShareableClbit> = clbits.iter().map(|c| current_clbitmap[c].clone()).collect();

        if circuit_data.num_qubits() != new_qubits.len() {
            return Err(QASM3ExporterError::Error(format!(
                "Tried to push a scope whose circuit needs {} qubits, but only provided {}.",
                circuit_data.num_qubits(),
                new_qubits.len()
            )));
        }

        if circuit_data.num_clbits() != new_clbits.len() {
            return Err(QASM3ExporterError::Error(format!(
                "Tried to push a scope whose circuit needs {} clbits, but only provided {}.",
                circuit_data.num_clbits(),
                new_clbits.len()
            )));
        }

        let new_qubit_map: HashMap<ShareableQubit, ShareableQubit> = circuit_data
            .qubits()
            .objects()
            .iter()
            .map(|q| (q.clone(), q.clone()))
            .collect();

        let new_clbit_map: HashMap<ShareableClbit, ShareableClbit> = circuit_data
            .clbits()
            .objects()
            .iter()
            .map(|c| (c.clone(), c.clone()))
            .collect();
        self.symbol_table.push_scope();
        let mut old_scope = std::mem::replace(
            &mut self.circuit_scope,
            BuildScope::with_mappings(circuit_data.clone(), new_qubit_map, new_clbit_map),
        );

        let result = f(self);
        std::mem::swap(&mut self.circuit_scope, &mut old_scope);

        self.symbol_table.pop_scope();

        result
    }

    fn new_context<F>(
        &mut self,
       body: &'a CircuitData,
        f: F
    ) -> ExporterResult<QuantumBlock>
    where
        F: FnOnce(&mut QASM3Builder) -> QuantumBlock,
    {
        let qubit_map :HashMap<ShareableQubit, ShareableQubit> = 
            body
            .qubits()
            .objects()
            .iter()
            .map(|q| (q.clone(), q.clone()))
            .collect();
        let clbit_map: HashMap<ShareableClbit, ShareableClbit> = body
            .clbits()
            .objects()
            .iter()
            .map(|c| (c.clone(), c.clone()))
            .collect();
        let new_table = self.symbol_table.new_context();
        let mut old_symbol_table = std::mem::replace(&mut self.symbol_table, new_table);
        let mut old_scope = std::mem::replace(&mut self.circuit_scope, BuildScope::with_mappings(body.clone(), qubit_map, clbit_map));
        let result = f(self);
        std::mem::swap(&mut self.circuit_scope, &mut old_scope);
        std::mem::swap(&mut self.symbol_table, &mut old_symbol_table);
        self.symbol_table.gates = old_symbol_table.gates;

        Ok(result)
    }

    fn lookup_qubit(&self, qubit: &ShareableQubit) -> ExporterResult<&IdentifierOrSubscripted> {
        let qubit_ref = self.circuit_scope.qubit_map.get(qubit).ok_or_else(|| {
            QASM3ExporterError::Error(format!("Qubit mapping not found for {:?}", qubit))
        })?;
        let id = self.symbol_table.get_qubitinfo(qubit_ref).ok_or_else(|| {
            QASM3ExporterError::Error(format!("Qubit not found: {:?}", qubit))
        })?;
        Ok(id)
    }

    fn lookup_clbit(&self, clbit: &ShareableClbit) -> ExporterResult<&IdentifierOrSubscripted> {
        let clbit_ref = self.circuit_scope.clbit_map.get(clbit).ok_or_else(|| {
            QASM3ExporterError::Error(format!("Clbit mapping not found for {:?}", clbit))
        })?;
        let id = self.symbol_table.get_clbitinfo(clbit_ref).ok_or_else(|| {
            QASM3ExporterError::Error(format!("Clbit not found: {:?}", clbit))
        })?;
        Ok(id)
    }

    pub fn build_program(&mut self) -> ExporterResult<Program> {
        self.register_basis_gates();
        let header = self.build_header();

        self.hoist_global_params()?;
        self.hoist_classical_bits()?;
        let qubit_decls = self.build_qubit_decls()?;
        let main_stmts = self.build_top_level_stmts()?;

        let mut all_stmts = Vec::new();
        for decl in &self.global_io_decls {
            all_stmts.push(Statement::IODeclaration(decl.clone()));
        }
        for gate in self.symbol_table.gates.values() {
            all_stmts.push(Statement::QuantumGateDefinition(gate.clone()));
        }
        for decl in &self.global_classical_decls {
            all_stmts.push(Statement::ClassicalDeclaration(decl.clone()));
        }
        for decl in qubit_decls {
            all_stmts.push(Statement::QuantumDeclaration(decl));
        }
        all_stmts.extend(main_stmts);

        Ok(Program {
            header,
            statements: all_stmts,
        })
    }

    fn register_basis_gates(&mut self) {
        for gate in &self.basis_gates {
            let _ = self.symbol_table.bind(gate);
        }
    }

    fn build_header(&mut self) -> Header {
        let includes = self
            .includes
            .iter()
            .map(|fname| {
                if *fname == "stdgates.inc" {
                    self.symbol_table.add_standard_library_gates();
                }
                Include {
                    filename: fname.to_string(),
                }
            })
            .collect();
        Header {
            version: Some(Version {
                version_number: "3.0".to_string(),
            }),
            includes,
        }
    }

    fn hoist_global_params(&mut self) -> ExporterResult<()> {
        Python::with_gil(|py| {
            for param in self.circuit_scope.circuit_data.get_parameters(py) {
                let raw_name: String = match param.getattr("name") {
                    Ok(attr) => match attr.extract() {
                        Ok(name) => name,
                        Err(err) => return Err(QASM3ExporterError::PyErr(err)),
                    },
                    Err(err) => return Err(QASM3ExporterError::PyErr(err)),
                };
                let identifier = Identifier {
                    string: raw_name.clone(),
                };
                let _ = self.symbol_table.bind(&raw_name);
                self.global_io_decls.push(IODeclaration {
                    modifier: IOModifier::Input,
                    type_: ClassicalType::Float(Float::Double),
                    identifier,
                });
            }
            Ok(())
        })
    }

    fn hoist_classical_bits(&mut self) -> ExporterResult<()> {
        let clbit_indices = self.circuit_scope.circuit_data.clbit_indices();
        let clbits = self.circuit_scope.circuit_data.clbits().objects();
        let has_multiple_registers = clbits.iter().any(
            |clbit| {
                let bit_info = self.circuit_scope.circuit_data.clbit_indices().get(clbit).unwrap();
                bit_info.registers().len() > 1
            }
        );

        if has_multiple_registers {
            return Err(QASM3ExporterError::Error(format!(
                "Some classical registers in this circuit overlap."
            )));
        }
        let mut decls = Vec::new();
        for (i, clbit) in clbits.iter().enumerate() {
            if clbit_indices
                .get(clbit)
                .map_or(true, |bit_info| bit_info.registers().is_empty())
            {
                let raw_name = format!("{}{}", self.loose_bit_prefix, i);
                let identifier = Identifier {
                    string: raw_name.clone(),
                };
                let _ = self.symbol_table.bind(&raw_name);
                self.symbol_table.set_clbitinfo(
                    IdentifierOrSubscripted::Identifier(identifier.clone()),
                    clbit.to_owned()
                );
                decls.push(ClassicalDeclaration {
                    type_: ClassicalType::Bit,
                    identifier,
                });
            }
        }
        for creg in self.circuit_scope.circuit_data.cregs() {
            let raw_name = creg.name();
            let _ = self.symbol_table.bind(&raw_name);
            for (i, clbit) in creg.bits().enumerate() {
                self.symbol_table.set_clbitinfo(
                    IdentifierOrSubscripted::Subscripted(SubscriptedIdentifier {
                        string: raw_name.to_string(),
                        subscript: Box::new(Expression::IntegerLiteral(IntegerLiteral(i as i32))),
                    }),
                    clbit
                )
            }
            let identifier = Identifier {
                string: raw_name.to_string(),
            };
            decls.push(
                ClassicalDeclaration {
                    type_: ClassicalType::BitArray(BitArray(creg.len() as u32)),
                    identifier,
                }
            )

        }
        self.global_classical_decls.extend(decls);
        Ok(())
    }
    

    fn build_qubit_decls(&mut self) -> ExporterResult<Vec<QuantumDeclaration>> {
        let qubit_indices = self.circuit_scope.circuit_data.qubit_indices();
        let qubits = self.circuit_scope.circuit_data.qubits().objects();
        let has_multiple_registers = qubits.iter().any(
            |qubit| {
                let bit_info = self.circuit_scope.circuit_data.qubit_indices().get(qubit).unwrap();
                bit_info.registers().len() > 1
            }
        );

        if has_multiple_registers {
            return Err(QASM3ExporterError::Error(format!(
                "Some quantum registers in this circuit overlap."
            )));
        }
        let mut decls = Vec::new();

        if self.is_layout {
            self.loose_qubit_prefix = "$";
        }
        for (i, qubit) in qubits.iter().enumerate() {
            if qubit_indices
                .get(qubit)
                .map_or(true, |bit_info| bit_info.registers().is_empty())
            {
                let raw_name = format!("{}{}", self.loose_qubit_prefix, i);
                let identifier = Identifier {
                    string: raw_name.clone(),
                };
                let _ = self.symbol_table.bind(&raw_name);
                self.symbol_table.set_qubitinfo(
                    IdentifierOrSubscripted::Identifier(identifier.clone()),
                    qubit.to_owned()
                );
                decls.push(QuantumDeclaration {
                    identifier,
                    designator: None,
                });
            }
        }
        for qreg in self.circuit_scope.circuit_data.qregs() {
            let raw_name = qreg.name();
            self.symbol_table.bind(&raw_name)?;
            for (i, qubit) in qreg.bits().enumerate() {
                self.symbol_table.set_qubitinfo(
                    IdentifierOrSubscripted::Subscripted(SubscriptedIdentifier {
                        string: raw_name.to_string(),
                        subscript: Box::new(Expression::IntegerLiteral(IntegerLiteral(i as i32))),
                    }),
                    qubit
                )
            }
            let identifier = Identifier {
                string: raw_name.to_string(),
            };
            decls.push(
                QuantumDeclaration {
                    identifier,
                    designator: Some(Designator{expression: Expression::IntegerLiteral(IntegerLiteral(qreg.len() as i32))}),
                }
            )

        }
        Ok(decls)
    }

    fn build_top_level_stmts(&mut self) -> ExporterResult<Vec<Statement>> {
        let mut stmts = Vec::new();
        let data = self.circuit_scope.circuit_data.data().to_vec();
        for instr in data {
            self.build_instruction(&instr, &mut stmts)?;
        }
        Ok(stmts)
    }

    fn build_instruction(
        &mut self,
        instruction: &PackedInstruction,
        stmts: &mut Vec<Statement>,
    ) -> ExporterResult<()> {
        let name = instruction.op.name();

        if instruction.op.control_flow() {
            return Err(QASM3ExporterError::Error(format!(
                "Control flow {} is not supported",
                name
            )));
        } else {
            match name {
                "barrier" => self.handle_barrier(instruction, stmts),
                "measure" => self.handle_measure(instruction, stmts),
                "reset" => self.handle_reset(instruction, stmts),
                "delay" => self.handle_delay(instruction, stmts),
                "break_loop" => {
                    stmts.push(Statement::Break(Break {}));
                    Ok(())
                }
                "continue_loop" => {
                    stmts.push(Statement::Continue(Continue {}));
                    Ok(())
                }
                "store" => {
                    panic!("Store is not yet supported");
                }
                _ => {
                    let gate_call = self.build_gate_call(instruction)?;
                    stmts.push(Statement::QuantumInstruction(QuantumInstruction::GateCall(
                        gate_call,
                    )));
                    Ok(())
                }
            }
        }
    }

    fn handle_barrier(
        &mut self,
        instr: &PackedInstruction,
        stmts: &mut Vec<Statement>,
    ) -> ExporterResult<()> {
        let qargs = self
            .circuit_scope
            .circuit_data
            .qargs_interner()
            .get(instr.qubits);
        let mut qubit_ids = Vec::new();
        let qubits_registry = self.circuit_scope.circuit_data.qubits();

        for q in qargs {
            let id = self.lookup_qubit(qubits_registry.get(*q).unwrap())?;
            qubit_ids.push(id.to_owned());
        }
        stmts.push(Statement::QuantumInstruction(QuantumInstruction::Barrier(
            Barrier {
                index_identifier_list: qubit_ids,
            },
        )));
        Ok(())
    }

    fn handle_measure(
        &mut self,
        instr: &PackedInstruction,
        stmts: &mut Vec<Statement>,
    ) -> ExporterResult<()> {
        let qargs = self
            .circuit_scope
            .circuit_data
            .qargs_interner()
            .get(instr.qubits);
        let mut qubits = Vec::new();
        let qubits_registry = self.circuit_scope.circuit_data.qubits();

        for q in qargs {
            let id = self.lookup_qubit(qubits_registry.get(*q).unwrap())?;
            qubits.push(id.to_owned());
        }
        let measurement = QuantumMeasurement {
            identifier_list: qubits,
        };

        let cargs = self
            .circuit_scope
            .circuit_data
            .cargs_interner()
            .get(instr.clbits);
        let clbits_registry = self.circuit_scope.circuit_data.clbits();
        let id = self.lookup_clbit(clbits_registry.get(cargs[0]).unwrap())?;
        stmts.push(Statement::QuantumMeasurementAssignment(
            QuantumMeasurementAssignment {
                identifier: id.to_owned(),
                quantum_measurement: measurement,
            },
        ));
        Ok(())
    }

    fn handle_reset(
        &mut self,
        instr: &PackedInstruction,
        stmts: &mut Vec<Statement>,
    ) -> ExporterResult<()> {
        let qargs = self
            .circuit_scope
            .circuit_data
            .qargs_interner()
            .get(instr.qubits);
        let qubits_registry = self.circuit_scope.circuit_data.qubits();

        for q in qargs {
            let id = self.lookup_qubit(qubits_registry.get(*q).unwrap())?;

            stmts.push(Statement::QuantumInstruction(QuantumInstruction::Reset(
                Reset {
                    identifier: id.to_owned(),
                },
            )));
        }
        Ok(())
    }

    fn handle_delay(
        &self,
        instr: &PackedInstruction,
        stmts: &mut Vec<Statement>,
    ) -> ExporterResult<()> {
        if instr.op.num_clbits() > 0 {
            return Err(QASM3ExporterError::Error(
                "Delay cannot have classical bits".to_string(),
            ));
        }
        let delay = self.build_delay(instr)?;
        stmts.push(Statement::QuantumInstruction(QuantumInstruction::Delay(
            delay,
        )));
        Ok(())
    }

    fn build_delay(&self, instr: &PackedInstruction) -> ExporterResult<Delay> {

        // let duration_value = instr.op.try_into();
        // let duration_value = Python::with_gil(|py| {
        //     let dur = instr.extra_attrs.duration().ok_or_else(|| {
        //         QASM3ExporterError::Error("Failed to extract duration".to_string())
        //     })?;
        //     dur.bind(py).extract::<f64>().map_err(|_| {
        //         QASM3ExporterError::Error("Failed to extract duration value".to_string())
        //     })
        // })?;
        let standard_instr = instr.op.standard_instruction();
        let delay_unit = if let StandardInstruction::Delay(delay) = standard_instr {
            delay
        } else {
            return Err(QASM3ExporterError::Error(
                "Expected Delay instruction, but got wrong instruction".to_string(),
            ));
        };
        let param = &instr.params_view()[0];
        let duration: f64 = Python::with_gil(|py| {
            match param {
                Param::Float(val) => *val,
                Param::ParameterExpression(p) => {
                    let py_obj = p.bind(py);
                    let py_str = py_obj.str().expect("Failed to call str() on Parameter");
                    let name = py_str.str().expect("Failed to convert PyString to &str").to_string();
                    match name.parse::<f64>() {
                        Ok(val) => val,
                        Err(_) => panic!("Failed to parse parameter value"),
                    }
                    // let name = Python::with_gil(|py| {
                    //     let py_obj = p.bind(py);
                    //     let py_str = py_obj.str().expect("Failed to call str() on Parameter");
                    //     py_str.str().expect("Failed to convert PyString to &str").to_string()
                    // });
                },
                Param::Obj(obj) => {
                    let py_obj = obj.bind(py);
                    let py_str = py_obj.str().expect("Failed to call str() on Parameter");
                    let name = py_str.str().expect("Failed to convert PyString to &str").to_string();
                    match name.parse::<f64>() {
                        Ok(val) => val,
                        Err(_) => panic!("Failed to parse parameter value"),
                    }
                    // Expression::Parameter(Parameter {
                    //     obj: py_str.str().expect("Failed to convert PyString to &str").to_string(),
                    // })
                },
            }
            });
        // let name = Python::with_gil(|py| {
        //     let py_obj = p.bind(py);
        //     let py_str = py_obj.str().expect("Failed to call str() on Parameter");
        //     py_str.str().expect("Failed to convert PyString to &str").to_string()
        // });

        let mut map = HashMap::new();
        map.insert(DelayUnit::NS, DurationUnit::Nanosecond);
        map.insert(DelayUnit::US, DurationUnit::Microsecond);
        map.insert(DelayUnit::MS, DurationUnit::Millisecond);
        map.insert(DelayUnit::S, DurationUnit::Second);
        map.insert(DelayUnit::DT, DurationUnit::Sample);

        let duration_literal: DurationLiteral = match map.get(&delay_unit) {
            Some(found) => DurationLiteral {
                value: duration,
                unit: found.clone(),
            },
            None => {
                if delay_unit == DelayUnit::PS {
                    DurationLiteral {
                        value: duration * 1000.0,
                        unit: DurationUnit::Nanosecond,
                    }
                } else {
                    return Err(QASM3ExporterError::Error(format!(
                        "Unknown delay unit: {}",
                        delay_unit
                    )));
                }
            }
        };


        let mut qubits = Vec::new();
        let qargs = self
            .circuit_scope
            .circuit_data
            .qargs_interner()
            .get(instr.qubits);
        let qubits_registry = self.circuit_scope.circuit_data.qubits();

        for q in qargs {
            let id = self.lookup_qubit(qubits_registry.get(*q).unwrap())?;
            qubits.push(id.to_owned());
        }
        Ok(Delay {
            duration: duration_literal,
            qubits,
        })
    }

    fn build_gate_call(&mut self, instr: &PackedInstruction) -> ExporterResult<GateCall> {
        let op_name = instr.op.name();
        if !self.symbol_table.contains_name(op_name) && !self.symbol_table.stdgates.contains(op_name) {
            let _ = self.define_gate(instr);
        }
        let params = if self.disable_constants {
            Python::with_gil(|_py| {
                instr
                    .params_view()
                    .iter()
                    .map(|param| match param {
                        Param::Float(val) => Expression::Parameter(Parameter {
                            obj: val.to_string(),
                        }),
                        Param::ParameterExpression(p) => {
                            let name = Python::with_gil(|py| {
                                let py_obj = p.bind(py);
                                let py_str = py_obj.str().expect("Failed to call str() on Parameter");
                                py_str.str().expect("Failed to convert PyString to &str").to_string()
                            });
                            Expression::Parameter(Parameter { obj: name })
                        },
                        Param::Obj(_) => panic!("Objects not supported yet"),
                    })
                    .collect::<Vec<_>>()
            })
        } else {
            panic!("Constant parameters not supported yet");
        };

        let qargs = self
            .circuit_scope
            .circuit_data
            .qargs_interner()
            .get(instr.qubits);
        let qubits_registry = self.circuit_scope.circuit_data.qubits();
        let mut qubit_ids = Vec::new();
        for q in qargs {
            let id = self.lookup_qubit(qubits_registry.get(*q).unwrap())?;
            qubit_ids.push(id.to_owned());
        }
        Ok(GateCall {
            quantum_gate_name: Identifier {
                string: op_name.to_string(),
            },
            index_identifier_list: qubit_ids,
            parameters: params,
            modifiers: None,
        })
    }

    #[allow(dead_code)]
    fn define_gate(&mut self, instr: &PackedInstruction) -> ExporterResult<()> {
        let operation = &instr.op;
        let params: Vec<Param> = Python::with_gil(|py| {
            let qiskit_circuit = PyModule::import(py, "qiskit.circuit").expect("Failed to import qiskit.circuit");
            let parameter_class = qiskit_circuit.getattr("Parameter").expect("No Parameter class in qiskit.circuit");
        
            (0..instr.params_view().len())
                .map(|i| {
                    let name = format!("{}{}", self._gate_param_prefix, i);
                    let py_param = parameter_class.call1((name,)).expect("Failed to create Parameter");
                    Param::ParameterExpression(py_param.into())
                })
                .collect()
        });
        if let Ok(stdgate) = StandardGate::try_from(operation) {
            if let Some(instruction) = stdgate.definition(&params) {
                let params_def = params.iter().enumerate().map(
                    |(i, p)| {
                        let name = format!("{}{}", self._gate_param_prefix, i);
                        Identifier {
                            string: name.clone(),
                        }
                    }
                ).collect::<Vec<_>>();
                let qubits = (0..instruction.num_qubits())
                .map(|i| {
                    let name = format!("{}{}", self._gate_qubit_prefix, i);
                    Identifier {
                        string: name.clone(),
                    }
                    // if !self.symbol_table.contains_name(&name) {
                        //     let _ = self.symbol_table.bind(&name);
                        // }
                        // identifier
                    })
                    .collect::<Vec<_>>();
                // let qubit_map :HashMap<ShareableQubit, ShareableQubit> = 
                //         instruction.qubits().objects()
                //         .iter()
                //         .cloned()
                //         .zip(instruction.qubits().objects().iter().cloned())
                //         .collect();
                // let clbit_map :HashMap<ShareableClbit, ShareableClbit> = 
                //                         instruction.clbits().objects()
                //                         .iter()
                //                         .cloned()
                //                         .zip(instruction.clbits().objects().iter().cloned())
                //                         .collect();
                // let new_table = self.symbol_table.new_context();
                // let mut old_symbol_table = std::mem::replace(&mut self.symbol_table, new_table);
                // let mut old_scope = std::mem::replace(&mut self.circuit_scope, BuildScope::with_mappings(instruction.clone(), qubit_map, clbit_map));

                // for param in &params_def {
                //     self.symbol_table.bind(&param.string)?;
                // }
                // for (i, q) in instruction.qubits().objects().iter().enumerate() {
                //     let name = format!("{}{}", self._gate_qubit_prefix, i);
                //     let qid = Identifier {
                //         string: name.clone(),
                //     };
                //     let _ = self.symbol_table.bind(&qid.string);
                //     self.symbol_table.set_qubitinfo(
                //         IdentifierOrSubscripted::Identifier(qid.clone()),
                //         q.to_owned(),
                //     );
                // }
                // let mut stmts_tmp = Vec::new();
                // for instr in instruction.data() {
                //     let _ = self.build_instruction(instr, &mut stmts_tmp)?;
                // }
                // let body = QuantumBlock {
                //     statements: stmts_tmp,
                // };
                // // self.circuit_scope = old_scope;
                // std::mem::swap(&mut self.circuit_scope, &mut old_scope);
                // std::mem::swap(&mut self.symbol_table, &mut old_symbol_table);
                // self.symbol_table.gates = old_symbol_table.gates;
                let body = self.new_context(&instruction, |builder| {
                    for param in &params_def {
                        builder.symbol_table.bind(&param.string);
                    }
                    for (i, q) in instruction.qubits().objects().iter().enumerate() {
                        let name = format!("{}{}", builder._gate_qubit_prefix, i);
                        let qid = Identifier {
                            string: name.clone(),
                        };
                        let _ = builder.symbol_table.bind(&qid.string);
                        builder.symbol_table.set_qubitinfo(
                            IdentifierOrSubscripted::Identifier(qid.clone()),
                            q.to_owned(),
                        );
                    }


                    let mut stmts_tmp = Vec::new();
                    for instr in instruction.data() {
                        let _ = builder.build_instruction(instr, &mut stmts_tmp);
                    }
                    QuantumBlock {
                        statements: stmts_tmp,
                    }
                })?;


                let _ = self.symbol_table.register_gate(
                    stdgate,
                    params_def,
                    qubits,
                    body
                );
               Ok(())
            } else {
                return Err(QASM3ExporterError::Error(format!(
                    "Failed to get definition for standard gate: {}",
                    operation.name()
                )));
            }
        } else {
            return Err(QASM3ExporterError::Error(format!(
                "Non StandardGate is not supported: {}",
                operation.name()
            )));
        }
    }
}
