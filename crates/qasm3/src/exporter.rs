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

use std::sync::Arc;

use crate::ast::{
    Alias, Barrier, BitArray, Break, ClassicalDeclaration, ClassicalType, Continue, Delay,
    Designator, DurationLiteral, DurationUnit, Expression, Float, GateCall, Header, IODeclaration,
    IOModifier, Identifier, Include, Index, IndexSet, IntegerLiteral, Node,
    Parameter, Program, QuantumBlock, QuantumDeclaration, QuantumGateDefinition,
    QuantumGateSignature, QuantumInstruction, QuantumMeasurementAssignment,
    Reset, Statement, Version,
};

use crate::error::QASM3ExporterError;
use crate::printer::BasicPrinter;
use crate::symbol_table::SymbolTable;
use crate::circuit_builder::CircuitBuilder;
use hashbrown::{HashMap, HashSet};
use indexmap::IndexMap;
use pyo3::prelude::*;
use pyo3::Python;
use qiskit_circuit::bit::{
    ClassicalRegister, QuantumRegister, Register,
};
use qiskit_circuit::{Clbit, Qubit};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{DelayUnit, StandardInstruction};
use qiskit_circuit::operations::{Operation, Param};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr;
use thiserror::Error;

use lazy_static::lazy_static;
use regex::Regex;

type ExporterResult<T> = Result<T, QASM3ExporterError>;

// These are the prefixes used for the loose qubit and bit names.
pub const BIT_PREFIX: &str = "_bit";
pub const QUBIT_PREFIX: &str = "_qubit";

// These are the prefixes used for the gate parameters and qubits.
pub const GATE_PARAM_PREFIX: &str = "_gate_p";
pub const GATE_QUBIT_PREFIX: &str = "_gate_q";

// These are the gates that are defined by the standard library.
pub const GATES_DEFINED_BY_STDGATES: &[&str] = &[
    "p", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "rx", "ry", "rz", "cx", "cy", "cz",
    "rzz", "cp", "crx", "cry", "crz", "ch", "swap", "ccx", "cswap", "cu", "CX", "phase",
    "cphase", "id", "u1", "u2", "u3",
];

// These are the reserved keywords in QASM3.
pub const RESERVED_KEYWORDS: &[&str] = &[
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
];

lazy_static! {
    pub static ref VALID_IDENTIFIER: Regex = Regex::new(r"(^[\w][\w\d]*$|^\$\d+$)").unwrap();
}

lazy_static! {
    pub static ref _BAD_IDENTIFIER_CHARACTERS: Regex = Regex::new(r"[^\w\d]").unwrap();
}

lazy_static! {
    pub static ref _VALID_HARDWARE_QUBIT: Regex = Regex::new(r"\$\d+").unwrap();
}


#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub enum BitType {
    Qubit(Qubit),
    Clbit(Clbit),
}

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub enum RegisterType {
    QuantumRegister(QuantumRegister),
    ClassicalRegister(ClassicalRegister),
}

impl RegisterType {
    pub fn name(&self) -> &str {
        match self {
            RegisterType::QuantumRegister(q) => q.name(),
            RegisterType::ClassicalRegister(c) => c.name(),
        }
    }

    pub fn bits(&self, circuit_data: &CircuitData) -> Vec<BitType> {
        match self {
            RegisterType::QuantumRegister(quantum_register) => quantum_register
                .bits()
                .filter_map(|shareable_qubit| {
                    circuit_data.qubits().find(&shareable_qubit).map(BitType::Qubit)
                })
                .collect(),
            RegisterType::ClassicalRegister(classical_register) => classical_register
                .bits()
                .filter_map(|shareable_clbit| {
                    circuit_data.clbits().find(&shareable_clbit).map(BitType::Clbit)
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Counter {
    current: usize,
}

impl Counter {
    pub fn new() -> Self {
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

struct BuildScope {
    circuit_data: CircuitData,
    bit_map: HashMap<BitType, BitType>,
}

impl BuildScope {
    fn new(
        circuit_data: &CircuitData,
    ) -> Self {
        let mut bit_map: HashMap<BitType, BitType> = HashMap::new();
        
        // Map all qubits in the circuit
        for i in 0..circuit_data.num_qubits() {
            let qubit = Qubit(i.try_into().unwrap_or_else(|_| panic!("Qubit index {} exceeds u32 range", i)));
            bit_map.insert(
                BitType::Qubit(qubit),
                BitType::Qubit(qubit),
            );
        }

        // Map all clbits in the circuit
        for i in 0..circuit_data.num_clbits() {
            let clbit = Clbit(i.try_into().unwrap_or_else(|_| panic!("Clbit index {} exceeds u32 range", i)));
            bit_map.insert(
                BitType::Clbit(clbit),
                BitType::Clbit(clbit),
            );
        }
        
        Self {
            circuit_data: circuit_data.clone(),
            bit_map,
        }
    }

    fn with_mappings(circuit_data: CircuitData, bit_map: HashMap<BitType, BitType>) -> Self {
        Self {
            circuit_data,
            bit_map,
        }
    }
}


pub struct Exporter {
    includes: Vec<String>,
    basis_gates: Vec<String>,
    disable_constants: bool,
    allow_aliasing: bool,
    indent: String,
}

impl Exporter {
    pub fn new(
        includes: Vec<String>,
        basis_gates: Vec<String>,
        disable_constants: bool,
        allow_aliasing: bool,
        indent: String,
    ) -> Self {
        Self {
            includes,
            basis_gates,
            disable_constants,
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

pub struct QASM3Builder {
    _builtin_instr: HashSet<&'static str>,
    is_layout: bool,
    circuit_scope: BuildScope,
    symbol_table: SymbolTable,
    global_io_decls: Vec<IODeclaration>,
    includes: Vec<String>,
    basis_gates: Vec<String>,
    disable_constants: bool,
    allow_aliasing: bool,
}

impl<'a> QASM3Builder {
    pub fn new(
        circuit_data: &'a CircuitData,
        is_layout: bool,
        includes: Vec<String>,
        basis_gates: Vec<String>,
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
            circuit_scope: BuildScope::new(
                circuit_data,
            ),
            is_layout,
            symbol_table: SymbolTable::new(),
            global_io_decls: Vec::new(),
            includes,
            basis_gates,
            disable_constants,
            allow_aliasing,
        }
    }

    #[allow(dead_code)]
    fn new_scope<F>(
        &mut self,
        circuit_data: &CircuitData,
        qubits: Vec<BitType>,
        clbits: Vec<BitType>,
        f: F,
    ) -> ExporterResult<()>
    where
        F: FnOnce(&mut QASM3Builder) -> ExporterResult<()>,
    {
        let current_bitmap = &self.circuit_scope.bit_map;
        let new_qubits: Vec<BitType> = qubits.iter().map(|q| current_bitmap[q].clone()).collect();
        let new_clbits: Vec<BitType> = clbits.iter().map(|c| current_bitmap[c].clone()).collect();

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

        let mut new_bit_map = HashMap::new();

        for i in 0..circuit_data.num_qubits() {
            let qubit = Qubit(i.try_into().unwrap_or_else(|_| panic!("Qubit index {} exceeds u32 range", i)));
            new_bit_map.insert(
                BitType::Qubit(qubit),
                BitType::Qubit(qubit),
            );
        }
        for i in 0..circuit_data.num_clbits() {
            let clbit = Clbit(i.try_into().unwrap_or_else(|_| panic!("Clbit index {} exceeds u32 range", i)));
            new_bit_map.insert(
                BitType::Clbit(clbit),
                BitType::Clbit(clbit),
            );
        }

        self.symbol_table.push_scope();
        let mut old_scope = std::mem::replace(
            &mut self.circuit_scope,
            BuildScope::with_mappings(circuit_data.clone(), new_bit_map),
        );

        let result = f(self);
        std::mem::swap(&mut self.circuit_scope, &mut old_scope);

        self.symbol_table.pop_scope();

        result
    }

    fn new_context<F>(&mut self, body: &'a CircuitData, f: F) -> ExporterResult<QuantumBlock>
    where
        F: FnOnce(&mut QASM3Builder) -> QuantumBlock,
    {
        let mut bit_map = HashMap::new();

        for i in 0..body.num_qubits() {
            let qubit = Qubit(i.try_into().unwrap_or_else(|_| panic!("Qubit index {} exceeds u32 range", i)));
            bit_map.insert(
                BitType::Qubit(qubit),
                BitType::Qubit(qubit),
            );
        }
        for i in 0..body.num_clbits() {
            let clbit = Clbit(i.try_into().unwrap_or_else(|_| panic!("Clbit index {} exceeds u32 range", i)));
            bit_map.insert(
                BitType::Clbit(clbit),
                BitType::Clbit(clbit),
            );
        }

        let new_table = self.symbol_table.new_context();
        let mut old_symbol_table = std::mem::replace(&mut self.symbol_table, new_table);
        let mut old_scope = std::mem::replace(
            &mut self.circuit_scope,
            BuildScope::with_mappings(body.clone(), bit_map),
        );
        let result = f(self);
        std::mem::swap(&mut self.circuit_scope, &mut old_scope);
        std::mem::swap(&mut self.symbol_table, &mut old_symbol_table);
        self.symbol_table.gates = old_symbol_table.gates;

        Ok(result)
    }

    fn lookup_bit(&self, bit: &BitType) -> ExporterResult<&Expression> {
        let qubit_ref = self.circuit_scope.bit_map.get(bit).ok_or_else(|| {
            QASM3ExporterError::Error(format!("Bit mapping not found for {bit:?}"))
        })?;
        let id = self
            .symbol_table
            .get_bitinfo(qubit_ref)
            .ok_or_else(|| QASM3ExporterError::Error(format!("Bit not found: {:?}, qubit_ref: {:?}", bit, qubit_ref)))?;
        Ok(id)
    }

    pub fn build_program(&mut self) -> ExporterResult<Program> {
        self.register_basis_gates();
        let header = self.build_header();

        self.hoist_global_params()?;
        
        let mut circuit_builder = CircuitBuilder::new(
            &self.circuit_scope.circuit_data,
            &mut self.symbol_table,
            self.is_layout,
            self.allow_aliasing,
        );
        let classical_decls = circuit_builder.build_classical_declarations()?;
        let qubit_decls = circuit_builder.build_qubit_declarations()?;
        
        let main_stmts = self.build_top_level_stmts()?;

        let mut all_stmts = Vec::new();
        for decl in &self.global_io_decls {
            all_stmts.push(Statement::IODeclaration(decl.clone()));
        }
        for gate in self.symbol_table.gates.values() {
            all_stmts.push(Statement::QuantumGateDefinition(gate.clone()));
        }
        for decl in classical_decls {
            all_stmts.push(decl);
        }
        for decl in qubit_decls {
            all_stmts.push(decl);
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
            for param in self.circuit_scope.circuit_data.get_parameters(py)? {
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
            Err(QASM3ExporterError::Error(format!(
                "Control flow {name} is not supported"
            )))
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
                    return Err(QASM3ExporterError::Error("Store is not yet supported".to_string()));
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

        for q in qargs {
            let id = self.lookup_bit(&BitType::Qubit(*q))?;
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

        for q in qargs {
            let id = self.lookup_bit(&BitType::Qubit(*q))?;
            qubits.push(id.to_owned());
        }

        let cargs = self
            .circuit_scope
            .circuit_data
            .cargs_interner()
            .get(instr.clbits);
        let id = self.lookup_bit(&BitType::Clbit(cargs[0]))?;
        stmts.push(Statement::QuantumMeasurementAssignment(
            QuantumMeasurementAssignment {
                target: id.to_owned(),
                qubits,
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

        for q in qargs {
            let id = self.lookup_bit(&BitType::Qubit(*q))?;

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
        let standard_instr = instr.op.standard_instruction();
        let delay_unit = if let StandardInstruction::Delay(delay) = standard_instr {
            delay
        } else {
            return Err(QASM3ExporterError::Error(
                "Expected Delay instruction, but got wrong instruction".to_string(),
            ));
        };
        let param = &instr.params_view()[0];
        let duration: f64 = Python::with_gil(|py| match param {
            Param::Float(val) => *val,
            Param::ParameterExpression(p) => {
                if let Ok(symbol_expr::Value::Real(val)) = p.try_to_value(true) {
                    val
                } else {
                    panic!("Failed to parse parameter value")
                }
                Param::Obj(obj) => {
                    let py_obj = obj.bind(py);
                    let py_str = py_obj.str().map_err(|e| QASM3ExporterError::Error(format!("Failed to call str() on Parameter: {}", e)))?;
                    let name = py_str
                        .str()
                        .map_err(|e| QASM3ExporterError::Error(format!("Failed to convert PyString to &str: {}", e)))?
                        .to_string();
                    name.parse::<f64>().map_err(|e| QASM3ExporterError::Error(format!("Failed to parse parameter value: {}", e)))?
                }
            })
        })?;

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
                        "Unknown delay unit: {delay_unit}"
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

        for q in qargs {
            let id = self.lookup_bit(&BitType::Qubit(*q))?;
            qubits.push(id.to_owned());
        }
        Ok(Delay {
            duration: duration_literal,
            qubits,
        })
    }

    fn build_gate_call(&mut self, instr: &PackedInstruction) -> ExporterResult<GateCall> {
        let mut op_name = instr.op.name();
        if op_name == "u" {
            op_name = "U";
        }
        if !self.symbol_table.contains_name(op_name)
            && !self.symbol_table.stdgates.contains(op_name)
        {
            self.define_gate(instr)?;
        }
        let params = if self.disable_constants {
            Python::with_gil(|_py| -> Result<Vec<Expression>, QASM3ExporterError> {
                instr
                    .params_view()
                    .iter()
                    .map(|param| match param {
                        Param::Float(val) => Ok(Expression::Parameter(Parameter {
                            obj: val.to_string(),
                        })),
                        Param::ParameterExpression(p) => {
                            let name = p.to_string();
                            Expression::Parameter(Parameter { obj: name })
                        }
                        Param::Obj(_) => Err(QASM3ExporterError::Error("Objects not supported yet".to_string())),
                    })
                    .collect::<Result<Vec<_>, _>>()
            })?
        } else {
            return Err(QASM3ExporterError::Error(
                "Constant parameters not supported yet".to_string(),
            ));
        };

        let qargs = self
            .circuit_scope
            .circuit_data
            .qargs_interner()
            .get(instr.qubits);
        let mut qubit_ids = Vec::new();
        for q in qargs {
            let id = self.lookup_bit(&BitType::Qubit(*q))?;
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
        let params: Vec<Param> = (0..instr.params_view().len())
            .map(|i| {
                let name = format!("{}_{}", self._gate_param_prefix, i);
                // TODO this need to be achievable more easily
                let symbol = symbol_expr::Symbol::new(name.as_str(), None, None);
                let symbol_expr = symbol_expr::SymbolExpr::Symbol(Arc::new(symbol));
                let expr = ParameterExpression::from_symbol_expr(symbol_expr);
                Param::ParameterExpression(Arc::new(expr))
            })
            .collect();
        if let Some(instruction) = operation.definition(&params) {
            let params_def = params
                .iter()
                .enumerate()
                .map(|(i, _p)| {
                    let name = format!("{}_{}", GATE_PARAM_PREFIX, i);
                    Identifier {
                        string: name.clone(),
                    }
                })
                .collect::<Vec<_>>();
            let qubits = (0..instruction.num_qubits())
                .map(|i| {
                    let name = format!("{}_{}", GATE_QUBIT_PREFIX, i);
                    Identifier {
                        string: name.clone(),
                    }
                })
                .collect::<Vec<_>>();

            let body = self.new_context(&instruction, |builder| {
                for param in &params_def {
                    let _ = builder.symbol_table.bind(&param.string);
                }
                for (i, _q) in instruction.qubits().objects().iter().enumerate() {
                    let name = format!("{}_{}", GATE_QUBIT_PREFIX, i);
                    let qid = Identifier {
                        string: name.clone(),
                    };
                    let _ = builder.symbol_table.bind(&qid.string);
                    builder.symbol_table.set_bitinfo(
                        Expression::Parameter(Parameter {
                            obj: qid.string.clone(),
                        }),
                        BitType::Qubit(Qubit(i as u32)),
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
                operation.name().to_string(),
                params_def,
                qubits,
                body,
            );
            Ok(())
        } else {
            Err(QASM3ExporterError::Error(format!(
                "Failed to get definition for this gate: {}",
                operation.name()
            )))
        }
    }
}
