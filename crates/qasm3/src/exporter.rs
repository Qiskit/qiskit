use crate::ast::{
    Barrier, Break, ClassicalDeclaration, ClassicalType, Constant, Continue, Delay,
    DurationLiteral, DurationUnit, Expression, Float, GateCall, Header, IODeclaration, IOModifier,
    Identifier, Include, IntegerLiteral, Node, Parameter, Program, QuantumBlock,
    QuantumDeclaration, QuantumGateDefinition, QuantumGateModifier, QuantumGateModifierName,
    QuantumGateSignature, QuantumInstruction, QuantumMeasurement, QuantumMeasurementAssignment,
    Reset, Statement, Version,
};
use crate::printer::BasicPrinter;
use crate::symbols::SymbolTable;
use hashbrown::{HashMap, HashSet};
use oq3_semantics::types::{IsConst, Type};
use pyo3::prelude::*;
use pyo3::Python;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Operation, Param};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::{Clbit, Qubit};
use std::sync::Mutex;
use thiserror::Error;

use lazy_static::lazy_static;
use regex::Regex;

type ExporterResult<T> = std::result::Result<T, QASM3ExporterError>;

#[derive(Error, Debug)]
pub enum QASM3ExporterError {
    #[error("Error: {0}")]
    Error(String),
    #[error("Symbol '{0}' is not found in the table")]
    SymbolNotFound(String),
    #[error("Failed to rename symbol: {0}")]
    RenameFailed(String),
    #[error("PyError: {0}")]
    PyErr(PyErr),
    #[error("Not in global scope")]
    NotInGlobalScopeError,
}

impl From<PyErr> for QASM3ExporterError {
    fn from(err: PyErr) -> Self {
        QASM3ExporterError::PyErr(err)
    }
}

lazy_static! {
    static ref GLOBAL_COUNTER: Mutex<usize> = Mutex::new(0);
}

fn get_next_counter_value() -> usize {
    let mut counter = GLOBAL_COUNTER.lock().unwrap();
    let value = *counter;
    *counter += 1;
    value
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

pub struct Exporter {
    includes: Vec<&'static str>,
    basis_gates: Vec<&'static str>,
    disable_constants: bool,
    alias_classical_registers: bool,
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
        Exporter {
            includes,
            basis_gates,
            disable_constants,
            alias_classical_registers,
            allow_aliasing,
            indent,
        }
    }

    pub fn dump(
        &self,
        circuit_data: &CircuitData,
        islayout: bool,
    ) -> String {
        let mut stream = String::new();
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
                BasicPrinter::new(&mut stream, self.indent.to_string(), false)
                    .visit(&Node::Program(&program));
                stream
            }
            Err(e) => format!("Error: {:?}", e),
        }
    }
}
#[derive(Debug, Clone)]
struct BuildScope<'a> {
    circuit_data: &'a CircuitData,
    qubit_map: HashMap<Qubit, Qubit>,
    clbit_map: HashMap<Clbit, Clbit>,
}

impl<'a> BuildScope<'a> {
    fn new(circuit_data: &'a CircuitData) -> Self {
        let qubit_map: HashMap<Qubit, Qubit> = Python::with_gil(|py| {
            let qubits = circuit_data.qubits();
            qubits
                .bits()
                .iter()
                .map(|bit| {
                    let bound_bit = bit.bind(py);
                    let qubit = qubits.find(bound_bit).unwrap_or_else(|| {
                        panic!("Qubit not found");
                    });
                    (qubit, qubit)
                })
                .collect()
        });

        let clbit_map: HashMap<Clbit, Clbit> = Python::with_gil(|py| {
            let clbits = circuit_data.clbits();
            clbits
                .bits()
                .iter()
                .map(|bit| {
                    let bound_bit = bit.bind(py);
                    let clbit = clbits.find(bound_bit).unwrap_or_else(|| {
                        panic!("Clbit not found");
                    });
                    (clbit, clbit)
                })
                .collect()
        });

        BuildScope {
            circuit_data,
            qubit_map,
            clbit_map,
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
        let next_value = self.current;
        self.current = self.current.wrapping_add(1);
        Some(next_value)
    }
}

pub struct QASM3Builder<'a> {
    buildins: HashSet<&'static str>,
    loose_bit_prefix: &'static str,
    loose_qubit_prefix: &'static str,
    gate_parameter_prefix: &'static str,
    gate_qubit_prefix: &'static str,
    circuit_scope: BuildScope<'a>,
    islayout: bool,
    symbol_table: SymbolTable,
    global_io_declarations: Vec<IODeclaration>,
    global_classical_forward_declarations: Vec<ClassicalDeclaration>,
    includes: Vec<&'static str>,
    basis_gates: Vec<&'static str>,
    disable_constants: bool,
    allow_aliasing: bool,
    counter: Counter,
}

impl<'a> QASM3Builder<'a> {
    pub fn new(
        circuit_data: &'a CircuitData,
        islayout: bool,
        includes: Vec<&'static str>,
        basis_gates: Vec<&'static str>,
        disable_constants: bool,
        allow_aliasing: bool,
    ) -> Self {
        QASM3Builder {
            buildins: [
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
            loose_bit_prefix: "_bit",
            loose_qubit_prefix: "_qubit",
            gate_parameter_prefix: "_gate_p",
            gate_qubit_prefix: "_gate_q",
            circuit_scope: BuildScope::new(circuit_data),
            islayout,
            symbol_table: SymbolTable::new(),
            global_io_declarations: Vec::new(),
            global_classical_forward_declarations: Vec::new(),
            includes,
            basis_gates,
            disable_constants,
            allow_aliasing,
            counter: Counter::new(),
        }
    }

    #[allow(dead_code)]
    fn define_gate(
        &mut self,
        instruction: &PackedInstruction,
    ) -> ExporterResult<QuantumGateDefinition> {
        let operation = &instruction.op;
        if operation.name() == "cx" && operation.num_qubits() == 2 {
            let (control, target) = (
                Identifier {
                    string: "c".to_string(),
                },
                Identifier {
                    string: "t".to_string(),
                },
            );
            let call = GateCall {
                quantum_gate_name: Identifier {
                    string: "U".to_string(),
                },
                index_identifier_list: vec![control.clone(), target.clone()],
                parameters: vec![
                    Expression::Constant(Constant::PI),
                    Expression::IntegerLiteral(IntegerLiteral { value: 0 }),
                    Expression::Constant(Constant::PI),
                ],
                modifiers: Some(vec![QuantumGateModifier {
                    modifier: QuantumGateModifierName::Ctrl,
                    argument: None,
                }]),
            };

            Ok(QuantumGateDefinition {
                quantum_gate_signature: QuantumGateSignature {
                    name: Identifier {
                        string: "cx".to_string(),
                    },
                    qarg_list: vec![control, target],
                    params: None,
                },
                quantum_block: QuantumBlock {
                    statements: vec![Statement::QuantumInstruction(QuantumInstruction::GateCall(
                        call,
                    ))],
                },
            })
        } else {
            let qubit_map: HashMap<Qubit, Qubit> = Python::with_gil(|py| {
                let qubits = self.circuit_scope.circuit_data.qubits();
                qubits
                    .bits()
                    .iter()
                    .map(|bit| {
                        let bound_bit = bit.bind(py);
                        let qubit = qubits.find(bound_bit).unwrap_or_else(|| {
                            panic!("Qubit not found");
                        });
                        (qubit, qubit)
                    })
                    .collect()
            });

            let clbit_map: HashMap<Clbit, Clbit> = Python::with_gil(|py| {
                let clbits = self.circuit_scope.circuit_data.clbits();
                clbits
                    .bits()
                    .iter()
                    .map(|bit| {
                        let bound_bit = bit.bind(py);
                        let clbit = clbits.find(bound_bit).unwrap_or_else(|| {
                            panic!("Clbit not found");
                        });
                        (clbit, clbit)
                    })
                    .collect()
            });

            let operation_def_owned = operation.definition(instruction.params_view()).ok_or(
                QASM3ExporterError::Error("Failed to get operation definition".to_string()),
            )?;

            let body = {
                let tmp_scope = BuildScope {
                    circuit_data: unsafe { std::mem::transmute::<&_, &'a _>(&operation_def_owned) },
                    qubit_map,
                    clbit_map,
                };

                let old_scope = std::mem::replace(&mut self.circuit_scope, tmp_scope);
                let body = QuantumBlock {
                    statements: self.build_current_scope()?,
                };
                self.circuit_scope = old_scope;
                body
            };

            let defn = &operation_def_owned;
            let params: Vec<Expression> = (0..defn.num_parameters())
                .map(|i| {
                    let name = format!("{}_{}", self.gate_parameter_prefix, i);
                    let symbol = if self
                        .symbol_table
                        .new_binding(&name, &Type::Float(Some(64), IsConst::False))
                        .is_ok()
                    {
                        self.symbol_table
                            .lookup(&name)
                            .map_err(|_| QASM3ExporterError::SymbolNotFound(name))?
                    } else {
                        let rename =
                            escape_invalid_identifier(&self.symbol_table, &name, true, false);
                        if self
                            .symbol_table
                            .new_binding(&rename, &Type::Float(Some(64), IsConst::False))
                            .is_err()
                        {
                            return Err(QASM3ExporterError::RenameFailed(name));
                        }

                        self.symbol_table
                            .lookup(&rename)
                            .map_err(|_| QASM3ExporterError::SymbolNotFound(name))?
                    };
                    Ok(Expression::Parameter(Parameter {
                        obj: symbol.symbol().name().to_string(),
                    }))
                })
                .collect::<ExporterResult<Vec<Expression>>>()?;

            let qubits: Vec<Identifier> = (0..defn.num_parameters())
                .map(|i| {
                    let name = format!("{}_{}", self.gate_qubit_prefix, i);
                    let symbol = if self.symbol_table.new_binding(&name, &Type::Qubit).is_ok() {
                        self.symbol_table
                            .lookup(&name)
                            .map_err(|_| QASM3ExporterError::SymbolNotFound(name))?
                    } else {
                        let rename =
                            escape_invalid_identifier(&self.symbol_table, &name, true, false);
                        if self
                            .symbol_table
                            .new_binding(&rename, &Type::Float(Some(64), IsConst::False))
                            .is_err()
                        {
                            return Err(QASM3ExporterError::RenameFailed(name));
                        }

                        self.symbol_table
                            .lookup(&rename)
                            .map_err(|_| QASM3ExporterError::SymbolNotFound(name))?
                    };

                    Ok(Identifier {
                        string: symbol.symbol().name().to_string(),
                    })
                })
                .collect::<ExporterResult<Vec<Identifier>>>()?;
            if self
                .symbol_table
                .new_binding(
                    operation.name(),
                    &Type::Gate(
                        operation.num_clbits() as usize,
                        operation.num_qubits() as usize,
                    ),
                )
                .is_ok()
            {}
            Ok(QuantumGateDefinition {
                quantum_gate_signature: QuantumGateSignature {
                    name: Identifier {
                        string: operation.name().to_string(),
                    },
                    qarg_list: qubits,
                    params: Some(params),
                },
                quantum_block: body,
            })
        }
    }

    pub fn build_program(&mut self) -> ExporterResult<Program> {
        for builtin in self.basis_gates.iter() {
            let _ = self.symbol_table.new_binding(builtin, &Type::Gate(0, 1));
        }
        let header = Header {
            version: Some(Version {
                version_number: "3.0".to_string(),
            }),
            includes: self.build_includes(),
        };
        let _ = self.hoist_global_parameter_declaration();
        let _ = self.hoist_classical_register_declarations();
        let qubit_declarations = self.build_quantum_declarations()?;

        let main_statements = self.build_current_scope()?;

        let mut statements = Vec::new();
        for global_io_declaration in self.global_io_declarations.iter() {
            statements.push(Statement::IODeclaration(global_io_declaration.to_owned()));
        }
        for global_classical_forward_declaration in
            self.global_classical_forward_declarations.iter()
        {
            statements.push(Statement::ClassicalDeclaration(
                global_classical_forward_declaration.to_owned(),
            ));
        }
        for qubit_declaration in qubit_declarations.into_iter() {
            statements.push(Statement::QuantumDeclaration(qubit_declaration));
        }
        statements.extend(main_statements);
        Ok(Program { header, statements })
    }

    fn build_includes(&mut self) -> Vec<Include> {
        let mut includes = Vec::new();
        for filename in &self.includes {
            if *filename == "stdgates.inc" {
                self.symbol_table.standard_library_gates();
            }
            includes.push(Include {
                filename: filename.to_string(),
            });
        }
        includes
    }

    fn assert_global_scope(&self) -> ExporterResult<()> {
        if !self.symbol_table.in_global_scope() {
            drop(QASM3ExporterError::NotInGlobalScopeError);
        }
        Ok(())
    }

    fn hoist_global_parameter_declaration(&mut self) -> ExporterResult<()> {
        let _ = self.assert_global_scope();
        let circuit_data = self.circuit_scope.circuit_data;

        Python::with_gil(|py| {
            for parameter in circuit_data.get_parameters(py) {
                let name: String = parameter.getattr("name")?.extract()?;

                let parameter = if self
                    .symbol_table
                    .new_binding(&name, &Type::Float(Some(64), IsConst::False))
                    .is_ok()
                {
                    self.symbol_table
                        .lookup(&name)
                        .map_err(|_| QASM3ExporterError::SymbolNotFound(name))
                } else {
                    let rename = escape_invalid_identifier(&self.symbol_table, &name, true, false);
                    if self
                        .symbol_table
                        .new_binding(&rename, &Type::Float(Some(64), IsConst::False))
                        .is_err()
                    {
                        return Err(QASM3ExporterError::RenameFailed(name));
                    }

                    self.symbol_table
                        .lookup(&rename)
                        .map_err(|_| QASM3ExporterError::SymbolNotFound(name))
                };

                self.global_io_declarations.push(IODeclaration {
                    modifier: IOModifier::Input,
                    type_: ClassicalType::Float(Float::Double),
                    identifier: Identifier {
                        string: parameter?.symbol().name().to_string(),
                    },
                });
            }
            Ok(())
        })
    }

    fn build_instruction(
        &mut self,
        instruction: &PackedInstruction,
        statements: &mut Vec<Statement>,
    ) -> ExporterResult<()> {
        let op_name = instruction.op.name();
        if instruction.op.control_flow() {
            if op_name == "for_loop" {
                return Err(QASM3ExporterError::Error(format!(
                    "Unsupported Python interface of control flow condition: {}",
                    op_name
                )));
            } else if op_name == "while_loop" {
                return Err(QASM3ExporterError::Error(format!(
                    "Unsupported Python interface of control flow condition: {}",
                    op_name
                )));
            } else if op_name == "if_else" {
                return Err(QASM3ExporterError::Error(format!(
                    "Unsupported Python interface of control flow condition: {}",
                    op_name
                )));
            } else if op_name == "switch_case" {
                return Err(QASM3ExporterError::Error(format!(
                    "Unsupported Python interface of control flow condition: {}",
                    op_name
                )));
            } else {
                return Err(QASM3ExporterError::Error(format!(
                    "Unsupported Python interface of control flow condition: {}",
                    op_name
                )));
            }
        } else {
            let qubits = self
                .circuit_scope
                .circuit_data
                .qargs_interner()
                .get(instruction.qubits);
            if instruction.op.name() == "barrier" {
                let mut barrier_qubits = Vec::new();
                for qubit in qubits {
                    let name = format!("{}{}", self.loose_qubit_prefix, qubit.0);
                    let qubit_symbol = self
                        .symbol_table
                        .lookup(&name)
                        .map_err(|_| QASM3ExporterError::SymbolNotFound(name))?;
                    barrier_qubits.push(Identifier {
                        string: qubit_symbol.symbol().name().to_string(),
                    });
                }
                statements.push(Statement::QuantumInstruction(QuantumInstruction::Barrier(
                    Barrier {
                        index_identifier_list: barrier_qubits,
                    },
                )));
            } else if instruction.op.name() == "measure" {
                let mut measured_qubits = Vec::new();
                for qubit in qubits {
                    let name = format!("{}{}", self.loose_qubit_prefix, qubit.0);
                    let qubit_symbol = self
                        .symbol_table
                        .lookup(&name)
                        .map_err(|_| QASM3ExporterError::SymbolNotFound(name))?;
                    measured_qubits.push(Identifier {
                        string: qubit_symbol.symbol().name().to_string(),
                    });
                }
                let measurement = QuantumMeasurement {
                    identifier_list: measured_qubits,
                };
                let clbits = self
                    .circuit_scope
                    .circuit_data
                    .cargs_interner()
                    .get(instruction.clbits);
                let name = format!("{}{}", self.loose_bit_prefix, clbits[0].0);
                let clbit_symbol = self
                    .symbol_table
                    .lookup(&name)
                    .map_err(|_| QASM3ExporterError::SymbolNotFound(name))?;
                statements.push(Statement::QuantumMeasurementAssignment(
                    QuantumMeasurementAssignment {
                        identifier: Identifier {
                            string: clbit_symbol.symbol().name().to_string(),
                        },
                        quantum_measurement: measurement,
                    },
                ));
            } else if instruction.op.name() == "reset" {
                for qubit in qubits {
                    let name = format!("{}{}", self.loose_qubit_prefix, qubit.0);
                    let qubit_symbol = self
                        .symbol_table
                        .lookup(&name)
                        .map_err(|_| QASM3ExporterError::SymbolNotFound(name))?;
                    statements.push(Statement::QuantumInstruction(QuantumInstruction::Reset(
                        Reset {
                            identifier: Identifier {
                                string: qubit_symbol.symbol().name().to_string(),
                            },
                        },
                    )));
                }
            } else if instruction.op.name() == "delay" {
                let delay = self.build_delay(instruction)?;
                statements.push(Statement::QuantumInstruction(QuantumInstruction::Delay(
                    delay,
                )));
            } else if instruction.op.name() == "break_loop" {
                statements.push(Statement::Break(Break {}));
            } else if instruction.op.name() == "continue_loop" {
                statements.push(Statement::Continue(Continue {}));
            } else if instruction.op.name() == "store" {
                panic!("Store is not yet supported");
            } else {
                statements.push(Statement::QuantumInstruction(QuantumInstruction::GateCall(
                    self.build_gate_call(instruction)?,
                )));
            }
        }
        Ok(())
    }

    fn build_gate_call(&mut self, instruction: &PackedInstruction) -> ExporterResult<GateCall> {
        let gate_identifier = if self
            .symbol_table
            .standard_library_gates()
            .contains(&instruction.op.name())
        {
            Identifier {
                string: instruction.op.name().to_string(),
            }
        } else {
            panic!(
                "Non-standard gate calls are not yet supported, but received: {}",
                instruction.op.name()
            );
        };
        let qubits = self
            .circuit_scope
            .circuit_data
            .qargs_interner()
            .get(instruction.qubits);
        let parameters: Vec<Expression> = if self.disable_constants {
            if instruction.params_view().is_empty() {
                Vec::<Expression>::new()
            } else {
                instruction
                    .params_view()
                    .iter()
                    .map(|param| match param {
                        Param::Float(value) => Expression::Parameter(Parameter {
                            obj: value.to_string(),
                        }),
                        Param::ParameterExpression(_) => {
                            panic!("Parameter expressions are not yet supported")
                        }
                        Param::Obj(_) => panic!("Objects are not yet supported"),
                    })
                    .collect()
            }
        } else {
            panic!("'disable_constant = true' are not yet supported");
        };

        let index_identifier_list: Vec<Identifier> = qubits
            .iter()
            .map(|qubit| {
                let name = format!("{}{}", self.loose_qubit_prefix, qubit.0);
                self.symbol_table
                    .lookup(&name)
                    .map_err(|_| QASM3ExporterError::SymbolNotFound(name))
                    .map(|qubit_symbol| Identifier {
                        string: qubit_symbol.symbol().name().to_string(),
                    })
            })
            .collect::<ExporterResult<Vec<Identifier>>>()?;

        Ok(GateCall {
            quantum_gate_name: gate_identifier,
            index_identifier_list,
            parameters,
            modifiers: None,
        })
    }

    fn hoist_classical_register_declarations(&mut self) -> ExporterResult<()> {
        let _ = self.assert_global_scope();
        let circuit_data = self.circuit_scope.circuit_data;
        let mut classical_declarations: Vec<ClassicalDeclaration> = Vec::new();
        for i in 0..circuit_data.num_clbits() {
            let name = format!("{}{}", self.loose_bit_prefix, i);
            let clbit = if self
                .symbol_table
                .new_binding(&name, &Type::Bit(IsConst::False))
                .is_ok()
            {
                self.symbol_table
                    .lookup(&name)
                    .map_err(|_| QASM3ExporterError::SymbolNotFound(name))?
            } else {
                let rename = escape_invalid_identifier(&self.symbol_table, &name, true, false);
                if self
                    .symbol_table
                    .new_binding(&rename, &Type::Float(Some(64), IsConst::False))
                    .is_err()
                {
                    return Err(QASM3ExporterError::RenameFailed(rename));
                }

                self.symbol_table
                    .lookup(&rename)
                    .map_err(|_| QASM3ExporterError::SymbolNotFound(name))?
            };

            classical_declarations.push(ClassicalDeclaration {
                type_: ClassicalType::Bit,
                identifier: Identifier {
                    string: clbit.symbol().name().to_string(),
                },
            });
        }
        self.global_classical_forward_declarations
            .extend(classical_declarations);
        Ok(())
    }

    fn build_quantum_declarations(&mut self) -> ExporterResult<Vec<QuantumDeclaration>> {
        let _ = self.assert_global_scope();
        let circuit_data = self.circuit_scope.circuit_data;
        let mut qubit_declarations: Vec<QuantumDeclaration> = Vec::new();
        if self.islayout {
            self.loose_qubit_prefix = "$";
            for i in 0..circuit_data.num_qubits() {
                let name = format!("{}{}", self.loose_qubit_prefix, i);
                let qubit = if self.symbol_table.new_binding(&name, &Type::Qubit).is_ok() {
                    self.symbol_table
                        .lookup(&name)
                        .map_err(|_| QASM3ExporterError::SymbolNotFound(name))?
                } else {
                    return Err(QASM3ExporterError::SymbolNotFound(name));
                };

                qubit_declarations.push(QuantumDeclaration {
                    identifier: Identifier {
                        string: qubit.symbol().name().to_string(),
                    },
                    designator: None,
                });
            }
            return Ok(qubit_declarations);
        }
        for i in 0..circuit_data.num_qubits() {
            let name = format!("{}{}", self.loose_qubit_prefix, i);
            let qubit = if self.symbol_table.new_binding(&name, &Type::Qubit).is_ok() {
                self.symbol_table
                    .lookup(&name)
                    .map_err(|_| QASM3ExporterError::SymbolNotFound(name))?
            } else {
                let rename = escape_invalid_identifier(&self.symbol_table, &name, true, false);
                if self
                    .symbol_table
                    .new_binding(&rename, &Type::Float(Some(64), IsConst::False))
                    .is_err()
                {
                    return Err(QASM3ExporterError::RenameFailed(rename));
                }

                self.symbol_table
                    .lookup(&rename)
                    .map_err(|_| QASM3ExporterError::SymbolNotFound(name))?
            };

            qubit_declarations.push(QuantumDeclaration {
                identifier: Identifier {
                    string: qubit.symbol().name().to_string(),
                },
                designator: None,
            });
        }
        Ok(qubit_declarations)
    }

    fn build_current_scope(&mut self) -> ExporterResult<Vec<Statement>> {
        let mut statements: Vec<Statement> = Vec::<Statement>::new();
        for instruction in self.circuit_scope.circuit_data.data() {
            self.build_instruction(instruction, &mut statements)?;
        }
        Ok(statements)
    }

    fn build_delay(&self, instruction: &PackedInstruction) -> ExporterResult<Delay> {
        if instruction.op.num_clbits() > 0 {
            return Err(QASM3ExporterError::Error(
                "Delay instruction cannot have classical bits".to_string(),
            ));
        }

        let duration_value = Python::with_gil(|py| {
            if let Some(duration) = instruction.extra_attrs.duration() {
                match duration.bind(py).extract::<f64>() {
                    Ok(value) => Ok(value),
                    Err(_) => Err(QASM3ExporterError::Error(
                        "Failed to extract duration value".to_string(),
                    )),
                }
            } else {
                Err(QASM3ExporterError::Error(
                    "Failed to extract duration value".to_string(),
                ))
            }
        })?;
        let default_duration_unit = "us";
        let duration_unit = instruction
            .extra_attrs
            .unit()
            .unwrap_or(default_duration_unit);

        let duration_literal = if duration_unit == "ps" {
            DurationLiteral {
                value: duration_value * 1000.0,
                unit: DurationUnit::Nanosecond,
            }
        } else {
            let unit_map: HashMap<&str, DurationUnit> = HashMap::from([
                ("ns", DurationUnit::Nanosecond),
                ("us", DurationUnit::Microsecond),
                ("ms", DurationUnit::Millisecond),
                ("s", DurationUnit::Second),
                ("dt", DurationUnit::Sample),
            ]);

            match unit_map.get(duration_unit) {
                Some(mapped_unit) => DurationLiteral {
                    value: duration_value,
                    unit: mapped_unit.clone(),
                },
                None => {
                    return Err(QASM3ExporterError::Error(format!(
                        "Unknown duration unit: {}",
                        duration_unit
                    )))
                }
            }
        };

        let mut qubits = Vec::new();

        for qubit in self
            .circuit_scope
            .circuit_data
            .qargs_interner()
            .get(instruction.qubits)
        {
            let name = format!("{}{}", self.loose_qubit_prefix, qubit.0);
            let qubit_symbol = self
                .symbol_table
                .lookup(&name)
                .map_err(|_| QASM3ExporterError::SymbolNotFound(name))?;
            qubits.push(Identifier {
                string: qubit_symbol.symbol().name().to_string(),
            });
        }

        Ok(Delay {
            duration: duration_literal,
            qubits,
        })
    }
}

fn name_allowed(symbol_table: &SymbolTable, name: &str, unique: bool) -> bool {
    if unique {
        RESERVED_KEYWORDS.contains(name) || symbol_table.all_scopes_contains_name(name)
    } else {
        RESERVED_KEYWORDS.contains(name)
    }
}

fn escape_invalid_identifier(
    symbol_table: &SymbolTable,
    name: &str,
    allow_rename: bool,
    unique: bool,
) -> String {
    let base = if allow_rename {
        format!(
            "_{}",
            name.chars()
                .map(|c| if c.is_alphanumeric() || c == '_' {
                    c
                } else {
                    '_'
                })
                .collect::<String>()
        )
    } else {
        name.to_string()
    };
    let mut updated_name = base.clone();
    while !name_allowed(symbol_table, &base, unique) {
        updated_name = format!("{}_{}", base, get_next_counter_value());
    }
    updated_name
}
