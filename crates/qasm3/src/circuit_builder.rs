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

use crate::ast::*;
use crate::error::QASM3ExporterError;
use crate::exporter::{BitType, RegisterType, BIT_PREFIX, QUBIT_PREFIX};
use crate::symbol_table::SymbolTable;
use hashbrown::{HashMap, HashSet};
use pyo3::prelude::*;
use qiskit_circuit::bit::{ClassicalRegister, QuantumRegister, Register};
use qiskit_circuit::{Clbit, Qubit};
use qiskit_circuit::circuit_data::CircuitData;

type ExporterResult<T> = Result<T, QASM3ExporterError>;

pub struct CircuitBuilder<'a> {
    circuit_data: &'a CircuitData,
    symbol_table: &'a mut SymbolTable,
    is_layout: bool,
    allow_aliasing: bool,
}

impl<'a> CircuitBuilder<'a> {
    pub fn new(
        circuit_data: &'a CircuitData,
        symbol_table: &'a mut SymbolTable,
        is_layout: bool,
        allow_aliasing: bool,
    ) -> Self {
        Self {
            circuit_data,
            symbol_table,
            is_layout,
            allow_aliasing,
        }
    }

    pub fn build_classical_declarations(&mut self) -> ExporterResult<Vec<Statement>> {
        let num_clbits = self.circuit_data.num_clbits();
        
        let mut has_multiple_registers = false;
        let registers: Vec<_> = self.circuit_data.cregs().to_vec();
        let mut bit_register_count = vec![0; num_clbits];
        for register in &registers {
            for shareable_clbit in register.bits() {
                if let Some(clbit_index) = self.circuit_data.clbits().find(&shareable_clbit) {
                    bit_register_count[clbit_index.0 as usize] += 1;
                    if bit_register_count[clbit_index.0 as usize] > 1 {
                        has_multiple_registers = true;
                        break;
                    }
                }
            }
            if has_multiple_registers {
                break;
            }
        }

        let mut decls = Vec::new();
        if has_multiple_registers {
            if !self.allow_aliasing {
                return Err(QASM3ExporterError::Error(
                    "Some Classical registers in this circuit overlap.".to_string(),
                ));
            }
            for i in 0..num_clbits {
                let clbit = Clbit(i as u32);
                let identifier = self.symbol_table.register_bits(
                    format!("{}{}", BIT_PREFIX, i),
                    &BitType::Clbit(clbit),
                    true,
                    false,
                )?;
                decls.push(Statement::ClassicalDeclaration(ClassicalDeclaration {
                    type_: ClassicalType::Bit,
                    identifier,
                }));
            }
            let registers: Vec<_> = self.circuit_data.cregs().to_vec();
            for register in registers {
                let aliased = self.build_aliases(&RegisterType::ClassicalRegister(register.clone()))?;
                decls.push(Statement::Alias(aliased));
            }
            return Ok(decls);
        }
        
        let mut clbits_in_registers = HashSet::new();
        for creg in self.circuit_data.cregs() {
            for shareable_clbit in creg.bits() {
                if let Some(clbit_index) = self.circuit_data.clbits().find(&shareable_clbit) {
                    clbits_in_registers.insert(clbit_index.0);
                }
            }
        }

        // Declare loose clbits (not in any register)
        for i in 0..num_clbits {
            if !clbits_in_registers.contains(&(i as u32)) {
                let clbit = Clbit(i as u32);
                let identifier = self.symbol_table.register_bits(
                    format!("{}{}", BIT_PREFIX, i),
                    &BitType::Clbit(clbit),
                    true,
                    false,
                )?;

                decls.push(Statement::ClassicalDeclaration(ClassicalDeclaration {
                    type_: ClassicalType::Bit,
                    identifier,
                }));
            }
        }
        
        // Declare registers
        for creg in self.circuit_data.cregs() {
            let identifier = self.symbol_table.register_registers(
                creg.name().to_string(),
                &RegisterType::ClassicalRegister(creg.clone()),
            )?;

            for (i, shareable_clbit) in creg.bits().enumerate() {
                if let Some(clbit) = self.circuit_data.clbits().find(&shareable_clbit) {
                    self.symbol_table.set_bitinfo(
                        Expression::Index(Index {
                            target: Box::new(Expression::Parameter(Parameter {
                                obj: identifier.string.to_string(),
                            })),
                            index: Box::new(Expression::IntegerLiteral(IntegerLiteral(i as i64))),
                        }),
                        BitType::Clbit(clbit),
                    )
                }
            }
            decls.push(Statement::ClassicalDeclaration(ClassicalDeclaration {
                type_: ClassicalType::BitArray(BitArray(creg.len())),
                identifier,
            }))
        }
        Ok(decls)
    }

    pub fn build_qubit_declarations(&mut self) -> ExporterResult<Vec<Statement>> {
        let num_qubits = self.circuit_data.num_qubits();
        
        let mut has_multiple_registers = false;
        let registers: Vec<_> = self.circuit_data.qregs().to_vec();
        let mut bit_register_count = vec![0; num_qubits];
        for register in &registers {
            for shareable_qubit in register.bits() {
                if let Some(qubit_index) = self.circuit_data.qubits().find(&shareable_qubit) {
                    bit_register_count[qubit_index.0 as usize] += 1;
                    if bit_register_count[qubit_index.0 as usize] > 1 {
                        has_multiple_registers = true;
                        break;
                    }
                }
            }
            if has_multiple_registers {
                break;
            }
        }

        let mut decls: Vec<Statement> = Vec::new();

        if self.is_layout {
            for i in 0..num_qubits {
                let qubit = Qubit(i as u32);
                self.symbol_table.register_bits(
                    format!("${}", i),
                    &BitType::Qubit(qubit),
                    false,
                    true,
                )?;
            }
            return Ok(decls);
        }

        if has_multiple_registers {
            if !self.allow_aliasing {
                return Err(QASM3ExporterError::Error(
                    "Some quantum registers in this circuit overlap.".to_string(),
                ));
            }

            for i in 0..num_qubits {
                let qubit = Qubit(i as u32);
                let identifier = self.symbol_table.register_bits(
                    format!("{}{}", if self.is_layout { "$" } else { QUBIT_PREFIX }, i),
                    &BitType::Qubit(qubit),
                    true,
                    false,
                )?;
                decls.push(Statement::QuantumDeclaration(QuantumDeclaration {
                    identifier,
                    designator: None,
                }));
            }
            let registers: Vec<_> = self.circuit_data.qregs().to_vec();
            for register in registers {
                let aliased = self.build_aliases(&RegisterType::QuantumRegister(register.clone()))?;
                decls.push(Statement::Alias(aliased));
            }
            return Ok(decls);
        }

        let mut qubits_in_registers = HashSet::new();
        for qreg in self.circuit_data.qregs() {
            for shareable_qubit in qreg.bits() {
                if let Some(qubit_index) = self.circuit_data.qubits().find(&shareable_qubit) {
                    qubits_in_registers.insert(qubit_index.0);
                }
            }
        }

        for i in 0..num_qubits {
            if !qubits_in_registers.contains(&(i as u32)) {
                let qubit = Qubit(i as u32);
                let identifier = self.symbol_table.register_bits(
                    format!("{}{}", if self.is_layout { "$" } else { QUBIT_PREFIX }, i),
                    &BitType::Qubit(qubit),
                    true,
                    false,
                )?;
                decls.push(Statement::QuantumDeclaration(QuantumDeclaration {
                    identifier,
                    designator: None,
                }));
            }
        }
        
        for qreg in self.circuit_data.qregs() {
            let identifier = self.symbol_table.register_registers(
                qreg.name().to_string(),
                &RegisterType::QuantumRegister(qreg.clone()),
            )?;
            for (i, shareable_qubit) in qreg.bits().enumerate() {
                if let Some(qubit) = self.circuit_data.qubits().find(&shareable_qubit) {
                    self.symbol_table.set_bitinfo(
                        Expression::Index(Index {
                            target: Box::new(Expression::Parameter(Parameter {
                                obj: identifier.string.to_string(),
                            })),
                            index: Box::new(Expression::IntegerLiteral(IntegerLiteral(i as i64))),
                        }),
                        BitType::Qubit(qubit),
                    )
                }
            }
            decls.push(Statement::QuantumDeclaration(QuantumDeclaration {
                identifier,
                designator: Some(Designator {
                    expression: Expression::IntegerLiteral(IntegerLiteral(qreg.len() as i64)),
                }),
            }))
        }
        Ok(decls)
    }

    fn build_aliases(&mut self, register: &RegisterType) -> ExporterResult<Alias> {
        let name = self.symbol_table.register_registers(
            register.name().to_string(), 
            register
        )?;
        let mut elements = Vec::new();
        
        for (i, bit) in register.bits(self.circuit_data).iter().enumerate() {
            let id = self.symbol_table.get_bitinfo(bit)
                .ok_or_else(|| QASM3ExporterError::Error(format!("Bit not found: {:?}", bit)))?
                .clone();
            
            let id2 = Expression::Index(Index {
                target: Box::new(Expression::Parameter(Parameter {
                    obj: name.string.clone(),
                })),
                index: Box::new(Expression::IntegerLiteral(IntegerLiteral(i as i64))),
            });
            self.symbol_table.set_bitinfo(id2, bit.clone());
            elements.push(id);
        }
        Ok(Alias {
            identifier: name,
            value: Expression::IndexSet(IndexSet { values: elements }),
        })
    }
}