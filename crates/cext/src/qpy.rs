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

use binrw::BinWrite;
use num_complex::ComplexFloat;
use std::io::Cursor;

use qiskit_circuit::bit::Register;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{Param, STANDARD_GATE_SIZE, StandardInstruction};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::parameter::parameter_expression::{ParameterExpression, ParameterValueType};
use qiskit_circuit::parameter::symbol_expr::Value;

use crate::pointers::const_ptr_as_ref;

// encode parameter expression elements
fn qpy_encode_parameter_expression_element(buf: &mut Cursor<Vec<u8>>, val: &ParameterValueType) {
    // write 1 byte type and 16 byte data
    match val {
        ParameterValueType::Int(i) => {
            let _ = "i".as_bytes().write(buf);
            let _ = 0u64.to_be_bytes().write(buf); // pad 8 bytes
            let _ = i.to_be_bytes().write(buf);
        }
        ParameterValueType::Float(f) => {
            let _ = "f".as_bytes().write(buf);
            let _ = 0u64.to_be_bytes().write(buf); // pad 8 bytes
            let _ = f.to_be_bytes().write(buf);
        }
        ParameterValueType::Complex(c) => {
            let _ = "c".as_bytes().write(buf);
            let _ = c.re().to_be_bytes().write(buf);
            let _ = c.im().to_be_bytes().write(buf);
        }
        ParameterValueType::Parameter(p) => {
            let _ = "p".as_bytes().write(buf);
            let _ = p.symbol().uuid.as_u128().to_be_bytes().write(buf);
        }
        ParameterValueType::VectorElement(v) => {
            let _ = "v".as_bytes().write(buf);
            let _ = v.symbol().uuid.as_u128().to_be_bytes().write(buf);
        }
    }
}

// encode ParameterExpression
fn qpy_enocde_parameter_expression(buf: &mut Cursor<Vec<u8>>, expr: &ParameterExpression) {
    let mut expr_data_buf = Cursor::new(Vec::<u8>::new());

    // write length of symbol table
    let _ = expr.num_symbols().to_be_bytes().write(buf);

    // construct expression data
    let replay = expr.qpy_replay();
    for op in &replay {
        // write OP code
        let _ = (op.op as u8).write(&mut expr_data_buf);
        // write lhs
        match &op.lhs {
            Some(lhs) => qpy_encode_parameter_expression_element(&mut expr_data_buf, lhs),
            None => {
                let _ = "n".as_bytes().write(&mut expr_data_buf);
                let _ = 0u8.write(&mut expr_data_buf);
            }
        }
        match &op.rhs {
            Some(rhs) => qpy_encode_parameter_expression_element(&mut expr_data_buf, rhs),
            None => {
                let _ = "n".as_bytes().write(&mut expr_data_buf);
                let _ = 0u8.write(&mut expr_data_buf);
            }
        }
    }
    let expr_data = expr_data_buf.into_inner();

    // write data length
    let _ = expr_data.len().to_be_bytes().write(buf);
    // write expression data
    let _ = expr_data.write(buf);

    // write symbols
    for symbol in expr.iter_symbols() {
        let _ = "pp".as_bytes().write(buf);
        let _ = 0u64.to_be_bytes().write(buf);
        // length of symbol name
        let _ = (symbol.name().len() as u16).to_be_bytes().write(buf);
        // UUID
        let _ = symbol.uuid.as_u128().to_be_bytes().write(buf);
        // name
        let _ = symbol.name().as_bytes().write(buf);
    }
}

// encode Param
// return tuple of value type and data
fn qpy_encode_param(param: &Param) -> Result<(u8, Vec<u8>), ()> {
    let mut buf = Cursor::new(Vec::<u8>::new());

    match param {
        Param::Float(f) => {
            let _ = f.to_be_bytes().write(&mut buf);
            Ok((b'f', buf.into_inner()))
        }
        Param::ParameterExpression(e) => {
            if let Ok(s) = e.try_to_symbol() {
                // length of name
                let _ = (s.name().len() as u16).to_be_bytes().write(&mut buf);
                // UUID
                let _ = s.uuid.as_u128().to_be_bytes().write(&mut buf);
                // name
                let _ = s.name().as_bytes().write(&mut buf);
                return Ok((b'p', buf.into_inner()));
            } else if let Ok(v) = e.try_to_value(true) {
                // write Value
                return match v {
                    Value::Int(i) => {
                        let _ = i.to_be_bytes().write(&mut buf);
                        Ok((b'i', buf.into_inner()))
                    }
                    Value::Real(r) => {
                        let _ = r.to_be_bytes().write(&mut buf);
                        Ok((b'f', buf.into_inner()))
                    }
                    Value::Complex(c) => {
                        let _ = c.re().to_be_bytes().write(&mut buf);
                        let _ = c.im().to_be_bytes().write(&mut buf);
                        Ok((b'c', buf.into_inner()))
                    }
                };
            }
            // parameter expression
            qpy_enocde_parameter_expression(&mut buf, &e);
            Ok((b'e', buf.into_inner()))
        }
        Param::Obj(_) => Err(()),
    }
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct QPYRegisterHeader {
    pub register_type: u8,
    pub standalone: u8,
    pub size: u32,
    pub name_size: u16,
    pub in_circuit: u8,
}

// write registers
fn qpy_encode_registers(buf: &mut Cursor<Vec<u8>>, circ: &CircuitData) {
    // write qregs
    for reg in circ.qregs() {
        let mut indices = Vec::<i64>::with_capacity(reg.len());
        let mut is_in_circuit = true;
        for (i, bit) in reg.bits().enumerate() {
            match circ.qubits().find(&bit) {
                Some(b) => indices[i] = b.index() as i64,
                None => {
                    indices[i] = -1i64;
                    is_in_circuit = false;
                }
            }
        }
        let _ = QPYRegisterHeader {
            register_type: b'q',
            standalone: (reg.len() == circ.num_qubits()) as u8,
            size: reg.len() as u32,
            name_size: reg.name().len() as u16,
            in_circuit: is_in_circuit as u8,
        }
        .write(buf);

        let _ = reg.name().as_bytes().write(buf);
        indices.iter().for_each(|i| {
            let _ = i.to_be_bytes().write(buf);
        });
    }

    // write cregs
    for reg in circ.cregs() {
        let mut indices = Vec::<i64>::with_capacity(reg.len());
        let mut is_in_circuit = true;
        for (i, bit) in reg.bits().enumerate() {
            match circ.clbits().find(&bit) {
                Some(b) => indices[i] = b.index() as i64,
                None => {
                    indices[i] = -1i64;
                    is_in_circuit = false;
                }
            }
        }
        let _ = QPYRegisterHeader {
            register_type: b'c',
            standalone: (reg.len() == circ.num_clbits()) as u8,
            size: reg.len() as u32,
            name_size: reg.name().len() as u16,
            in_circuit: is_in_circuit as u8,
        }
        .write(buf);

        let _ = reg.name().as_bytes().write(buf);
        indices.iter().for_each(|i| {
            let _ = i.to_be_bytes().write(buf);
        });
    }
}

static STANDARD_GATE_CLASS_NAME: [&str; STANDARD_GATE_SIZE] = [
    "GlobalPhaseGate", // 0
    "HGate",           // 1
    "IGate",           // 2
    "XGate",           // 3
    "YGate",           // 4
    "ZGate",           // 5
    "PhaseGate",       // 6
    "RGate",           // 7
    "RXGate",          // 8
    "RYGate",          // 9
    "RZGate",          // 10
    "SGate",           // 11
    "SdgGate",         // 12
    "SXGate",          // 13
    "SXdgGate",        // 14
    "TGate",           // 15
    "TdgGate",         // 16
    "UGate",           // 17
    "U1Gate",          // 18
    "U2Gate",          // 19
    "U3Gate",          // 20
    "CHGate",          // 21
    "CXGate",          // 22
    "CYGate",          // 23
    "CZGate",          // 24
    "DCXGate",         // 25
    "ECRGate",         // 26
    "SwapGate",        // 27
    "iSwapGate",       // 28
    "CPhaseGate",      // 29
    "CRXGate",         // 30
    "CRYGate",         // 31
    "CRZGate",         // 32
    "CSGate",          // 33
    "CSdgGate",        // 34
    "CSXGate",         // 35
    "CUGate",          // 36
    "CU1Gate",         // 37
    "CU3Gate",         // 38
    "RXXGate",         // 39
    "RYYGate",         // 40
    "RZZGate",         // 41
    "RZXGate",         // 42
    "XXMinusYYGate",   // 43
    "XXPlusYYGate",    // 44
    "CCXGate",         // 45
    "CCZGate",         // 46
    "CSwapGate",       // 47
    "RCCXGate",        // 48
    "C3XGate",         // 49 ("c3x")
    "C3SXGate",        // 50
    "RC3XGate",        // 51 ("rc3x")
];

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct QPYInstruction {
    pub name_size: u16,
    pub label_size: u16,
    pub num_parameters: u16,
    pub num_qargs: u32,
    pub num_cargs: u32,
    pub extras_key: u8,
    pub conditional_reg_name_size: u16,
    pub conditional_value: i64,
    pub num_ctrl_qubits: u32,
    pub ctrl_state: u32,
}

// write instruction
fn qpy_encode_instruction(buf: &mut Cursor<Vec<u8>>, circ: &CircuitData, inst: &PackedInstruction) {
    let qargs = circ.get_qargs(inst.qubits);
    let cargs = circ.get_cargs(inst.clbits);
    let num_ctrl_bits = match inst.standard_gate() {
        Some(g) => g.get_num_ctrl_qubits(),
        None => 0,
    };
    let gate_class_name = match inst.standard_gate() {
        Some(g) => STANDARD_GATE_CLASS_NAME[g as usize],
        None => match inst.op.try_standard_instruction() {
            Some(i) => match i {
                StandardInstruction::Barrier(_) => "Barrier",
                StandardInstruction::Delay(_) => "Delay",
                StandardInstruction::Measure => "Measure",
                StandardInstruction::Reset => "Reset",
            },
            None => "",
        },
    };

    // write instruction header
    // TO DO: implement conditional if conditional gates will be supported in C-API
    let _ = QPYInstruction {
        name_size: gate_class_name.len() as u16,
        label_size: match inst.label() {
            Some(l) => l.len() as u16,
            None => 0 as u16,
        },
        num_parameters: inst.params_view().len() as u16,
        num_qargs: qargs.len() as u32,
        num_cargs: cargs.len() as u32,
        extras_key: 0,
        conditional_reg_name_size: 0,
        conditional_value: 0,
        num_ctrl_qubits: num_ctrl_bits,
        ctrl_state: (1u32 << num_ctrl_bits) - 1,
    }
    .write(buf);

    // write class name
    let _ = gate_class_name.as_bytes().write(buf);

    // write label
    if let Some(l) = inst.label() {
        let _ = l.as_bytes().write(buf);
    }

    // TO DO: insert conditional params

    // write qargs
    for qubit in qargs {
        let _ = b'q'.write(buf);
        let _ = (qubit.index() as u32).to_be_bytes().write(buf);
    }
    // write cargs
    for clbit in cargs {
        let _ = b'c'.write(buf);
        let _ = (clbit.index() as u32).to_be_bytes().write(buf);
    }

    // write parameters
    for p in inst.params_view() {
        if let Ok((k, param_buf)) = qpy_encode_param(p) {
            let _ = k.write(buf);
            let _ = param_buf.write(buf);
        }
    }
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct QPYHeader {
    pub qiskit: [u8; 6],
    pub version: u8,
    pub qiskit_major_version: u8,
    pub qiskit_minor_version: u8,
    pub qiskit_patch_version: u8,
    pub num_circuits: u8,
    pub encoding: u8,
    pub program: u8,
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct QPYCircuitHeader {
    pub name_size: u16,
    pub global_phase_type: u8,
    pub global_phase_size: u16,
    pub num_qubits: u32,
    pub num_clbits: u32,
    pub metadata_size: u64,
    pub num_registers: u32,
    pub num_instructions: u64,
    pub num_vars: u32,
}

#[derive(BinWrite)]
#[brw(big)]
#[derive(Debug)]
pub struct QPYLayoutHeader {
    pub exists: u8,
    pub initial_layout_size: i32,
    pub input_mapping_size: i32,
    pub final_layout_size: i32,
    pub extra_registers_length: u32,
    pub input_qubit_count: i32,
}

// encode circuit header
fn qpy_encode_header(buf: &mut Cursor<Vec<u8>>, num_circuits: usize) {
    let _ = QPYHeader {
        qiskit: *b"QISKIT",
        version: 14,
        qiskit_major_version: 2,
        qiskit_minor_version: 2,
        qiskit_patch_version: 0,
        num_circuits: num_circuits as u8,
        encoding: b'p',
        program: b'q',
    }
    .write(buf);
}

// encode circuit to QPY binary
fn qpy_encode_circuit(buf: &mut Cursor<Vec<u8>>, circ: &CircuitData) {
    // get global phase data
    let global_phase = match qpy_encode_param(circ.global_phase()) {
        Ok(p) => p,
        Err(_) => (b'i', 0i64.to_be_bytes().to_vec()),
    };

    // circuit header
    let _ = QPYCircuitHeader {
        name_size: 0,
        global_phase_type: global_phase.0,
        global_phase_size: global_phase.1.len() as u16,
        num_qubits: circ.num_qubits() as u32,
        num_clbits: circ.num_clbits() as u32,
        metadata_size: 0,
        num_registers: circ.qregs().len() as u32 + circ.qregs().len() as u32,
        num_instructions: circ.data().len() as u64,
        num_vars: circ.identifiers().len() as u32,
    }
    .write(buf);

    // write global phase data
    let _ = global_phase.1.write(buf);

    // write registers
    qpy_encode_registers(buf, circ);

    // write instructions
    for inst in circ.data() {
        qpy_encode_instruction(buf, circ, inst);
    }

    // custom instructions
    // TO DO : implement custom instructions if C-API will support
    let _ = 0u64.to_be_bytes().write(buf);

    // no calibration
    let _ = 0u64.to_be_bytes().write(buf);

    // layout
    // TO DO : implement this by using transpiled layout ?
    let _ = QPYLayoutHeader {
        exists: 0u8,
        initial_layout_size: -1,
        input_mapping_size: -1,
        final_layout_size: -1,
        extra_registers_length: 0,
        input_qubit_count: 0,
    }
    .write(buf);
}

/// storage for QPY passed to C
#[repr(C)]
pub struct QkQPYContainer {
    /// A array of byte size ``len`` storage for QPY raw binary data
    qpy: *mut u8,
    /// The number of bytes in ``qpy``
    len: usize,
}

/// @ingroup QkCircuit
/// Encode QPY for a list of QkCircuit
///
/// @param circuits A pointer to the list of circuits to be encoded as QPY format
/// @param num_circuits number of QkCircuit in the list
///
/// @return encoded QPY binary data in ``QkQPYContainer`` struct
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(10, 10);
///     uint32_t qubit[1] = {0};
///     qk_circuit_gate(qc, QkGate_H, qubit, NULL);
///     QkQPYContainer* qpy = qk_circuit_to_qpy(&qc, 1);
///
///     qk_circuit_qpy_free(qpy);
/// ```
///
/// # Safety
/// The ``circuits`` should be valid pointer to the list of QkCircuit
/// and all the QkCircuit pointer should be valid pointer
/// The ``num_circuit`` should be no larger than the size of a list ``circuits``
///
/// Behavior is undefined if ``circuits`` is not a valid, non-null pointer to a ``QkCircuit``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_to_qpy(
    circuits: *const *const CircuitData,
    num_circuits: usize,
) -> QkQPYContainer {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuits = unsafe {
        std::slice::from_raw_parts(circuits, num_circuits)
            .iter()
            .map(|circ| const_ptr_as_ref(*circ))
    };
    let mut buf = Cursor::new(Vec::<u8>::new());

    qpy_encode_header(&mut buf, circuits.len());
    for circ in circuits {
        qpy_encode_circuit(&mut buf, circ);
    }

    let buf = buf.into_inner().into_boxed_slice();
    let len = buf.len();
    QkQPYContainer {
        qpy: Box::into_raw(buf) as *mut u8,
        len,
    }
}

/// @ingroup QkCircuit
/// Free the QPY data
///
/// @param qpy A pointer to the QPY binary data storage
///
/// # Example
/// ```c
///     QkCircuit *qc = qk_circuit_new(10, 10);
///     uint32_t qubit[1] = {0};
///     qk_circuit_gate(qc, QkGate_H, qubit, NULL);
///     QkQPYContainer* qpy = qk_circuit_to_qpy(&qc, 1);
///
///     qk_circuit_qpy_free(qpy);
/// ```
///
/// # Safety
///
/// Behavior is undefined if ``qpy`` is not either null or a valid pointer to a
/// ``Vec<u8>``.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_qpy_free(qpy: QkQPYContainer) {
    if !qpy.qpy.is_null() {
        if !qpy.qpy.is_aligned() {
            panic!("Attempted to free a non-aligned pointer.")
        }

        // SAFETY: We have verified the pointer is non-null and aligned, so it should be
        // readable by Box.
        unsafe {
            let _ = Box::from_raw(qpy.qpy);
        }
    }
}
