// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use crate::operations::{Operation, OperationRef, StandardGate, StandardInstruction};

#[test]
fn test_standard_gate_py_cls_name() {
    // Test a few representative gates
    let h_gate = StandardGate::H;
    let x_gate = StandardGate::X;
    let cx_gate = StandardGate::CX;
    let rz_gate = StandardGate::RZ;

    // Check that the py_cls_name method returns the expected values
    assert_eq!(
        h_gate.py_cls_name(),
        Some(("qiskit.circuit.library.standard_gates.h", "HGate"))
    );
    assert_eq!(
        x_gate.py_cls_name(),
        Some(("qiskit.circuit.library.standard_gates.x", "XGate"))
    );
    assert_eq!(
        cx_gate.py_cls_name(),
        Some(("qiskit.circuit.library.standard_gates.x", "CXGate"))
    );
    assert_eq!(
        rz_gate.py_cls_name(),
        Some(("qiskit.circuit.library.standard_gates.rz", "RZGate"))
    );
}

#[test]
fn test_standard_instruction_py_cls_name() {
    // Test all standard instructions
    let barrier = StandardInstruction::Barrier(2);
    let delay = StandardInstruction::Delay(crate::operations::DelayUnit::NS);
    let measure = StandardInstruction::Measure;
    let reset = StandardInstruction::Reset;

    // Check that the py_cls_name method returns the expected values
    assert_eq!(barrier.py_cls_name(), Some(("qiskit.circuit", "Barrier")));
    assert_eq!(delay.py_cls_name(), Some(("qiskit.circuit", "Delay")));
    assert_eq!(measure.py_cls_name(), Some(("qiskit.circuit", "Measure")));
    assert_eq!(reset.py_cls_name(), Some(("qiskit.circuit", "Reset")));
}

#[test]
fn test_operation_ref_py_cls_name() {
    // Test OperationRef with StandardGate and StandardInstruction
    let h_gate = StandardGate::H;
    let measure = StandardInstruction::Measure;

    let op_ref_gate = OperationRef::StandardGate(h_gate);
    let op_ref_inst = OperationRef::StandardInstruction(measure);

    // Check that the py_cls_name method returns the expected values
    assert_eq!(
        op_ref_gate.py_cls_name(),
        Some(("qiskit.circuit.library.standard_gates.h", "HGate"))
    );
    assert_eq!(op_ref_inst.py_cls_name(), Some(("qiskit.circuit", "Measure")));

    // Test that UnitaryGate returns the expected value
    let unitary_gate = crate::operations::UnitaryGate {
        array: crate::operations::ArrayType::OneQ(nalgebra::Matrix2::identity()),
    };
    let op_ref_unitary = OperationRef::Unitary(&unitary_gate);
    assert_eq!(
        op_ref_unitary.py_cls_name(),
        Some(("qiskit.circuit.library.generalized_gates.unitary", "UnitaryGate"))
    );

}
