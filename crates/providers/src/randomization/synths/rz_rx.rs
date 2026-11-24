// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::f64::consts::FRAC_PI_2;

use qiskit_circuit::Qubit;
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError};
use qiskit_circuit::operations::{Param, StandardGate};

use super::Synth;
use crate::ops::{Add, Constant, EulerAnglesZYZ};
use crate::randomization::flags::BitFlags;
use crate::randomization::virtual_type::VirtualType;
use crate::tensor::{Dim, Tensor, TensorType};
use crate::{FunctionError, ProgramFunction};

/// Express a single-qubit unitary as the template `rz rx rz`.
///
/// The template is `rz(p0) rx(p1) rz(p2)` in circuit order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RzRx;

impl RzRx {
    /// The number of parameters of the template on one qubit.
    const NUM_PARAMS: usize = 3;

    /// The affine offset between the template parameters and the ZYZ Euler angles.
    const OFFSET: [f64; Self::NUM_PARAMS] = [-FRAC_PI_2, 0.0, FRAC_PI_2];
}

impl Synth for RzRx {
    fn name(&self) -> &str {
        "rzrx"
    }

    fn virtual_types(&self) -> BitFlags<VirtualType> {
        // TODO: extend this to other single-qubit virtual types
        BitFlags::from_flag(VirtualType::U2)
    }

    fn num_params(&self, num_qubits: usize) -> Option<usize> {
        (num_qubits == 1).then_some(Self::NUM_PARAMS)
    }

    fn make_template(
        &self,
        circuit: &mut CircuitData,
        qubits: &[Qubit],
        params: &[Param],
    ) -> Result<(), CircuitDataError> {
        let ([_], [first, second, third]) = (qubits, params) else {
            panic!(
                "the rzrx template takes 1 qubit and 3 parameters, got {} and {}",
                qubits.len(),
                params.len()
            );
        };
        circuit.push_standard_gate(StandardGate::RZ, std::slice::from_ref(first), qubits)?;
        circuit.push_standard_gate(StandardGate::RX, std::slice::from_ref(second), qubits)?;
        circuit.push_standard_gate(StandardGate::RZ, std::slice::from_ref(third), qubits)
    }

    fn program_function(
        &self,
        ty: VirtualType,
        num_qubits: usize,
        leading: &[Dim],
    ) -> Result<ProgramFunction, FunctionError> {
        assert!(
            self.accepts(ty) && self.num_params(num_qubits).is_some(),
            "the rzrx synth expresses U2 on 1 qubit, not {ty:?} on {num_qubits}"
        );
        let gate = ty
            .tensor_type(num_qubits)
            .expect("U2 has a gate on 1 qubit");
        let mut shape = leading.to_vec();
        shape.extend(gate.shape);

        let mut function = ProgramFunction::new();
        let unitaries = function.add_parameter(TensorType {
            dtype: gate.dtype,
            shape,
        });
        let angles = function.add_op(EulerAnglesZYZ::default(), &[unitaries])?[0];
        let offset = function.add_op(Constant::new(Tensor::from(Self::OFFSET)), &[])?[0];
        let params = function.add_op(Add, &[angles, offset])?[0];
        function.add_result(params)?;
        Ok(function)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, ArrayD, IxDyn};
    use num_complex::Complex64;

    use super::*;
    use crate::tensor::{DType, dim_shape};

    /// Return the value of a float parameter.
    fn float(param: &Param) -> f64 {
        let Param::Float(value) = param else {
            panic!("{param:?} is not a float")
        };
        *value
    }

    /// Return the matrix of `circuit`, which is on one qubit.
    fn unitary_of(circuit: &CircuitData) -> Array2<Complex64> {
        circuit.iter().fold(Array2::eye(2), |product, instruction| {
            let gate = instruction.op.standard_gate();
            let matrix = gate
                .matrix(instruction.params_view())
                .expect("a standard gate with float parameters has a matrix");
            matrix.dot(&product)
        })
    }

    /// Assert that two 2x2 unitaries are equal up to a global phase.
    fn assert_equal_up_to_phase(actual: &Array2<Complex64>, expected: &Array2<Complex64>) {
        // The trace of `actual^dagger expected` has magnitude 2 exactly when they differ by a phase.
        let overlap: Complex64 = actual.iter().zip(expected).map(|(a, b)| a.conj() * b).sum();
        assert!(
            (overlap.norm() - 2.0).abs() < 1e-9,
            "{actual} is not {expected} up to a phase"
        );
    }

    #[test]
    fn test_rz_rx_declares_u2_on_one_qubit() {
        assert_eq!(RzRx.name(), "rzrx");
        assert_eq!(RzRx.virtual_types(), BitFlags::from_flag(VirtualType::U2));
        assert!(RzRx.accepts(VirtualType::U2));
        assert!(!RzRx.accepts(VirtualType::Pauli));
        assert_eq!(RzRx.num_params(1), Some(3));
        assert_eq!(RzRx.num_params(2), None);
    }

    #[test]
    fn test_template_is_rz_rx_rz() {
        let mut circuit = CircuitData::with_capacity(2, 0, 3, Param::Float(0.0)).unwrap();
        let params = [0.1, 0.2, 0.3].map(Param::Float);
        RzRx.make_template(&mut circuit, &[Qubit(1)], &params)
            .unwrap();

        let gates: Vec<_> = circuit
            .iter()
            .map(|instruction| {
                (
                    instruction.op.standard_gate(),
                    circuit.get_qargs(instruction.qubits).to_vec(),
                    instruction
                        .params_view()
                        .iter()
                        .map(float)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        assert_eq!(
            gates,
            [
                (StandardGate::RZ, vec![Qubit(1)], vec![0.1]),
                (StandardGate::RX, vec![Qubit(1)], vec![0.2]),
                (StandardGate::RZ, vec![Qubit(1)], vec![0.3]),
            ]
        );
    }

    #[test]
    fn test_program_function_keeps_leading_axes() {
        let leading = [Dim::Fixed(4), Dim::Bounded { max: 8 }];
        let function = RzRx.program_function(VirtualType::U2, 1, &leading).unwrap();
        let signature = function.signature();
        assert_eq!(
            signature.inputs,
            [TensorType {
                dtype: DType::C128,
                shape: [&leading[..], &dim_shape([2, 2])].concat(),
            }]
        );
        assert_eq!(
            signature.outputs,
            [TensorType {
                dtype: DType::F64,
                shape: [&leading[..], &[Dim::Fixed(3)]].concat(),
            }]
        );
    }

    #[test]
    fn test_template_reproduces_the_unitary() {
        let phase = Complex64::from_polar(1.0, 0.7);
        let generic = [1.1, -2.0, 0.3].map(Param::Float);
        let unitaries = [
            StandardGate::I.matrix(&[]).unwrap(),
            StandardGate::X.matrix(&[]).unwrap(),
            StandardGate::S.matrix(&[]).unwrap(),
            StandardGate::H.matrix(&[]).unwrap(),
            StandardGate::SX.matrix(&[]).unwrap(),
            StandardGate::U.matrix(&generic).unwrap() * phase,
        ];
        let elements = unitaries.iter().flat_map(|u| u.iter().copied()).collect();
        let stack = ArrayD::from_shape_vec(IxDyn(&[2, 3, 2, 2]), elements).unwrap();

        let function = RzRx
            .program_function(VirtualType::U2, 1, &dim_shape([2, 3]))
            .unwrap();
        let outputs = function.eval(&[Tensor::from(stack)]).unwrap();
        let Tensor::F64(params) = &outputs[0] else {
            panic!("the parameters are F64")
        };
        assert_eq!(params.shape(), [2, 3, 3]);

        let params = params.as_slice().expect("a contiguous result");
        for (expected, values) in unitaries.iter().zip(params.chunks_exact(3)) {
            let mut circuit = CircuitData::with_capacity(1, 0, 3, Param::Float(0.0)).unwrap();
            let values = values.iter().copied().map(Param::Float).collect::<Vec<_>>();
            RzRx.make_template(&mut circuit, &[Qubit(0)], &values)
                .unwrap();
            assert_equal_up_to_phase(&unitary_of(&circuit), expected);
        }
    }
}
