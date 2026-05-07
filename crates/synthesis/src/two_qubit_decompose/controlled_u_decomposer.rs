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

use approx::abs_diff_eq;
use num_complex::Complex64;
use smallvec::{SmallVec, smallvec};
use std::f64::consts::FRAC_PI_2;

use nalgebra::{Matrix2, MatrixView2, U2};
use ndarray::prelude::*;
use numpy::PyReadonlyArray2;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::QiskitError;
use crate::euler_one_qubit_decomposer::{
    EulerBasis, EulerBasisSet, unitary_to_gate_sequence_inner,
};
use crate::linalg::nalgebra_array_view;

use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::operations::{Operation, OperationRef, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{NoBlocks, Qubit};

use super::common::{DEFAULT_FIDELITY, HGATE, SDGGATE, SGATE};
use super::gate_sequence::{TwoQubitGateSequence, TwoQubitSequenceVec};
use super::weyl_decomposition::{Specialization, TwoQubitWeylDecomposition};

type TwoQubitUnitary = (
    PackedOperation,
    SmallVec<[f64; 3]>,
    f64,
    [Matrix2<Complex64>; 4],
);

#[derive(Clone, Debug, FromPyObject)]
pub enum RXXEquivalent {
    Standard(StandardGate),
    CustomPython(Py<PyType>),
}

impl RXXEquivalent {
    fn matrix(&self, param: f64) -> PyResult<Array2<Complex64>> {
        match self {
            Self::Standard(gate) => Ok(gate.matrix(&[Param::Float(param)]).unwrap()),
            Self::CustomPython(gate_cls) => Python::attach(|py: Python| {
                let gate_obj = gate_cls.bind(py).call1((param,))?;
                let raw_matrix = gate_obj
                    .call_method0(intern!(py, "to_matrix"))?
                    .extract::<PyReadonlyArray2<Complex64>>()?;
                Ok(raw_matrix.as_array().to_owned())
            }),
        }
    }
}
impl<'a, 'py> IntoPyObject<'py> for &'a RXXEquivalent {
    type Target = PyAny;
    type Output = Borrowed<'a, 'py, Self::Target>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            RXXEquivalent::Standard(gate) => Ok(gate.get_gate_class(py)?.bind_borrowed(py)),
            RXXEquivalent::CustomPython(gate) => Ok(gate.as_any().bind_borrowed(py)),
        }
    }
}
impl<'py> IntoPyObject<'py> for RXXEquivalent {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            RXXEquivalent::Standard(gate) => Ok(gate.get_gate_class(py)?.bind(py).clone()),
            RXXEquivalent::CustomPython(gate) => Ok(gate.bind(py).clone().into_any()),
        }
    }
}

#[derive(Clone, Debug)]
#[pyclass(
    module = "qiskit._accelerate.two_qubit_decompose",
    subclass,
    skip_from_py_object
)]
pub struct TwoQubitControlledUDecomposer {
    rxx_equivalent_gate: RXXEquivalent,
    euler_basis: EulerBasis,
    #[pyo3(get)]
    scale: f64,
}

const DEFAULT_ATOL: f64 = 1e-12;
type InverseReturn = (PackedOperation, SmallVec<[f64; 3]>, SmallVec<[u8; 2]>);

///  Decompose two-qubit unitary in terms of a desired
///  :math:`U \sim U_d(\alpha, 0, 0) \sim \text{Ctrl-U}`
///  gate that is locally equivalent to an :class:`.RXXGate`.
impl TwoQubitControlledUDecomposer {
    /// Compute the number of basis gates needed for a given unitary
    pub fn num_basis_gates_inner(&self, unitary: ArrayView2<Complex64>) -> PyResult<usize> {
        let target_decomposed =
            TwoQubitWeylDecomposition::new_inner(unitary, Some(DEFAULT_FIDELITY), None)?;
        let num_basis_gates = (((target_decomposed.a).abs() > DEFAULT_ATOL) as usize)
            + (((target_decomposed.b).abs() > DEFAULT_ATOL) as usize)
            + (((target_decomposed.c).abs() > DEFAULT_ATOL) as usize);
        Ok(num_basis_gates)
    }

    /// invert 2q gate sequence
    fn invert_2q_gate(
        &self,
        gate: (PackedOperation, SmallVec<[f64; 3]>, SmallVec<[u8; 2]>),
    ) -> PyResult<InverseReturn> {
        let (gate, params, qubits) = gate;
        let inv_gate = match gate.view() {
            OperationRef::StandardGate(gate) => {
                let res = gate
                    .inverse(&params.into_iter().map(Param::Float).collect::<Vec<_>>())
                    .unwrap();
                (res.0.into(), res.1)
            }
            OperationRef::Gate(gate) => {
                Python::attach(|py: Python| -> PyResult<(PackedOperation, SmallVec<_>)> {
                    let raw_inverse = gate.instruction.call_method0(py, intern!(py, "inverse"))?;
                    let mut inverse: OperationFromPython<NoBlocks> = raw_inverse.extract(py)?;
                    let params = inverse.take_params().unwrap_or_default();
                    Ok((inverse.operation, params))
                })?
            }
            // UnitaryGate isn't applicable here as the 2q gate here is the parameterized
            // ControlledU equivalent used in the decomposition. This precludes UnitaryGate
            _ => panic!("Only 2q gate objects can be inverted in the decomposer"),
        };
        let inv_gate_params = inv_gate
            .1
            .into_iter()
            .map(|param| match param {
                Param::Float(val) => val,
                _ => {
                    unreachable!("Parameterized inverse generated from non-parameterized gate.")
                }
            })
            .collect::<SmallVec<_>>();
        Ok((inv_gate.0, inv_gate_params, qubits))
    }

    ///  Takes an angle and returns the circuit equivalent to an RXXGate with the
    ///  RXX equivalent gate as the two-qubit unitary.
    ///  Args:
    ///      angle: Rotation angle (in this case one of the Weyl parameters a, b, or c)
    ///      is_inv_rxx: Whether the RXX equivalent gate should be inverted or not
    ///  Returns:
    ///      out_gate_name: An equivalent 2-qubit gate (or its inverse)
    ///      out_gate_params: The equivalent 2-qubit gate params
    ///      global_phase: The global phase of the equivalent circuit
    ///      k_mats: Four 1-qubit gates in the equivalent circuit (before and after the 2-qubit gate)
    ///  Raises:
    ///      QiskitError: If the circuit is not equivalent to an RXXGate.
    fn to_rxx_gate(&self, angle: f64, is_inv_rxx: bool) -> PyResult<TwoQubitUnitary> {
        // The user-provided RXX equivalent gate may be locally equivalent to the RXX gate
        // but with some scaling in the rotation angle. For example, RXX(angle) has Weyl
        // parameters (angle, 0, 0) for angle in [0, pi/2] but the user provided gate, i.e.
        // :code:`self.rxx_equivalent_gate(angle)` might produce the Weyl parameters
        // (scale * angle, 0, 0) where scale != 1. This is the case for the CPhase gate.

        let mat = self.rxx_equivalent_gate.matrix(self.scale * angle)?;
        let decomposer_inv =
            TwoQubitWeylDecomposition::new_inner(mat.view(), Some(DEFAULT_FIDELITY), None)?;

        // Express the RXX in terms of the user-provided RXX equivalent gate.
        let global_phase = -decomposer_inv.global_phase;

        let k1r = decomposer_inv.K1r;
        let k2r = decomposer_inv.K2r;
        let k1l = decomposer_inv.K1l;
        let k2l = decomposer_inv.K2l;

        // the k matrices where RXX is inverted
        let mut k_mats = [k1r, k1l, k2r, k2l];

        if !is_inv_rxx {
            // 1-qubit gates before the rxx_op, on qubits 0 and 1 respectively
            if !k_mats[2].try_inverse_mut() {
                panic!("TwoQubitWeylDecomposition failed. Matrix K2r is not unitary");
            }
            if !k_mats[3].try_inverse_mut() {
                panic!("TwoQubitWeylDecomposition failed. Matrix K2l is not unitary");
            }
            // 1-qubit gates after the rxx_op, on qubits 0 and 1 respectively
            if !k_mats[0].try_inverse_mut() {
                panic!("TwoQubitWeylDecomposition failed. Matrix K1R is not unitary");
            }
            if !k_mats[1].try_inverse_mut() {
                panic!("TwoQubitWeylDecomposition failed. Matrix K1l is not unitary");
            }
            // k_mats = [k2r_inv, k2l_inv, k1r_inv, k1l_inv];
            k_mats.swap(0, 2);
            k_mats.swap(1, 3);
        }
        let rxx_op = match &self.rxx_equivalent_gate {
            RXXEquivalent::Standard(gate) => PackedOperation::from_standard_gate(*gate),
            RXXEquivalent::CustomPython(gate_cls) => {
                Python::attach(|py| -> PyResult<PackedOperation> {
                    let op: OperationFromPython<NoBlocks> =
                        gate_cls.bind(py).call1((self.scale * angle,))?.extract()?;
                    Ok(op.operation)
                })?
            }
        };
        let mut out_gate_name = rxx_op;
        let mut out_gate_params = smallvec![self.scale * angle];

        if is_inv_rxx {
            // invert the rxx_op
            let (inv_gate_name, inv_gate_params, _inv_gate_qubits) =
                self.invert_2q_gate((out_gate_name, out_gate_params, smallvec![0, 1]))?;
            out_gate_name = inv_gate_name;
            out_gate_params = inv_gate_params;
        }

        Ok((out_gate_name, out_gate_params, global_phase, k_mats))
    }

    /// Appends U_d(a, b, c) to the circuit.
    fn weyl_gate(
        &self,
        circ: &mut TwoQubitGateSequence,
        target_decomposed: &TwoQubitWeylDecomposition,
        atol: f64,
        c_mats: [Matrix2<Complex64>; 4],
        target_1q_basis_list: EulerBasisSet,
    ) -> PyResult<()> {
        // RXX(a)
        let (circ_a_name, circ_a_params, global_phase_a, rxx_mats) =
            self.to_rxx_gate(-2.0 * target_decomposed.a, false)?;
        let mut global_phase = global_phase_a;

        let mut c2r = c_mats[0]; // before weyl_gate, qubit 0
        let mut c2l = c_mats[1]; // before weyl_gate, qubit 1
        let mut c1r = c_mats[2]; // after weyl_gate, qubit 0
        let mut c1l = c_mats[3]; // after weyl_gate, qubit 1

        let rxx_k2r = rxx_mats[0]; // before RXX(a), qubit 0
        let rxx_k2l = rxx_mats[1]; // before RXX(a), qubit 1
        let rxx_k1r = rxx_mats[2]; // after RXX(a), qubit 0
        let rxx_k1l = rxx_mats[3]; // after RXX(a), qubit 1

        let mut ryy_k2r: Matrix2<Complex64>; // before RYY(b), qubit 0
        let mut ryy_k2l: Matrix2<Complex64>; // before RYY(b), qubit 1
        let mut ryy_k1r: Matrix2<Complex64>; // after RYY(b), qubit 0
        let mut ryy_k1l: Matrix2<Complex64>; // after RYY(b), qubit 1

        let mut rzz_k2r: Matrix2<Complex64>; // before RZZ(c), qubit 0
        let mut rzz_k2l: Matrix2<Complex64>; // before RZZ(c), qubit 1
        let mut rzz_k1r: Matrix2<Complex64>; // after RZZ(c), qubit 0
        let mut rzz_k1l: Matrix2<Complex64>; // after RZZ(c), qubit 1

        // before the weyl_gate
        c2r = rxx_k2r * c2r;
        c2l = rxx_k2l * c2l;

        self.append_1q_sequence(
            &mut circ.gates,
            &mut global_phase,
            c2r.as_view(),
            0,
            target_1q_basis_list,
        );
        self.append_1q_sequence(
            &mut circ.gates,
            &mut global_phase,
            c2l.as_view(),
            1,
            target_1q_basis_list,
        );
        circ.gates
            .push((circ_a_name, circ_a_params, smallvec![0, 1]));

        // translate RYY(b) into a circuit based on the desired Ctrl-U gate.
        if (target_decomposed.b).abs() > atol {
            let (circ_b_name, circ_b_params, global_phase_b, ryy_mats) =
                self.to_rxx_gate(-2.0 * target_decomposed.b, false)?;
            global_phase += global_phase_b;

            ryy_k2r = ryy_mats[0]; // before RYY(b), qubit 0
            ryy_k2l = ryy_mats[1]; // before RYY(b), qubit 1
            ryy_k1r = ryy_mats[2]; // after RYY(b), qubit 0
            ryy_k1l = ryy_mats[3]; // after RYY(b), qubit 1

            ryy_k2r = ryy_k2r * SDGGATE * rxx_k1r; // between RXX(a) and RYY(b), qubit 0
            ryy_k2l = ryy_k2l * SDGGATE * rxx_k1l; // between RXX(a) and RYY(b), qubit 1
            ryy_k1r = SGATE * ryy_k1r; // between RYY(b) and RZZ(c), qubit 0
            ryy_k1l = SGATE * ryy_k1l; // between RYY(b) and RZZ(c), qubit 1

            self.append_1q_sequence(
                &mut circ.gates,
                &mut global_phase,
                ryy_k2r.as_view(),
                0,
                target_1q_basis_list,
            );
            self.append_1q_sequence(
                &mut circ.gates,
                &mut global_phase,
                ryy_k2l.as_view(),
                1,
                target_1q_basis_list,
            );
            circ.gates
                .push((circ_b_name, circ_b_params, smallvec![0, 1]));
        } else {
            // no circ_b
            ryy_k1r = rxx_k1r;
            ryy_k1l = rxx_k1l;
        }

        // translate RZZ(c) into a circuit based on the desired Ctrl-U gate.
        if (target_decomposed.c).abs() > atol {
            // Since the Weyl chamber is here defined as a > b > |c| we may have
            // negative c. This will cause issues in _to_rxx_gate
            // as TwoQubitWeylControlledEquiv will map (c, 0, 0) to (|c|, 0, 0).
            // We therefore produce RZZ(|c|) and append its inverse to the
            // circuit if c < 0.
            let mut gamma = -2.0 * target_decomposed.c;
            if gamma <= 0.0 {
                let (circ_c_name, circ_c_params, global_phase_c, rzz_mats) =
                    self.to_rxx_gate(gamma, false)?;
                global_phase += global_phase_c;

                rzz_k2r = rzz_mats[0]; // before RZZ(c), qubit 0
                rzz_k2l = rzz_mats[1]; // before RZZ(c), qubit 1
                rzz_k1r = rzz_mats[2]; // after RZZ(c), qubit 0
                rzz_k1l = rzz_mats[3]; // after RZZ(c), qubit 1

                rzz_k2r = rzz_k2r * HGATE * ryy_k1r; // between RYY(b) and RZZ(c), qubit 0
                rzz_k2l = rzz_k2l * HGATE * ryy_k1l; // between RYY(b) and RZZ(c), qubit 1
                rzz_k1r = HGATE * rzz_k1r; // after RZZ(c), qubit 0
                rzz_k1l = HGATE * rzz_k1l; // after RZZ(c), qubit 1

                self.append_1q_sequence(
                    &mut circ.gates,
                    &mut global_phase,
                    rzz_k2r.as_view(),
                    0,
                    target_1q_basis_list,
                );
                self.append_1q_sequence(
                    &mut circ.gates,
                    &mut global_phase,
                    rzz_k2l.as_view(),
                    1,
                    target_1q_basis_list,
                );
                circ.gates
                    .push((circ_c_name, circ_c_params, smallvec![0, 1]));
            } else {
                // invert the circuit above
                gamma *= -1.0;
                // the inverted 2-qubit RXX gate
                let (circ_c_name, circ_c_params, global_phase_c, rzz_mats) =
                    self.to_rxx_gate(gamma, true)?;
                global_phase -= global_phase_c;

                // the inverted 1-qubit matrices
                rzz_k2r = rzz_mats[0]; // before RZZ(c), qubit 0
                rzz_k2l = rzz_mats[1]; // before RZZ(c), qubit 1
                rzz_k1r = rzz_mats[2]; // after RZZ(c), qubit 0
                rzz_k1l = rzz_mats[3]; // after RZZ(c), qubit 1

                rzz_k2r = rzz_k2r * HGATE * ryy_k1r; // between RYY(b) and RZZ(c), qubit 0
                rzz_k2l = rzz_k2l * HGATE * ryy_k1l; // between RYY(b) and RZZ(c), qubit 1
                rzz_k1r = HGATE * rzz_k1r; // after RZZ(c), qubit 0
                rzz_k1l = HGATE * rzz_k1l; // after RZZ(c), qubit 1

                self.append_1q_sequence(
                    &mut circ.gates,
                    &mut global_phase,
                    rzz_k2r.as_view(),
                    0,
                    target_1q_basis_list,
                );
                self.append_1q_sequence(
                    &mut circ.gates,
                    &mut global_phase,
                    rzz_k2l.as_view(),
                    1,
                    target_1q_basis_list,
                );
                circ.gates
                    .push((circ_c_name, circ_c_params, smallvec![0, 1]));
            }
        } else {
            // no circ_c
            rzz_k1r = ryy_k1r;
            rzz_k1l = ryy_k1l;
        }

        // after the weyl_gate
        c1r *= rzz_k1r;
        c1l *= rzz_k1l;

        self.append_1q_sequence(
            &mut circ.gates,
            &mut global_phase,
            c1r.as_view(),
            0,
            target_1q_basis_list,
        );
        self.append_1q_sequence(
            &mut circ.gates,
            &mut global_phase,
            c1l.as_view(),
            1,
            target_1q_basis_list,
        );

        circ.global_phase = global_phase;
        Ok(())
    }

    ///  Returns the Weyl decomposition in circuit form.
    ///  Note: atol is passed to OneQubitEulerDecomposer.
    pub fn call_inner(
        &self,
        unitary: ArrayView2<Complex64>,
        atol: Option<f64>,
    ) -> PyResult<TwoQubitGateSequence> {
        let target_decomposed =
            TwoQubitWeylDecomposition::new_inner(unitary, Some(DEFAULT_FIDELITY), None)?;

        let mut target_1q_basis_list = EulerBasisSet::new();
        target_1q_basis_list.add_basis(self.euler_basis);

        let c2r: Matrix2<Complex64> = target_decomposed.K2r; // qubit 0, before weyl_gate
        let c2l: Matrix2<Complex64> = target_decomposed.K2l; // qubit 1, before weyl_gate
        let c1r: Matrix2<Complex64> = target_decomposed.K1r; // qubit 0, after weyl_gate
        let c1l: Matrix2<Complex64> = target_decomposed.K1l; // qubit 1, after weyl_gate

        // Capacity = 5*8 + 3 = 43
        // 3 2-qubit gates
        // 8 1-qubit unitaries
        // max 5 1-qubit gates per 1-qubit unitary
        let gates = Vec::with_capacity(43);
        let global_phase = target_decomposed.global_phase;

        let mut weyl_gates = TwoQubitGateSequence {
            gates,
            global_phase,
        };
        self.weyl_gate(
            &mut weyl_gates,
            &target_decomposed,
            atol.unwrap_or(DEFAULT_ATOL),
            [c2r, c2l, c1r, c1l],
            target_1q_basis_list,
        )?;
        weyl_gates.global_phase += global_phase;

        Ok(weyl_gates)
    }

    /// Initialize the KAK decomposition.
    pub fn new_inner(rxx_equivalent_gate: RXXEquivalent, euler_basis: &str) -> PyResult<Self> {
        let atol = DEFAULT_ATOL;
        let test_angles = [0.2, 0.3, FRAC_PI_2];

        let scales: PyResult<Vec<f64>> = test_angles
            .into_iter()
            .map(|test_angle| {
                match &rxx_equivalent_gate {
                    RXXEquivalent::Standard(gate) => {
                        if gate.num_params() != 1 {
                            return Err(QiskitError::new_err(
                                "Equivalent gate needs to take exactly 1 angle parameter.",
                            ));
                        }
                    }
                    RXXEquivalent::CustomPython(gate_cls) => {
                        let takes_param = Python::attach(|py: Python| {
                            gate_cls.bind(py).call1((test_angle,)).ok().is_none()
                        });
                        if takes_param {
                            return Err(QiskitError::new_err(
                                "Equivalent gate needs to take exactly 1 angle parameter.",
                            ));
                        }
                    }
                };
                let mat = rxx_equivalent_gate.matrix(test_angle)?;
                let decomp =
                    TwoQubitWeylDecomposition::new_inner(mat.view(), Some(DEFAULT_FIDELITY), None)?;
                let mat_rxx = StandardGate::RXX
                    .matrix(&[Param::Float(test_angle)])
                    .unwrap();
                let decomposer_rxx = TwoQubitWeylDecomposition::new_inner(
                    mat_rxx.view(),
                    None,
                    Some(Specialization::ControlledEquiv),
                )?;
                let decomposer_equiv = TwoQubitWeylDecomposition::new_inner(
                    mat.view(),
                    Some(DEFAULT_FIDELITY),
                    Some(Specialization::ControlledEquiv),
                )?;
                let scale_a = decomposer_rxx.a / decomposer_equiv.a;
                if (decomp.a * 2.0 - test_angle / scale_a).abs() > atol {
                    return Err(QiskitError::new_err(
                        "The provided gate is not equivalent to an RXX.",
                    ));
                }
                Ok(scale_a)
            })
            .collect();
        let scales = scales?;

        let scale = scales[0];

        // Check that all three tested angles give the same scale
        for scale_val in &scales {
            if !abs_diff_eq!(scale_val, &scale, epsilon = atol) {
                return Err(QiskitError::new_err(
                    "Inconsistent scaling parameters in check.",
                ));
            }
        }

        Ok(TwoQubitControlledUDecomposer {
            scale,
            rxx_equivalent_gate,
            euler_basis: EulerBasis::__new__(euler_basis)?,
        })
    }

    // Note: this function also appears in TwoQubitBasisDecomposer)
    fn append_1q_sequence(
        &self,
        gates: &mut TwoQubitSequenceVec,
        global_phase: &mut f64,
        unitary: MatrixView2<Complex64>,
        qubit: u8,
        target_1q_basis_list: EulerBasisSet,
    ) {
        let sequence = unitary_to_gate_sequence_inner(
            nalgebra_array_view::<Complex64, U2, U2>(unitary),
            &target_1q_basis_list,
            qubit as usize,
            None,
            true,
            None,
        );
        if let Some(sequence) = sequence {
            *global_phase += sequence.global_phase;
            for gate in sequence.gates {
                gates.push((gate.0.into(), gate.1, smallvec![qubit]));
            }
        }
    }
}

#[pymethods]
impl TwoQubitControlledUDecomposer {
    fn __getnewargs__(&self) -> (&RXXEquivalent, &str) {
        (&self.rxx_equivalent_gate, self.euler_basis.as_str())
    }

    ///  Initialize the KAK decomposition.
    ///  Args:
    ///      rxx_equivalent_gate: Gate that is locally equivalent to an :class:`.RXXGate`:
    ///      :math:`U \sim U_d(\alpha, 0, 0) \sim \text{Ctrl-U}` gate.
    ///     euler_basis: Basis string to be provided to :class:`.OneQubitEulerDecomposer`
    ///     for 1Q synthesis.
    ///  Raises:
    ///      QiskitError: If the gate is not locally equivalent to an :class:`.RXXGate`.
    #[new]
    #[pyo3(signature=(rxx_equivalent_gate, euler_basis="ZXZ"))]
    pub fn new(rxx_equivalent_gate: RXXEquivalent, euler_basis: &str) -> PyResult<Self> {
        TwoQubitControlledUDecomposer::new_inner(rxx_equivalent_gate, euler_basis)
    }

    #[pyo3(signature=(unitary, atol=None))]
    fn __call__(
        &self,
        unitary: PyReadonlyArray2<Complex64>,
        atol: Option<f64>,
    ) -> PyResult<PyCircuitData> {
        let sequence = self.call_inner(unitary.as_array(), atol)?;
        Ok(CircuitData::from_packed_operations(
            2,
            0,
            sequence.gates.into_iter().map(|(gate, params, qubits)| {
                Ok((
                    gate,
                    params.into_iter().map(Param::Float).collect(),
                    qubits.into_iter().map(|x| Qubit(x as u32)).collect(),
                    vec![],
                ))
            }),
            Param::Float(sequence.global_phase),
        )?
        .into())
    }
}
