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

use nalgebra::U2;
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
use qiskit_circuit::gate_matrix::{H_GATE, S_GATE, SDG_GATE};
use qiskit_circuit::operations::{Operation, OperationRef, Param, StandardGate};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{NoBlocks, Qubit};

use super::common::DEFAULT_FIDELITY;
use super::gate_sequence::TwoQubitGateSequence;
use super::weyl_decomposition::{Specialization, TwoQubitWeylDecomposition};

/// invert 1q gate sequence
fn invert_1q_gate(
    gate: (StandardGate, SmallVec<[f64; 3]>),
) -> (PackedOperation, SmallVec<[f64; 3]>) {
    let gate_params = gate.1.into_iter().map(Param::Float).collect::<Vec<_>>();
    let inv_gate = gate
        .0
        .inverse(&gate_params)
        .expect("An unexpected standard gate was inverted");
    let inv_gate_params = inv_gate
        .1
        .into_iter()
        .map(|param| match param {
            Param::Float(val) => val,
            _ => unreachable!("Parameterized inverse generated from non-parameterized gate."),
        })
        .collect::<SmallVec<_>>();
    (inv_gate.0.into(), inv_gate_params)
}

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
    ///  Returns:
    ///      Circuit: Circuit equivalent to an RXXGate.
    ///  Raises:
    ///      QiskitError: If the circuit is not equivalent to an RXXGate.
    fn to_rxx_gate(&self, angle: f64) -> PyResult<TwoQubitGateSequence> {
        // The user-provided RXX equivalent gate may be locally equivalent to the RXX gate
        // but with some scaling in the rotation angle. For example, RXX(angle) has Weyl
        // parameters (angle, 0, 0) for angle in [0, pi/2] but the user provided gate, i.e.
        // :code:`self.rxx_equivalent_gate(angle)` might produce the Weyl parameters
        // (scale * angle, 0, 0) where scale != 1. This is the case for the CPhase gate.

        let mat = self.rxx_equivalent_gate.matrix(self.scale * angle)?;
        let decomposer_inv =
            TwoQubitWeylDecomposition::new_inner(mat.view(), Some(DEFAULT_FIDELITY), None)?;

        let mut target_1q_basis_list = EulerBasisSet::new();
        target_1q_basis_list.add_basis(self.euler_basis);

        // Express the RXX in terms of the user-provided RXX equivalent gate.
        let mut gates = Vec::with_capacity(13);
        let mut global_phase = -decomposer_inv.global_phase;

        let decomp_k1r = nalgebra_array_view::<Complex64, U2, U2>(decomposer_inv.K1r.as_view());
        let decomp_k2r = nalgebra_array_view::<Complex64, U2, U2>(decomposer_inv.K2r.as_view());
        let decomp_k1l = nalgebra_array_view::<Complex64, U2, U2>(decomposer_inv.K1l.as_view());
        let decomp_k2l = nalgebra_array_view::<Complex64, U2, U2>(decomposer_inv.K2l.as_view());

        let unitary_k1r =
            unitary_to_gate_sequence_inner(decomp_k1r, &target_1q_basis_list, 0, None, true, None);
        let unitary_k2r =
            unitary_to_gate_sequence_inner(decomp_k2r, &target_1q_basis_list, 0, None, true, None);
        let unitary_k1l =
            unitary_to_gate_sequence_inner(decomp_k1l, &target_1q_basis_list, 0, None, true, None);
        let unitary_k2l =
            unitary_to_gate_sequence_inner(decomp_k2l, &target_1q_basis_list, 0, None, true, None);

        if let Some(unitary_k2r) = unitary_k2r {
            global_phase -= unitary_k2r.global_phase;
            for gate in unitary_k2r.gates.into_iter().rev() {
                let (inv_gate_name, inv_gate_params) = invert_1q_gate(gate);
                gates.push((inv_gate_name, inv_gate_params, smallvec![0]));
            }
        }
        if let Some(unitary_k2l) = unitary_k2l {
            global_phase -= unitary_k2l.global_phase;
            for gate in unitary_k2l.gates.into_iter().rev() {
                let (inv_gate_name, inv_gate_params) = invert_1q_gate(gate);
                gates.push((inv_gate_name, inv_gate_params, smallvec![1]));
            }
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
        gates.push((rxx_op, smallvec![self.scale * angle], smallvec![0, 1]));

        if let Some(unitary_k1r) = unitary_k1r {
            global_phase -= unitary_k1r.global_phase;
            for gate in unitary_k1r.gates.into_iter().rev() {
                let (inv_gate_name, inv_gate_params) = invert_1q_gate(gate);
                gates.push((inv_gate_name, inv_gate_params, smallvec![0]));
            }
        }
        if let Some(unitary_k1l) = unitary_k1l {
            global_phase -= unitary_k1l.global_phase;
            for gate in unitary_k1l.gates.into_iter().rev() {
                let (inv_gate_name, inv_gate_params) = invert_1q_gate(gate);
                gates.push((inv_gate_name, inv_gate_params, smallvec![1]));
            }
        }

        Ok(TwoQubitGateSequence {
            gates,
            global_phase,
        })
    }

    /// Appends U_d(a, b, c) to the circuit.
    fn weyl_gate(
        &self,
        circ: &mut TwoQubitGateSequence,
        target_decomposed: TwoQubitWeylDecomposition,
        atol: f64,
    ) -> PyResult<()> {
        let circ_a = self.to_rxx_gate(-2.0 * target_decomposed.a)?;
        circ.gates.extend(circ_a.gates);
        let mut global_phase = circ_a.global_phase;

        let mut target_1q_basis_list = EulerBasisSet::new();
        target_1q_basis_list.add_basis(self.euler_basis);

        let s_decomp = unitary_to_gate_sequence_inner(
            aview2(&S_GATE),
            &target_1q_basis_list,
            0,
            None,
            true,
            None,
        );
        let sdg_decomp = unitary_to_gate_sequence_inner(
            aview2(&SDG_GATE),
            &target_1q_basis_list,
            0,
            None,
            true,
            None,
        );
        let h_decomp = unitary_to_gate_sequence_inner(
            aview2(&H_GATE),
            &target_1q_basis_list,
            0,
            None,
            true,
            None,
        );

        // translate RYY(b) into a circuit based on the desired Ctrl-U gate.
        if (target_decomposed.b).abs() > atol {
            let circ_b = self.to_rxx_gate(-2.0 * target_decomposed.b)?;
            global_phase += circ_b.global_phase;
            if let Some(sdg_decomp) = sdg_decomp {
                global_phase += 2.0 * sdg_decomp.global_phase;
                for gate in sdg_decomp.gates.into_iter() {
                    let gate_params = gate.1;
                    circ.gates
                        .push((gate.0.into(), gate_params.clone(), smallvec![0]));
                    circ.gates.push((gate.0.into(), gate_params, smallvec![1]));
                }
            }
            circ.gates.extend(circ_b.gates);
            if let Some(s_decomp) = s_decomp {
                global_phase += 2.0 * s_decomp.global_phase;
                for gate in s_decomp.gates.into_iter() {
                    let gate_params = gate.1;
                    circ.gates
                        .push((gate.0.into(), gate_params.clone(), smallvec![0]));
                    circ.gates.push((gate.0.into(), gate_params, smallvec![1]));
                }
            }
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
                let circ_c = self.to_rxx_gate(gamma)?;
                global_phase += circ_c.global_phase;

                if let Some(ref h_decomp) = h_decomp {
                    global_phase += 2.0 * h_decomp.global_phase;
                    for gate in h_decomp.gates.clone().into_iter() {
                        let gate_params = gate.1;
                        circ.gates
                            .push((gate.0.into(), gate_params.clone(), smallvec![0]));
                        circ.gates.push((gate.0.into(), gate_params, smallvec![1]));
                    }
                }
                circ.gates.extend(circ_c.gates);
                if let Some(ref h_decomp) = h_decomp {
                    global_phase += 2.0 * h_decomp.global_phase;
                    for gate in h_decomp.gates.clone().into_iter() {
                        let gate_params = gate.1;
                        circ.gates
                            .push((gate.0.into(), gate_params.clone(), smallvec![0]));
                        circ.gates.push((gate.0.into(), gate_params, smallvec![1]));
                    }
                }
            } else {
                // invert the circuit above
                gamma *= -1.0;
                let circ_c = self.to_rxx_gate(gamma)?;
                global_phase -= circ_c.global_phase;
                if let Some(ref h_decomp) = h_decomp {
                    global_phase += 2.0 * h_decomp.global_phase;
                    for gate in h_decomp.gates.clone().into_iter() {
                        let gate_params = gate.1;
                        circ.gates
                            .push((gate.0.into(), gate_params.clone(), smallvec![0]));
                        circ.gates.push((gate.0.into(), gate_params, smallvec![1]));
                    }
                }
                for gate in circ_c.gates.into_iter().rev() {
                    let (inv_gate_name, inv_gate_params, inv_gate_qubits) =
                        self.invert_2q_gate(gate)?;
                    circ.gates
                        .push((inv_gate_name, inv_gate_params, inv_gate_qubits));
                }
                if let Some(ref h_decomp) = h_decomp {
                    global_phase += 2.0 * h_decomp.global_phase;
                    for gate in h_decomp.gates.clone().into_iter() {
                        let gate_params = gate.1;
                        circ.gates
                            .push((gate.0.into(), gate_params.clone(), smallvec![0]));
                        circ.gates.push((gate.0.into(), gate_params, smallvec![1]));
                    }
                }
            }
        }

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

        let c1r = nalgebra_array_view::<Complex64, U2, U2>(target_decomposed.K1r.as_view());
        let c2r = nalgebra_array_view::<Complex64, U2, U2>(target_decomposed.K2r.as_view());
        let c1l = nalgebra_array_view::<Complex64, U2, U2>(target_decomposed.K1l.as_view());
        let c2l = nalgebra_array_view::<Complex64, U2, U2>(target_decomposed.K2l.as_view());

        let unitary_c1r =
            unitary_to_gate_sequence_inner(c1r, &target_1q_basis_list, 0, None, true, None);
        let unitary_c2r =
            unitary_to_gate_sequence_inner(c2r, &target_1q_basis_list, 0, None, true, None);
        let unitary_c1l =
            unitary_to_gate_sequence_inner(c1l, &target_1q_basis_list, 0, None, true, None);
        let unitary_c2l =
            unitary_to_gate_sequence_inner(c2l, &target_1q_basis_list, 0, None, true, None);

        let mut gates = Vec::with_capacity(59);
        let mut global_phase = target_decomposed.global_phase;

        if let Some(unitary_c2r) = unitary_c2r {
            global_phase += unitary_c2r.global_phase;
            for gate in unitary_c2r.gates.into_iter() {
                gates.push((gate.0.into(), gate.1, smallvec![0]));
            }
        }
        if let Some(unitary_c2l) = unitary_c2l {
            global_phase += unitary_c2l.global_phase;
            for gate in unitary_c2l.gates.into_iter() {
                gates.push((gate.0.into(), gate.1, smallvec![1]));
            }
        }
        let mut gates1 = TwoQubitGateSequence {
            gates,
            global_phase,
        };
        self.weyl_gate(&mut gates1, target_decomposed, atol.unwrap_or(DEFAULT_ATOL))?;
        global_phase += gates1.global_phase;

        if let Some(unitary_c1r) = unitary_c1r {
            global_phase += unitary_c1r.global_phase;
            for gate in unitary_c1r.gates.into_iter() {
                gates1.gates.push((gate.0.into(), gate.1, smallvec![0]));
            }
        }
        if let Some(unitary_c1l) = unitary_c1l {
            global_phase += unitary_c1l.global_phase;
            for gate in unitary_c1l.gates.into_iter() {
                gates1.gates.push((gate.0.into(), gate.1, smallvec![1]));
            }
        }

        gates1.global_phase = global_phase;
        Ok(gates1)
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
