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

use std::f64::consts::PI;
use std::ops::Neg;
#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

use approx::abs_diff_eq;
use hashbrown::HashMap;
use nalgebra::{DMatrix, DVector, Matrix4};
use ndarray::prelude::*;
use num_complex::Complex64;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::Python;
use smallvec::smallvec;

use crate::euler_one_qubit_decomposer::{
    unitary_to_gate_sequence_inner, EulerBasis, EulerBasisSet,
};
use crate::linalg::{cos_sin_decomp, is_hermitian_matrix};
use crate::two_qubit_decompose::{two_qubit_decompose_up_to_diagonal, TwoQubitBasisDecomposer};
use qiskit_circuit::bit::ShareableQubit;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::interner::Interned;
use qiskit_circuit::operations::{
    ArrayType, Operation, OperationRef, Param, StandardGate, UnitaryGate,
};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::{Qubit, VarsMode};
use qiskit_quantum_info::convert_2q_block_matrix::instructions_to_matrix;

const EPS: f64 = 1e-10;

pub fn quantum_shannon_decomposition(
    py: Python,
    mat: &DMatrix<Complex64>,
    opt_a1: bool,
    opt_a2: bool,
    two_qubit_decomposer: Option<&TwoQubitBasisDecomposer>,
    one_qubit_decomposer: Option<&EulerBasisSet>,
) -> PyResult<CircuitData> {
    let dim = mat.shape().0;
    let num_qubits = dim.ilog2() as usize;
    if abs_diff_eq!(DMatrix::identity(dim, dim), mat) {
        let out_qubits = (0..num_qubits)
            .map(|_| ShareableQubit::new_anonymous())
            .collect::<Vec<_>>();
        return CircuitData::new(Some(out_qubits), None, None, 0, Param::Float(0.));
    }
    qsd_inner(
        py,
        mat,
        opt_a1,
        opt_a2,
        two_qubit_decomposer,
        one_qubit_decomposer,
        0,
    )
}

fn qsd_inner(
    py: Python,
    mat: &DMatrix<Complex64>,
    opt_a1: bool,
    opt_a2: bool,
    two_qubit_decomposer: Option<&TwoQubitBasisDecomposer>,
    one_qubit_decomposer: Option<&EulerBasisSet>,
    depth: usize,
) -> PyResult<CircuitData> {
    let dim = mat.shape().0;
    let num_qubits = dim.ilog2() as usize;
    let mut default_1q_basis = EulerBasisSet::new();
    default_1q_basis.add_basis(EulerBasis::U);
    let default_2q_decomposer = TwoQubitBasisDecomposer::new_inner(
        StandardGate::CX.into(),
        smallvec![],
        StandardGate::CX.matrix(&[]).unwrap().view(),
        1.0,
        "U",
        None,
    )?;
    let one_qubit_decomposer = one_qubit_decomposer.unwrap_or(&default_1q_basis);
    let two_qubit_decomposer = two_qubit_decomposer.unwrap_or(&default_2q_decomposer);
    if dim == 2 {
        let array = Array2::from_shape_fn((dim, dim), |(i, j)| mat[(i, j)]);
        let sequence =
            unitary_to_gate_sequence_inner(array.view(), one_qubit_decomposer, 0, None, true, None);

        return match sequence {
            Some(seq) => CircuitData::from_standard_gates(
                1,
                seq.gates.into_iter().map(|(gate, params)| {
                    (
                        gate,
                        params.into_iter().map(Param::Float).collect(),
                        smallvec![Qubit(0)],
                    )
                }),
                Param::Float(seq.global_phase),
            ),
            None => {
                let out_qubits = (0..num_qubits)
                    .map(|_| ShareableQubit::new_anonymous())
                    .collect::<Vec<_>>();
                CircuitData::new(Some(out_qubits), None, None, 0, Param::Float(0.))
            }
        };
    } else if dim == 4 {
        if opt_a2 && depth > 0 {
            let out_qubits = (0..num_qubits)
                .map(|_| ShareableQubit::new_anonymous())
                .collect::<Vec<_>>();
            let mut out = CircuitData::new(Some(out_qubits), None, None, 0, Param::Float(0.))?;
            let two_q_mat: Matrix4<Complex64> = Matrix4::from_fn(|i, j| mat[(i, j)]);
            let packed_inst = PackedInstruction {
                op: PackedOperation::from_unitary(Box::new(UnitaryGate {
                    array: ArrayType::TwoQ(two_q_mat),
                })),
                qubits: out.add_qargs(&[Qubit(0), Qubit(1)]),
                clbits: Default::default(),
                params: None,
                label: Some(Box::new("qsd2q".to_string())),
                #[cfg(feature = "cache_pygates")]
                py_op: OnceLock::new(),
            };
            out.push(packed_inst)?;
            return Ok(out);
        }
        let array = Array2::from_shape_fn((dim, dim), |(i, j)| mat[(i, j)]);
        let sequence = two_qubit_decomposer
            .call_inner(array.view(), None, false, None)
            .unwrap();
        return CircuitData::from_packed_operations(
            num_qubits as u32,
            0,
            sequence.gates().iter().map(|(op, params, qubits)| {
                Ok((
                    op.clone(),
                    params.iter().map(|x| Param::Float(*x)).collect(),
                    qubits.iter().map(|q| Qubit(*q as u32)).collect(),
                    vec![],
                ))
            }),
            Param::Float(sequence.global_phase()),
        );
    }
    // Check whether the matrix is equivalent to a block diagonal w.r.t ctrl_index
    if !opt_a2 {
        for ctrl_index in 0..num_qubits {
            let [um00, um11, um01, um10] = extract_multiplex_blocks(mat, ctrl_index);

            if off_diagonals_are_zero(&um01, &um10, None) {
                return demultiplex(
                    py,
                    &um00,
                    &um11,
                    opt_a1,
                    opt_a2,
                    depth,
                    Some(num_qubits - 1 - ctrl_index),
                );
            }
        }
    }
    let out_qubits = (0..num_qubits)
        .map(|_| ShareableQubit::new_anonymous())
        .collect::<Vec<_>>();
    let mut out = CircuitData::new(Some(out_qubits), None, None, 0, Param::Float(0.)).unwrap();
    let res = cos_sin_decomp::cos_sin_decomposition(mat);
    let u1 = res.l0;
    let mut u2 = res.l1;
    let vtheta = res.thetas;
    let v1h = res.r0;
    let v2h = res.r1;
    let left_circuit = demultiplex(py, &v1h, &v2h, opt_a1, opt_a2, depth, None)?;
    let qr = (0..num_qubits).map(Qubit::new).collect::<Vec<_>>();
    append(&mut out, left_circuit, &qr)?;
    if opt_a1 {
        let num_angles = vtheta.len();
        let half_size = num_angles / 2;
        let mut angles = vtheta.iter().map(|x| 2. * x).collect::<Vec<_>>();
        let circ_cz = get_ucry_cz(num_qubits as u32, &mut angles)?;
        append(&mut out, circ_cz, &qr)?;
        // merge final cz with right-side generic multiplexer
        //
        // u2[:, half_size:] = np.negative(u2[:, half_size:])
        u2.view_mut((0, half_size), (num_angles, half_size))
            .iter_mut()
            .for_each(|x| *x = x.neg());
    } else {
        let mut angles = vtheta.iter().map(|x| 2. * x).collect::<Vec<_>>();
        let ucry_circ = decompose_uc_rotation(&mut angles, StandardGate::RY)?;
        append(
            &mut out,
            ucry_circ,
            &[
                std::slice::from_ref(qr.last().unwrap()),
                &qr[..num_qubits - 1],
            ]
            .concat(),
        )?;
    }
    let right_circuit = demultiplex(py, &u1, &u2, opt_a1, opt_a2, depth, None)?;
    append(&mut out, right_circuit, &qr)?;
    if opt_a2 && depth == 0 && dim > 4 {
        apply_a2(py, &out, two_qubit_decomposer)
    } else {
        Ok(out)
    }
}

fn demultiplex(
    py: Python,
    um0: &DMatrix<Complex64>,
    um1: &DMatrix<Complex64>,
    opt_a1: bool,
    opt_a2: bool,
    depth: usize,
    _ctrl_index: Option<usize>,
) -> PyResult<CircuitData> {
    let dim = um0.shape().0 + um1.shape().0;
    let num_qubits = dim.ilog2() as usize;
    let _ctrl_index = _ctrl_index.unwrap_or_else(|| num_qubits - 1);
    let layout: Vec<Qubit> = (0.._ctrl_index)
        .chain(_ctrl_index + 1..num_qubits)
        .chain([_ctrl_index])
        .map(Qubit::new)
        .collect();
    let um0um1 = um0 * um1.adjoint();
    let (eigvals, vmat) = if is_hermitian_matrix(&um0um1) {
        let eigh = um0um1.symmetric_eigen();
        let evals = eigh.eigenvalues;
        let eigvals = evals.map(|x| Complex64::new(x, 0.));
        (eigvals, eigh.eigenvectors)
    } else {
        let shur = nalgebra::linalg::Schur::try_new(um0um1, 1e-12, 100000).unwrap();
        let (vmat, evals) = shur.unpack();
        let eigvals = evals.diagonal();
        (eigvals, vmat)
    };
    let d_values: DVector<Complex64> = eigvals.map(|x| x.sqrt());
    let d_mat: DMatrix<Complex64> = DMatrix::from_diagonal(&d_values);
    let w_mat = d_mat * vmat.adjoint() * um1;
    let left_circuit = qsd_inner(py, &w_mat, opt_a1, opt_a2, None, None, depth + 1)?;
    let out_qubits = (0..num_qubits)
        .map(|_| ShareableQubit::new_anonymous())
        .collect::<Vec<_>>();
    let mut out = CircuitData::new(Some(out_qubits), None, None, 0, Param::Float(0.))?;
    append(&mut out, left_circuit, &layout[..num_qubits - 1])?;
    let mut angles = d_values
        .iter()
        .map(|x| 2. * x.conj().arg())
        .collect::<Vec<_>>();
    let multiplexed_rz = decompose_uc_rotation(&mut angles, StandardGate::RZ)?;
    append(
        &mut out,
        multiplexed_rz,
        &[
            std::slice::from_ref(layout.last().unwrap()),
            &layout[..num_qubits - 1],
        ]
        .concat(),
    )?;
    let right_circuit = qsd_inner(py, &vmat, opt_a1, opt_a2, None, None, depth + 1)?;
    append(&mut out, right_circuit, &layout[..num_qubits - 1])?;
    Ok(out)
}

fn decompose_uc_rotation(
    angle_list: &mut [f64],
    rotation_axis: StandardGate,
) -> PyResult<CircuitData> {
    let num_controls = angle_list.len().ilog2();
    let num_qubits = num_controls + 1;
    let out_qubits = (0..num_qubits)
        .map(|_| ShareableQubit::new_anonymous())
        .collect::<Vec<_>>();
    let mut out = CircuitData::new(Some(out_qubits), None, None, 0, Param::Float(0.))?;
    if num_qubits < 2 {
        if angle_list[0].abs() > 0. {
            out.push_standard_gate(rotation_axis, &[Param::Float(angle_list[0])], &[Qubit(0)]);
        }
    } else {
        decompose_uc_rotations(angle_list, 0, angle_list.len(), false);
        for (i, angle) in angle_list.iter().enumerate() {
            if angle.abs() > EPS {
                out.push_standard_gate(rotation_axis, &[Param::Float(*angle)], &[Qubit(0)]);
            }
            let q_ctrl_index = if i != angle_list.len() - 1 {
                (i + 1).trailing_zeros()
            } else {
                num_controls
            };
            if matches!(rotation_axis, StandardGate::X) {
                out.push_standard_gate(StandardGate::RY, &[Param::Float(PI / 2.)], &[Qubit(0)]);
            }
            out.push_standard_gate(StandardGate::CX, &[], &[Qubit(q_ctrl_index), Qubit(0)]);
            if matches!(rotation_axis, StandardGate::X) {
                out.push_standard_gate(StandardGate::RY, &[Param::Float(-PI / 2.)], &[Qubit(0)]);
            }
        }
    }
    Ok(out)
}

fn decompose_uc_rotations(
    angles: &mut [f64],
    start_index: usize,
    end_index: usize,
    reversed_decomposition: bool,
) {
    let interval_len_half = (end_index - start_index) / 2;
    for i in start_index..start_index + interval_len_half {
        if !reversed_decomposition {
            let new_angles = update_angle(angles[i], angles[i + interval_len_half]);
            angles[i] = new_angles[0];
            angles[i + interval_len_half] = new_angles[1];
        } else {
            let new_angles = update_angle(angles[i], angles[i + interval_len_half]);
            angles[i + interval_len_half] = new_angles[0];
            angles[i] = new_angles[1];
        }
    }
    if interval_len_half > 1 {
        decompose_uc_rotations(angles, start_index, start_index + interval_len_half, false);
        decompose_uc_rotations(angles, start_index + interval_len_half, end_index, true);
    }
}

fn update_angle(angle_1: f64, angle_2: f64) -> [f64; 2] {
    [(angle_1 + angle_2) / 2., (angle_1 - angle_2) / 2.]
}

fn get_ucry_cz(num_qubits: u32, angles: &mut [f64]) -> PyResult<CircuitData> {
    let out_qubits = (0..num_qubits)
        .map(|_| ShareableQubit::new_anonymous())
        .collect::<Vec<_>>();
    let mut out = CircuitData::new(Some(out_qubits), None, None, 0, Param::Float(0.))?;
    if num_qubits < 2 {
        if angles[0].abs() > EPS {
            out.push_standard_gate(StandardGate::RY, &[Param::Float(angles[0])], &[Qubit(0)]);
        }
    } else {
        decompose_uc_rotations(angles, 0, angles.len(), false);
        for (i, angle) in angles.iter().enumerate() {
            if angle.abs() > EPS {
                out.push_standard_gate(StandardGate::RY, &[Param::Float(*angle)], &[Qubit(0)]);
            }
            let q_ctrl_index = if i != angles.len() - 1 {
                (i + 1).trailing_zeros()
            } else {
                num_qubits - 1
            };
            if i < angles.len() - 1 {
                out.push_standard_gate(
                    StandardGate::CZ,
                    &[],
                    &[Qubit(q_ctrl_index), Qubit(num_qubits - 1)],
                );
            }
        }
    }
    Ok(out)
}

fn apply_a2(
    py: Python,
    circ: &CircuitData,
    two_qubit_decomposer: &TwoQubitBasisDecomposer,
) -> PyResult<CircuitData> {
    let ind2q: Vec<usize> = circ
        .data()
        .iter()
        .enumerate()
        .filter_map(|(idx, inst)| {
            if matches!(inst.op.view(), OperationRef::Unitary(_)) {
                if let Some(ref label) = inst.label {
                    if label.as_str() == "qsd2q" {
                        return Some(idx);
                    }
                }
            }
            None
        })
        .collect();
    if ind2q.is_empty() {
        return Ok(circ.clone());
    } else if ind2q.len() == 1 {
        let mut circ = circ.clone();
        circ.data_mut()[ind2q[0]].label = None;
        return Ok(circ);
    }
    let mut diagonal_rollover: HashMap<usize, CircuitData> = HashMap::with_capacity(ind2q.len());
    let mut new_matrices: HashMap<usize, Array2<Complex64>> = ind2q
        .iter()
        .map(|idx| {
            let OperationRef::Unitary(unitary) = circ.data()[*idx].op.view() else {
                unreachable!("diagonal unitary is not a unitary gate");
            };
            (*idx, unitary.matrix_view().to_owned())
        })
        .collect();
    for (ind1, ind2) in ind2q[0..ind2q.len() - 1].iter().zip(ind2q[1..].iter()) {
        let mat1 = match diagonal_rollover.get(ind1) {
            Some(circ) => instructions_to_matrix(
                py,
                circ.data().iter(),
                [Qubit(0), Qubit(1)],
                circ.qargs_interner(),
            )?,
            None => new_matrices[ind1].to_owned(),
        };
        let mat2 = match diagonal_rollover.get(ind2) {
            Some(circ) => instructions_to_matrix(
                py,
                circ.data().iter(),
                [Qubit(0), Qubit(1)],
                circ.qargs_interner(),
            )?,
            None => new_matrices[ind2].to_owned(),
        };
        let (diagonal_mat, qc2cx) = two_qubit_decompose_up_to_diagonal(mat1.view())?;
        diagonal_rollover.insert(*ind1, qc2cx);
        let new_mat2 = mat2.dot(&diagonal_mat);
        new_matrices.insert(*ind2, new_mat2);
    }
    let last_idx = ind2q.last().unwrap();
    let qc3_seq =
        two_qubit_decomposer.call_inner(new_matrices[last_idx].view(), None, true, None)?;
    let qc3 = CircuitData::from_packed_operations(
        2,
        0,
        qc3_seq.gates().iter().map(|(gate, params, qubits)| {
            Ok((
                gate.clone(),
                params.iter().map(|x| Param::Float(*x)).collect(),
                qubits.iter().map(|q| Qubit(*q as u32)).collect(),
                vec![],
            ))
        }),
        Param::Float(qc3_seq.global_phase()),
    )?;
    diagonal_rollover.insert(*last_idx, qc3);
    let mut out_circ = CircuitData::clone_empty_like(
        circ,
        Some(
            circ.data().len()
                + diagonal_rollover
                    .values()
                    .map(|x| x.data().len())
                    .sum::<usize>()
                - ind2q.len(),
        ),
        VarsMode::Alike,
    )?;
    for (idx, inst) in circ.data().iter().enumerate() {
        if let Some(new_circ) = diagonal_rollover.get(&idx) {
            let block_index_map = circ.get_qargs(circ.data()[idx].qubits);

            // Build interned key map for possible combinations of 2q unitary qargs in out circuit
            let qarg_vals = [
                out_circ.add_qargs(&[block_index_map[0]]),
                out_circ.add_qargs(&[block_index_map[1]]),
                out_circ.add_qargs(&[block_index_map[0], block_index_map[1]]),
                out_circ.add_qargs(&[block_index_map[1], block_index_map[0]]),
            ];
            // Build lookup table for possible interned qubits in custom decomposition
            // in 2q new circuit
            let interned_default = out_circ.qargs_interner().get_default();
            let interned_map: [&[Qubit]; 4] = [
                &[Qubit(0)],
                &[Qubit(1)],
                &[Qubit(0), Qubit(1)],
                &[Qubit(1), Qubit(0)],
            ];
            let interned_map = interned_map.map(|qubits| {
                new_circ
                    .qargs_interner()
                    .try_key(qubits)
                    .unwrap_or(interned_default)
            });
            // Map new circuit interned qubits -> out_circuit interned qubits
            let qarg_lookup = |qargs: Interned<[Qubit]>| {
                qarg_vals[interned_map
                    .iter()
                    .position(|key| qargs == *key)
                    .expect("not a 1q or 2q gate in the qargs block")]
            };
            for inst in new_circ.data() {
                let out_inst = PackedInstruction {
                    op: inst.op.clone(),
                    qubits: qarg_lookup(inst.qubits),
                    clbits: inst.clbits,
                    params: inst.params.clone(),
                    label: inst.label.clone(),
                    #[cfg(feature = "cache_pygates")]
                    py_op: OnceLock::new(),
                };
                out_circ.push(out_inst)?;
            }
        } else {
            out_circ.push(inst.clone())?;
        }
    }
    Ok(out_circ)
}

fn append(circ: &mut CircuitData, new: CircuitData, qubit_map: &[Qubit]) -> PyResult<()> {
    for inst in new.data() {
        let qubits_map: Vec<Qubit> = new
            .get_qargs(inst.qubits)
            .iter()
            .map(|x| qubit_map[x.index()])
            .collect();
        let out_inst = PackedInstruction {
            op: inst.op.clone(),
            params: inst.params.clone(),
            qubits: circ.add_qargs(&qubits_map),
            clbits: Default::default(),
            label: inst.label.clone(),
            #[cfg(feature = "cache_pygates")]
            py_op: inst.py_op.clone(),
        };
        circ.push(out_inst)?;
    }
    Ok(())
}

fn extract_multiplex_blocks(umat: &DMatrix<Complex64>, k: usize) -> [DMatrix<Complex64>; 4] {
    let dim = umat.shape().0;
    let num_qubits = dim.ilog2() as usize;
    let half_dim = dim / 2;
    let umat_array: Array2<Complex64> = Array2::from_shape_fn(umat.shape(), |(i, j)| umat[(i, j)]);
    let mut utensor = umat_array.to_shape(vec![2; 2 * num_qubits]).unwrap();
    if k != 0 {
        utensor.swap_axes(k, 0);
        utensor.swap_axes(k + num_qubits, num_qubits)
    }
    let ud4 = utensor.to_shape([2, half_dim, 2, half_dim]).unwrap();
    let um00 = ud4.slice(s![0, .., 0, ..]);
    let um11 = ud4.slice(s![1, .., 1, ..]);
    let um01 = ud4.slice(s![0, .., 1, ..]);
    let um10 = ud4.slice(s![1, .., 0, ..]);
    [
        DMatrix::from_fn(um00.shape()[0], um00.shape()[1], |i, j| um00[[i, j]]),
        DMatrix::from_fn(um11.shape()[0], um11.shape()[1], |i, j| um11[[i, j]]),
        DMatrix::from_fn(um01.shape()[0], um01.shape()[1], |i, j| um01[[i, j]]),
        DMatrix::from_fn(um10.shape()[0], um10.shape()[1], |i, j| um01[[i, j]]),
    ]
}

fn off_diagonals_are_zero(
    um01: &DMatrix<Complex64>,
    um10: &DMatrix<Complex64>,
    atol: Option<f64>,
) -> bool {
    let atol = atol.unwrap_or(1e-12);
    um01.iter()
        .all(|x| abs_diff_eq!(*x, Complex64::ZERO, epsilon = atol))
        && um10
            .iter()
            .all(|x| abs_diff_eq!(*x, Complex64::ZERO, epsilon = atol))
}

#[pyfunction]
pub fn qs_decomposition(
    py: Python,
    mat: PyReadonlyArray2<Complex64>,
    opt_a1: bool,
    opt_a2: bool,
) -> PyResult<CircuitData> {
    let array: ArrayView2<Complex64> = mat.as_array();
    let mat = DMatrix::from_fn(array.shape()[0], array.shape()[1], |i, j| array[[i, j]]);
    quantum_shannon_decomposition(py, &mat, opt_a1, opt_a2, None, None)
}

pub fn qsd_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(qs_decomposition, m)?)?;
    Ok(())
}
