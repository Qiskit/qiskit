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

use std::thread::available_parallelism;

use qiskit_circuit::imports::UNITARY_GATE;

use pyo3::prelude::*;

use crate::getenv_use_multiple_threads;
use faer_ext::{IntoFaerComplex, IntoNdarrayComplex};
use ndarray::prelude::*;
use num_complex::Complex64;
use numpy::IntoPyArray;
use rand::prelude::*;
use rand_distr::Normal;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::Param;
use qiskit_circuit::operations::PyInstruction;
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::Qubit;
use smallvec::smallvec;

#[inline]
fn random_unitaries(seed: u64, size: usize) -> impl Iterator<Item = Array2<Complex64>> {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let dist = Normal::new(0., 1.0).unwrap();

    (0..size).map(move |_| {
        let mut z: Array2<Complex64> = Array2::from_shape_simple_fn((4, 4), || {
            Complex64::new(dist.sample(&mut rng), dist.sample(&mut rng))
        });
        z.mapv_inplace(|x| x * std::f64::consts::FRAC_1_SQRT_2);
        let qr = z.view().into_faer_complex().qr();
        let r = qr.compute_r().as_ref().into_ndarray_complex().to_owned();
        let mut d = r.into_diag();
        d.mapv_inplace(|d| d / d.norm());
        let mut q = qr.compute_q().as_ref().into_ndarray_complex().to_owned();
        q.axis_iter_mut(Axis(0)).for_each(|mut row| {
            row.iter_mut()
                .enumerate()
                .for_each(|(index, val)| *val *= d.diag()[index])
        });
        q
    })
}

#[pyfunction]
pub fn quantum_volume(
    py: Python,
    num_qubits: u32,
    depth: usize,
    seed: Option<u64>,
) -> PyResult<CircuitData> {
    let width = num_qubits as usize / 2;
    let num_unitaries = width * depth;
    let mut permutation: Vec<Qubit> = (0..num_qubits).map(Qubit).collect();

    let mut build_instruction = |(unitary_index, unitary_array): (usize, Array2<Complex64>),
                                 rng: &mut Pcg64Mcg| {
        let layer_index = unitary_index % width;
        if layer_index == 0 {
            permutation.shuffle(rng);
        }
        let unitary = unitary_array.into_pyarray_bound(py);
        let unitary_gate = UNITARY_GATE
            .get_bound(py)
            .call1((unitary.clone(), py.None(), false))
            .unwrap();
        let instruction = PyInstruction {
            qubits: 2,
            clbits: 0,
            params: 1,
            op_name: "unitary".to_string(),
            control_flow: false,
            instruction: unitary_gate.unbind(),
        };
        let qubit = layer_index * 2;
        (
            PackedOperation::from_instruction(Box::new(instruction)),
            smallvec![Param::Obj(unitary.unbind().into())],
            vec![permutation[qubit], permutation[qubit + 1]],
            vec![],
        )
    };

    if getenv_use_multiple_threads() {
        let mut per_thread = num_unitaries / available_parallelism().unwrap();
        if per_thread == 0 {
            if num_unitaries > 10 {
                per_thread = 10
            } else {
                per_thread = num_unitaries
            }
        }

        let mut outer_rng = match seed {
            Some(seed) => Pcg64Mcg::seed_from_u64(seed),
            None => Pcg64Mcg::from_entropy(),
        };
        let seed_vec: Vec<u64> = outer_rng
            .clone()
            .sample_iter(&rand::distributions::Standard)
            .take(num_unitaries)
            .collect();
        let unitaries: Vec<Array2<Complex64>> = seed_vec
            .into_par_iter()
            .chunks(per_thread)
            .flat_map_iter(|seeds| random_unitaries(seeds[0], seeds.len()))
            .collect();
        CircuitData::from_packed_operations(
            py,
            num_qubits,
            0,
            unitaries
                .into_iter()
                .enumerate()
                .map(|x| build_instruction(x, &mut outer_rng)),
            Param::Float(0.),
        )
    } else {
        let mut outer_rng = match seed {
            Some(seed) => Pcg64Mcg::seed_from_u64(seed),
            None => Pcg64Mcg::from_entropy(),
        };
        let seed: u64 = outer_rng.sample(rand::distributions::Standard);
        CircuitData::from_packed_operations(
            py,
            num_qubits,
            0,
            random_unitaries(seed, num_unitaries)
                .enumerate()
                .map(|x| build_instruction(x, &mut outer_rng)),
            Param::Float(0.),
        )
    }
}
