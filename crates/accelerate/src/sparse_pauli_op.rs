// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use hashbrown::HashMap;
use ndarray::{ArrayView1, ArrayView2, Axis};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};

/// Find the unique elements of an array.
///
/// This function is a drop-in replacement of
/// ``np.unique(array, return_index=True, return_inverse=True, axis=0)``
/// where ``array`` is a ``numpy.ndarray`` of ``dtype=u16`` and ``ndim=2``.
///
/// Note that the order of the output of this function is not sorted while ``numpy.unique``
/// returns the sorted elements.
///
/// Args:
///     array (numpy.ndarray): An array of ``dtype=u16`` and ``ndim=2``
///
/// Returns:
///     (indexes, inverses): A tuple of the following two indices.
///
///         - the indices of the input array that give the unique values
///         - the indices of the unique array that reconstruct the input array
///
#[pyfunction]
pub fn unordered_unique(py: Python, array: PyReadonlyArray2<u16>) -> (PyObject, PyObject) {
    let array = array.as_array();
    let shape = array.shape();
    let mut table = HashMap::<ArrayView1<u16>, usize>::with_capacity(shape[0]);
    let mut indices = Vec::new();
    let mut inverses = vec![0; shape[0]];
    for (i, v) in array.axis_iter(Axis(0)).enumerate() {
        match table.get(&v) {
            Some(id) => inverses[i] = *id,
            None => {
                let new_id = table.len();
                table.insert(v, new_id);
                inverses[i] = new_id;
                indices.push(i);
            }
        }
    }
    (
        indices.into_pyarray(py).into(),
        inverses.into_pyarray(py).into(),
    )
}

/// A complete ZX-convention representation of a Pauli decomposition.  This is all the components
/// necessary to construct a Qiskit-space :class:`.SparsePauliOp`, where :attr:`phases` is in the
/// ZX convention.
#[pyclass(module = "qiskit._accelerate.sparse_pauli_op")]
pub struct ZXPaulis {
    #[pyo3(get)]
    pub z: Py<PyArray2<bool>>,
    #[pyo3(get)]
    pub x: Py<PyArray2<bool>>,
    #[pyo3(get)]
    pub phases: Py<PyArray1<u8>>,
    #[pyo3(get)]
    pub coeffs: Py<PyArray1<Complex64>>,
}

struct DecomposeOut {
    z: Vec<bool>,
    x: Vec<bool>,
    phases: Vec<u8>,
    coeffs: Vec<Complex64>,
    scale: f64,
    tol: f64,
    num_qubits: usize,
}

/// Decompose a dense complex operator into the symplectic Pauli representation in the
/// ZX-convention.
///
/// This is an implementation of the "tensorized Pauli decomposition" presented in
/// `Hantzko, Binkowski and Gupta (2023) <https://arxiv.org/abs/2310.13421>`__.
#[pyfunction]
pub fn decompose_dense(
    py: Python,
    operator: PyReadonlyArray2<Complex64>,
    tolerance: f64,
) -> PyResult<ZXPaulis> {
    let num_qubits = operator.shape()[0].ilog2() as usize;
    let side = 1 << num_qubits;
    if operator.shape() != [side, side] {
        return Err(PyValueError::new_err(format!(
            "bad input shape for {} qubits: {:?}",
            num_qubits,
            operator.shape()
        )));
    }
    let (stack, mut out_list, mut scratch) =
        decompose_first_level(operator.as_array(), num_qubits);
    decompose_middle_levels(stack, &mut out_list, &mut scratch, num_qubits);
    let out = decompose_last_level(&mut out_list, &scratch, num_qubits, tolerance)?;
    Ok(ZXPaulis {
        z: PyArray1::from_vec(py, out.z)
            .reshape([out.phases.len(), num_qubits])?
            .into(),
        x: PyArray1::from_vec(py, out.x)
            .reshape([out.phases.len(), num_qubits])?
            .into(),
        phases: PyArray1::from_vec(py, out.phases).into(),
        coeffs: PyArray1::from_vec(py, out.coeffs).into(),
    })
}

/// Apply the matrix-addition decomposition at the first level.  This is split out because it acts
/// on an `ArrayView2`, and is responsible for populating the initial scratch space.  We can't
/// write over the operator the user gave us (it's not ours to do that to), and anyway, we want to
/// drop to a chunk of memory that we can 100% guarantee is contiguous, so we can elide all the
/// stride checking.
fn decompose_first_level(
    in_op: ArrayView2<Complex64>,
    num_qubits: usize,
) -> (Vec<PauliLocation>, Vec<PauliLocation>, Vec<Complex64>) {
    let side = 1 << num_qubits;
    let zero = Complex64::new(0.0, 0.0);
    let mut stack = Vec::<PauliLocation>::with_capacity(4);
    let mut out_list = Vec::<PauliLocation>::new();
    let mut scratch = Vec::<Complex64>::with_capacity(side * side);
    match num_qubits {
        0 => {}
        1 => {
            // If we've only got one qubit, we just want to copy the data over in the correct
            // continuity and let the base case of the iteration take care of outputting it.
            scratch.extend(in_op.iter());
            out_list.push(PauliLocation::begin(num_qubits));
        }
        _ => {
            unsafe { scratch.set_len(side * side) };
            let mut ptr = 0usize;

            let cur_qubit = num_qubits - 1;
            let mid = 1 << cur_qubit;
            let loc = PauliLocation::begin(num_qubits);
            let mut i_nonzero = false;
            let mut x_nonzero = false;
            let mut y_nonzero = false;
            let mut z_nonzero = false;

            let i_row_0 = loc.row();
            let i_col_0 = loc.col();

            let x_row_0 = loc.row();
            let x_col_0 = loc.col() + mid;

            let y_row_0 = loc.row() + mid;
            let y_col_0 = loc.col();

            let z_row_0 = loc.row() + mid;
            let z_col_0 = loc.col() + mid;

            for off_row in 0..mid {
                let i_row = i_row_0 + off_row;
                let z_row = z_row_0 + off_row;
                for off_col in 0..mid {
                    let i_col = i_col_0 + off_col;
                    let z_col = z_col_0 + off_col;
                    let value = in_op[[i_row, i_col]] + in_op[[z_row, z_col]];
                    scratch[ptr] = value;
                    ptr += 1;
                    i_nonzero = i_nonzero || (value != zero);
                }

                let x_row = x_row_0 + off_row;
                let y_row = y_row_0 + off_row;
                for off_col in 0..mid {
                    let x_col = x_col_0 + off_col;
                    let y_col = y_col_0 + off_col;
                    let value = in_op[[x_row, x_col]] + in_op[[y_row, y_col]];
                    scratch[ptr] = value;
                    ptr += 1;
                    x_nonzero = x_nonzero || (value != zero);
                }
            }
            for off_row in 0..mid {
                let x_row = x_row_0 + off_row;
                let y_row = y_row_0 + off_row;
                for off_col in 0..mid {
                    let x_col = x_col_0 + off_col;
                    let y_col = y_col_0 + off_col;
                    let value = in_op[[x_row, x_col]] - in_op[[y_row, y_col]];
                    scratch[ptr] = value;
                    ptr += 1;
                    y_nonzero = y_nonzero || (value != zero);
                }
                let i_row = i_row_0 + off_row;
                let z_row = z_row_0 + off_row;
                for off_col in 0..mid {
                    let i_col = i_col_0 + off_col;
                    let z_col = z_col_0 + off_col;
                    let value = in_op[[i_row, i_col]] - in_op[[z_row, z_col]];
                    scratch[ptr] = value;
                    ptr += 1;
                    z_nonzero = z_nonzero || (value != zero);
                }
            }
            // The middle-levels `stack` is a LIFO, so if we push in this order, we'll consider the
            // Pauli terms in lexicographical order, which is the canonical order from
            // `SparsePauliOp.sort`.  Populating the `out_list` (an initially empty `Vec`) effectively
            // reverses the stack, so we want to push its elements in the IXYZ order.
            if loc.qubit() == 1 {
                i_nonzero.then(|| out_list.push(loc.push_i()));
                x_nonzero.then(|| out_list.push(loc.push_x()));
                y_nonzero.then(|| out_list.push(loc.push_y()));
                z_nonzero.then(|| out_list.push(loc.push_z()));
            } else {
                z_nonzero.then(|| stack.push(loc.push_z()));
                y_nonzero.then(|| stack.push(loc.push_y()));
                x_nonzero.then(|| stack.push(loc.push_x()));
                i_nonzero.then(|| stack.push(loc.push_i()));
            }
        }
    }
    (stack, out_list, scratch)
}

/// Iteratively decompose the matrix at all levels other than the first and last.  This populates
/// the `out_stack` with locations
fn decompose_middle_levels(
    mut stack: Vec<PauliLocation>,
    out_list: &mut Vec<PauliLocation>,
    scratch: &mut [Complex64],
    num_qubits: usize,
) {
    let side = 1 << num_qubits;
    let zero = Complex64::new(0.0, 0.0);
    // This the stack is a is a LIFO queue, which is how we implement the depth-first iteration.
    // Depth-first means `stack` never grows very large; it reaches at most `3*num_qubits - 2`
    // elements (if all terms are zero all the way through the first subblock decomposition).
    // `out_stack`, on the other hand, can be `4 ** (num_qubits - 1)` entries in the worst-case
    // scenario of a completely dense (in Pauli terms) operator.
    while let Some(loc) = stack.pop() {
        // Here we work pairwise, writing out the new values into both I and Z simultaneously (etc
        // for X and Y) so we can re-use their scratch space and avoid re-allocating.  We're doing
        // the multiple assignment `(I, Z) = (I + Z, I - Z)`.
        let mid = 1 << loc.qubit();
        let mut i_nonzero = false;
        let mut z_nonzero = false;
        let i_row_0 = loc.row();
        let i_col_0 = loc.col();
        let z_row_0 = loc.row() + mid;
        let z_col_0 = loc.col() + mid;
        for off_row in 0..mid {
            let i_loc_0 = (i_row_0 + off_row) * side + i_col_0;
            let z_loc_0 = (z_row_0 + off_row) * side + z_col_0;
            for off_col in 0..mid {
                let i_loc = i_loc_0 + off_col;
                let z_loc = z_loc_0 + off_col;
                let add = scratch[i_loc] + scratch[z_loc];
                let sub = scratch[i_loc] - scratch[z_loc];
                scratch[i_loc] = add;
                scratch[z_loc] = sub;
                i_nonzero = i_nonzero || (add != zero);
                z_nonzero = z_nonzero || (sub != zero);
            }
        }

        let mut x_nonzero = false;
        let mut y_nonzero = false;
        let x_row_0 = loc.row();
        let x_col_0 = loc.col() + mid;
        let y_row_0 = loc.row() + mid;
        let y_col_0 = loc.col();
        for off_row in 0..mid {
            let x_loc_0 = (x_row_0 + off_row) * side + x_col_0;
            let y_loc_0 = (y_row_0 + off_row) * side + y_col_0;
            for off_col in 0..mid {
                let x_loc = x_loc_0 + off_col;
                let y_loc = y_loc_0 + off_col;
                let add = scratch[x_loc] + scratch[y_loc];
                let sub = scratch[x_loc] - scratch[y_loc];
                scratch[x_loc] = add;
                scratch[y_loc] = sub;
                x_nonzero = x_nonzero || (add != zero);
                y_nonzero = y_nonzero || (sub != zero);
            }
        }
        // The middle-levels `stack` is a LIFO, so if we push in this order, we'll consider the
        // Pauli terms in lexicographical order, which is the canonical order from
        // `SparsePauliOp.sort`.  Populating the `out_list` (an initially empty `Vec`) effectively
        // reverses the stack, so we want to push its elements in the IXYZ order.
        if loc.qubit() == 1 {
            i_nonzero.then(|| out_list.push(loc.push_i()));
            x_nonzero.then(|| out_list.push(loc.push_x()));
            y_nonzero.then(|| out_list.push(loc.push_y()));
            z_nonzero.then(|| out_list.push(loc.push_z()));
        } else {
            z_nonzero.then(|| stack.push(loc.push_z()));
            y_nonzero.then(|| stack.push(loc.push_y()));
            x_nonzero.then(|| stack.push(loc.push_x()));
            i_nonzero.then(|| stack.push(loc.push_i()));
        }
    }
}

fn decompose_last_level(
    out_list: &mut Vec<PauliLocation>,
    scratch: &[Complex64],
    num_qubits: usize,
    tolerance: f64,
) -> PyResult<DecomposeOut> {
    let side = 1 << num_qubits;
    let scale = 0.5f64.powi(num_qubits.try_into()?);
    // Pessimistically allocate assuming that there will be no zero terms in the out stack.  We
    // don't really pay much cost if we overallocate, but underallocating means that all four
    // outputs have to copy their data across to a new allocation.
    let mut out = DecomposeOut {
        z: Vec::with_capacity(4 * num_qubits * out_list.len()),
        x: Vec::with_capacity(4 * num_qubits * out_list.len()),
        phases: Vec::with_capacity(4 * out_list.len()),
        coeffs: Vec::with_capacity(4 * out_list.len()),
        scale,
        tol: (tolerance * tolerance) / (scale * scale),
        num_qubits,
    };

    for loc in out_list.drain(..) {
        let row = loc.row();
        let col = loc.col();
        let base = row * side + col;
        let i_value = scratch[base] + scratch[base + side + 1];
        let z_value = scratch[base] - scratch[base + side + 1];
        let x_value = scratch[base + 1] + scratch[base + side];
        let y_value = scratch[base + 1] - scratch[base + side];

        let x = row ^ col;
        let z = row;
        let phase = (x & z).count_ones() as u8;
        // ... and pushing the last Pauli onto the chain happens forwards, since this is the
        // construction of the final object.
        push_pauli_if_nonzero(x, z, phase, i_value, &mut out);
        push_pauli_if_nonzero(x | 1, z, phase, x_value, &mut out);
        push_pauli_if_nonzero(x | 1, z | 1, phase + 1, y_value, &mut out);
        push_pauli_if_nonzero(x, z | 1, phase, z_value, &mut out);
    }
    // If we _wildly_ overallocated, then shrink back to a sensible size to avoid tying up too much
    // memory as we return to Python space.
    if out.z.capacity() / 4 > out.z.len() {
        out.z.shrink_to_fit();
        out.x.shrink_to_fit();
        out.phases.shrink_to_fit();
        out.coeffs.shrink_to_fit();
    }
    Ok(out)
}

// This generates lookup tables of the form
//      const LOOKUP: [[bool; 2] 4] = [[false, false], [true, false], [false, true], [true, true]];
// when called `pauli_lookup!(LOOKUP, 2, [_, _])`.  The last argument is like a dummy version of
// an individual lookup rule, which is consumed to make an inner "loop" with a declarative macro.
macro_rules! pauli_lookup {
    ($name:ident, $n:literal, [$head:expr$ (, $($tail:expr),*)?]) => {
        const $name: [[bool; $n]; 1<<$n] = pauli_lookup!(@acc, [$($($tail),*)?], [[false], [true]]);
    };
    (@acc, [$head:expr $(, $($tail:expr),*)?], [$([$($bools:tt),*]),+]) => {
        pauli_lookup!(@acc, [$($($tail),*)?], [$([$($bools),*, false]),+, $([$($bools),*, true]),+])
    };
    (@acc, [], $init:expr) => { $init };
}
pauli_lookup!(PAULI_LOOKUP_2, 2, [(), ()]);
pauli_lookup!(PAULI_LOOKUP_4, 4, [(), (), (), ()]);
pauli_lookup!(PAULI_LOOKUP_8, 8, [(), (), (), (), (), (), (), ()]);

/// Push a complete Pauli chain into the output (`out`), if the corresponding entry is non-zero.
/// `x` and `z` represent the symplectic X and Z bitvectors, packed into `usize`, where LSB n
/// corresponds to qubit `n`.
fn push_pauli_if_nonzero(
    mut x: usize,
    mut z: usize,
    phase: u8,
    value: Complex64,
    out: &mut DecomposeOut,
) {
    if value.norm_sqr() <= out.tol {
        return;
    }

    // This set of `extend` calls is effectively an 8-fold unrolling of the "natural" loop through
    // each bit, where the initial `if` statements are handling the remainder (the up-to 7
    // least-significant bits).  In practice, it's probably unlikely that people are decomposing
    // 16q+ operators, since that's a pretty huge matrix already.
    //
    // The 8-fold loop unrolling is because going bit-by-bit all the way would be dominated by loop
    // and bitwise-operation overhead.

    if out.num_qubits & 1 == 1 {
        out.x.push(x & 1 == 1);
        out.z.push(z & 1 == 1);
        x >>= 1;
        z >>= 1;
    }
    if out.num_qubits & 2 == 2 {
        out.x.extend(&PAULI_LOOKUP_2[x & 0b11]);
        out.z.extend(&PAULI_LOOKUP_2[z & 0b11]);
        x >>= 2;
        z >>= 2;
    }
    if out.num_qubits & 4 == 4 {
        out.x.extend(&PAULI_LOOKUP_4[x & 0b1111]);
        out.z.extend(&PAULI_LOOKUP_4[z & 0b1111]);
        x >>= 4;
        z >>= 4;
    }
    for _ in 0..(out.num_qubits / 8) {
        out.x.extend(&PAULI_LOOKUP_8[x & 0b1111_1111]);
        out.z.extend(&PAULI_LOOKUP_8[z & 0b1111_1111]);
        x >>= 8;
        z >>= 8;
    }

    let phase = phase % 4;
    let value = match phase {
        0 => Complex64::new(out.scale, 0.0) * value,
        1 => Complex64::new(0.0, out.scale) * value,
        2 => Complex64::new(-out.scale, 0.0) * value,
        3 => Complex64::new(0.0, -out.scale) * value,
        _ => unreachable!("'x % 4' has only four possible values"),
    };
    out.phases.push(phase);
    out.coeffs.push(value);
}

/// Pack the information about which row, column and qubit we're considering into a single `usize`.
/// Complex64 data is 16 bytes long and the operators are square and must be addressable in memory,
/// so the row and column are hardware limited to be of width `usize::BITS / 2 - 2` each.  However,
/// we don't need to store at a granularity of 1, because the last 2x2 block we handle manually, so
/// we can remove an extra least significant bit from the row and column.  Regardless of the width
/// of `usize`, we can therefore track the state for up to 64 qubits losslessly, which is greater
/// than the maximum addressable memory on a 64-bit system.
///
/// For a 64-bit usize, the bit pattern is stored like this:
///
///    0b__000101__11111111111111111111111110000__11111111111111111111111110000
///        <-6-->  <------------29------------->  <------------29------------->
///          |                  |                              |
///          |         uint of the input row         uint of the input column
///          |         (once a 0 is appended)        (once a 0 is appended)
///          |
///        current qubit under consideration
///
/// The `qubit` field encodes the depth in the call stack that the user of the `PauliLocation`
/// should consider.  When the stack is initialised (before any calculation is done), it starts at
/// the highest qubit index (`num_qubits - 1`) and decreases from there until 0.
///
/// The `row` and `col` methods form the top-left corner of a `(2**(qubit + 1), 2**(qubit + 1))`
/// submatrix (where the top row and leftmost column are 0).  The least significant `qubit + 1`
/// bits of the of row and column are therefore always zero; the 0-indexed qubit still corresponds
/// to a 2x2 block.  This is why we needn't store it.
#[derive(Debug, Clone, Copy)]
struct PauliLocation(usize);

impl PauliLocation {
    const QUBIT_SHIFT: u32 = usize::BITS - 6;
    const QUBIT_MASK: usize = (usize::MAX >> Self::QUBIT_SHIFT) << Self::QUBIT_SHIFT;
    const ROW_SHIFT: u32 = usize::BITS / 2 - 3;
    const ROW_MASK: usize =
        ((usize::MAX >> Self::ROW_SHIFT) << Self::ROW_SHIFT) & !Self::QUBIT_MASK;
    const COL_SHIFT: u32 = 0;
    const COL_MASK: usize = usize::MAX & !Self::ROW_MASK & !Self::QUBIT_MASK;

    /// Create the base `PauliLocation` for an entire matrix with `num_qubits` qubits.  The initial
    /// Pauli chain is empty.
    #[inline(always)]
    fn begin(num_qubits: usize) -> Self {
        Self::new(0, 0, num_qubits - 1)
    }

    /// Manually create a new `PauliLocation` with the given information.  The logic in the rest of
    /// the class assumes that `row` and `col` will end with at least `qubit + 1` zeros, since
    /// these are the only valid locations.
    #[inline(always)]
    fn new(row: usize, col: usize, qubit: usize) -> Self {
        debug_assert!(row & 1 == 0);
        debug_assert!(col & 1 == 0);
        debug_assert!(row < 2 * Self::ROW_SHIFT as usize);
        debug_assert!(col < 2 * Self::ROW_SHIFT as usize);
        debug_assert!(qubit < 64);
        Self(
            (qubit << Self::QUBIT_SHIFT)
                | (row << Self::ROW_SHIFT >> 1)
                | (col << Self::COL_SHIFT >> 1),
        )
    }

    /// The row in the dense matrix that this location corresponds to.
    #[inline(always)]
    fn row(&self) -> usize {
        ((self.0 & Self::ROW_MASK) >> Self::ROW_SHIFT) << 1
    }

    /// The column in the dense matrix that this location corresponds to.
    #[inline(always)]
    fn col(&self) -> usize {
        ((self.0 & Self::COL_MASK) >> Self::COL_SHIFT) << 1
    }

    /// Which qubit in the Pauli chain we're currently considering.
    #[inline(always)]
    fn qubit(&self) -> usize {
        (self.0 & Self::QUBIT_MASK) >> Self::QUBIT_SHIFT
    }

    /// Create a new location corresponding to the Pauli chain so far, plus an identity on the
    /// currently considered qubit.
    #[inline(always)]
    fn push_i(&self) -> Self {
        Self::new(self.row(), self.col(), self.qubit() - 1)
    }

    /// Create a new location corresponding to the Pauli chain so far, plus an X on the currently
    /// considered qubit.
    #[inline(always)]
    fn push_x(&self) -> Self {
        Self::new(
            self.row(),
            self.col() | (1 << self.qubit()),
            self.qubit() - 1,
        )
    }

    /// Create a new location corresponding to the Pauli chain so far, plus a Y on the currently
    /// considered qubit.
    #[inline(always)]
    fn push_y(&self) -> Self {
        Self::new(
            self.row() | (1 << self.qubit()),
            self.col(),
            self.qubit() - 1,
        )
    }

    /// Create a new location corresponding to the Pauli chain so far, plus a Z on the currently
    /// considered qubit.
    #[inline(always)]
    fn push_z(&self) -> Self {
        Self::new(
            self.row() | (1 << self.qubit()),
            self.col() | (1 << self.qubit()),
            self.qubit() - 1,
        )
    }
}

#[pymodule]
pub fn sparse_pauli_op(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(unordered_unique))?;
    m.add_wrapped(wrap_pyfunction!(decompose_dense))?;
    m.add_class::<ZXPaulis>()?;
    Ok(())
}
