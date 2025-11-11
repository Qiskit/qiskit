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

use crate::sparse_observable::{BitTerm, SparseObservable};
use ahash::RandomState;
use pyo3::Python;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use numpy::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};

use hashbrown::HashMap;
use indexmap::IndexMap;
use ndarray::{ArrayView1, ArrayView2, Axis, s};
use num_complex::Complex64;
use num_traits::Zero;
use thiserror::Error;

use qiskit_circuit::util::{C_ZERO, c64};

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
pub fn unordered_unique(py: Python, array: PyReadonlyArray2<u16>) -> (Py<PyAny>, Py<PyAny>) {
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
        indices.into_pyarray(py).into_any().unbind(),
        inverses.into_pyarray(py).into_any().unbind(),
    )
}

/// Pack a 2D array of Booleans into a given width.  Returns an error if the input array is
/// too large to be packed into u64.
fn pack_bits(bool_arr: ArrayView2<bool>) -> Result<Vec<u64>, ()> {
    let num_qubits = bool_arr.shape()[1];
    if num_qubits > (u64::BITS as usize) {
        return Err(());
    }
    let slack = num_qubits % 8;
    let pack_row = |row: ArrayView1<bool>| -> u64 {
        let mut val: u64 = 0;
        let mut shift = 0;
        for chunk in row.exact_chunks(8) {
            val |= ((chunk[0] as u8
                | ((chunk[1] as u8) << 1)
                | ((chunk[2] as u8) << 2)
                | ((chunk[3] as u8) << 3)
                | ((chunk[4] as u8) << 4)
                | ((chunk[5] as u8) << 5)
                | ((chunk[6] as u8) << 6)
                | ((chunk[7] as u8) << 7)) as u64)
                << shift;
            shift += 8;
        }
        if slack > 0 {
            for (i, b) in row
                .slice(s![num_qubits - slack..num_qubits])
                .iter()
                .enumerate()
            {
                val |= (*b as u64) << (shift + i);
            }
        }
        val
    };
    Ok(bool_arr
        .axis_iter(Axis(0))
        .map(pack_row)
        .collect::<Vec<_>>())
}

/// A complete ZX-convention representation of a Pauli decomposition.  This is all the components
/// necessary to construct a Qiskit-space :class:`.SparsePauliOp`, where :attr:`phases` is in the
/// ZX convention.  This class is just meant for interoperation between Rust and Python.
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

#[pymethods]
impl ZXPaulis {
    #[new]
    fn __new__(
        x: &Bound<PyArray2<bool>>,
        z: &Bound<PyArray2<bool>>,
        phases: &Bound<PyArray1<u8>>,
        coeffs: &Bound<PyArray1<Complex64>>,
    ) -> PyResult<Self> {
        let &[num_ops, num_qubits] = x.shape() else {
            unreachable!("PyArray2 must be 2D")
        };
        if z.shape() != [num_ops, num_qubits] {
            return Err(PyValueError::new_err(format!(
                "'x' and 'z' have different shapes: {:?} and {:?}",
                [num_ops, num_qubits],
                z.shape()
            )));
        }
        if phases.len() != num_ops || coeffs.len() != num_ops {
            return Err(PyValueError::new_err(format!(
                "mismatched dimensions: 'x' and 'z' have {} operator(s), 'phase' has {} and 'coeffs' has {}",
                num_ops,
                phases.len(),
                coeffs.len(),
            )));
        }

        Ok(Self {
            x: x.to_owned().unbind(),
            z: z.to_owned().unbind(),
            phases: phases.to_owned().unbind(),
            coeffs: coeffs.to_owned().unbind(),
        })
    }
}

/// Intermediate structure that represents [ndarray] views onto the Python-space sparse Pauli data
/// in the ZX convention.  This can be used directly by Rust methods if desired, or bit-packed into
/// a matrix-representation format [MatrixCompressedPaulis] using the [compress] method.
pub struct ZXPaulisView<'py> {
    x: ArrayView2<'py, bool>,
    z: ArrayView2<'py, bool>,
    phases: ArrayView1<'py, u8>,
    coeffs: ArrayView1<'py, Complex64>,
}

impl ZXPaulisView<'_> {
    /// The number of qubits this operator acts on.
    pub fn num_qubits(&self) -> usize {
        self.x.shape()[1]
    }

    /// Convert the ZX representation into a bitpacked internal representation.  See the
    /// documentation of [MatrixCompressedPaulis] for details of the changes to the Pauli
    /// convention and representation.
    pub fn matrix_compress(&self) -> PyResult<MatrixCompressedPaulis> {
        let num_qubits = self.num_qubits();
        // This is obviously way too big for a dense operator, and SciPy limits us to using `i64`
        // for the index and indptr types, so (except for some synthetic cases), it's not possible
        // for us to work with a larger matrix than this.
        if num_qubits > 63 {
            return Err(PyValueError::new_err(format!(
                "{num_qubits} is too many qubits to convert to a matrix"
            )));
        }
        if num_qubits == 0 {
            return Ok(MatrixCompressedPaulis {
                num_qubits: 0,
                x_like: Vec::new(),
                z_like: Vec::new(),
                coeffs: self.coeffs.to_vec(),
            });
        }
        let x_like = pack_bits(self.x).expect("x should already be validated as <64 qubits");
        let z_like = pack_bits(self.z).expect("z should already be validated as <64 qubits");
        let coeffs = x_like
            .iter()
            .zip(z_like.iter())
            .zip(self.phases.iter().zip(self.coeffs.iter()))
            .map(|((xs, zs), (&phase, &coeff))| {
                let ys = (xs & zs).count_ones();
                match (phase as u32 + ys) % 4 {
                    0 => coeff,
                    1 => c64(coeff.im, -coeff.re),
                    2 => c64(-coeff.re, -coeff.im),
                    3 => c64(-coeff.im, coeff.re),
                    _ => unreachable!(),
                }
            })
            .collect::<Vec<_>>();
        Ok(MatrixCompressedPaulis {
            num_qubits: num_qubits as u8,
            x_like,
            z_like,
            coeffs,
        })
    }
}

/// Temporary bit-compressed storage of the Pauli string.  The [coeffs] are reinterpreted to
/// include the old `phase` component in them directly, plus the factors of `-i` stemming from `Y`
/// components.  The result is that the [coeffs] now more directly represent entries in a matrix,
/// while [x_like] and [z_like] are no longer direct measures of `X` and `Z` elements (as in the ZX
/// convention), but are instead only a marker of the column and parity respectively.
///
/// In other words, `row_num ^ x_like` gives the column number of an element, while
/// `(row_num & z_like).count_ones()` counts multiplicative factors of `-1` to be applied to
/// `coeff` when placing it at `(row_num, col_num)` in an output matrix.
pub struct MatrixCompressedPaulis {
    num_qubits: u8,
    x_like: Vec<u64>,
    z_like: Vec<u64>,
    coeffs: Vec<Complex64>,
}

impl MatrixCompressedPaulis {
    /// The number of qubits this operator acts on.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits as usize
    }

    /// The number of explicitly stored operators in the sum.
    pub fn num_ops(&self) -> usize {
        self.coeffs.len()
    }

    /// Sum coefficients that correspond to the same Pauli operator; this reduces the number of
    /// explicitly stored operations, if there are duplicates.  After the summation, any terms that
    /// have become zero are dropped.
    pub fn combine(&mut self) {
        let mut hash_table =
            IndexMap::<(u64, u64), Complex64, RandomState>::with_capacity_and_hasher(
                self.coeffs.len(),
                RandomState::new(),
            );
        for (key, coeff) in self
            .x_like
            .drain(..)
            .zip(self.z_like.drain(..))
            .zip(self.coeffs.drain(..))
        {
            *hash_table.entry(key).or_insert(C_ZERO) += coeff;
        }
        for ((x, z), coeff) in hash_table {
            if coeff.is_zero() {
                continue;
            }
            self.x_like.push(x);
            self.z_like.push(z);
            self.coeffs.push(coeff);
        }
    }
}

#[derive(Clone, Debug)]
struct DecomposeOut {
    z: Vec<bool>,
    x: Vec<bool>,
    phases: Vec<u8>,
    coeffs: Vec<Complex64>,
    scale: f64,
    tol: f64,
    num_qubits: usize,
}

#[derive(Error, Debug)]
enum DecomposeError {
    #[error("operators must have two dimensions, not {0}")]
    BadDimension(usize),
    #[error("operators must be square with a power-of-two side length, not {0:?}")]
    BadShape([usize; 2]),
}
impl From<DecomposeError> for PyErr {
    fn from(value: DecomposeError) -> PyErr {
        PyValueError::new_err(value.to_string())
    }
}

/// Decompose a dense complex operator into the symplectic Pauli representation in the
/// ZX-convention.
///
/// This is an implementation of the "tensorized Pauli decomposition" presented in
/// `Hantzko, Binkowski and Gupta (2023) <https://arxiv.org/abs/2310.13421>`__.
///
/// Implementation
/// --------------
///
/// The original algorithm was described recurisvely, allocating new matrices for each of the
/// block-wise sums (e.g. `op[top_left] + op[bottom_right]`).  This implementation differs in two
/// major ways:
///
/// - We do not allocate new matrices recursively, but instead produce a single copy of the input
///   and repeatedly overwrite subblocks of it at each point of the decomposition.
/// - The implementation is rewritten as an iteration rather than a recursion.  The current "state"
///   of the iteration is encoded in a single machine word (the `PauliLocation` struct below).
///
/// We do the decomposition in three "stages", with the stage changing whenever we need to change
/// the input/output types.  The first level is mathematically the same as the middle levels, it
/// just gets handled separately because it does the double duty of moving the data out of the
/// Python-space strided array into a Rust-space contiguous array that we can modify in-place.
/// The middle levels all act in-place on this newly created scratch space.  Finally, at the last
/// level, we've completed the decomposition and need to be writing the result into the output
/// data structures rather than into the scratch space.
///
/// Each "level" is handling one qubit in the operator, equivalently to the recursive procedure
/// described in the paper referenced in the docstring.  This implementation is iterative
/// stack-based and in place, rather than recursive.
///
/// We can get away with overwriting our scratch-space matrix at each point, because each
/// element of a given subblock is used exactly twice during each decomposition - once for the `a +
/// b` case, and once for the `a - b` case.  The second operand is the same in both cases.
/// Illustratively, at each step we're decomposing a submatrix blockwise, where we label the blocks
/// like this:
///
///   +---------+---------+          +---------+---------+
///   |         |         |          |         |         |
///   |    I    |    X    |          |  I + Z  |  X + Y  |
///   |         |         |          |         |         |
///   +---------+---------+  =====>  +---------+---------+
///   |         |         |          |         |         |
///   |    Y    |    Z    |          |  X - Y  |  I - Z  |
///   |         |         |          |         |         |
///   +---------+---------+          +---------+---------+
///
/// Each addition or subtraction is done elementwise, so as long as we iterate through the two pairs
/// of coupled blocks in order in lockstep, we can write out the answers together without
/// overwriting anything we need again.  We ignore all factors of 1/2 until the very last step, and
/// apply them all at once.  This minimises the number of floating-point operations we have to do.
///
/// We store the iteration order as a stack of `PauliLocation`s, whose own docstring explains how it
/// tracks the top-left corner and the size of the submatrix it represents.
#[pyfunction]
pub fn decompose_dense(
    py: Python,
    operator: PyReadonlyArray2<Complex64>,
    tolerance: f64,
) -> PyResult<ZXPaulis> {
    let array_view = operator.as_array();
    let out = py.detach(|| decompose_dense_inner(array_view, tolerance))?;
    Ok(ZXPaulis {
        z: PyArray1::from_vec(py, out.z)
            .reshape([out.phases.len(), out.num_qubits])?
            .into(),
        x: PyArray1::from_vec(py, out.x)
            .reshape([out.phases.len(), out.num_qubits])?
            .into(),
        phases: PyArray1::from_vec(py, out.phases).into(),
        coeffs: PyArray1::from_vec(py, out.coeffs).into(),
    })
}

/// Rust-only inner component of the `SparsePauliOp` decomposition.
///
/// See the top-level documentation of [decompose_dense] for more information on the internal
/// algorithm at play.
fn decompose_dense_inner(
    operator: ArrayView2<Complex64>,
    tolerance: f64,
) -> Result<DecomposeOut, DecomposeError> {
    let op_shape = match operator.shape() {
        [a, b] => [*a, *b],
        shape => return Err(DecomposeError::BadDimension(shape.len())),
    };
    if op_shape[0].is_zero() {
        return Err(DecomposeError::BadShape(op_shape));
    }
    let num_qubits = op_shape[0].ilog2() as usize;
    let side = 1 << num_qubits;
    if op_shape != [side, side] {
        return Err(DecomposeError::BadShape(op_shape));
    }
    if num_qubits.is_zero() {
        // We have to special-case the zero-qubit operator because our `decompose_last_level` still
        // needs to "consume" a qubit.
        return Ok(DecomposeOut {
            z: vec![],
            x: vec![],
            phases: vec![],
            coeffs: vec![operator[[0, 0]]],
            scale: 1.0,
            tol: tolerance,
            num_qubits: 0,
        });
    }
    let (stack, mut out_list, mut scratch) = decompose_first_level(operator, num_qubits);
    decompose_middle_levels(stack, &mut out_list, &mut scratch, num_qubits);
    Ok(decompose_last_level(
        &mut out_list,
        &scratch,
        num_qubits,
        tolerance,
    ))
}

/// Apply the matrix-addition decomposition at the first level.
///
/// This is split out from the middle levels because it acts on an `ArrayView2`, and is responsible
/// for copying the operator over into the contiguous scratch space.  We can't write over the
/// operator the user gave us (it's not ours to do that to), and anyway, we want to drop to a chunk
/// of memory that we can 100% guarantee is contiguous, so we can elide all the stride checking.
/// We split this out so we can do the first decomposition at the same time as scanning over the
/// operator to copy it.
///
/// # Panics
///
/// If the number of qubits in the operator is zero.
fn decompose_first_level(
    in_op: ArrayView2<Complex64>,
    num_qubits: usize,
) -> (Vec<PauliLocation>, Vec<PauliLocation>, Vec<Complex64>) {
    let side = 1 << num_qubits;
    let mut stack = Vec::<PauliLocation>::with_capacity(4);
    let mut out_list = Vec::<PauliLocation>::new();
    let mut scratch = Vec::<Complex64>::with_capacity(side * side);
    match num_qubits {
        0 => panic!("number of qubits must be greater than zero"),
        1 => {
            // If we've only got one qubit, we just want to copy the data over in the correct
            // continuity and let the base case of the iteration take care of outputting it.
            scratch.extend(in_op.iter());
            out_list.push(PauliLocation::begin(num_qubits));
        }
        _ => {
            // We don't write out the operator in contiguous-index order, but we can easily
            // guarantee that we'll write to each index exactly once without reading it - we still
            // visit every index, just in 2x2 blockwise order, not row-by-row.
            unsafe { scratch.set_len(scratch.capacity()) };
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
                    i_nonzero = i_nonzero || (value != C_ZERO);
                }

                let x_row = x_row_0 + off_row;
                let y_row = y_row_0 + off_row;
                for off_col in 0..mid {
                    let x_col = x_col_0 + off_col;
                    let y_col = y_col_0 + off_col;
                    let value = in_op[[x_row, x_col]] + in_op[[y_row, y_col]];
                    scratch[ptr] = value;
                    ptr += 1;
                    x_nonzero = x_nonzero || (value != C_ZERO);
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
                    y_nonzero = y_nonzero || (value != C_ZERO);
                }
                let i_row = i_row_0 + off_row;
                let z_row = z_row_0 + off_row;
                for off_col in 0..mid {
                    let i_col = i_col_0 + off_col;
                    let z_col = z_col_0 + off_col;
                    let value = in_op[[i_row, i_col]] - in_op[[z_row, z_col]];
                    scratch[ptr] = value;
                    ptr += 1;
                    z_nonzero = z_nonzero || (value != C_ZERO);
                }
            }
            // The middle-levels `stack` is a LIFO, so if we push in this order, we'll consider the
            // Pauli terms in lexicographical order, which is the canonical order from
            // `SparsePauliOp.sort`.  Populating the `out_list` (an initially empty `Vec`)
            // effectively reverses the stack, so we want to push its elements in the IXYZ order.
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

/// Iteratively decompose the matrix at all levels other than the first and last.
///
/// This populates the `out_list` with locations.  This is mathematically the same as the first
/// level of the decomposition, except now we're acting in-place on our Rust-space contiguous
/// scratch space, rather than the strided Python-space array we were originally given.
fn decompose_middle_levels(
    mut stack: Vec<PauliLocation>,
    out_list: &mut Vec<PauliLocation>,
    scratch: &mut [Complex64],
    num_qubits: usize,
) {
    let side = 1 << num_qubits;
    // The stack is a LIFO, which is how we implement the depth-first iteration.  Depth-first
    // means `stack` never grows very large; it reaches at most `3*num_qubits - 2` elements (if all
    // terms are zero all the way through the first subblock decomposition).  `out_list`, on the
    // other hand, can be `4 ** (num_qubits - 1)` entries in the worst-case scenario of a
    // completely dense (in Pauli terms) operator.
    while let Some(loc) = stack.pop() {
        // Here we work pairwise, writing out the new values into both I and Z simultaneously (etc
        // for X and Y) so we can re-use their scratch space and avoid re-allocating.  We're doing
        // the multiple assignment `(I, Z) = (I + Z, I - Z)`.
        //
        // See the documentation of `decompose_dense` for more information on how this works.
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
                i_nonzero = i_nonzero || (add != C_ZERO);
                z_nonzero = z_nonzero || (sub != C_ZERO);
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
                x_nonzero = x_nonzero || (add != C_ZERO);
                y_nonzero = y_nonzero || (sub != C_ZERO);
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

/// Write out the results of the final decomposition into the Pauli ZX form.
///
/// The calculation here is the same as the previous two sets of decomposers, but we don't want to
/// write the result out into the scratch space to iterate needlessly once more; we want to
/// associate each non-zero coefficient with the final Pauli in the ZX format.
///
/// This function applies all the factors of 1/2 that we've been skipping during the intermediate
/// decompositions.  This means that the factors are applied to the output with `2 * output_len`
/// floating-point operations (real and imaginary), which is a huge reduction compared to repeatedly
/// doing it during the decomposition.
fn decompose_last_level(
    out_list: &mut Vec<PauliLocation>,
    scratch: &[Complex64],
    num_qubits: usize,
    tolerance: f64,
) -> DecomposeOut {
    let side = 1 << num_qubits;
    let scale = 0.5f64.powi(num_qubits as i32);
    // Pessimistically allocate assuming that there will be no zero terms in the out list.  We
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
        // Pushing the last Pauli onto the `loc` happens "forwards" to maintain lexicographical
        // ordering in `out`, since this is the construction of the final object.
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
    out
}

// This generates lookup tables of the form
//      const LOOKUP: [[bool; 2] 4] = [[false, false], [true, false], [false, true], [true, true]];
// when called `pauli_lookup!(LOOKUP, 2, [_, _])`.  The last argument is like a dummy version of
// an individual lookup rule, which is consumed to make an inner "loop" with a declarative macro.
macro_rules! pauli_lookup {
    ($name:ident, $n:literal, [$head:expr$ (, $($tail:expr),*)?]) => {
        static $name: [[bool; $n]; 1<<$n] = pauli_lookup!(@acc, [$($($tail),*)?], [[false], [true]]);
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
///
/// `x` and `z` represent the symplectic X and Z bitvectors, packed into `usize`, where LSb n
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

/// The "state" of an iteration step of the dense-operator decomposition routine.
///
/// Pack the information about which row, column and qubit we're considering into a single `usize`.
/// Complex64 data is 16 bytes long and the operators are square and must be addressable in memory,
/// so the row and column are hardware limited to be of width `usize::BITS / 2 - 2` each.  However,
/// we don't need to store at a granularity of 1, because the last 2x2 block we handle manually, so
/// we can remove an extra least significant bit from the row and column.  Regardless of the width
/// of `usize`, we can therefore track the state for up to 30 qubits losslessly, which is greater
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
    // These shifts and masks are used to access the three components of the bit-packed state.
    const QUBIT_SHIFT: u32 = usize::BITS - 6;
    const QUBIT_MASK: usize = (usize::MAX >> Self::QUBIT_SHIFT) << Self::QUBIT_SHIFT;
    const ROW_SHIFT: u32 = usize::BITS / 2 - 3;
    const ROW_MASK: usize =
        ((usize::MAX >> Self::ROW_SHIFT) << Self::ROW_SHIFT) & !Self::QUBIT_MASK;
    const COL_SHIFT: u32 = 0; // Just for consistency.
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

pub fn sparse_pauli_op(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(unordered_unique))?;
    m.add_wrapped(wrap_pyfunction!(decompose_dense))?;
    m.add_class::<ZXPaulis>()?;
    Ok(())
}

// inner function to convert MatrixCompressedPaulis to SparseObservable
impl From<&MatrixCompressedPaulis> for SparseObservable {
    fn from(paulis: &MatrixCompressedPaulis) -> Self {
        let mut coeffs = Vec::new();
        let mut bit_terms_flat = Vec::new();
        let mut indices_flat = Vec::new();
        let mut boundaries = vec![0];

        for i in 0..paulis.num_ops() {
            let phase = ((paulis.z_like[i] & paulis.x_like[i]).count_ones() % 4) as u8;
            let coeff = paulis.coeffs[i];
            //matching to match the SparseObservable convention, which applies the phase to the coefficient
            //converting back from the phase corrected coefficient of MatrixCompressedPaulis to the standard form
            coeffs.push(match phase {
                0 => coeff,
                1 => Complex64::new(-coeff.im, coeff.re), //  i
                2 => -coeff,                              // -1
                3 => Complex64::new(coeff.im, -coeff.re), // -i
                _ => unreachable!(),
            });

            let mut term_bit_terms = Vec::new();
            let mut term_indices = Vec::new();
            for q in 0..paulis.num_qubits() {
                let x = (paulis.x_like[i] >> q) & 1;
                let z = (paulis.z_like[i] >> q) & 1;
                if x == 1 && z == 1 {
                    term_bit_terms.push(BitTerm::Y);
                    term_indices.push(q as u32);
                } else if x == 1 {
                    term_bit_terms.push(BitTerm::X);
                    term_indices.push(q as u32);
                } else if z == 1 {
                    term_bit_terms.push(BitTerm::Z);
                    term_indices.push(q as u32);
                }
            }
            bit_terms_flat.extend(term_bit_terms);
            indices_flat.extend(term_indices);
            boundaries.push(bit_terms_flat.len());
        }

        SparseObservable::new(
            paulis.num_qubits as u32,
            coeffs,
            bit_terms_flat,
            indices_flat,
            boundaries,
        )
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, aview2};

    use super::*;
    use crate::sparse_observable::SparseObservable;

    #[cfg(miri)]
    use approx::AbsDiffEq;

    // The purpose of these tests is more about exercising the `unsafe` code under Miri; we test for
    // full numerical correctness from Python space.

    /// Helper struct for the decomposition testing.  This is a subset of the `DecomposeOut`
    /// struct, skipping the unnecessary algorithm-state components of it.
    ///
    /// If we add a more Rust-friendly interface to `SparsePauliOp` in the future, hopefully this
    /// can be removed.
    #[derive(Clone, PartialEq, Debug)]
    struct DecomposeMinimal {
        z: Vec<bool>,
        x: Vec<bool>,
        phases: Vec<u8>,
        coeffs: Vec<Complex64>,
        num_qubits: usize,
    }
    impl From<DecomposeOut> for DecomposeMinimal {
        fn from(value: DecomposeOut) -> Self {
            Self {
                z: value.z,
                x: value.x,
                phases: value.phases,
                coeffs: value.coeffs,
                num_qubits: value.num_qubits,
            }
        }
    }
    impl From<MatrixCompressedPaulis> for DecomposeMinimal {
        fn from(value: MatrixCompressedPaulis) -> Self {
            let phases = value
                .z_like
                .iter()
                .zip(value.x_like.iter())
                .map(|(z, x)| ((z & x).count_ones() % 4) as u8)
                .collect::<Vec<_>>();
            let coeffs = value
                .coeffs
                .iter()
                .zip(phases.iter())
                .map(|(c, phase)| match phase {
                    0 => *c,
                    1 => Complex64::new(-c.im, c.re),
                    2 => Complex64::new(-c.re, -c.im),
                    3 => Complex64::new(c.im, -c.re),
                    _ => panic!("phase should only in [0, 4)"),
                })
                .collect();
            let z = value
                .z_like
                .iter()
                .flat_map(|digit| (0..value.num_qubits).map(move |i| (digit & (1 << i)) != 0))
                .collect();
            let x = value
                .x_like
                .iter()
                .flat_map(|digit| (0..value.num_qubits).map(move |i| (digit & (1 << i)) != 0))
                .collect();
            Self {
                z,
                x,
                phases,
                coeffs,
                num_qubits: value.num_qubits as usize,
            }
        }
    }

    #[test]
    fn decompose_empty_operator_fails() {
        assert!(matches!(
            decompose_dense_inner(aview2::<Complex64, 0>(&[]), 0.0),
            Err(DecomposeError::BadShape(_)),
        ));
    }

    #[test]
    fn decompose_0q_operator() {
        let coeff = Complex64::new(1.5, -0.5);
        let arr = [[coeff]];
        let out = decompose_dense_inner(aview2(&arr), 0.0).unwrap();
        let expected = DecomposeMinimal {
            z: vec![],
            x: vec![],
            phases: vec![],
            coeffs: vec![coeff],
            num_qubits: 0,
        };
        assert_eq!(DecomposeMinimal::from(out), expected);
    }

    #[test]
    fn decompose_1q_operator() {
        // Be sure that any sums are given in canonical order of the output, or there will be
        // spurious test failures.
        let paulis = [
            (vec![0], vec![0]),             // I
            (vec![1], vec![0]),             // X
            (vec![1], vec![1]),             // Y
            (vec![0], vec![1]),             // Z
            (vec![0, 1], vec![0, 0]),       // I, X
            (vec![0, 1], vec![0, 1]),       // I, Y
            (vec![0, 0], vec![0, 1]),       // I, Z
            (vec![1, 1], vec![0, 1]),       // X, Y
            (vec![1, 0], vec![1, 1]),       // X, Z
            (vec![1, 0], vec![1, 1]),       // Y, Z
            (vec![1, 1, 0], vec![0, 1, 1]), // X, Y, Z
        ];
        let coeffs = [
            Complex64::new(1.5, -0.5),
            Complex64::new(-0.25, 2.0),
            Complex64::new(0.75, 0.75),
        ];
        for (x_like, z_like) in paulis {
            let paulis = MatrixCompressedPaulis {
                num_qubits: 1,
                coeffs: coeffs[0..x_like.len()].to_owned(),
                x_like,
                z_like,
            };
            let observable = SparseObservable::from(&paulis);
            let arr_vec = observable
                .to_matrix_dense(false)
                .expect("Failed to create dense matrix");
            let arr = Array1::from_vec(arr_vec)
                .into_shape_with_order((2, 2))
                .unwrap();
            let expected: DecomposeMinimal = paulis.into();
            let actual: DecomposeMinimal = decompose_dense_inner(arr.view(), 0.0).unwrap().into();
            #[cfg(not(miri))]
            assert_eq!(actual, expected);
            #[cfg(miri)]
            {
                assert!(actual.coeffs.abs_diff_eq(&expected.coeffs, 1e-8));
                assert_eq!(actual.z, expected.z);
                assert_eq!(actual.x, expected.x);
                assert_eq!(actual.phases, expected.phases);
                assert_eq!(actual.num_qubits, expected.num_qubits);
            }
        }
    }

    #[test]
    fn decompose_3q_operator() {
        // Be sure that any sums are given in canonical order of the output, or there will be
        // spurious test failures.
        let paulis = [
            (vec![0], vec![0]),             // III
            (vec![1], vec![0]),             // IIX
            (vec![2], vec![2]),             // IYI
            (vec![0], vec![4]),             // ZII
            (vec![6], vec![6]),             // YYI
            (vec![7], vec![7]),             // YYY
            (vec![1, 6, 7], vec![1, 6, 7]), // IIY, YYI, YYY
            (vec![1, 2, 0], vec![0, 2, 4]), // IIX, IYI, ZII
        ];
        let coeffs = [
            Complex64::new(1.5, -0.5),
            Complex64::new(-0.25, 2.0),
            Complex64::new(0.75, 0.75),
        ];
        for (x_like, z_like) in paulis {
            let paulis = MatrixCompressedPaulis {
                num_qubits: 3,
                coeffs: coeffs[0..x_like.len()].to_owned(),
                x_like,
                z_like,
            };
            let observable = SparseObservable::from(&paulis);
            let arr_vec = observable
                .to_matrix_dense(false)
                .expect("Failed to create dense matrix");
            let arr = Array1::from_vec(arr_vec)
                .into_shape_with_order((8, 8))
                .unwrap();
            let expected: DecomposeMinimal = paulis.into();
            let actual: DecomposeMinimal = decompose_dense_inner(arr.view(), 0.0).unwrap().into();
            #[cfg(not(miri))]
            assert_eq!(actual, expected);
            #[cfg(miri)]
            {
                assert!(actual.coeffs.abs_diff_eq(&expected.coeffs, 1e-8));
                assert_eq!(actual.z, expected.z);
                assert_eq!(actual.x, expected.x);
                assert_eq!(actual.phases, expected.phases);
                assert_eq!(actual.num_qubits, expected.num_qubits);
            }
        }
    }
}
