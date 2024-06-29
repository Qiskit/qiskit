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

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use numpy::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};

use hashbrown::HashMap;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_complex::Complex64;
use num_traits::Zero;
use qiskit_circuit::util::{c64, C_ONE, C_ZERO};
use rayon::prelude::*;

use crate::rayon_ext::*;

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
        indices.into_pyarray_bound(py).into(),
        inverses.into_pyarray_bound(py).into(),
    )
}

#[derive(Clone, Copy)]
enum Pauli {
    I,
    X,
    Y,
    Z,
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

impl ZXPaulis {
    /// Attempt to acquire a Rust-enforced Rust-only immutable borrow onto the underlying
    /// Python-space data. This returns `None` if any of the underlying arrays already has a
    /// mutable borrow taken out onto it.
    pub fn try_readonly<'a, 'py>(&'a self, py: Python<'py>) -> Option<ZXPaulisReadonly<'py>>
    where
        'a: 'py,
    {
        Some(ZXPaulisReadonly {
            x: self.x.bind(py).try_readonly().ok()?,
            z: self.z.bind(py).try_readonly().ok()?,
            phases: self.phases.bind(py).try_readonly().ok()?,
            coeffs: self.coeffs.bind(py).try_readonly().ok()?,
        })
    }
}

/// Intermediate structure that represents readonly views onto the Python-space sparse Pauli data.
/// This is used in the chained methods so that the syntactical temporary lifetime extension can
/// occur; we can't have the readonly array temporaries only live within a method that returns
/// [ZXPaulisView], because otherwise the lifetimes of the [PyReadonlyArray] elements will be too
/// short.
pub struct ZXPaulisReadonly<'a> {
    x: PyReadonlyArray2<'a, bool>,
    z: PyReadonlyArray2<'a, bool>,
    phases: PyReadonlyArray1<'a, u8>,
    coeffs: PyReadonlyArray1<'a, Complex64>,
}

impl ZXPaulisReadonly<'_> {
    /// Get a [ndarray] view of the data of these [rust-numpy] objects.
    fn as_array(&self) -> ZXPaulisView {
        ZXPaulisView {
            x: self.x.as_array(),
            z: self.z.as_array(),
            phases: self.phases.as_array(),
            coeffs: self.coeffs.as_array(),
        }
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

impl<'py> ZXPaulisView<'py> {
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
        let mut hash_table = HashMap::<(u64, u64), Complex64>::with_capacity(self.coeffs.len());
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
    let size = 1 << num_qubits;
    if operator.shape() != [size, size] {
        return Err(PyValueError::new_err(format!(
            "input with shape {:?} cannot be interpreted as a multiqubit operator",
            operator.shape()
        )));
    }
    let mut paulis = vec![];
    let mut coeffs = vec![];
    if num_qubits > 0 {
        decompose_dense_inner(
            C_ONE,
            num_qubits,
            &[],
            operator.as_array(),
            &mut paulis,
            &mut coeffs,
            tolerance * tolerance,
        );
    }
    if coeffs.is_empty() {
        Ok(ZXPaulis {
            z: PyArray2::zeros_bound(py, [0, num_qubits], false).into(),
            x: PyArray2::zeros_bound(py, [0, num_qubits], false).into(),
            phases: PyArray1::zeros_bound(py, [0], false).into(),
            coeffs: PyArray1::zeros_bound(py, [0], false).into(),
        })
    } else {
        // Constructing several arrays of different shapes at once is rather awkward in iterator
        // logic, so we just loop manually.
        let mut z = Array2::<bool>::uninit([paulis.len(), num_qubits]);
        let mut x = Array2::<bool>::uninit([paulis.len(), num_qubits]);
        let mut phases = Array1::<u8>::uninit(paulis.len());
        for (i, paulis) in paulis.drain(..).enumerate() {
            let mut phase = 0u8;
            for (j, pauli) in paulis.into_iter().rev().enumerate() {
                match pauli {
                    Pauli::I => {
                        z[[i, j]].write(false);
                        x[[i, j]].write(false);
                    }
                    Pauli::X => {
                        z[[i, j]].write(false);
                        x[[i, j]].write(true);
                    }
                    Pauli::Y => {
                        z[[i, j]].write(true);
                        x[[i, j]].write(true);
                        phase = phase.wrapping_add(1);
                    }
                    Pauli::Z => {
                        z[[i, j]].write(true);
                        x[[i, j]].write(false);
                    }
                }
            }
            phases[i].write(phase % 4);
        }
        // These are safe because the above loops write into every element.  It's guaranteed that
        // each of the elements of the `paulis` vec will have `num_qubits` because they're all
        // reading from the same base array.
        let z = unsafe { z.assume_init() };
        let x = unsafe { x.assume_init() };
        let phases = unsafe { phases.assume_init() };
        Ok(ZXPaulis {
            z: z.into_pyarray_bound(py).into(),
            x: x.into_pyarray_bound(py).into(),
            phases: phases.into_pyarray_bound(py).into(),
            coeffs: PyArray1::from_vec_bound(py, coeffs).into(),
        })
    }
}

/// Recurse worker routine of `decompose_dense`.  Should be called with at least one qubit.
fn decompose_dense_inner(
    factor: Complex64,
    num_qubits: usize,
    paulis: &[Pauli],
    block: ArrayView2<Complex64>,
    out_paulis: &mut Vec<Vec<Pauli>>,
    out_coeffs: &mut Vec<Complex64>,
    square_tolerance: f64,
) {
    if num_qubits == 0 {
        // It would be safe to `return` here, but if it's unreachable then LLVM is allowed to
        // optimize out this branch entirely in release mode, which is good for a ~2% speedup.
        unreachable!("should not call this with an empty operator")
    }
    // Base recursion case.
    if num_qubits == 1 {
        let mut push_if_nonzero = |extra: Pauli, value: Complex64| {
            if value.norm_sqr() <= square_tolerance {
                return;
            }
            let paulis = {
                let mut vec = Vec::with_capacity(paulis.len() + 1);
                vec.extend_from_slice(paulis);
                vec.push(extra);
                vec
            };
            out_paulis.push(paulis);
            out_coeffs.push(value);
        };
        push_if_nonzero(Pauli::I, 0.5 * factor * (block[[0, 0]] + block[[1, 1]]));
        push_if_nonzero(Pauli::X, 0.5 * factor * (block[[0, 1]] + block[[1, 0]]));
        push_if_nonzero(
            Pauli::Y,
            0.5 * Complex64::i() * factor * (block[[0, 1]] - block[[1, 0]]),
        );
        push_if_nonzero(Pauli::Z, 0.5 * factor * (block[[0, 0]] - block[[1, 1]]));
        return;
    }
    let mut recurse_if_nonzero = |extra: Pauli, factor: Complex64, values: Array2<Complex64>| {
        let mut is_zero = true;
        for value in values.iter() {
            if !value.is_zero() {
                is_zero = false;
                break;
            }
        }
        if is_zero {
            return;
        }
        let mut new_paulis = Vec::with_capacity(paulis.len() + 1);
        new_paulis.extend_from_slice(paulis);
        new_paulis.push(extra);
        decompose_dense_inner(
            factor,
            num_qubits - 1,
            &new_paulis,
            values.view(),
            out_paulis,
            out_coeffs,
            square_tolerance,
        );
    };
    let mid = 1usize << (num_qubits - 1);
    recurse_if_nonzero(
        Pauli::I,
        0.5 * factor,
        &block.slice(s![..mid, ..mid]) + &block.slice(s![mid.., mid..]),
    );
    recurse_if_nonzero(
        Pauli::X,
        0.5 * factor,
        &block.slice(s![..mid, mid..]) + &block.slice(s![mid.., ..mid]),
    );
    recurse_if_nonzero(
        Pauli::Y,
        0.5 * Complex64::i() * factor,
        &block.slice(s![..mid, mid..]) - &block.slice(s![mid.., ..mid]),
    );
    recurse_if_nonzero(
        Pauli::Z,
        0.5 * factor,
        &block.slice(s![..mid, ..mid]) - &block.slice(s![mid.., mid..]),
    );
}

/// Convert the given [ZXPaulis] object to a dense 2D Numpy matrix.
#[pyfunction]
#[pyo3(signature = (/, paulis, force_serial=false))]
pub fn to_matrix_dense<'py>(
    py: Python<'py>,
    paulis: &ZXPaulis,
    force_serial: bool,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    let paulis_readonly = paulis
        .try_readonly(py)
        .ok_or_else(|| PyRuntimeError::new_err("could not produce a safe view onto the data"))?;
    let mut paulis = paulis_readonly.as_array().matrix_compress()?;
    paulis.combine();
    let side = 1usize << paulis.num_qubits();
    let parallel = !force_serial && crate::getenv_use_multiple_threads();
    let out = to_matrix_dense_inner(&paulis, parallel);
    PyArray1::from_vec_bound(py, out).reshape([side, side])
}

/// Inner worker of the Python-exposed [to_matrix_dense].  This is separate primarily to allow
/// Rust-space unit testing even if Python isn't available for execution.  This returns a C-ordered
/// [Vec] of the 2D matrix.
fn to_matrix_dense_inner(paulis: &MatrixCompressedPaulis, parallel: bool) -> Vec<Complex64> {
    let side = 1usize << paulis.num_qubits();
    #[allow(clippy::uninit_vec)]
    let mut out = {
        let mut out = Vec::with_capacity(side * side);
        // SAFETY: we iterate through the vec in chunks of `side`, and start each row by filling it
        // with zeros before ever reading from it.  It's fine to overwrite the uninitialised memory
        // because `Complex64: !Drop`.
        unsafe { out.set_len(side * side) };
        out
    };
    let write_row = |(i_row, row): (usize, &mut [Complex64])| {
        // Doing the initialization here means that when we're in parallel contexts, we do the
        // zeroing across the whole threadpool.  This also seems to give a speed-up in serial
        // contexts, but I don't understand that. ---Jake
        row.fill(C_ZERO);
        for ((&x_like, &z_like), &coeff) in paulis
            .x_like
            .iter()
            .zip(paulis.z_like.iter())
            .zip(paulis.coeffs.iter())
        {
            // Technically this discards part of the storable data, but in practice, a dense
            // operator with more than 32 qubits needs in the region of 1 ZiB memory.  We still use
            // `u64` to help sparse-matrix construction, though.
            let coeff = if (i_row as u32 & z_like as u32).count_ones() % 2 == 0 {
                coeff
            } else {
                -coeff
            };
            row[i_row ^ (x_like as usize)] += coeff;
        }
    };
    if parallel {
        out.par_chunks_mut(side).enumerate().for_each(write_row);
    } else {
        out.chunks_mut(side).enumerate().for_each(write_row);
    }
    out
}

type CSRData<T> = (Vec<Complex64>, Vec<T>, Vec<T>);
type ToCSRData<T> = fn(&MatrixCompressedPaulis) -> CSRData<T>;

/// Convert the given [ZXPaulis] object to the three-array CSR form.  The output type of the
/// `indices` and `indptr` matrices will be `i32` if that type is guaranteed to be able to hold the
/// number of non-zeros, otherwise it will be `i64`.  Signed types are used to match Scipy.  `i32`
/// is preferentially returned, because Scipy will downcast to this on `csr_matrix` construction if
/// all array elements would fit.  For large operators with significant cancellation, it is
/// possible that `i64` will be returned when `i32` would suffice, but this will not cause
/// unsoundness, just a copy overhead when constructing the Scipy matrix.
#[pyfunction]
#[pyo3(signature = (/, paulis, force_serial=false))]
pub fn to_matrix_sparse(
    py: Python,
    paulis: &ZXPaulis,
    force_serial: bool,
) -> PyResult<Py<PyTuple>> {
    let paulis_readonly = paulis
        .try_readonly(py)
        .ok_or_else(|| PyRuntimeError::new_err("could not produce a safe view onto the data"))?;
    let mut paulis = paulis_readonly.as_array().matrix_compress()?;
    paulis.combine();

    // This deliberately erases the Rust types in the output so we can return either 32- or 64-bit
    // indices as appropriate without breaking Rust's typing.
    fn to_py_tuple<T>(py: Python, csr_data: CSRData<T>) -> Py<PyTuple>
    where
        T: numpy::Element,
    {
        let (values, indices, indptr) = csr_data;
        (
            PyArray1::from_vec_bound(py, values),
            PyArray1::from_vec_bound(py, indices),
            PyArray1::from_vec_bound(py, indptr),
        )
            .into_py(py)
    }

    // Pessimistic estimation of whether we can fit in `i32`.  If there's any risk of overflowing
    // `i32`, we use `i64`, but Scipy will always try to downcast to `i32`, so we try to match it.
    let max_entries_per_row = (paulis.num_ops() as u64).min(1u64 << (paulis.num_qubits() - 1));
    let use_32_bit =
        max_entries_per_row.saturating_mul(1u64 << paulis.num_qubits()) <= (i32::MAX as u64);
    if use_32_bit {
        let to_sparse: ToCSRData<i32> = if crate::getenv_use_multiple_threads() && !force_serial {
            to_matrix_sparse_parallel_32
        } else {
            to_matrix_sparse_serial_32
        };
        Ok(to_py_tuple(py, to_sparse(&paulis)))
    } else {
        let to_sparse: ToCSRData<i64> = if crate::getenv_use_multiple_threads() && !force_serial {
            to_matrix_sparse_parallel_64
        } else {
            to_matrix_sparse_serial_64
        };
        Ok(to_py_tuple(py, to_sparse(&paulis)))
    }
}

/// Copy several slices into a single flat vec, in parallel.  Allocates a temporary `Vec<usize>` of
/// the same length as the input slice to track the chunking.
fn copy_flat_parallel<T, U>(slices: &[U]) -> Vec<T>
where
    T: Copy + Send + Sync,
    U: AsRef<[T]> + Sync,
{
    let lens = slices
        .iter()
        .map(|slice| slice.as_ref().len())
        .collect::<Vec<_>>();
    let size = lens.iter().sum();
    #[allow(clippy::uninit_vec)]
    let mut out = {
        let mut out = Vec::with_capacity(size);
        // SAFETY: we've just calculated that the lengths of the given vecs add up to the right
        // thing, and we're about to copy in the data from each of them into this uninitialised
        // array.  It's guaranteed safe to write `T` to the uninitialised space, because `Copy`
        // implies `!Drop`.
        unsafe { out.set_len(size) };
        out
    };
    out.par_uneven_chunks_mut(&lens)
        .zip(slices.par_iter().map(|x| x.as_ref()))
        .for_each(|(out_slice, in_slice)| out_slice.copy_from_slice(in_slice));
    out
}

macro_rules! impl_to_matrix_sparse {
    ($serial_fn:ident, $parallel_fn:ident, $int_ty:ty, $uint_ty:ty $(,)?) => {
        /// Build CSR data arrays for the matrix-compressed set of the Pauli operators, using a
        /// completely serial strategy.
        fn $serial_fn(paulis: &MatrixCompressedPaulis) -> CSRData<$int_ty> {
            let side = 1 << paulis.num_qubits();
            let num_ops = paulis.num_ops();
            if num_ops == 0 {
                return (vec![], vec![], vec![0; side + 1]);
            }

            let mut order = (0..num_ops).collect::<Vec<_>>();
            let mut values = Vec::<Complex64>::with_capacity(side * (num_ops + 1) / 2);
            let mut indices = Vec::<$int_ty>::with_capacity(side * (num_ops + 1) / 2);
            let mut indptr: Vec<$int_ty> = vec![0; side + 1];
            let mut nnz = 0;
            for i_row in 0..side {
                order.sort_unstable_by(|&a, &b| {
                    ((i_row as $uint_ty) ^ (paulis.x_like[a] as $uint_ty))
                        .cmp(&((i_row as $uint_ty) ^ (paulis.x_like[b] as $uint_ty)))
                });
                let mut running = C_ZERO;
                let mut prev_index = i_row ^ (paulis.x_like[order[0]] as usize);
                for (x_like, z_like, coeff) in order
                    .iter()
                    .map(|&i| (paulis.x_like[i], paulis.z_like[i], paulis.coeffs[i]))
                {
                    let coeff =
                        if ((i_row as $uint_ty) & (z_like as $uint_ty)).count_ones() % 2 == 0 {
                            coeff
                        } else {
                            -coeff
                        };
                    let index = i_row ^ (x_like as usize);
                    if index == prev_index {
                        running += coeff;
                    } else {
                        nnz += 1;
                        values.push(running);
                        indices.push(prev_index as $int_ty);
                        running = coeff;
                        prev_index = index;
                    }
                }
                nnz += 1;
                values.push(running);
                indices.push(prev_index as $int_ty);
                indptr[i_row + 1] = nnz;
            }
            (values, indices, indptr)
        }

        /// Build CSR data arrays for the matrix-compressed set of the Pauli operators, using a
        /// parallel strategy.  This involves more data copying than the serial form, so there is a
        /// nontrivial amount of parallel overhead.
        fn $parallel_fn(paulis: &MatrixCompressedPaulis) -> CSRData<$int_ty> {
            let side = 1 << paulis.num_qubits();
            let num_ops = paulis.num_ops();
            if num_ops == 0 {
                return (vec![], vec![], vec![0; side + 1]);
            }

            let mut indptr = Vec::<$int_ty>::with_capacity(side + 1);
            indptr.push(0);
            // SAFETY: we allocate the space for the `indptr` array here, then each thread writes
            // in the number of nonzero entries for each row it was responsible for.  We know ahead
            // of time exactly how many entries we need (one per row, plus an explicit 0 to start).
            // It's also important that `$int_ty` does not implement `Drop`, since otherwise it
            // will be called on uninitialised memory (all primitive int types satisfy this).
            unsafe {
                indptr.set_len(side + 1);
            }

            // The parallel overhead from splitting a subtask is fairly high (allocating and
            // potentially growing a couple of vecs), so we're trading off some of Rayon's ability
            // to keep threads busy by subdivision with minimizing overhead; we're setting the
            // chunk size such that the iterator will have as many elements as there are threads.
            let num_threads = rayon::current_num_threads();
            let chunk_size = (side + num_threads - 1) / num_threads;
            let mut values_chunks = Vec::with_capacity(num_threads);
            let mut indices_chunks = Vec::with_capacity(num_threads);
            // SAFETY: the slice here is uninitialised data; it must not be read.
            indptr[1..]
                .par_chunks_mut(chunk_size)
                .enumerate()
                .map(|(i, indptr_chunk)| {
                    let start = chunk_size * i;
                    let end = (chunk_size * (i + 1)).min(side);
                    let mut order = (0..num_ops).collect::<Vec<_>>();
                    // Since we compressed the Paulis by summing equal elements, we're
                    // lower-bounded on the number of elements per row by this value, up to
                    // cancellations.  This should be a reasonable trade-off between sometimes
                    // expanding the vector and overallocation.
                    let mut values =
                        Vec::<Complex64>::with_capacity(chunk_size * (num_ops + 1) / 2);
                    let mut indices = Vec::<$int_ty>::with_capacity(chunk_size * (num_ops + 1) / 2);
                    let mut nnz = 0;
                    for i_row in start..end {
                        order.sort_unstable_by(|&a, &b| {
                            (i_row as $uint_ty ^ paulis.x_like[a] as $uint_ty)
                                .cmp(&(i_row as $uint_ty ^ paulis.x_like[b] as $uint_ty))
                        });
                        let mut running = C_ZERO;
                        let mut prev_index = i_row ^ (paulis.x_like[order[0]] as usize);
                        for (x_like, z_like, coeff) in order
                            .iter()
                            .map(|&i| (paulis.x_like[i], paulis.z_like[i], paulis.coeffs[i]))
                        {
                            let coeff =
                                if (i_row as $uint_ty & z_like as $uint_ty).count_ones() % 2 == 0 {
                                    coeff
                                } else {
                                    -coeff
                                };
                            let index = i_row ^ (x_like as usize);
                            if index == prev_index {
                                running += coeff;
                            } else {
                                nnz += 1;
                                values.push(running);
                                indices.push(prev_index as $int_ty);
                                running = coeff;
                                prev_index = index;
                            }
                        }
                        nnz += 1;
                        values.push(running);
                        indices.push(prev_index as $int_ty);
                        // When we write it, this is a cumulative `nnz` _within the chunk_.  We
                        // turn that into a proper cumulative sum in serial afterwards.
                        indptr_chunk[i_row - start] = nnz;
                    }
                    (values, indices)
                })
                .unzip_into_vecs(&mut values_chunks, &mut indices_chunks);
            // Turn the chunkwise nnz counts into absolute nnz counts.
            let mut start_nnz = 0usize;
            let chunk_nnz = values_chunks
                .iter()
                .map(|chunk| {
                    let prev = start_nnz;
                    start_nnz += chunk.len();
                    prev as $int_ty
                })
                .collect::<Vec<_>>();
            indptr[1..]
                .par_chunks_mut(chunk_size)
                .zip(chunk_nnz)
                .for_each(|(indptr_chunk, start_nnz)| {
                    indptr_chunk.iter_mut().for_each(|nnz| *nnz += start_nnz);
                });
            // Concatenate the chunkwise values and indices togther.
            let values = copy_flat_parallel(&values_chunks);
            let indices = copy_flat_parallel(&indices_chunks);
            (values, indices, indptr)
        }
    };
}

impl_to_matrix_sparse!(
    to_matrix_sparse_serial_32,
    to_matrix_sparse_parallel_32,
    i32,
    u32
);
impl_to_matrix_sparse!(
    to_matrix_sparse_serial_64,
    to_matrix_sparse_parallel_64,
    i64,
    u64
);

#[pymodule]
pub fn sparse_pauli_op(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(unordered_unique))?;
    m.add_wrapped(wrap_pyfunction!(decompose_dense))?;
    m.add_wrapped(wrap_pyfunction!(to_matrix_dense))?;
    m.add_wrapped(wrap_pyfunction!(to_matrix_sparse))?;
    m.add_class::<ZXPaulis>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::*;

    // The purpose of these tests is more about exercising the `unsafe` code; we test for full
    // correctness from Python space.

    fn example_paulis() -> MatrixCompressedPaulis {
        MatrixCompressedPaulis {
            num_qubits: 4,
            x_like: vec![0b0000, 0b0001, 0b0010, 0b1100, 0b1010, 0b0000],
            z_like: vec![0b1000, 0b0110, 0b1001, 0b0100, 0b1010, 0b1000],
            // Deliberately using multiples of small powers of two so the floating-point addition
            // of them is associative.
            coeffs: vec![
                c64(0.25, 0.5),
                c64(0.125, 0.25),
                c64(0.375, 0.125),
                c64(-0.375, 0.0625),
                c64(-0.5, -0.25),
            ],
        }
    }

    #[test]
    fn dense_threaded_and_serial_equal() {
        let paulis = example_paulis();
        let parallel = in_scoped_thread_pool(|| to_matrix_dense_inner(&paulis, true)).unwrap();
        let serial = to_matrix_dense_inner(&paulis, false);
        assert_eq!(parallel, serial);
    }

    #[test]
    fn sparse_threaded_and_serial_equal_32() {
        let paulis = example_paulis();
        let parallel = in_scoped_thread_pool(|| to_matrix_sparse_parallel_32(&paulis)).unwrap();
        let serial = to_matrix_sparse_serial_32(&paulis);
        assert_eq!(parallel, serial);
    }

    #[test]
    fn sparse_threaded_and_serial_equal_64() {
        let paulis = example_paulis();
        let parallel = in_scoped_thread_pool(|| to_matrix_sparse_parallel_64(&paulis)).unwrap();
        let serial = to_matrix_sparse_serial_64(&paulis);
        assert_eq!(parallel, serial);
    }
}
