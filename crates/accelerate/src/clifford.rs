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

use ndarray::ArrayView1;
use pyo3::prelude::*;

use numpy::prelude::*;

use numpy::{PyArray2, PyReadonlyArray2};

use numpy::ndarray::{aview2, Array2, ArrayView2};
use smallvec::SmallVec;
use crate::QiskitError;


// /// Symplectic tableau
// #[pyclass(module = "qiskit._accelerate.clifford")]
// pub struct Tableau {
//     pub tableau: Array2<bool>,
// }
//
// #[pymethods]
// impl Tableau {
//     fn __new__(tableau: &Bound<PyReadonlyArray2<bool>>) -> Self {
//         Self {
//             tableau: tableau.asarray(),
//         }
//     }
//
//     ///
//     fn get_tableau(&self, py: Python) -> PyObject {
//         self.tableau.to_object(py)
//     }
//
// }

// the idea is to create a function to decompose cliffords



// Global arrays of the 16 pairs of Pauli operators
// divided into 5 equivalence classes under the action of single-qubit Cliffords.


// ToDo: replace this crap by a HashMap !!!!

// Class A - canonical representative is 'XZ'
static A_CLASS: [[[bool; 2]; 2]; 6] =
[
    [[false, true], [true, true]],   // 'XY'
    [[false, true], [true, false]],  // 'XZ'
    [[true, true], [false, true]],   // 'YX'
    [[true, true], [true, false]],   // 'YZ'
    [[true, false], [false, true]],  // 'ZX'
    [[true, false], [true, true]],   // 'ZY'
];

// Class B - canonical representative is 'XX'
static B_CLASS: [[[bool; 2]; 2]; 3] = [
    [[true, false], [true, false]],  // 'ZZ'
    [[false, true], [false, true]],  // 'XX'
    [[true, true], [true, true]],    // 'YY'
];

// Class C - canonical representative is 'XI'
static C_CLASS: [[[bool; 2]; 2]; 3] = [
    [[true, false], [false, false]],  // 'ZI'
    [[false, true], [false, false]],  // 'XI'
    [[true, true], [false, false]],   // 'YI'
];

// Class D - canonical representative is 'IZ'
static D_CLASS: [[[bool; 2]; 2]; 3] = [
    [[false, false], [false, true]],  // 'IX'
    [[false, false], [true, false]],  // 'IZ'
    [[false, false], [true, true]],   // 'IY'
];

// Class E - only 'II'
static E_CLASS: [[[bool; 2]; 2]; 1] = [
    [[false, false], [false, false]]  // 'II'
];


fn compute_greedy_cost(pairs: &Vec<[[bool; 2]; 2]>) -> PyResult<usize>{
    let mut a_num = 0;
    let mut b_num = 0;
    let mut c_num = 0;
    let mut d_num = 0;

    for pair in pairs {
        if A_CLASS.contains(&pair) {
            a_num += 1;
        }
        else if B_CLASS.contains(&pair) {
            b_num += 1;
        }
        else if C_CLASS.contains(&pair) {
            c_num += 1;
        }
        else if D_CLASS.contains(&pair) {
            d_num += 1;
        }
    }

    if a_num % 2 == 0 {
        return Err(QiskitError::new_err(
            "Symplectic Gaussian elimination fails",
        ));
    }

    let mut cnot_cost: usize =  3 * (a_num - 1) / 2 + (b_num + 1) * ((b_num > 0) as usize) + c_num + d_num;

    if !A_CLASS.contains(&pairs[0]) {
        cnot_cost += 3;
    }
    Ok(cnot_cost)
}



// hack: for single-qubit gates second arg is always 0
type CliffordSequenceVec = Vec<(String, SmallVec<[u8; 2]>)>;

fn from_pair_paulis_to_type(pauli_x: ArrayView1<bool>, pauli_z: ArrayView1<bool>, qubit: usize) -> [[bool; 2]; 2] {
    let num_qubits: usize = pauli_x.len() / 2;
    [[pauli_x[num_qubits + qubit], pauli_x[qubit]], [pauli_z[num_qubits + qubit], pauli_z[qubit]]]
}


// for now modify clifford in-place
fn synth_clifford_greedy_inner(clifford: &Array2<bool>) -> PyResult<()> {
    let mut clifford_gates = CliffordSequenceVec::new();

    println!("I AM IN SYNTH_CLIFFORD_INNER!");
    println!("{:?}", clifford);

    let num_qubits = clifford.shape()[0] / 2;
    println!("num_qubits = {:?}", num_qubits);

    // ToDo: this is a vector for now to be compatible with the python
    // implementation, but we should really turn it into a set
    let qubit_list: Vec<usize> = (0..num_qubits).collect();

    // while

    let mut list_greedy_cost  = Vec::<(usize, usize)>::new();

    for qubit in &qubit_list {
        println!("HERE qubit = {}" , qubit);
        let pauli_x = clifford.column(*qubit);
        println!("HERE pauli_x = {:?}" , pauli_x);
        let pauli_z = clifford.column(*qubit + num_qubits);
        println!("HERE pauli_z = {:?}" , pauli_z);

        let list_pairs: Vec<[[bool; 2]; 2]> = qubit_list
            .iter()
            .map(|i| from_pair_paulis_to_type(pauli_x, pauli_z, *i))
            .collect();

        println!("{:?}", list_pairs);

        let cost = compute_greedy_cost(&list_pairs)?;
        println!("{}", cost);
        list_greedy_cost.push((cost, *qubit));
    }
    println!("list_greedy_cost = {:?}", list_greedy_cost);
    let min_qubit = list_greedy_cost.iter().min_by_key(|(cost, qubit)| cost).unwrap().0;
    println!("min_qubit = {:?}", min_qubit);
    // clifford[0 : self.num_qubits, :]

    let pauli_x = clifford.column(min_qubit);
    let pauli_z = clifford.column(min_qubit + num_qubits);



    Ok(())
}


#[pyfunction]
#[pyo3(signature = (tableau))]
fn synth_clifford_greedy_new(tableau: PyReadonlyArray2<bool>) {
    let mut clifford_gates = CliffordSequenceVec::new();
    let clifford = tableau.as_array().to_owned();

    println!("I AM IN SYNTH_CLIFFORD_GREEDY_NEW!");

    synth_clifford_greedy_inner(&clifford);

}






#[pymodule]
pub fn clifford(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(synth_clifford_greedy_new, m)?)?;
    Ok(())
}



#[cfg(test)]
mod tests {
    use ndarray::arr2;
    use super::*;
    use crate::test::*;

    // for quicker development
    fn example_clifford() -> Array2<bool> {
        arr2(
        &[[false, false, true, true, false, true, false, false, false, true, true],
         [false, false, false, true, false, false, false, true, true, false, false],
         [false, true, true, true, false, false, false, false, false, false, true],
         [false, false, true, true, false, false, false, false, false, false, true],
         [false, false, true, false, false, true, false, true, true, false, true],
         [false, true, false, true, true, true, false, true, true, false, false],
         [true, false, false, false, true, true, false, true, true, false, false],
         [false, true, true, false, false, false, true, true, true, true, true],
         [true, false, true, true, false, true, true, true, false, false, false],
         [true, true, true, false, true, true, false, true, true, true, true]])
    }

    #[test]
    fn test_example_clifford() {
        println!("===========================");
        println!("TEST!!!");
        let cliff = example_clifford();
        synth_clifford_greedy_inner(&cliff);
        println!("===========================");

    }
}
