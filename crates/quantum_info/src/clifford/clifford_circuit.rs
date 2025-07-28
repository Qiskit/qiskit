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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CliffordGate {
    CNOT(usize, usize),
    CZ(usize, usize),
    H(usize),
    S(usize),
    Sd(usize),
    SqrtX(usize),
    SqrtXd(usize),
}
impl CliffordGate {
    pub fn dagger(&self) -> Self {
        match self {
            Self::S(i) => Self::Sd(*i),
            Self::SqrtX(i) => Self::SqrtXd(*i),
            Self::Sd(i) => Self::S(*i),
            Self::SqrtXd(i) => Self::SqrtX(*i),
            _ => *self,
        }
    }
    pub fn to_vec(&self) -> (String, Vec<usize>) {
        match self {
            CliffordGate::CNOT(i, j) => ("CNOT".to_owned(), vec![*i, *j]),
            CliffordGate::CZ(i, j) => ("CZ".to_owned(), vec![*i, *j]),
            CliffordGate::H(i) => ("H".to_owned(), vec![*i]),
            CliffordGate::S(i) => ("S".to_owned(), vec![*i]),
            CliffordGate::Sd(i) => ("Sd".to_owned(), vec![*i]),
            CliffordGate::SqrtX(i) => ("SqrtX".to_owned(), vec![*i]),
            CliffordGate::SqrtXd(i) => ("SqrtXd".to_owned(), vec![*i]),
        }
    }
    pub fn from_vec(gate: &str, qbits: &[usize]) -> Self {
        match gate {
            "H" => Self::H(qbits[0]),
            "S" => Self::S(qbits[0]),
            "Sd" => Self::Sd(qbits[0]),
            "SqrtX" => Self::SqrtX(qbits[0]),
            "SqrtXd" => Self::SqrtXd(qbits[0]),
            "CX" => Self::CNOT(qbits[0], qbits[1]),
            "CNOT" => Self::CNOT(qbits[0], qbits[1]),
            "CZ" => Self::CZ(qbits[0], qbits[1]),
            _ => panic!("Unknown gate {}", gate),
        }
    }
    pub fn arity(&self) -> usize {
        match self {
            CliffordGate::CNOT(_, _) => 2,
            CliffordGate::CZ(_, _) => 2,
            _ => 1,
        }
    }
}
#[derive(Debug, Clone)]
pub struct CliffordCircuit {
    pub nqbits: usize,
    pub gates: Vec<CliffordGate>,
}

impl CliffordCircuit {
    pub fn new(nqbits: usize) -> Self {
        Self {
            nqbits,
            gates: Vec::new(),
        }
    }
    pub fn from_vec(gates: Vec<(String, Vec<usize>)>) -> Self {
        let mut nqbits = 0;
        for (_, qbits) in gates.iter() {
            for qbit in qbits {
                if qbit + 1 > nqbits {
                    nqbits = qbit + 1;
                }
            }
        }
        Self {
            nqbits,
            gates: gates
                .iter()
                .map(|(gate, qbits)| CliffordGate::from_vec(gate, qbits))
                .collect(),
        }
    }

    pub fn extend_with(&mut self, other: &CliffordCircuit) {
        self.gates.extend_from_slice(&other.gates);
    }

    /// Returns the inverse of the circuit
    pub fn dagger(&self) -> Self {
        let new_gates = self.gates.iter().rev().map(|gate| gate.dagger()).collect();
        Self {
            nqbits: self.nqbits,
            gates: new_gates,
        }
    }
}
