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

use qiskit_circuit::nlayout::{NLayout, PhysicalQubit, VirtualQubit};

/// The "layout" caused by transpilation
///
/// In general Qiskit's transpiler is unitary-preserving up to the initial layout
/// and routing permutations. The initial layout permutation is caused by
/// setting and applying the initial layout (the mapping from virtual circuit
/// qubits to physical qubits on the target) and the routing permtuations are
/// caused by swap gate insertion or permutation ellision prior to the initial
/// layout. This struct tracks these details and provide an interface to reason
/// about these permutations.
pub struct TranspileLayout {
    /// The initial layout which is mapping the virtual qubits in the input circuit to the
    /// transpiler to the physical qubits used on the transpilation target
    initial_layout: NLayout,
    /// The optional routing permutation that
    /// represents the permutation caused by routing or permutation elision during
    /// transpilation. This vector maps the qubits at the start of the circuit to their
    /// final position/physical qubit at the end of the circuit.
    routing_permutation: Option<Vec<PhysicalQubit>>,
    /// The number of qubits in the input circuit to the transpiler.
    input_qubit_count: u32,
    /// The number of qubits in the output circuit, this will differ from `input_qubit_count`
    /// when the transpiler allocates ancilla qubits when the target has more physical qubits
    /// than the input circuit has virtual qubits.
    output_qubit_count: u32,
}

impl TranspileLayout {

    /// Construct a new [`TranspileLayout`] object
    ///
    /// # Args
    ///
    /// - `initial_layout` - The initial layout which is mapping from the virtual qubits in the
    ///     input circuit to the transpiler to the physical qubits used on the transpilation target
    /// - `routing_permutation` - The optional routing permutation that
    ///     represents the permutation caused by routing or permutation elision during
    ///     transpilation. This vector maps the qubits at the start of the circuit to their
    ///     final position/physical qubit at the end of the circuit.
    /// - `input_qubit_count` - The number of qubits in the input circuit
    /// - `output_qubit_count` - The number of qubits in the output circuit,
    pub fn new(
        initial_layout: NLayout,
        routing_permutation: Option<Vec<PhysicalQubit>>,
        input_qubit_count: u32,
        output_qubit_count: u32,
    ) -> Self {
        TranspileLayout {
            initial_layout,
            routing_permutation,
            input_qubit_count,
            output_qubit_count,
        }
    }

    /// The number of input circuit qubits
    pub fn num_input_qubits(&self) -> u32 {
        self.input_qubit_count
    }

    /// The number of output circuit qubits
    pub fn num_output_qubits(&self) -> u32 {
        self.output_qubit_count
    }

    /// Generate an initial layout as an array of PhysicalQubit indices.
    ///
    /// # Args
    ///     `filter_ancillas` - If set to `true` any ancilla qubits added to
    ///     the circuit by the transpiler will not be included in the output
    ///     array.
    pub fn initial_layout(&self, filter_ancillas: bool) -> Vec<PhysicalQubit> {
        if filter_ancillas {
            (0..self.input_qubit_count)
                .map(|x| {
                    self.initial_layout
                        .virtual_to_physical(VirtualQubit::new(x))
                })
                .collect()
        } else {
            (0..self.output_qubit_count)
                .map(|x| {
                    self.initial_layout
                        .virtual_to_physical(VirtualQubit::new(x))
                })
                .collect()
        }
    }

    /// Return the routing permutation
    ///
    /// This method returns an option slice, if it is `Some` then there
    /// was a permutation introduced by the transpiler typically caused
    /// by either routing or permutation elision. The slice contained
    /// in the return represents the mapping of the qubits at the start
    /// of the circuit for each index to their final position/physical
    /// qubit at the end of the circuit.
    ///
    /// If you would instead prefer to represent no permutation case with
    /// an explicit trivial permutation (e.g. `[0, 1, 2, 3]`) then you can
    /// use [`explicit_routing_permutation`] which always returns a slice.
    pub fn routing_permutation(&self) -> Option<&[PhysicalQubit]> {
        self.routing_permutation.as_deref()
    }

    /// Return the routing permutation explicitly
    ///
    /// This method returns a slice, of the permutation introduced by the
    /// transpiler typically caused by either routing or permutation elision.
    /// The slice contained in the return represents the mapping of the qubits
    /// at the start of the circuit for each index to their final position/physical
    /// qubit at the end of the circuit.
    ///
    /// If you would instead prefer to represent no permutation case with
    /// a `None` then you can use [`routing_permutation`] which returns an
    /// `Option<&[PhysicalQubit]>`.
    pub fn explicit_routing_permutation(&self) -> std::borrow::Cow<'_, [PhysicalQubit]> {
        match self.routing_permutation {
            Some(ref perm) => std::borrow::Cow::Borrowed(perm),
            None => std::borrow::Cow::Owned(
                (0..self.output_qubit_count)
                    .map(PhysicalQubit::new)
                    .collect(),
            ),
        }
    }

    /// Generate the final layout as an array of PhysicalQubits
    ///
    /// This method will generate an array of final positions for each qubit in the input circuit.
    /// For example, if you had an input circuit like:
    ///
    /// ```python
    /// qc = QuantumCircuit(3)
    /// qc.h(0)
    /// qc.cx(0, 1)
    /// qc.cx(0, 2)
    /// ```
    ///
    /// and then the output from the transpiler was:
    ///
    /// ```python
    /// tqc = QuantumCircuit(3)
    /// tqc.h(2)
    /// tqc.cx(2, 1)
    /// tqc.swap(0, 1)
    /// tqc.cx(2, 1)
    /// ```
    ///
    /// then the `final_index_layout` method returns:
    ///
    /// ```python
    /// [2, 0, 1]
    /// ```
    ///
    /// This can be seen as follows. Qubit 0 in the original circuit is mapped to qubit 2
    /// in the output circuit during the layout stage, which is mapped to qubit 2 during the
    /// routing stage. Qubit 1 in the original circuit is mapped to qubit 1 in the output
    /// circuit during the layout stage, which is mapped to qubit 0 during the routing
    /// stage. Qubit 2 in the original circuit is mapped to qubit 0 in the output circuit
    /// during the layout stage, which is mapped to qubit 1 during the routing stage.
    /// The output list length will be as wide as the input circuit's number of qubits,
    /// as the output list from this method is for tracking the permutation of qubits in the
    /// original circuit caused by the transpiler.
    ///
    /// # Args
    ///     `filter_ancillas` - If set to `true` any ancilla qubits added to
    ///     the circuit by the transpiler will not be included in the output
    ///     array.
    pub fn final_index_layout(&self, filter_ancillas: bool) -> Vec<PhysicalQubit> {
        let qubit_range = if filter_ancillas {
            0..self.input_qubit_count
        } else {
            0..self.output_qubit_count
        };
        qubit_range
            .map(|idx| {
                let mut qubit_idx = self
                    .initial_layout
                    .virtual_to_physical(VirtualQubit::new(idx));
                if let Some(ref routing_permutation) = self.routing_permutation {
                    qubit_idx = routing_permutation[qubit_idx.index()]
                }
                qubit_idx
            })
            .collect()
    }

    /// Generate the final layout
    ///
    /// This method will generate the final layout which is the mapping of the
    /// the virtual qubits in the input circuit to the transpiler to the
    /// physical qubits with that virtual qubit's state at the end of the circuit.
    /// For example, if you had an input circuit like:
    ///
    /// ```python
    /// qc = QuantumCircuit(3)
    /// qc.h(0)
    /// qc.cx(0, 1)
    /// qc.cx(0, 2)
    /// ```
    ///
    /// and then the output from the transpiler was:
    ///
    /// ```python
    /// tqc = QuantumCircuit(3)
    /// tqc.h(2)
    /// tqc.cx(2, 1)
    /// tqc.swap(0, 1)
    /// tqc.cx(2, 1)
    /// ```
    ///
    /// then the `final_layout` method returns:
    ///
    /// | Virtual Qubit | Physical Qunit |
    /// |---------------|----------------|
    /// | 0             | 2              |
    /// | 1             | 0              |
    /// | 2             | 1              |
    ///
    /// This can be seen as follows. Qubit 0 in the original circuit is mapped to qubit 2
    /// in the output circuit during the layout stage, which is mapped to qubit 2 during the
    /// routing stage. Qubit 1 in the original circuit is mapped to qubit 1 in the output
    /// circuit during the layout stage, which is mapped to qubit 0 during the routing
    /// stage. Qubit 2 in the original circuit is mapped to qubit 0 in the output circuit
    /// during the layout stage, which is mapped to qubit 1 during the routing stage.
    /// The output list length will be as wide as the input circuit's number of qubits,
    /// as the output list from this method is for tracking the permutation of qubits in the
    /// original circuit caused by the transpiler.
    ///
    /// # Returns
    ///
    /// An [`NLayout`] object. This will always include all ancilla qubits
    /// allocated by the transpiler because an `NLayout` must have an equal
    /// number of [`VirtualQubit`] and [`PhysicalQubit`]. If you want a view
    /// with ancillas filtered you should use [`final_index_layout`] instead
    /// which returns the layout as a `Vec<PhysicalQubit>`.
    pub fn final_layout(&self) -> NLayout {
        NLayout::from_virtual_to_physical(self.final_index_layout(false)).unwrap()
    }

    /// Compose another routing permutation into the contained in this layout
    ///
    /// # Args
    ///
    /// * `other` - The other permutation array to compose with
    /// * `reverse` - Whether to compose in reverse order
    pub fn compose_routing_permutation(&mut self, other: &[PhysicalQubit], reverse: bool) {
        if let Some(ref routing_permutation) = self.routing_permutation {
            let new_perm = if !reverse {
                let mut new_perm = routing_permutation.clone();
                routing_permutation
                    .iter()
                    .enumerate()
                    .for_each(|(idx, qubit)| {
                        new_perm[idx] = other[qubit.index()];
                    });
                new_perm
            } else {
                let mut new_perm = other.to_vec();
                other.iter().enumerate().for_each(|(idx, qubit)| {
                    new_perm[idx] = routing_permutation[qubit.index()];
                });
                new_perm
            };
            self.routing_permutation = Some(new_perm);
        } else {
            self.routing_permutation = Some(other.to_vec());
        }
    }
}

#[cfg(test)]
mod test_transpile_layout {
    use super::TranspileLayout;
    use qiskit_circuit::nlayout::{NLayout, PhysicalQubit};

    #[test]
    fn test_final_index_layout() {
        let initial_layout_vec = vec![PhysicalQubit(2), PhysicalQubit(1), PhysicalQubit(0)];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![PhysicalQubit(1), PhysicalQubit(0), PhysicalQubit(2)];
        let layout = TranspileLayout::new(initial_layout, Some(routing_permutation), 3, 3);
        let result = layout.final_index_layout(false);
        assert_eq!(
            vec![PhysicalQubit(2), PhysicalQubit(0), PhysicalQubit(1)],
            result
        );
    }

    #[test]
    fn test_final_layout() {
        let initial_layout_vec = vec![PhysicalQubit(2), PhysicalQubit(1), PhysicalQubit(0)];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![PhysicalQubit(1), PhysicalQubit(0), PhysicalQubit(2)];
        let layout = TranspileLayout::new(initial_layout, Some(routing_permutation), 3, 3);
        let result = layout.final_layout();
        assert_eq!(
            NLayout::from_virtual_to_physical(vec![
                PhysicalQubit(2),
                PhysicalQubit(0),
                PhysicalQubit(1)
            ])
            .unwrap(),
            result
        );
    }

    #[test]
    fn test_initial_layout() {
        let initial_layout_vec = vec![PhysicalQubit(2), PhysicalQubit(1), PhysicalQubit(0)];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![PhysicalQubit(1), PhysicalQubit(0), PhysicalQubit(2)];
        let layout = TranspileLayout::new(initial_layout, Some(routing_permutation), 3, 3);
        let result = layout.initial_layout(false);
        assert_eq!(
            vec![PhysicalQubit(2), PhysicalQubit(1), PhysicalQubit(0)],
            result
        );
    }

    #[test]
    fn test_routing_permutation() {
        let initial_layout_vec = vec![PhysicalQubit(2), PhysicalQubit(1), PhysicalQubit(0)];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![PhysicalQubit(1), PhysicalQubit(0), PhysicalQubit(2)];
        let layout = TranspileLayout::new(initial_layout, Some(routing_permutation), 3, 3);
        let result = layout.routing_permutation();
        assert_eq!(
            Some([PhysicalQubit(1), PhysicalQubit(0), PhysicalQubit(2)].as_slice()),
            result
        );
    }

    #[test]
    fn test_routing_permutation_explicit() {
        let initial_layout_vec = vec![PhysicalQubit(2), PhysicalQubit(1), PhysicalQubit(0)];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![PhysicalQubit(1), PhysicalQubit(0), PhysicalQubit(2)];
        let layout = TranspileLayout::new(initial_layout, Some(routing_permutation), 3, 3);
        let result = layout.explicit_routing_permutation();
        assert_eq!(
            std::borrow::Cow::Borrowed(&[PhysicalQubit(1), PhysicalQubit(0), PhysicalQubit(2)]),
            result
        );
    }

    #[test]
    fn test_final_index_layout_with_ancillas() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![
            PhysicalQubit(2),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(4),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
            PhysicalQubit(9),
            PhysicalQubit(3),
        ];
        let layout = TranspileLayout::new(initial_layout, Some(routing_permutation), 3, 10);
        let result = layout.final_index_layout(true);
        assert_eq!(
            vec![PhysicalQubit(3), PhysicalQubit(5), PhysicalQubit(2)],
            result
        )
    }

    #[test]
    fn test_initial_layout_with_ancillas() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![
            PhysicalQubit(2),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(4),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
            PhysicalQubit(9),
            PhysicalQubit(3),
        ];
        let layout = TranspileLayout::new(initial_layout, Some(routing_permutation), 3, 10);
        let result = layout.initial_layout(true);
        assert_eq!(
            vec![PhysicalQubit(9), PhysicalQubit(4), PhysicalQubit(0)],
            result
        )
    }

    #[test]
    fn test_routing_permutation_with_ancillas() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![
            PhysicalQubit(2),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(4),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
            PhysicalQubit(9),
            PhysicalQubit(3),
        ];
        let expected = routing_permutation.clone();
        let layout = TranspileLayout::new(initial_layout, Some(routing_permutation), 3, 10);
        let result = layout.routing_permutation();
        assert_eq!(Some(expected.as_slice()), result)
    }

    #[test]
    fn test_routing_permutation_with_ancillas_explicit() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![
            PhysicalQubit(2),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(4),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
            PhysicalQubit(9),
            PhysicalQubit(3),
        ];
        let expected_vec = routing_permutation.clone();
        let expected: std::borrow::Cow<_> = expected_vec.as_slice().into();
        let layout = TranspileLayout::new(initial_layout, Some(routing_permutation), 3, 10);
        let result = layout.explicit_routing_permutation();
        assert_eq!(expected, result)
    }

    #[test]
    fn test_final_index_layout_with_ancillas_no_filter() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![
            PhysicalQubit(2),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(4),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
            PhysicalQubit(9),
            PhysicalQubit(3),
        ];
        let layout = TranspileLayout::new(initial_layout, Some(routing_permutation), 3, 10);
        let result = layout.final_index_layout(false);
        let expected = vec![
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(2),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(4),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
            PhysicalQubit(9),
        ];
        assert_eq!(expected, result)
    }

    #[test]
    fn test_final_layout_with_ancillas_no_filter() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![
            PhysicalQubit(2),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(4),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
            PhysicalQubit(9),
            PhysicalQubit(3),
        ];
        let layout = TranspileLayout::new(initial_layout, Some(routing_permutation), 3, 10);
        let result = layout.final_layout();
        let expected = NLayout::from_virtual_to_physical(vec![
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(2),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(4),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
            PhysicalQubit(9),
        ])
        .unwrap();
        assert_eq!(expected, result)
    }

    #[test]
    fn test_initial_layout_with_ancillas_no_filter() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let expected = initial_layout_vec.clone();
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let routing_permutation = vec![
            PhysicalQubit(2),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(4),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
            PhysicalQubit(9),
            PhysicalQubit(3),
        ];
        let layout = TranspileLayout::new(initial_layout, Some(routing_permutation), 3, 10);
        let result = layout.initial_layout(false);
        assert_eq!(expected, result)
    }

    #[test]
    fn test_final_index_layout_no_routing() {
        let initial_layout_vec = vec![PhysicalQubit(1), PhysicalQubit(2), PhysicalQubit(0)];
        let expected = initial_layout_vec.clone();
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let layout = TranspileLayout::new(initial_layout, None, 3, 3);
        let result = layout.final_index_layout(false);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_final_layout_no_routing() {
        let initial_layout_vec = vec![PhysicalQubit(1), PhysicalQubit(2), PhysicalQubit(0)];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let expected = initial_layout.clone();
        let layout = TranspileLayout::new(initial_layout, None, 3, 3);
        let result = layout.final_layout();
        assert_eq!(expected, result);
    }

    #[test]
    fn test_initial_layout_no_routing() {
        let initial_layout_vec = vec![PhysicalQubit(1), PhysicalQubit(2), PhysicalQubit(0)];
        let expected = initial_layout_vec.clone();
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let layout = TranspileLayout::new(initial_layout, None, 3, 3);
        let result = layout.initial_layout(false);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_routing_permutation_no_routing() {
        let initial_layout_vec = vec![PhysicalQubit(1), PhysicalQubit(2), PhysicalQubit(0)];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let layout = TranspileLayout::new(initial_layout, None, 3, 3);
        let result = layout.routing_permutation();
        assert_eq!(None, result);
    }

    #[test]
    fn test_routing_permutation_no_routing_explicit() {
        let initial_layout_vec = vec![PhysicalQubit(1), PhysicalQubit(2), PhysicalQubit(0)];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let layout = TranspileLayout::new(initial_layout, None, 3, 3);
        let result = layout.explicit_routing_permutation();
        let expected: std::borrow::Cow<[PhysicalQubit]> =
            (0..3u32).map(PhysicalQubit::new).collect();
        assert_eq!(expected, result);
    }

    #[test]
    fn test_final_index_layout_no_routing_with_ancillas() {
        let initial_layout_vec = vec![
            PhysicalQubit(2),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(3),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let layout = TranspileLayout::new(initial_layout, None, 3, 5);
        let result = layout.final_index_layout(true);
        assert_eq!(
            vec![PhysicalQubit(2), PhysicalQubit(4), PhysicalQubit(0)],
            result
        );
    }

    #[test]
    fn test_initial_layout_no_routing_with_ancillas() {
        let initial_layout_vec = vec![
            PhysicalQubit(2),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(3),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let layout = TranspileLayout::new(initial_layout, None, 3, 5);
        let result = layout.initial_layout(true);
        assert_eq!(
            vec![PhysicalQubit(2), PhysicalQubit(4), PhysicalQubit(0)],
            result
        );
    }

    #[test]
    fn test_routing_permutation_no_routing_with_ancillas() {
        let initial_layout_vec = vec![
            PhysicalQubit(2),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(3),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let layout = TranspileLayout::new(initial_layout, None, 3, 5);
        let result = layout.routing_permutation();
        assert_eq!(None, result);
    }

    #[test]
    fn test_routing_permutation_no_routing_with_ancillas_explicit() {
        let initial_layout_vec = vec![
            PhysicalQubit(2),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(3),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let layout = TranspileLayout::new(initial_layout, None, 3, 5);
        let result = layout.explicit_routing_permutation();
        let expected = std::borrow::Cow::from_iter((0..5u32).map(PhysicalQubit::new));
        assert_eq!(expected, result);
    }

    #[test]
    fn test_final_index_layout_no_routing_with_ancillas_no_filter() {
        let initial_layout_vec = vec![
            PhysicalQubit(2),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(3),
        ];
        let expected = initial_layout_vec.clone();
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let layout = TranspileLayout::new(initial_layout, None, 3, 5);
        let result = layout.final_index_layout(false);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_final_layout_no_routing_with_ancillas_no_filter() {
        let initial_layout_vec = vec![
            PhysicalQubit(2),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(3),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let expected = initial_layout.clone();
        let layout = TranspileLayout::new(initial_layout, None, 3, 5);
        let result = layout.final_layout();
        assert_eq!(expected, result);
    }

    #[test]
    fn test_initial_layout_no_routing_with_ancillas_no_filter() {
        let initial_layout_vec = vec![
            PhysicalQubit(2),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(3),
        ];
        let expected = initial_layout_vec.clone();
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let layout = TranspileLayout::new(initial_layout, None, 3, 5);
        let result = layout.initial_layout(false);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_compose() {
        let first = vec![
            PhysicalQubit(0),
            PhysicalQubit(3),
            PhysicalQubit(1),
            PhysicalQubit(2),
        ];
        let second = vec![
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(1),
            PhysicalQubit(0),
        ];
        let mut layout =
            TranspileLayout::new(NLayout::generate_trivial_layout(4), Some(first), 4, 4);
        layout.compose_routing_permutation(&second, true);
        let result = layout.routing_permutation();
        let expected = Some(
            [
                PhysicalQubit(1),
                PhysicalQubit(2),
                PhysicalQubit(3),
                PhysicalQubit(0),
            ]
            .as_slice(),
        );
        assert_eq!(expected, result);
        let first = vec![
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(0),
        ];
        let second = vec![
            PhysicalQubit(0),
            PhysicalQubit(2),
            PhysicalQubit(1),
            PhysicalQubit(3),
        ];
        let mut layout =
            TranspileLayout::new(NLayout::generate_trivial_layout(4), Some(first), 4, 4);
        layout.compose_routing_permutation(&second, false);
        let result = layout.routing_permutation();
        let expected = Some(
            [
                PhysicalQubit(2),
                PhysicalQubit(1),
                PhysicalQubit(3),
                PhysicalQubit(0),
            ]
            .as_slice(),
        );
        assert_eq!(expected, result);
    }

    #[test]
    fn test_compose_no_permutation_original() {
        let second = vec![
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(1),
            PhysicalQubit(0),
        ];
        let mut layout = TranspileLayout::new(NLayout::generate_trivial_layout(4), None, 4, 4);
        layout.compose_routing_permutation(&second, false);
        let result = layout.routing_permutation();
        let expected = Some(
            [
                PhysicalQubit(2),
                PhysicalQubit(3),
                PhysicalQubit(1),
                PhysicalQubit(0),
            ]
            .as_slice(),
        );
        assert_eq!(expected, result);
    }

    #[test]
    fn test_compose_no_permutation_second() {
        let second = [
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(1),
            PhysicalQubit(0),
        ];
        let mut layout = TranspileLayout::new(NLayout::generate_trivial_layout(4), None, 4, 4);
        layout.compose_routing_permutation(&second, true);
        let result = layout.routing_permutation();
        let expected = Some(
            [
                PhysicalQubit(2),
                PhysicalQubit(3),
                PhysicalQubit(1),
                PhysicalQubit(0),
            ]
            .as_slice(),
        );
        assert_eq!(expected, result);
    }
}
