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
// Data tree serialization code
#![allow(dead_code)] // Currently in progress
use crate::data_tree::*;
use crate::serialization::formats::*;
use hashbrown::{HashMap, HashSet};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum UnpackTreeError {
    #[error("a packed data tree contains no nodes")]
    Empty,
    #[error("node {node} has more children than can be represented on this platform")]
    ChildCountOverflow { node: usize },
    #[error("node {node} expects {expected} children, but only {remaining} nodes remain")]
    MissingChildren {
        node: usize,
        expected: usize,
        remaining: usize,
    },
    #[error("the packed data tree contains {count} trailing nodes")]
    TrailingNodes { count: usize },
    #[error(
        "key {key:?} on node {node} refers to child {child}, but the node has {count} children"
    )]
    KeyOutOfRange {
        node: usize,
        key: String,
        child: u64,
        count: usize,
    },
    #[error("node {node} assigns more than one key to child {child}")]
    DuplicateChildKey { node: usize, child: usize },
    #[error("node {node} uses key {key:?} more than once")]
    DuplicateKey { node: usize, key: String },
    #[error("invalid key on node {node}: {source}")]
    InvalidKey {
        node: usize,
        #[source]
        source: InvalidName,
    },
}

pub fn pack_tree<T>(tree: &DataTree<T>) -> DataTreePack {
    let mut nodes = Vec::new();
    let mut node_queue = vec![tree];
    let mut next = 0;
    while next < node_queue.len() {
        let node_to_pack = node_queue[next];
        next += 1;
        match node_to_pack {
            DataTree::Leaf(_) => nodes.push(DataTreeNodePack::Leaf),
            DataTree::Branch(branch) => {
                let num_children = branch.len() as u64;
                let mut keys = Vec::new();

                for (i, (name, child)) in node_to_pack.iter_children().enumerate() {
                    node_queue.push(child);

                    if let Some(name) = name {
                        keys.push((
                            StringU16Pack {
                                value: name.to_owned(),
                            },
                            i as u64,
                        ));
                    }
                }
                let keys_pack = DataTreeBranchKeysPack { keys };
                nodes.push(DataTreeNodePack::Branch(DataTreeBranchPack {
                    num_children,
                    keys_pack,
                }))
            }
        }
    }
    DataTreePack { nodes }
}

/// Reconstruct a data-tree structure from its breadth-first packed representation.
pub fn unpack_tree(pack: DataTreePack) -> Result<DataTree<()>, UnpackTreeError> {
    if pack.nodes.is_empty() {
        return Err(UnpackTreeError::Empty);
    }

    // Associate every branch with the contiguous range containing its children.  Since the
    // packed representation is breadth-first, these ranges are assigned in node order.
    let mut children = vec![None; pack.nodes.len()];
    let mut next_child = 1usize;
    for (node_index, node) in pack.nodes.iter().enumerate() {
        if node_index >= next_child {
            return Err(UnpackTreeError::TrailingNodes {
                count: pack.nodes.len() - node_index,
            });
        }
        let DataTreeNodePack::Branch(branch) = node else {
            continue;
        };
        let count = usize::try_from(branch.num_children)
            .map_err(|_| UnpackTreeError::ChildCountOverflow { node: node_index })?;
        let end = next_child
            .checked_add(count)
            .ok_or(UnpackTreeError::ChildCountOverflow { node: node_index })?;
        if end > pack.nodes.len() {
            return Err(UnpackTreeError::MissingChildren {
                node: node_index,
                expected: count,
                remaining: pack.nodes.len().saturating_sub(next_child),
            });
        }
        children[node_index] = Some(next_child..end);
        next_child = end;
    }
    if next_child != pack.nodes.len() {
        return Err(UnpackTreeError::TrailingNodes {
            count: pack.nodes.len() - next_child,
        });
    }

    let mut unpacked: Vec<Option<DataTree<()>>> = std::iter::repeat_with(|| None)
        .take(pack.nodes.len())
        .collect();
    for node_index in (0..pack.nodes.len()).rev() {
        unpacked[node_index] = Some(match &pack.nodes[node_index] {
            DataTreeNodePack::Leaf => DataTree::Leaf(()),
            DataTreeNodePack::Branch(branch) => {
                let child_range = children[node_index].clone().unwrap();
                let child_count = child_range.len();
                let mut names = HashMap::with_capacity(branch.keys_pack.keys.len());
                let mut seen_names = HashSet::with_capacity(branch.keys_pack.keys.len());
                for (name, child) in &branch.keys_pack.keys {
                    let child_index = usize::try_from(*child).ok();
                    let Some(child_index) = child_index.filter(|&index| index < child_count) else {
                        return Err(UnpackTreeError::KeyOutOfRange {
                            node: node_index,
                            key: name.value.clone(),
                            child: *child,
                            count: child_count,
                        });
                    };
                    if names.insert(child_index, name.value.clone()).is_some() {
                        return Err(UnpackTreeError::DuplicateChildKey {
                            node: node_index,
                            child: child_index,
                        });
                    }
                    if !seen_names.insert(name.value.clone()) {
                        return Err(UnpackTreeError::DuplicateKey {
                            node: node_index,
                            key: name.value.clone(),
                        });
                    }
                }

                let mut tree = DataTree::with_capacity(child_count);
                for (position, packed_index) in child_range.enumerate() {
                    let child = unpacked[packed_index].take().unwrap();
                    if let Some(name) = names.remove(&position) {
                        tree.insert_branch(name, child).map_err(|source| {
                            UnpackTreeError::InvalidKey {
                                node: node_index,
                                source,
                            }
                        })?;
                    } else {
                        tree.push_branch(child);
                    }
                }
                tree
            }
        });
    }

    Ok(unpacked[0].take().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mixed_tree() -> DataTree<()> {
        DataTree::mapping([
            (
                "first",
                DataTree::sequence([
                    DataTree::Leaf(()),
                    DataTree::Leaf(()),
                    DataTree::mapping([
                        ("nested_first", DataTree::Leaf(())),
                        ("nested_second", DataTree::Leaf(())),
                    ])
                    .unwrap(),
                ]),
            ),
            ("second", DataTree::Leaf(())),
        ])
        .unwrap()
    }

    #[test]
    fn round_trip_mixed_tree() {
        let tree = mixed_tree();
        assert_eq!(unpack_tree(pack_tree(&tree)).unwrap(), tree);
    }

    #[test]
    fn round_trip_root_leaf() {
        let tree = DataTree::Leaf(());
        assert_eq!(unpack_tree(pack_tree(&tree)).unwrap(), tree);
    }

    #[test]
    fn round_trip_empty_and_nested_empty_branches() {
        let tree = DataTree::sequence([
            DataTree::new(),
            DataTree::mapping([("empty", DataTree::new())]).unwrap(),
        ]);
        assert_eq!(unpack_tree(pack_tree(&tree)).unwrap(), tree);
    }
}
