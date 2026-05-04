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

use hashbrown::HashMap;

/// A path entry used for tracking a path through a [`DataTree`]
///
/// Each entry can either be an index or a key. A slice of `PathEntry` are used to form
/// a traversal path through the [`DataTree`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PathEntry<'a> {
    Index(usize),
    Key(&'a str),
}

/// A struct representing a branch in a [`DataTree`].
///
/// Each branch contains a vec of [`DataTree`] that can also be assigned a
/// string key for accessing it. Typically you will not create these directly
/// but instead create them via the [`DataTree`] API.
#[derive(Debug, Clone)]
pub struct DataTreeBranch<T> {
    data: Vec<DataTree<T>>,
    keys: HashMap<String, usize>,
}

impl<T> Default for DataTreeBranch<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> DataTreeBranch<T> {
    /// Construct a new empty [`DataTreeBranch`]
    pub fn new() -> Self {
        DataTreeBranch {
            data: Vec::new(),
            keys: HashMap::new(),
        }
    }

    /// Construct a new empty [`DataTreeBranch`] with a set capacity
    pub fn with_capacity(capacity: usize) -> Self {
        DataTreeBranch {
            data: Vec::with_capacity(capacity),
            keys: HashMap::with_capacity(capacity),
        }
    }

    /// Take a path slice and return the entry at the given path
    ///
    /// This will return `None` if a path can not be found. This includes an
    /// invalid path, such as a path a leaf node in the middle.
    fn get_by_path(&self, path: &[PathEntry]) -> Option<&DataTree<T>> {
        let start = match path[0] {
            PathEntry::Index(idx) => Some(&self.data[idx]),
            PathEntry::Key(key) => self.keys.get(key).map(|idx| &self.data[*idx]),
        }?;
        match start {
            DataTree::Leaf(_) => {
                if path.len() > 1 {
                    // If there are more components in the path this is an invalid path
                    None
                } else {
                    Some(start)
                }
            }
            DataTree::Branch(inner_tree) => {
                if path.len() > 1 {
                    inner_tree.get_by_path(&path[1..])
                } else {
                    Some(start)
                }
            }
        }
    }

    /// Return an iterator over the leaves in the `DataTree`
    ///
    /// This method will return an iterator over all leaf nodes in the tree by traversing the tree
    /// in a DFS order.
    fn iter_path(&self) -> IterDataTree<'_, T> {
        IterDataTree {
            tree: None,
            branch: Some(self),
            index: 0,
            inner: None,
            inner_next: None,
            path: vec![],
        }
    }

    /// The number of items in this `DataTree`. This length is just the number of items in this
    /// local tree object and will not recurse through the tree to compute the total number of
    /// leaves. If you want to do that you should use [`DataTree::iter_leaves`].
    fn iter_leaves(&self) -> IterLeaves<'_, T> {
        IterLeaves {
            tree: None,
            branch: Some(self),
            index: 0,
            inner: None,
            inner_next: None,
        }
    }

    /// The number of [`DataTree`] in this branch.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if there are any [`DataTree`] in this branch.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The number of string keys set on this branch.
    pub fn num_keys(&self) -> usize {
        self.keys.len()
    }

    /// Check if the branch has any string keys set.
    pub fn has_keys(&self) -> bool {
        !self.keys.is_empty()
    }
}

impl<T> From<DataTree<T>> for DataTreeBranch<T> {
    fn from(input: DataTree<T>) -> Self {
        DataTreeBranch {
            data: vec![input],
            keys: HashMap::new(),
        }
    }
}

/// A generic tree that is addressable either by either indices or string keys
#[derive(Debug, Clone)]
pub enum DataTree<T> {
    Leaf(T),
    Branch(DataTreeBranch<T>),
}

impl<T> Default for DataTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> DataTree<T> {
    /// Consume the entry and return the leaf value otherwise panic
    pub fn unwrap_leaf(self) -> T {
        match self {
            Self::Leaf(data) => data,
            Self::Branch(_) => panic!("called TreeEntry::unwrap_leaf() on a `Tree` value"),
        }
    }

    /// Create a new empty data tree
    pub fn new() -> Self {
        DataTree::Branch(DataTreeBranch::new())
    }

    /// Create a new leaf data tree
    pub fn new_leaf(value: T) -> Self {
        DataTree::Leaf(value)
    }

    /// Create a new empty data tree with an underlying allocation of a given size.
    ///
    /// The specified capacity is the number of items of type T stored in the `DataTree`
    /// along with an associated `String` key for each element in the tree. This does not
    /// account for nesting in the allocation as each layer in the tree is a separate
    /// `DataTree` object.
    pub fn with_capacity(capacity: usize) -> Self {
        DataTree::Branch(DataTreeBranch::with_capacity(capacity))
    }

    /// The number of items in this `DataTree`. This length is just the number of items in this
    /// local tree object and will not recurse through the tree to compute the total number of
    /// leaves. If you want to do that you should use [`DataTree::iter_leaves`].
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::DataTree;
    /// let mut inner_tree = DataTree::with_capacity(5);
    /// inner_tree.insert_leaf("y", 10);
    /// inner_tree.insert_leaf("z", 11);
    /// inner_tree.insert_leaf("a", 12);
    /// inner_tree.insert_leaf("b", 13);
    /// inner_tree.push_leaf(15);
    ///
    /// let mut tree = DataTree::new();
    /// tree.insert_branch("x", inner_tree);
    /// assert_eq!(tree.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::Branch(branch) => branch.data.len(),
        }
    }

    /// Return whether this `DataTree` has an items in it.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Take a string key and return the entry at the given key.
    ///
    /// The "." character is reserved in keys and used to indicate a path
    /// through the graph.
    ///
    /// This will return `None` if the string key can not be found. This includes
    /// an invalid path, such as a path containing component or a leaf node in the
    /// middle. An empty string for the path will return `self`.
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::DataTree;
    /// let mut inner_tree = DataTree::new();
    /// inner_tree.insert_leaf("y", 10);
    /// let mut tree = DataTree::new();
    /// tree.insert_branch("x", inner_tree);
    /// let result = tree.get_by_str_key("x.y").unwrap().clone().unwrap_leaf();
    /// assert_eq!(result, 10);
    /// ```
    pub fn get_by_str_key(&self, key: &str) -> Option<&Self> {
        if key.is_empty() {
            return Some(self);
        }
        if key.contains(".") {
            let path: Vec<PathEntry> = key.split(".").map(PathEntry::Key).collect();
            self.get_by_path(&path)
        } else {
            match self {
                Self::Leaf(_) => None,
                Self::Branch(branch) => branch.keys.get(key).map(|value| &branch.data[*value]),
            }
        }
    }

    /// Take a path slice and return the entry at the given path
    ///
    /// This will return `None` if a path can not be found. This includes an
    /// invalid path, such as a path a leaf node in the middle. An empty path
    /// will also return `self`.
    pub fn get_by_path(&self, path: &[PathEntry]) -> Option<&Self> {
        if path.is_empty() {
            return Some(self);
        }
        let Self::Branch(branch) = self else {
            return None;
        };
        let start = match path[0] {
            PathEntry::Index(idx) => Some(&branch.data[idx]),
            PathEntry::Key(key) => branch.keys.get(key).map(|idx| &branch.data[*idx]),
        }?;
        match start {
            DataTree::Leaf(_) => {
                if path.len() > 1 {
                    // If there are more components in the path this is an invalid path
                    None
                } else {
                    Some(start)
                }
            }
            DataTree::Branch(inner_tree) => {
                if path.len() > 1 {
                    inner_tree.get_by_path(&path[1..])
                } else {
                    Some(start)
                }
            }
        }
    }

    /// Get an item from the `DataTree` by index.
    ///
    /// This will return `None` if the index is not valid.
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::DataTree;
    /// let mut inner_tree = DataTree::new();
    /// inner_tree.insert_leaf("y", 10);
    /// let mut tree = DataTree::new();
    /// tree.insert_branch("x", inner_tree);
    /// tree.push_leaf(124);
    /// let result = tree.get(1).unwrap().clone().unwrap_leaf();
    /// assert_eq!(result, 124);
    /// let subtree = tree.get(0).unwrap();
    /// let subtree_result = subtree.get(0).unwrap().clone().unwrap_leaf();
    /// assert_eq!(subtree_result, 10);
    /// ```
    pub fn get(&self, index: usize) -> Option<&DataTree<T>> {
        match self {
            Self::Leaf(_) => panic!("Called get() on a leaf node"),
            Self::Branch(branch) => branch.data.get(index),
        }
    }

    /// Insert a new leaf node with an associated string key
    ///
    /// If a key is provided that is already in the tree the new value will be associated with
    /// with the key and the old value will remain in the tree but without a string key.
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::DataTree;
    /// let mut tree = DataTree::new();
    /// tree.insert_leaf("y", 10);
    /// tree.insert_leaf("y", 1000);
    /// let result = tree.get_by_str_key("y").unwrap().clone().unwrap_leaf();
    /// assert_eq!(result, 1000);
    /// ```
    pub fn insert_leaf(&mut self, key: &str, value: T) {
        match self {
            Self::Leaf(_) => panic!("Called insert_leaf() on a leaf node"),
            Self::Branch(branch) => {
                branch.data.push(Self::Leaf(value));
                branch.keys.insert(key.to_string(), branch.data.len() - 1);
            }
        }
    }

    /// Add a new leaf to the tree
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::DataTree;
    /// let mut tree = DataTree::new();
    /// tree.push_leaf(10);
    /// tree.push_leaf(1000);
    /// assert_eq!(vec![10, 1000], tree.iter_leaves().copied().collect::<Vec<_>>());
    /// ```
    pub fn push_leaf(&mut self, value: T) {
        match self {
            Self::Leaf(_) => panic!("Called push_leaf() on a leaf_node"),
            Self::Branch(branch) => branch.data.push(DataTree::Leaf(value)),
        }
    }

    /// Add a subtree to the tree with an associated string key
    ///
    /// If a key is provided that is already in the tree the new value will be associated with
    /// with the key and the old value will remain in the tree but without a string key.
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::DataTree;
    /// let mut tree = DataTree::new();
    /// tree.insert_leaf("y", 10);
    /// let mut subtree = DataTree::with_capacity(2);
    /// subtree.push_leaf(123);
    /// subtree.push_leaf(456);
    /// tree.insert_branch("y", subtree);
    /// let result = tree.get_by_str_key("y").unwrap();
    /// let leaves: Vec<_> = result.iter_leaves().copied().collect();
    /// assert_eq!(leaves, vec![123, 456]);
    /// ```
    pub fn insert_branch(&mut self, key: &str, value: DataTree<T>) {
        match self {
            Self::Leaf(_) => panic!("Called insert_branch() on a leaf_node"),
            Self::Branch(branch) => {
                branch.data.push(value);
                branch.keys.insert(key.to_string(), branch.data.len() - 1);
            }
        }
    }

    /// Add a new branch to the tree
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::DataTree;
    /// let mut tree = DataTree::new();
    /// tree.push_leaf(10);
    /// let mut subtree = DataTree::with_capacity(2);
    /// subtree.push_leaf(123);
    /// subtree.push_leaf(456);
    /// tree.push_branch(subtree);
    /// assert_eq!(vec![10, 123, 456], tree.iter_leaves().copied().collect::<Vec<_>>());
    /// ```
    pub fn push_branch(&mut self, value: DataTree<T>) {
        match self {
            Self::Leaf(_) => panic!("Called insert_branch() on a leaf_node"),
            Self::Branch(branch) => branch.data.push(value),
        }
    }

    /// Return an iterator over the leaves in the `DataTree`
    ///
    /// This method will return an iterator over all leaf nodes in the tree by traversing the tree
    /// in a DFS order.
    ///
    /// # Example
    ///
    /// Traversing this tree:
    ///
    /// ```rust
    /// use qiskit_providers::DataTree;
    /// let mut subsubsubtree = DataTree::new();
    /// subsubsubtree.push_leaf(3);
    /// subsubsubtree.push_leaf(4);
    /// let mut subsubtree = DataTree::new();
    /// subsubtree.push_branch(subsubsubtree);
    /// subsubtree.insert_leaf("b", 5);
    /// let mut subsubtree_prime = DataTree::new();
    /// subsubtree_prime.push_leaf(7);
    /// let mut subtree = DataTree::new();
    /// subtree.insert_branch("c", subsubtree);
    /// subtree.insert_leaf("d", 6);
    /// subtree.push_branch(subsubtree_prime);
    /// let mut tree = DataTree::new();
    /// tree.insert_leaf("a", 0);
    /// tree.insert_branch("root", subtree);
    /// tree.insert_leaf("z", 26);
    /// let leaves: Vec<_> = tree.iter_leaves().copied().collect();
    /// let expected = vec![0, 3, 4, 5, 6, 7, 26];
    /// assert_eq!(leaves, expected);
    /// ```
    pub fn iter_leaves(&self) -> IterLeaves<'_, T> {
        IterLeaves {
            tree: Some(self),
            branch: None,
            index: 0,
            inner: None,
            inner_next: None,
        }
    }

    /// Return an iterator over the leaves in the `DataTree` that returns the path and leaf value.
    ///
    /// This method will return an iterator over all the leaf nodes in the tree in a DFS order.
    /// Unlike [`iter_leaves`] which just returns the value this will return an owned `Vec` of the
    /// path through the data tree to get to that value. This has allocation overhead for each leaf
    /// node in the tree and should only be used if you need the path along with the value.
    ///
    /// ```rust
    /// use qiskit_providers::{DataTree, PathEntry};
    /// let mut subsubsubtree = DataTree::new();
    /// subsubsubtree.push_leaf(3);
    /// subsubsubtree.push_leaf(4);
    /// let mut subsubtree = DataTree::new();
    /// subsubtree.push_branch(subsubsubtree);
    /// subsubtree.insert_leaf("b", 5);
    /// let mut subsubtree_prime = DataTree::new();
    /// subsubtree_prime.push_leaf(7);
    /// let mut subtree = DataTree::new();
    /// subtree.insert_branch("c", subsubtree);
    /// subtree.insert_leaf("d", 6);
    /// subtree.push_branch(subsubtree_prime);
    /// let mut tree = DataTree::new();
    /// tree.insert_leaf("a", 0);
    /// tree.insert_branch("root", subtree);
    /// tree.insert_leaf("z", 26);
    /// let result: Vec<_> = tree.iter_path().map(|(a, b)| (a, *b)).collect();
    /// let expected_paths: Vec<Vec<usize>> = vec![
    ///     vec![0],
    ///     vec![1, 0, 0, 0],
    ///     vec![1, 0, 0, 1],
    ///     vec![1, 0, 1],
    ///     vec![1, 1],
    ///     vec![1, 2, 0],
    ///     vec![2],
    /// ];
    /// let expected_vals = vec![0, 3, 4, 5, 6, 7, 26];
    /// let expected: Vec<_> = expected_paths
    ///     .into_iter()
    ///     .map(|x| x.into_iter().map(PathEntry::Index).collect::<Vec<_>>())
    ///     .zip(expected_vals)
    ///     .collect();
    /// assert_eq!(result, expected);
    /// ```
    pub fn iter_path(&self) -> IterDataTree<'_, T> {
        IterDataTree {
            tree: Some(self),
            branch: None,
            index: 0,
            inner: None,
            inner_next: None,
            path: Vec::new(),
        }
    }
}

pub struct IterDataTree<'a, T> {
    tree: Option<&'a DataTree<T>>,
    branch: Option<&'a DataTreeBranch<T>>,
    index: usize,
    inner: Option<Box<IterDataTree<'a, T>>>,
    inner_next: Option<(Vec<PathEntry<'a>>, &'a T)>,
    path: Vec<PathEntry<'a>>,
}

impl<'a, T> Iterator for IterDataTree<'a, T> {
    type Item = (Vec<PathEntry<'a>>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(tree) = self.tree {
            if let DataTree::Leaf(val) = tree {
                if self.index == 0 {
                    self.index += 1;
                    return Some((vec![], val));
                } else {
                    return None;
                }
            }
            let DataTree::Branch(branch) = tree else {
                unreachable!("Must be a branch variant");
            };
            if self.index >= branch.data.len() {
                return None;
            }
            let entry = &branch.data[self.index];
            match entry {
                DataTree::Leaf(val) => {
                    self.index += 1;
                    let mut out_path = self.path.clone();
                    out_path.push(PathEntry::Index(self.index - 1));
                    Some((out_path, val))
                }
                DataTree::Branch(sub_branch) => {
                    if let Some(ref mut inner) = self.inner {
                        if let Some(val) = inner.next() {
                            let (return_path, return_val) = self.inner_next.replace(val).unwrap();
                            Some((return_path, return_val))
                        } else {
                            self.inner = None;
                            self.index += 1;
                            let (return_path, return_val) = self.inner_next.take().unwrap();
                            self.inner_next = None;
                            Some((return_path, return_val))
                        }
                    } else {
                        let mut inner = sub_branch.iter_path();
                        let mut inner_path = self.path.clone();
                        inner_path.push(PathEntry::Index(self.index));
                        inner.path = inner_path;
                        self.inner = Some(Box::new(inner));
                        let (inner_path, val) = self.inner.as_mut().map(|x| x.next().unwrap())?;
                        self.inner_next = self.inner.as_mut().and_then(|x| x.next());
                        if self.inner_next.is_none() {
                            self.index += 1;
                            self.inner = None;
                            self.inner_next = None;
                        }
                        Some((inner_path, val))
                    }
                }
            }
        } else if let Some(subtree) = self.branch {
            if self.index >= subtree.data.len() {
                return None;
            }
            let entry = &subtree.data[self.index];
            match entry {
                DataTree::Leaf(val) => {
                    self.index += 1;
                    let mut out_path = self.path.clone();
                    out_path.push(PathEntry::Index(self.index - 1));
                    Some((out_path, val))
                }
                DataTree::Branch(subtree) => match self.inner {
                    Some(ref mut inner) => {
                        if let Some(val) = inner.next() {
                            let (return_path, return_val) = self.inner_next.replace(val).unwrap();
                            Some((return_path, return_val))
                        } else {
                            self.inner = None;
                            self.index += 1;
                            let (return_path, return_val) = self.inner_next.take().unwrap();
                            self.inner_next = None;
                            Some((return_path, return_val))
                        }
                    }
                    None => {
                        let mut inner = subtree.iter_path();
                        let mut inner_path = self.path.clone();
                        inner_path.push(PathEntry::Index(self.index));
                        inner.path = inner_path;
                        self.inner = Some(Box::new(inner));
                        let (inner_path, val) = self.inner.as_mut().map(|x| x.next().unwrap())?;
                        self.inner_next = self.inner.as_mut().and_then(|x| x.next());
                        if self.inner_next.is_none() {
                            self.index += 1;
                            self.inner = None;
                            self.inner_next = None;
                        }
                        Some((inner_path, val))
                    }
                },
            }
        } else {
            None
        }
    }
}

pub struct IterLeaves<'a, T> {
    tree: Option<&'a DataTree<T>>,
    branch: Option<&'a DataTreeBranch<T>>,
    index: usize,
    inner: Option<Box<IterLeaves<'a, T>>>,
    inner_next: Option<&'a T>,
}

impl<'a, T: std::fmt::Debug> Iterator for IterLeaves<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(tree) = self.tree {
            if let DataTree::Leaf(val) = tree {
                if self.index == 0 {
                    self.index += 1;
                    return Some(val);
                } else {
                    return None;
                }
            }
            let DataTree::Branch(branch) = tree else {
                unreachable!("Must be a branch variant");
            };
            if self.index >= branch.data.len() {
                return None;
            }
            let entry = &branch.data[self.index];
            match entry {
                DataTree::Leaf(val) => {
                    self.index += 1;
                    Some(val)
                }
                DataTree::Branch(sub_branch) => {
                    if let Some(ref mut inner) = self.inner {
                        if let Some(val) = inner.next() {
                            let return_val = self.inner_next.replace(val).unwrap();
                            Some(return_val)
                        } else {
                            self.inner = None;
                            self.index += 1;
                            let return_val = self.inner_next.take().unwrap();
                            self.inner_next = None;
                            Some(return_val)
                        }
                    } else {
                        let inner = sub_branch.iter_leaves();
                        self.inner = Some(Box::new(inner));
                        let val = self.inner.as_mut().map(|x| x.next().unwrap())?;
                        self.inner_next = self.inner.as_mut().and_then(|x| x.next());
                        if self.inner_next.is_none() {
                            self.index += 1;
                            self.inner = None;
                            self.inner_next = None;
                        }
                        Some(val)
                    }
                }
            }
        } else if let Some(subtree) = self.branch {
            if self.index >= subtree.data.len() {
                return None;
            }
            let entry = &subtree.data[self.index];
            match entry {
                DataTree::Leaf(val) => {
                    self.index += 1;
                    Some(val)
                }
                DataTree::Branch(subtree) => match self.inner {
                    Some(ref mut inner) => {
                        if let Some(val) = inner.next() {
                            let return_val = self.inner_next.replace(val).unwrap();
                            Some(return_val)
                        } else {
                            self.inner = None;
                            self.index += 1;
                            let return_val = self.inner_next.take().unwrap();
                            self.inner_next = None;
                            Some(return_val)
                        }
                    }
                    None => {
                        let inner = subtree.iter_leaves();
                        self.inner = Some(Box::new(inner));
                        let val = self.inner.as_mut().map(|x| x.next().unwrap())?;
                        self.inner_next = self.inner.as_mut().and_then(|x| x.next());
                        if self.inner_next.is_none() {
                            self.index += 1;
                            self.inner = None;
                            self.inner_next = None;
                        }
                        Some(val)
                    }
                },
            }
        } else {
            None
        }
    }
}

impl<T: PartialEq> PartialEq for DataTree<T> {
    fn eq(&self, other: &DataTree<T>) -> bool {
        match self {
            Self::Leaf(val) => {
                let Self::Leaf(other_val) = other else {
                    return false;
                };
                val == other_val
            }
            Self::Branch(branch) => {
                let Self::Branch(other) = other else {
                    return false;
                };
                branch.data == other.data && branch.keys == other.keys
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_data_leaf() {
        let mut tree = DataTree::new();
        tree.push_leaf(42);
        let result = tree.get(0).unwrap().clone();
        assert_eq!(result.unwrap_leaf(), 42);
    }

    #[test]
    fn test_flat_dict() {
        let mut tree = DataTree::with_capacity(3);
        tree.insert_leaf("a", 1);
        tree.insert_leaf("b", 2);
        let result = tree.get_by_str_key("b").unwrap().clone();
        assert_eq!(result.unwrap_leaf(), 2);
        let result = tree.get_by_str_key("a").unwrap().clone();
        assert_eq!(result.unwrap_leaf(), 1);
    }

    #[test]
    fn test_nested_dict() {
        let mut inner_tree = DataTree::new();
        inner_tree.insert_leaf("y", 10);
        let mut tree = DataTree::new();
        tree.insert_branch("x", inner_tree.clone());
        tree.insert_leaf("z", 100);
        assert_eq!(None, tree.get_by_str_key("z.y"));
        assert_eq!(Some(&inner_tree), tree.get_by_str_key("x"));
    }

    #[test]
    fn test_nested_dict_iter() {
        let mut inner_tree = DataTree::new();
        inner_tree.insert_leaf("y", 10);
        inner_tree.insert_leaf("yy", 1);
        let mut inner_inner_tree = DataTree::new();
        inner_inner_tree.push_leaf(2);
        inner_inner_tree.push_leaf(3);
        inner_inner_tree.push_leaf(4);
        inner_inner_tree.push_leaf(5);
        inner_tree.push_branch(inner_inner_tree);
        let mut tree = DataTree::new();
        tree.insert_branch("x", inner_tree.clone());
        tree.insert_leaf("z", 100);
        assert_eq!(
            vec![10, 1, 2, 3, 4, 5, 100],
            tree.iter_leaves().copied().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_nested_dict_iter_path() {
        let mut inner_tree = DataTree::new();
        inner_tree.insert_leaf("y", 10);
        inner_tree.insert_leaf("yy", 1);
        let mut inner_inner_tree = DataTree::new();
        inner_inner_tree.push_leaf(2);
        inner_inner_tree.push_leaf(3);
        inner_inner_tree.push_leaf(4);
        inner_inner_tree.push_leaf(5);
        inner_tree.push_branch(inner_inner_tree);
        let mut tree = DataTree::new();
        tree.insert_branch("x", inner_tree.clone());
        tree.insert_leaf("z", 100);
        let expected_paths = vec![
            vec![PathEntry::Index(0), PathEntry::Index(0)],
            vec![PathEntry::Index(0), PathEntry::Index(1)],
            vec![
                PathEntry::Index(0),
                PathEntry::Index(2),
                PathEntry::Index(0),
            ],
            vec![
                PathEntry::Index(0),
                PathEntry::Index(2),
                PathEntry::Index(1),
            ],
            vec![
                PathEntry::Index(0),
                PathEntry::Index(2),
                PathEntry::Index(2),
            ],
            vec![
                PathEntry::Index(0),
                PathEntry::Index(2),
                PathEntry::Index(3),
            ],
            vec![PathEntry::Index(1)],
        ];
        let expected_vals = vec![10, 1, 2, 3, 4, 5, 100];
        let expected = expected_paths
            .into_iter()
            .zip(expected_vals)
            .collect::<Vec<_>>();
        assert_eq!(
            expected,
            tree.iter_path().map(|(a, b)| (a, *b)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_get_by_str() {
        let mut inner_tree = DataTree::new();
        inner_tree.insert_leaf("y", 10);
        inner_tree.insert_leaf("yy", 1);
        let mut inner_inner_tree = DataTree::new();
        inner_inner_tree.push_leaf(2);
        inner_inner_tree.push_leaf(3);
        inner_inner_tree.insert_leaf("a", 4);
        inner_inner_tree.push_leaf(5);
        let inner_inner_tree_expected = inner_inner_tree.clone();
        inner_tree.insert_branch("yyy", inner_inner_tree);
        let mut tree = DataTree::new();
        tree.insert_branch("x", inner_tree.clone());
        tree.insert_leaf("z", 100);
        let result = tree.get_by_str_key("x.yyy.a");
        assert_eq!(result, Some(&DataTree::Leaf(4)));
        assert_eq!(tree.get_by_str_key("z"), Some(&DataTree::Leaf(100)));
        assert_eq!(
            tree.get_by_str_key("x.yyy"),
            Some(&inner_inner_tree_expected),
        );
        assert_eq!(tree.get_by_str_key("x.yy"), Some(&DataTree::Leaf(1)));
    }

    #[test]
    fn test_get_by_str_no_match() {
        let mut inner_tree = DataTree::new();
        inner_tree.insert_leaf("y", 10);
        inner_tree.insert_leaf("yy", 1);
        let mut inner_inner_tree = DataTree::new();
        inner_inner_tree.push_leaf(2);
        inner_inner_tree.push_leaf(3);
        inner_inner_tree.insert_leaf("a", 4);
        inner_inner_tree.push_leaf(5);
        inner_tree.insert_branch("yyy", inner_inner_tree);
        let mut tree = DataTree::new();
        tree.insert_branch("x", inner_tree.clone());
        tree.insert_leaf("z", 100);
        assert_eq!(None, tree.get_by_str_key("a"));
        assert_eq!(None, tree.get_by_str_key("x.yyyy"));
        assert_eq!(None, tree.get_by_str_key("x.yy.a"));
        assert_eq!(None, tree.get_by_str_key("🎩"));
        assert_eq!(None, tree.get_by_str_key("z.yyy.a"));
    }
}
