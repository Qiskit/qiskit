// Copyright 2019 Jared Samet
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// struct MultiAxisIterator<'a, A> {
//     carrying: bool,
//     ndim: usize,
//     // axes: Vec<usize>,
//     renumbered_axes: Vec<usize>,
//     shape: Vec<usize>,
//     positions: Vec<usize>,
//     underlying: &'a ArrayViewD<'a, A>,
//     // subviews: Vec<ArrayViewD<'a, A>>,
// }
//
// impl<'a, A> MultiAxisIterator<'a, A> {
//     fn new(base: &'a ArrayViewD<'a, A>, axes: &[usize]) -> MultiAxisIterator<'a, A> {
//         let ndim = axes.len();
//         // let axes: Vec<usize> = axes.to_vec();
//         let renumbered_axes: Vec<usize> = axes
//             .iter()
//             .enumerate()
//             .map(|(i, &v)| v - axes[0..i].iter().filter(|&&x| x < v).count())
//             .collect();
//         let shape: Vec<usize> = axes
//             .iter()
//             .map(|&x| base.shape().get(x).unwrap())
//             .cloned()
//             .collect();
//         let positions = vec![0; shape.len()];
//
//         // let mut subviews = Vec::new();
//         // let mut axis_iters = Vec::new();
//         //
//         // for (ax_num, &ax) in axes.iter().enumerate() {
//         //     let mut subview = base.view();
//         //     for i in 0..ax_num {
//         //         subview = subview.index_axis_move(Axis(0), 0);
//         //     }
//         //     subviews.push(subview);
//         // }
//
//         MultiAxisIterator {
//             underlying: base,
//             carrying: false,
//             ndim,
//             // axes,
//             renumbered_axes,
//             shape,
//             positions,
//             // subviews,
//         }
//     }
// }
//
// impl<'a, A> Iterator for MultiAxisIterator<'a, A> {
//     type Item = ArrayViewD<'a, A>;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         if !self.carrying {
//             let mut view = self.underlying.view();
//             for (&ax, &pos) in self.renumbered_axes.iter().zip(&self.positions) {
//                 view = view.index_axis_move(Axis(ax), pos);
//             }
//             self.carrying = true;
//             for i in 0..self.ndim {
//                 let axis = self.ndim - i - 1;
//                 if self.positions[axis] == self.shape[axis] - 1 {
//                     self.positions[axis] = 0;
//                 } else {
//                     self.positions[axis] += 1;
//                     self.carrying = false;
//                     break;
//                 }
//             }
//             Some(view)
//         } else {
//             None
//         }
//     }
// }
//
