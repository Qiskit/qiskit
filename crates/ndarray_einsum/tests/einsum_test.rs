use ndarray::prelude::*;
use ndarray::Data;
use ndarray_einsum::*;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
const TOL: f64 = 1e-10;

trait AllClose {
    fn my_all_close<S2, E>(&self, rhs: &ArrayBase<S2, E>, tol: f64) -> bool
    where
        S2: Data<Elem = f64>,
        E: Dimension;
}

impl<S, D> AllClose for ArrayBase<S, D>
where
    D: Dimension,
    S: Data<Elem = f64>,
{
    fn my_all_close<S2, E>(&self, rhs: &ArrayBase<S2, E>, tol: f64) -> bool
    where
        S2: Data<Elem = f64>,
        E: Dimension,
    {
        let self_dyn_view = self.view().into_dyn();
        let rhs_dyn_view = rhs.view().into_dyn();

        self_dyn_view.abs_diff_eq(&rhs_dyn_view, tol)
    }
}

#[test]
fn tp1() {
    let contraction = Contraction::new("i->").unwrap();
    assert_eq!(contraction.operand_indices, &[vec!['i']]);
    assert_eq!(contraction.output_indices.len(), 0);
    assert_eq!(contraction.summation_indices, &['i']);
}

#[test]
fn tp2() {
    let contraction = Contraction::new("ij->").unwrap();
    assert_eq!(contraction.operand_indices, &[vec!['i', 'j']]);
    assert_eq!(contraction.output_indices.len(), 0);
    assert_eq!(contraction.summation_indices, &['i', 'j']);
}

#[test]
fn tp3() {
    let contraction = Contraction::new("i->i").unwrap();
    assert_eq!(contraction.operand_indices, &[vec!['i']]);
    assert_eq!(contraction.output_indices, &['i']);
    assert_eq!(contraction.summation_indices.len(), 0);
}

#[test]
fn tp4() {
    let contraction = Contraction::new("ij,ij->ij").unwrap();
    assert_eq!(
        contraction.operand_indices,
        &[vec!['i', 'j'], vec!['i', 'j']]
    );
    assert_eq!(contraction.output_indices, &['i', 'j']);
    assert_eq!(contraction.summation_indices.len(), 0);
}

#[test]
fn tp5() {
    let contraction = Contraction::new("ij,ij->").unwrap();
    assert_eq!(
        contraction.operand_indices,
        &[vec!['i', 'j'], vec!['i', 'j']]
    );
    assert_eq!(contraction.output_indices.len(), 0);
    assert_eq!(contraction.summation_indices, &['i', 'j']);
}

#[test]
fn tp6() {
    let contraction = Contraction::new("ij,kl->").unwrap();
    assert_eq!(
        contraction.operand_indices,
        &[vec!['i', 'j'], vec!['k', 'l']]
    );
    assert_eq!(contraction.output_indices.len(), 0);
    assert_eq!(contraction.summation_indices, &['i', 'j', 'k', 'l']);
}

#[test]
fn tp7() {
    let contraction = Contraction::new("ij,jk->ik").unwrap();
    assert_eq!(
        contraction.operand_indices,
        &[vec!['i', 'j'], vec!['j', 'k']]
    );
    assert_eq!(contraction.output_indices, &['i', 'k']);
    assert_eq!(contraction.summation_indices, &['j']);
}

#[test]
fn tp8() {
    let contraction = Contraction::new("ijk,jkl,klm->im").unwrap();
    assert_eq!(
        contraction.operand_indices,
        &[
            vec!['i', 'j', 'k'],
            vec!['j', 'k', 'l'],
            vec!['k', 'l', 'm']
        ]
    );
    assert_eq!(contraction.output_indices, &['i', 'm']);
    assert_eq!(contraction.summation_indices, &['j', 'k', 'l']);
}

#[test]
fn tp9() {
    let contraction = Contraction::new("ij,jk->ki").unwrap();
    assert_eq!(
        contraction.operand_indices,
        &[vec!['i', 'j'], vec!['j', 'k']]
    );
    assert_eq!(contraction.output_indices, &['k', 'i']);
    assert_eq!(contraction.summation_indices, &['j']);
}

#[test]
fn tp10() {
    let contraction = Contraction::new("ij,ja->ai").unwrap();
    assert_eq!(
        contraction.operand_indices,
        &[vec!['i', 'j'], vec!['j', 'a']]
    );
    assert_eq!(contraction.output_indices, &['a', 'i']);
    assert_eq!(contraction.summation_indices, &['j']);
}

#[test]
fn tp11() {
    let contraction = Contraction::new("ii->i").unwrap();
    assert_eq!(contraction.operand_indices, &[vec!['i', 'i']]);
    assert_eq!(contraction.output_indices, &['i']);
    assert_eq!(contraction.summation_indices.len(), 0);
}

#[test]
fn tp12() {
    let contraction = Contraction::new("ij,k").unwrap();
    assert_eq!(contraction.operand_indices, &[vec!['i', 'j'], vec!['k']]);
    assert_eq!(contraction.output_indices, &['i', 'j', 'k']);
    assert_eq!(contraction.summation_indices.len(), 0);
}

#[test]
fn tp13() {
    let contraction = Contraction::new("i").unwrap();
    assert_eq!(contraction.operand_indices, &[vec!['i']]);
    assert_eq!(contraction.output_indices, &['i']);
    assert_eq!(contraction.summation_indices.len(), 0);
}

#[test]
fn tp14() {
    let contraction = Contraction::new("ii").unwrap();
    assert_eq!(contraction.operand_indices, &[vec!['i', 'i']]);
    assert_eq!(contraction.output_indices.len(), 0);
    assert_eq!(contraction.summation_indices, &['i']);
}

#[test]
fn tp15() {
    let contraction = Contraction::new("ijj").unwrap();
    assert_eq!(contraction.operand_indices, &[vec!['i', 'j', 'j']]);
    assert_eq!(contraction.output_indices, &['i']);
    assert_eq!(contraction.summation_indices, &['j']);
}

#[test]
fn tp16() {
    let contraction = Contraction::new("i,j,klm,nop").unwrap();
    assert_eq!(
        contraction.operand_indices,
        &[
            vec!['i'],
            vec!['j'],
            vec!['k', 'l', 'm'],
            vec!['n', 'o', 'p']
        ]
    );
    assert_eq!(
        contraction.output_indices,
        &['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',]
    );
    assert_eq!(contraction.summation_indices.len(), 0);
}

#[test]
fn tp17() {
    let contraction = Contraction::new("ij,jk").unwrap();
    assert_eq!(
        contraction.operand_indices,
        &[vec!['i', 'j'], vec!['j', 'k']]
    );
    assert_eq!(contraction.output_indices, &['i', 'k']);
    assert_eq!(contraction.summation_indices, &['j']);
}

#[test]
fn tp18() {
    let contraction = Contraction::new("ij,ja").unwrap();
    assert_eq!(
        contraction.operand_indices,
        &[vec!['i', 'j'], vec!['j', 'a']]
    );
    assert_eq!(contraction.output_indices, &['a', 'i']);
    assert_eq!(contraction.summation_indices, &['j']);
}

#[test]
fn bad_parses_1() {
    for s in vec!["->i", "i,", "->", "i,,,j->k"].iter() {
        let contraction_result = Contraction::new(s);
        assert!(contraction_result.is_err());
    }
}

#[test]
fn bad_outputs_1() {
    for s in vec!["i,j,k,l,m->p", "i,j->ijj"].iter() {
        let contraction_result = Contraction::new(s);
        assert!(contraction_result.is_err());
    }
}

fn rand_array<Sh, D: Dimension>(shape: Sh) -> ArrayBase<ndarray::OwnedRepr<f64>, D>
where
    Sh: ShapeBuilder<Dim = D>,
{
    Array::random(shape, Uniform::new(-5., 5.))
}

#[test]
fn it_multiplies_two_matrices() {
    let a = rand_array((3, 4));
    let b = rand_array((4, 5));

    let correct_answer = a.dot(&b).into_dyn();
    let lib_output = einsum("ij,jk->ik", &[&a, &b]).unwrap();

    assert!(correct_answer.my_all_close(&lib_output, TOL));
}

#[test]
fn it_computes_the_trace() {
    let square_matrix = rand_array((5, 5));

    let diag: Vec<_> = (0..square_matrix.shape()[0])
        .map(|i| square_matrix[[i, i]])
        .collect();
    let correct_answer = arr1(&diag).into_dyn();

    let lib_output = einsum("ii->i", &[&square_matrix]).unwrap();

    assert!(correct_answer.my_all_close(&lib_output, TOL));
}

#[test]
fn it_transposes_a_matrix() {
    let rect_matrix = rand_array((2, 5));

    let correct_answer = rect_matrix.t();

    let tr1 = einsum("ji", &[&rect_matrix]).unwrap();
    let tr2 = einsum("ij->ji", &[&rect_matrix]).unwrap();
    let tr3 = einsum("ji->ij", &[&rect_matrix]).unwrap();

    assert!(correct_answer.my_all_close(&tr1, TOL));
    assert!(correct_answer.my_all_close(&tr2, TOL));
    assert!(correct_answer.my_all_close(&tr3, TOL));
}

#[test]
fn it_clones_a_matrix() {
    let rect_matrix = rand_array((2, 5));

    let correct_answer = rect_matrix.view().into_dyn();

    let cloned = einsum("ij->ij", &[&rect_matrix]).unwrap();
    assert!(correct_answer.my_all_close(&cloned, TOL));
}

#[test]
fn it_collapses_a_singleton_with_noncontiguous_strides() {
    // [?] -> ii -> <empty>
    let cube = rand_array((4, 4, 4));

    let data_slice = cube.as_slice_memory_order().unwrap();
    let output_shape = vec![4, 4];
    let strides = vec![17, 4];
    let square =
        ArrayView::from_shape(IxDyn(&output_shape).strides(IxDyn(&strides)), &data_slice).unwrap();

    let correct_answer = arr0(square.diag().sum());

    let ep = einsum_path("ii->", &[&square], OptimizationMethod::Naive).unwrap();
    let collapsed = ep.contract_operands(&[&square]);

    assert!(correct_answer.into_dyn().my_all_close(&collapsed, TOL));
}

#[test]
fn it_collapses_a_singleton_without_repeats() {
    // ijkl->lij
    let s = rand_array((4, 2, 3, 5));

    let mut correct_answer: Array3<f64> = Array::zeros((5, 4, 2));
    for l in 0..5 {
        for i in 0..4 {
            for j in 0..2 {
                let mut r = 0.;
                for k in 0..3 {
                    r += s[[i, j, k, l]];
                }
                correct_answer[[l, i, j]] = r;
            }
        }
    }

    let ep = einsum_path("ijkl->lij", &[&s], OptimizationMethod::Naive).unwrap();
    let collapsed = ep.contract_operands(&[&s]);

    assert!(correct_answer.into_dyn().my_all_close(&collapsed, TOL));
}

#[test]
fn it_diagonalizes_a_singleton() {
    // jkiii
    let s = rand_array((2, 3, 4, 4, 4));

    let mut correct_answer: Array3<f64> = Array::zeros((2, 4, 3));
    for i in 0..4 {
        for j in 0..2 {
            for k in 0..3 {
                correct_answer[[j, i, k]] = s[[j, k, i, i, i]];
            }
        }
    }

    let ep = einsum_path("jkiii->jik", &[&s], OptimizationMethod::Naive).unwrap();
    let collapsed = ep.contract_operands(&[&s]);

    assert!(correct_answer.into_dyn().my_all_close(&collapsed, TOL));
}

#[test]
fn it_collapses_a_singleton_with_multiple_repeats() {
    // kjiji->ijk
    let s = rand_array((4, 3, 2, 3, 2));

    let mut correct_answer: Array3<f64> = Array::zeros((2, 3, 4));
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                correct_answer[[i, j, k]] = s[[k, j, i, j, i]];
            }
        }
    }

    let ep = einsum_path("kjiji->ijk", &[&s], OptimizationMethod::Naive).unwrap();
    let collapsed = ep.contract_operands(&[&s]);

    assert!(correct_answer.into_dyn().my_all_close(&collapsed, TOL));
}

#[test]
fn it_collapses_a_singleton_with_a_repeat_that_gets_summed() {
    // iij->j
    let s = rand_array((2, 2, 3));

    let mut correct_answer: Array1<f64> = Array::zeros((3,));
    for j in 0..3 {
        let mut res = 0.;
        for i in 0..2 {
            res += s[[i, i, j]];
        }
        correct_answer[j] = res;
    }

    let ep = einsum_path("iij->j", &[&s], OptimizationMethod::Naive).unwrap();
    let collapsed = ep.contract_operands(&[&s]);
    assert!(correct_answer.into_dyn().my_all_close(&collapsed, TOL));
}

#[test]
fn it_collapses_a_singleton_with_multiple_repeats_that_get_summed() {
    // iijkk->j
    let s = rand_array((2, 2, 3, 4, 4));

    let mut correct_answer: Array1<f64> = Array::zeros((3,));
    for j in 0..3 {
        let mut res = 0.;
        for i in 0..2 {
            for k in 0..4 {
                res += s[[i, i, j, k, k]];
            }
        }
        correct_answer[j] = res;
    }

    let ep = einsum_path("iijkk->j", &[&s], OptimizationMethod::Naive).unwrap();
    let collapsed = ep.contract_operands(&[&s]);
    assert!(correct_answer.into_dyn().my_all_close(&collapsed, TOL));
}

#[test]
fn it_sums_a_singleton() {
    // ijk->
    let s = rand_array((2, 3, 4));

    let correct_answer = arr0(s.sum());

    let ep = einsum_path("ijk->", &[&s], OptimizationMethod::Naive).unwrap();
    let collapsed = ep.contract_operands(&[&s]);
    assert!(correct_answer.into_dyn().my_all_close(&collapsed, TOL));
}

#[test]
fn it_collapses_a_singleton_with_multiple_sums() {
    // ijk->k
    let s = rand_array((2, 3, 4));

    let mut correct_answer: Array1<f64> = Array::zeros((4,));
    for k in 0..4 {
        let mut res = 0.;
        for i in 0..2 {
            for j in 0..3 {
                res += s[[i, j, k]];
            }
        }
        correct_answer[k] = res;
    }

    let ep = einsum_path("ijk->k", &[&s], OptimizationMethod::Naive).unwrap();
    let collapsed = ep.contract_operands(&[&s]);
    assert!(correct_answer.into_dyn().my_all_close(&collapsed, TOL));
}

#[test]
fn it_collapses_a_singleton() {
    // iijkl->lij
    let s = rand_array((4, 4, 2, 3, 5));

    let mut correct_answer: Array3<f64> = Array::zeros((5, 4, 2));
    for l in 0..5 {
        for i in 0..4 {
            for j in 0..2 {
                let mut r = 0.;
                for k in 0..3 {
                    r += s[[i, i, j, k, l]];
                }
                correct_answer[[l, i, j]] = r;
            }
        }
    }

    let ep = einsum_path("iijkl->lij", &[&s], OptimizationMethod::Naive).unwrap();
    let collapsed = ep.contract_operands(&[&s]);

    assert!(correct_answer.into_dyn().my_all_close(&collapsed, TOL));
}

#[test]
fn tensordot_handles_degenerate_lhs() {
    let lhs = arr0(1.);
    let rhs = rand_array((2, 3));

    let dotted = tensordot(&lhs, &rhs, &[], &[]);
    assert!(rhs.into_dyn().my_all_close(&dotted, TOL));
}

#[test]
fn tensordot_handles_degenerate_rhs() {
    let lhs = rand_array((2, 3));
    let rhs = arr0(1.);

    let dotted = tensordot(&lhs, &rhs, &[], &[]);
    assert!(lhs.into_dyn().my_all_close(&dotted, TOL));
}

#[test]
fn tensordot_handles_degenerate_both() {
    let lhs = arr0(1.);
    let rhs = arr0(1.);

    let dotted = tensordot(&lhs, &rhs, &[], &[]);
    assert!(dotted[[]] == 1.);
}

#[test]
fn deduped_handles_dot_product() {
    let lhs = rand_array((3,));
    let rhs = rand_array((3,));

    let correct_answer = arr0(lhs.dot(&rhs));
    let ep = einsum_path("i,i->", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_hadamard_product() {
    let lhs = rand_array((3,));
    let rhs = rand_array((3,));

    let correct_answer = (&lhs * &rhs).into_dyn();
    let ep = einsum_path("i,i->i", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_outer_product_vec_vec() {
    let lhs = rand_array((3,));
    let rhs = rand_array((4,));

    let mut correct_answer: Array2<f64> = Array::zeros((3, 4));
    for i in 0..3 {
        for j in 0..4 {
            correct_answer[[i, j]] = lhs[[i]] * rhs[[j]];
        }
    }

    let ep = einsum_path("i,j->ij", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_matrix_vector_1() {
    let lhs = rand_array((3, 4));
    let rhs = rand_array((4,));

    let mut correct_answer: Array1<f64> = Array::zeros((3,));
    for i in 0..3 {
        let mut res = 0.;
        for j in 0..4 {
            res += lhs[[i, j]] * rhs[[j]];
        }
        correct_answer[i] = res;
    }

    let ep = einsum_path("ij,j->i", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
    assert!(correct_answer.my_all_close(&(lhs.dot(&rhs)), TOL));
}

#[test]
fn deduped_handles_matrix_vector_2() {
    let lhs = rand_array((3,));
    let rhs = rand_array((3, 4));

    let mut correct_answer: Array1<f64> = Array::zeros((4,));
    for j in 0..4 {
        let mut res = 0.;
        for i in 0..3 {
            res += lhs[[i]] * rhs[[i, j]];
        }
        correct_answer[j] = res;
    }

    let ep = einsum_path("i,ij->j", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
    assert!(correct_answer.my_all_close(&(lhs.dot(&rhs)), TOL));
}

#[test]
fn deduped_handles_matrix_vector_3() {
    let lhs = rand_array((3, 4));
    let rhs = rand_array((3,));

    let mut correct_answer: Array1<f64> = Array::zeros((4,));
    for j in 0..4 {
        let mut res = 0.;
        for i in 0..3 {
            res += lhs[[i, j]] * rhs[[i]];
        }
        correct_answer[j] = res;
    }

    let ep = einsum_path("ij,i->j", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
    assert!(correct_answer.my_all_close(&(rhs.dot(&lhs)), TOL));
}

#[test]
fn deduped_handles_matrix_vector_4() {
    let lhs = rand_array((4,));
    let rhs = rand_array((3, 4));

    let mut correct_answer: Array1<f64> = Array::zeros((3,));
    for i in 0..3 {
        let mut res = 0.;
        for j in 0..4 {
            res += lhs[[j]] * rhs[[i, j]];
        }
        correct_answer[i] = res;
    }

    let ep = einsum_path("j,ij->i", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
    assert!(correct_answer.my_all_close(&(rhs.dot(&lhs)), TOL));
}

#[test]
fn deduped_handles_stacked_scalar_vector_product_1() {
    let lhs = rand_array((3, 4));
    let rhs = rand_array((3,));

    let mut correct_answer: Array2<f64> = Array::zeros((3, 4));
    for i in 0..3 {
        for j in 0..4 {
            correct_answer[[i, j]] = lhs[[i, j]] * rhs[[i]];
        }
    }

    let ep = einsum_path("ij,i->ij", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_stacked_scalar_vector_product_2() {
    let lhs = rand_array((3, 4));
    let rhs = rand_array((3,));

    let mut correct_answer: Array2<f64> = Array::zeros((4, 3));
    for i in 0..3 {
        for j in 0..4 {
            correct_answer[[j, i]] = lhs[[i, j]] * rhs[[i]];
        }
    }

    let ep = einsum_path("ij,i->ji", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_stacked_scalar_vector_product_3() {
    let lhs = rand_array((3,));
    let rhs = rand_array((4, 3));

    let mut correct_answer: Array2<f64> = Array::zeros((3, 4));
    for i in 0..3 {
        for j in 0..4 {
            correct_answer[[i, j]] = lhs[[i]] * rhs[[j, i]];
        }
    }

    let ep = einsum_path("i,ji->ij", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_summed_hadamard_product_aka_stacked_1d_tensordot_1() {
    let lhs = rand_array((3, 4));
    let rhs = rand_array((3, 4));

    let correct_answer = (&lhs * &rhs).sum_axis(Axis(1));

    let ep = einsum_path("ij,ij->i", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_summed_hadamard_product_aka_stacked_1d_tensordot_2() {
    let lhs = rand_array((3, 4));
    let rhs = rand_array((4, 3));

    let correct_answer = (&lhs * &rhs.t()).sum_axis(Axis(1));

    let ep = einsum_path("ij,ji->i", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_summed_hadamard_product_aka_stacked_1d_tensordot_3() {
    let lhs = rand_array((3, 4));
    let rhs = rand_array((4, 3));

    let correct_answer = (&lhs * &rhs.t()).sum_axis(Axis(0));

    let ep = einsum_path("ji,ij->i", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_summed_hadamard_product_multiple_stacked_axes() {
    let lhs = rand_array((2, 3, 4, 5));
    let rhs = rand_array((3, 5, 4, 2));

    let mut correct_answer: Array3<f64> = Array::zeros((4, 3, 2));
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                for l in 0..5 {
                    correct_answer[[k, j, i]] += lhs[[i, j, k, l]] * rhs[[j, l, k, i]];
                }
            }
        }
    }

    let ep = einsum_path("ijkl,jlki->kji", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_2d_hadamard_product() {
    let lhs = rand_array((2, 3));
    let rhs = rand_array((2, 3));

    let correct_answer = (&lhs * &rhs).into_dyn();

    let ep = einsum_path("ij,ij->ij", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_2d_hadamard_product_2() {
    let lhs = rand_array((2, 3));
    let rhs = rand_array((2, 3));

    let correct_answer_t = (&lhs * &rhs).into_dyn();
    let correct_answer = correct_answer_t.t();

    let ep = einsum_path("ij,ij->ji", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_2d_hadamard_product_3() {
    let lhs = rand_array((3, 2));
    let rhs = rand_array((2, 3));

    let correct_answer = (&lhs.t() * &rhs).into_dyn();

    let ep = einsum_path("ji,ij->ij", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_double_dot_product_mat_mat() {
    let lhs = rand_array((2, 3));
    let rhs = rand_array((2, 3));

    let correct_answer = arr0((&lhs * &rhs).sum());
    let ep = einsum_path("ij,ij->", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_outer_product_vect_mat_1() {
    let lhs = rand_array((2, 3));
    let rhs = rand_array((4,));

    let mut correct_answer: Array3<f64> = Array::zeros((4, 2, 3));
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                correct_answer[[k, i, j]] = lhs[[i, j]] * rhs[[k]];
            }
        }
    }
    let ep = einsum_path("ij,k->kij", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_outer_product_vect_mat_2() {
    let lhs = rand_array((4,));
    let rhs = rand_array((2, 3));

    let mut correct_answer: Array3<f64> = Array::zeros((4, 2, 3));
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                correct_answer[[k, i, j]] = rhs[[i, j]] * lhs[[k]];
            }
        }
    }
    let ep = einsum_path("k,ij->kij", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_outer_product_mat_mat() {
    let lhs = rand_array((5, 4));
    let rhs = rand_array((2, 3));

    let mut correct_answer: Array4<f64> = Array::zeros((4, 2, 5, 3));
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                for l in 0..5 {
                    correct_answer[[k, i, l, j]] = lhs[[l, k]] * rhs[[i, j]];
                }
            }
        }
    }
    let ep = einsum_path("lk,ij->kilj", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_matrix_product_1() {
    let lhs = rand_array((2, 3));
    let rhs = rand_array((3, 4));

    let correct_answer = lhs.dot(&rhs);
    let ep = einsum_path("ij,jk", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn it_handles_stacked_matrix_product_1() {
    let lhs = rand_array((4, 2, 5));
    let rhs = rand_array((4, 5, 3));

    let mut correct_answer: Array3<f64> = Array::zeros((4, 2, 3));
    for n in 0..4 {
        for i in 0..2 {
            for j in 0..5 {
                for k in 0..3 {
                    correct_answer[[n, i, k]] += lhs[[n, i, j]] * rhs[[n, j, k]];
                }
            }
        }
    }
    let ep = einsum_path("nij,njk->nik", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn it_handles_stacked_matrix_product_2() {
    let lhs = rand_array((4, 5, 2));
    let rhs = rand_array((4, 5, 3));

    let mut correct_answer: Array3<f64> = Array::zeros((4, 2, 3));
    for n in 0..4 {
        for i in 0..2 {
            for j in 0..5 {
                for k in 0..3 {
                    correct_answer[[n, i, k]] += lhs[[n, j, i]] * rhs[[n, j, k]];
                }
            }
        }
    }
    let ep = einsum_path("nji,njk->nik", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn it_handles_stacked_matrix_product_3() {
    let lhs = rand_array((5, 2, 4));
    let rhs = rand_array((4, 5, 3));

    let mut correct_answer: Array3<f64> = Array::zeros((4, 2, 3));
    for n in 0..4 {
        for i in 0..2 {
            for j in 0..5 {
                for k in 0..3 {
                    correct_answer[[n, i, k]] += lhs[[j, i, n]] * rhs[[n, j, k]];
                }
            }
        }
    }
    let ep = einsum_path("jin,njk->nik", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_matrix_product_2() {
    let lhs = rand_array((2, 3));
    let rhs = rand_array((4, 3));

    let correct_answer = lhs.dot(&rhs.t());
    let ep = einsum_path("ij,kj", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn deduped_handles_stacked_outer_product_1() {
    let lhs = rand_array((2, 3));
    let rhs = rand_array((4, 2));

    let mut correct_answer: Array3<f64> = Array::zeros((2, 3, 4));
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                correct_answer[[i, j, k]] = lhs[[i, j]] * rhs[[k, i]];
            }
        }
    }

    let ep = einsum_path("ij,ki->ijk", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn diagonals_product() {
    let lhs = rand_array((2, 2));
    let rhs = rand_array((2, 2));

    let mut correct_answer: Array1<f64> = Array::zeros((2,));
    for i in 0..2 {
        correct_answer[[i]] = lhs[[i, i]] * rhs[[i, i]];
    }

    let ep = einsum_path("ii,ii->i", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn diagonals_product_lhs() {
    let lhs = rand_array((2, 2));
    let rhs = rand_array((2,));

    let mut correct_answer: Array1<f64> = Array::zeros((2,));
    for i in 0..2 {
        correct_answer[[i]] = lhs[[i, i]] * rhs[[i]];
    }

    let ep = einsum_path("ii,i->i", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn diagonals_product_rhs() {
    let lhs = rand_array((2,));
    let rhs = rand_array((2, 2));

    let mut correct_answer: Array1<f64> = Array::zeros((2,));
    for i in 0..2 {
        correct_answer[[i]] = lhs[[i]] * rhs[[i, i]];
    }

    let ep = einsum_path("i,ii->i", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn presum_lhs() {
    let lhs = rand_array((2, 3));
    let rhs = rand_array((2, 2));

    let mut correct_answer: Array1<f64> = Array::zeros((2,));
    for i in 0..2 {
        for j in 0..3 {
            correct_answer[[i]] += lhs[[i, j]] * rhs[[i, i]];
        }
    }

    let ep = einsum_path("ij,ii->i", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn it_tolerates_permuted_axes() {
    let lhs = rand_array((2, 3, 4));
    let mut rhs = rand_array((5, 3, 4));
    rhs = rhs.permuted_axes([1, 2, 0]);

    let mut correct_answer: Array2<f64> = Array::zeros((2, 5));
    for i in 0..2 {
        for l in 0..5 {
            for j in 0..3 {
                for k in 0..4 {
                    correct_answer[[i, l]] += lhs[[i, j, k]] * rhs[[j, k, l]];
                }
            }
        }
    }

    let ep = einsum_path("ijk,jkl->il", &[&lhs, &rhs], OptimizationMethod::Naive).unwrap();
    let dotted = ep.contract_operands(&[&lhs, &rhs]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn it_contracts_three_matrices() {
    let op1 = rand_array((2, 3));
    let op2 = rand_array((3, 4));
    let op3 = rand_array((4, 5));

    let mut correct_answer: Array2<f64> = Array::zeros((2, 5));
    for i in 0..2 {
        for l in 0..5 {
            for j in 0..3 {
                for k in 0..4 {
                    correct_answer[[i, l]] += op1[[i, j]] * op2[[j, k]] * op3[[k, l]];
                }
            }
        }
    }

    let ep = einsum_path(
        "ij,jk,kl->il",
        &[&op1, &op2, &op3],
        OptimizationMethod::Naive,
    )
    .unwrap();
    let dotted = ep.contract_operands(&[&op1, &op2, &op3]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn it_contracts_three_matrices_with_repeats_1() {
    let op1 = rand_array((2, 3));
    let op2 = rand_array((3, 6, 6, 7, 4));
    let op3 = rand_array((4, 5));

    let mut correct_answer: Array2<f64> = Array::zeros((2, 5));
    for i in 0..2 {
        for l in 0..5 {
            for j in 0..3 {
                for k in 0..4 {
                    for m in 0..6 {
                        for n in 0..7 {
                            correct_answer[[i, l]] +=
                                op1[[i, j]] * op2[[j, m, m, n, k]] * op3[[k, l]];
                        }
                    }
                }
            }
        }
    }

    let ep = einsum_path(
        "ij,jmmnk,kl->il",
        &[&op1, &op2, &op3],
        OptimizationMethod::Naive,
    )
    .unwrap();
    let dotted = ep.contract_operands(&[&op1, &op2, &op3]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn it_contracts_three_matrices_with_repeats_2() {
    let op1 = rand_array((2, 6, 6, 7, 3));
    let op2 = rand_array((3, 4));
    let op3 = rand_array((4, 6, 5));

    let mut correct_answer: Array2<f64> = Array::zeros((2, 5));
    for i in 0..2 {
        for l in 0..5 {
            for j in 0..3 {
                for k in 0..4 {
                    for m in 0..6 {
                        for n in 0..7 {
                            correct_answer[[i, l]] +=
                                op1[[i, m, m, n, j]] * op2[[j, k]] * op3[[k, m, l]];
                        }
                    }
                }
            }
        }
    }

    let ep = einsum_path(
        "immnj,jk,kml->il",
        &[&op1, &op2, &op3],
        OptimizationMethod::Naive,
    )
    .unwrap();
    let dotted = ep.contract_operands(&[&op1, &op2, &op3]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn it_contracts_three_matrices_with_repeats_3() {
    let op1 = rand_array((2, 6, 6, 7, 3));
    let op2 = rand_array((3, 4));
    let op3 = rand_array((4, 5));

    let mut correct_answer: Array3<f64> = Array::zeros((6, 2, 5));
    for i in 0..2 {
        for l in 0..5 {
            for j in 0..3 {
                for k in 0..4 {
                    for m in 0..6 {
                        for n in 0..7 {
                            correct_answer[[m, i, l]] +=
                                op1[[i, m, m, n, j]] * op2[[j, k]] * op3[[k, l]];
                        }
                    }
                }
            }
        }
    }

    let ep = einsum_path(
        "immnj,jk,kl->mil",
        &[&op1, &op2, &op3],
        OptimizationMethod::Naive,
    )
    .unwrap();
    let dotted = ep.contract_operands(&[&op1, &op2, &op3]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}

#[test]
fn it_contracts_four_matrices() {
    let op1 = rand_array((2, 3));
    let op2 = rand_array((3, 4));
    let op3 = rand_array((4, 5));
    let op4 = rand_array((5, 6));

    let mut correct_answer: Array2<f64> = Array::zeros((2, 5));
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                for l in 0..5 {
                    for m in 0..6 {
                        correct_answer[[i, l]] +=
                            op1[[i, j]] * op2[[j, k]] * op3[[k, l]] * op4[[l, m]];
                    }
                }
            }
        }
    }

    let ep = einsum_path(
        "ij,jk,kl,lm->il",
        &[&op1, &op2, &op3, &op4],
        OptimizationMethod::Naive,
    )
    .unwrap();
    let dotted = ep.contract_operands(&[&op1, &op2, &op3, &op4]);
    assert!(correct_answer.my_all_close(&dotted, TOL));
}
