// from paths.py
use hashbrown::HashMap;
const PI: f64 = std::f64::consts::PI;

use crate::xx_decompose::polytopes::{Polytope, SpecialPolytope, AlcoveDetails, AF, Reflection};

// TODO: Do we know that target_coordinate has length 3?
/// Assembles a coordinate in the system used by `xx_region_polytope`.
fn get_augmented_coordinate(target_coordinate: &[f64], strengths: &[f64]) -> Vec<f64> {
    let (beta, strengths) = strengths.split_last().unwrap();
    let mut strengths = Vec::from(strengths);
    strengths.extend([0., 0.]);
    strengths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ssum: f64 = strengths.iter().sum();
    let n = strengths.len();
    let interaction_coordinate = [ssum, strengths[n-1], strengths[n-2], *beta];
    let mut target_coordinate = Vec::from(target_coordinate);
    target_coordinate.extend(interaction_coordinate);
    target_coordinate
}

// Given a `target_coordinate` and a list of interaction `strengths`, produces a new canonical
// coordinate which is one step back along `strengths`.
//
// `target_coordinate` is taken to be in positive canonical coordinates, and the entries of
// strengths are taken to be [0, pi], so that (sj / 2, 0, 0) is a positive canonical coordinate.
fn decomposition_hop(target_coordinate: &[f64; 3], strengths: &[f64]) -> Vec<f64> {
    let target_coordinate: Vec<_> = target_coordinate.iter().map(|x| x / (2. * PI)).collect();
    let strengths: Vec<_> = strengths.iter().map(|x| x / PI).collect();
    let augmented_coordinate = get_augmented_coordinate(&target_coordinate, &strengths);

    let (xx_lift_polytope, xx_region_polytope) = make_polys1();

    let mut special_polytope = None;
    for cp in xx_region_polytope.iter() {
        if ! cp.has_element(&augmented_coordinate) {
            continue
        }
        let af = match cp.get_AF() {
            Some(AF::B1) => target_coordinate[0],
            Some(AF::B3) => target_coordinate[2],
            None => panic!("Static data is incorrect") // halfway between programming error and data error.
        };
        let mut coefficient_dict = HashMap::<(i32, i32), f64>::new();
        let raw_convex_polytope = xx_lift_polytope
            .iter()
            .find(|p| p.alcove_details.as_ref() == cp.alcove_details.as_ref()).unwrap(); // TODO: handle None
        for ineq in raw_convex_polytope.ineqs.iter() {
            if ineq[1] == 0 && ineq[2] == 0 {
                continue
            }
            let offset = (
                ineq[0] as f64  // old constant term
                + (ineq[3] as f64) * af
                + (ineq[4] as f64) * augmented_coordinate[0]  // b1
                + (ineq[5] as f64) * augmented_coordinate[1]  // b2
                + (ineq[6] as f64) * augmented_coordinate[2]  // b3
                + (ineq[7] as f64) * augmented_coordinate[3]  // s+
                + (ineq[8] as f64) * augmented_coordinate[4]  // s1
                + (ineq[9] as f64) * augmented_coordinate[5]  // s2
                    + (ineq[10] as f64) * augmented_coordinate[6]);  // beta
            
            let _offset = match coefficient_dict.get(&(ineq[1], ineq[2])) {
                Some(_offset) => *_offset,
                None => offset,
            };
            if offset <= _offset {
                coefficient_dict.insert((ineq[1], ineq[2]), offset);
            }
        }
        special_polytope = Some(SpecialPolytope {
                ineqs: coefficient_dict.iter()
                .map(|((h, l), v)| [*h as f64, *l as f64, *v]).collect()
        });
        break;
        let special_polytope = special_polytope.unwrap_or_else(|| panic!("Failed to match a constrained_polytope summand."));
        let vertex = special_polytope.manual_get_vertex();
        let the_as = [vertex[0], vertex[1], af];
        // Note that we sort in reverse order.
        the_as.sort_by(|&a, &b| b.partial_cmp(&a).unwrap());
        let retval: Vec<f64> = the_as.iter()
            .map(|x| x * PI / 2.0)
            .collect();
        return retval
    }
    panic!()
}

fn make_polys1() ->  (Vec<Polytope<'static, MLIFT>>, Vec<Polytope<'static, MREGION>>) {
    let xx_lift_polytope: Vec<Polytope<'static, MLIFT>> =
        vec! [
            Polytope {
                ineqs: &LIFT_INEQ1,
                eqs: Some(&LIFT_EQ1),
                alcove_details: Some(AlcoveDetails::new(Reflection::Unreflected, AF::B3))
            },
            Polytope {
                ineqs: &LIFT_INEQ2,
                eqs: Some(&LIFT_EQ2),
                alcove_details: Some(AlcoveDetails::new(Reflection::Unreflected, AF::B1))
            },
            Polytope {
                ineqs: &LIFT_INEQ3,
                eqs: Some(&LIFT_EQ3),
                alcove_details: Some(AlcoveDetails::new(Reflection::Reflected, AF::B1))
            },
            Polytope {
                ineqs: &LIFT_INEQ4,
                eqs: Some(&LIFT_EQ4),
                alcove_details: Some(AlcoveDetails::new(Reflection::Reflected, AF::B3))
            },
        ];

    let xx_region_polytope: Vec<Polytope<'static, MREGION>> =
        vec! [
            Polytope {
                ineqs: &REGION_INEQ1,
                eqs: None,
                alcove_details: Some(AlcoveDetails::new(Reflection::Unreflected, AF::B3))
            },
            Polytope {
                ineqs: &REGION_INEQ2,
                eqs: None,
                alcove_details: Some(AlcoveDetails::new(Reflection::Reflected, AF::B3))
            },
            Polytope {
                ineqs: &REGION_INEQ3,
                eqs: None,
                alcove_details: Some(AlcoveDetails::new(Reflection::Unreflected, AF::B1))
            },
            Polytope {
                ineqs: &REGION_INEQ4,
                eqs: None,
                alcove_details: Some(AlcoveDetails::new(Reflection::Reflected, AF::B1))
            },
        ];

    (xx_lift_polytope, xx_region_polytope)
}

const MLIFT: usize = 11;

static LIFT_INEQ1: [[i32; MLIFT]; 29] = [
        [0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, -1, -1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, -1, -1, 0, 0, 0, 1, -2, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 1, -1, -1, 0],
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, -1, -1, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, -1, -1, 1, -2, 0, 1],
        [0, 0, 0, 0, 1, -1, -1, 1, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, -1, 1, -1, -1, 1],
        [0, 0, 0, 0, 0, 0, -1, 1, -1, 0, 0],
        [0, -1, -1, 0, 1, 1, 0, 0, 0, 0, 1],
        [2, -1, -1, 0, -1, -1, 0, 0, 0, 0, -1],
        [0, 1, 1, 0, -1, -1, 0, 0, 0, 0, 1],
        [0, -1, 1, 0, 1, -1, 0, 0, 0, 0, 1],
        [0, 1, -1, 0, -1, 1, 0, 0, 0, 0, 1],
        [0, 1, -1, 0, 1, -1, 0, 0, 0, 0, -1],
];

static LIFT_EQ1: [[i32; MLIFT]; 1] = [[0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0]];
// equalities=[[0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0]],


static LIFT_INEQ2: [[i32; MLIFT]; 29] = [
                [0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                [0, -1, -1, -1, 0, 0, 0, 1, 0, 0, 0],
                [0, -1, -1, 1, 0, 0, 0, 1, -2, 0, 0],
                [0, 0, -1, 0, 0, 0, 0, 1, -1, -1, 0],
                [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, -1, -1, 1, 0, 0, 1],
                [0, 0, 0, 0, 1, -1, -1, 1, -2, 0, 1],
                [0, 0, 0, 0, 1, -1, -1, 1, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, -1, 1, -1, -1, 1],
                [0, 0, 0, 0, 0, 0, -1, 1, -1, 0, 0],
                [0, -1, -1, 0, 0, 1, 1, 0, 0, 0, 1],
                [2, -1, -1, 0, 0, -1, -1, 0, 0, 0, -1],
                [0, 1, 1, 0, 0, -1, -1, 0, 0, 0, 1],
                [0, -1, 1, 0, 0, 1, -1, 0, 0, 0, 1],
                [0, 1, -1, 0, 0, -1, 1, 0, 0, 0, 1],
                [0, 1, -1, 0, 0, 1, -1, 0, 0, 0, -1],
            ];

static LIFT_EQ2: [[i32; MLIFT]; 1] = [[0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0]];
                                     

static LIFT_INEQ3: [[i32; MLIFT]; 29] = [
                [0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, -1, -1, 0, 0, 0, 1, -2, 0, 0],
                [-1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, -1, 0, 0, 0, 0, 1, -1, -1, 0],
                [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 1, -1, -1, 1, 0, 0, 1],
                [1, 0, 0, 0, -1, -1, -1, 1, 0, 0, -1],
                [1, 0, 0, 0, -1, -1, -1, 1, -2, 0, 1],
                [0, 0, 0, 0, 0, 0, -1, 1, -1, -1, 1],
                [0, 0, 0, 0, 0, 0, -1, 1, -1, 0, 0],
                [0, -1, -1, 0, 0, 1, 1, 0, 0, 0, 1],
                [2, -1, -1, 0, 0, -1, -1, 0, 0, 0, -1],
                [0, 1, 1, 0, 0, -1, -1, 0, 0, 0, 1],
                [0, -1, 1, 0, 0, 1, -1, 0, 0, 0, 1],
                [0, 1, -1, 0, 0, -1, 1, 0, 0, 0, 1],
                [0, 1, -1, 0, 0, 1, -1, 0, 0, 0, -1],
            ];

static LIFT_EQ3: [[i32; MLIFT]; 1] = [[0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0]];

static LIFT_INEQ4: [[i32; MLIFT]; 29] = [
                [0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, -1, -1, 0, 0, 0, 1, -2, 0, 0],
                [-1, 1, -1, -1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, -1, 0, 0, 0, 0, 1, -1, -1, 0],
                [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 1, -1, -1, 1, 0, 0, 1],
                [1, 0, 0, 0, -1, -1, -1, 1, 0, 0, -1],
                [1, 0, 0, 0, -1, -1, -1, 1, -2, 0, 1],
                [0, 0, 0, 0, 0, 0, -1, 1, -1, -1, 1],
                [0, 0, 0, 0, 0, 0, -1, 1, -1, 0, 0],
                [0, -1, -1, 0, 1, 1, 0, 0, 0, 0, 1],
                [2, -1, -1, 0, -1, -1, 0, 0, 0, 0, -1],
                [0, 1, 1, 0, -1, -1, 0, 0, 0, 0, 1],
                [0, -1, 1, 0, 1, -1, 0, 0, 0, 0, 1],
                [0, 1, -1, 0, -1, 1, 0, 0, 0, 0, 1],
                [0, 1, -1, 0, 1, -1, 0, 0, 0, 0, -1],
            ];

static LIFT_EQ4: [[i32; MLIFT]; 1] = [[0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0]];

const MREGION: usize = 8;

static REGION_INEQ1: [[i32; MREGION]; 15] = [
    [0, 0, 0, 0, 0, 1, -1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, -2, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, -2],
    [0, 0, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, -1, -1, 0],
    [1, -1, -1, 0, 0, 0, 0, 0],
    [0, -1, -1, -1, 1, 0, 0, 1],
    [0, 1, -1, 0, 0, 0, 0, 0],
    [0, 1, -1, -1, 1, -2, 0, 1],
    [0, 1, -1, -1, 1, 0, 0, -1],
    [0, 0, 0, -1, 1, -1, 0, 0],
    [0, 0, -1, 0, 1, -1, -1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1],
];

static REGION_INEQ2: [[i32; MREGION]; 15] = [
                [0, 0, 0, 0, 0, 1, -1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, -2, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, -2],
                [0, 0, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, -1, 0],
                [1, -1, -1, 0, 0, 0, 0, 0],
                [1, -1, -1, -1, 1, -2, 0, 1],
                [0, 1, -1, 0, 0, 0, 0, 0],
                [-1, 1, -1, -1, 1, 0, 0, 1],
                [1, -1, -1, -1, 1, 0, 0, -1],
                [0, 0, 0, -1, 1, -1, 0, 0],
                [0, 0, -1, 0, 1, -1, -1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
    ];

static REGION_INEQ3: [[i32; MREGION]; 16] = [
                [0, 0, 0, 0, 0, 1, -1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, -2, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, -2],
                [0, 1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [1, -1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, -1, 0],
                [0, 1, -1, -1, 1, -2, 0, 1],
                [0, -1, -1, -1, 1, 0, 0, 1],
                [0, 0, 1, -1, 0, 0, 0, 0],
                [1, -1, 1, -1, 0, 0, 0, -1],
                [0, 1, 1, -1, 1, -2, 0, -1],
                [0, -1, 1, -1, 1, 0, 0, -1],
                [0, 0, 0, -1, 1, -1, -1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],

];

static REGION_INEQ4: [[i32; MREGION]; 16] = [
                [0, 0, 0, 0, 0, 1, -1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, -2, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, -2],
                [0, 1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [1, -1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, -1, 0],
                [-1, 1, -1, -1, 1, 0, 0, 1],
                [1, -1, -1, -1, 1, -2, 0, 1],
                [0, 0, 1, -1, 0, 0, 0, 0],
                [1, -1, 1, -1, 0, 0, 0, -1],
                [-1, 1, 1, -1, 1, 0, 0, -1],
                [1, -1, 1, -1, 1, -2, 0, -1],
                [0, 0, 0, -1, 1, -1, -1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],

    ];
