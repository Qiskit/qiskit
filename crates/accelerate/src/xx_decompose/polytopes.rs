use crate::xx_decompose::types::Coordinate;
use crate::xx_decompose::utilities::EPSILON;

/// The raw data underlying a ConvexPolytope.  Describes a single /// polytope, specified by families of `inequalities` and `equalities`, each
/// entry of which respectively corresponds to
///
///     inequalities[j][0] + sum_i inequalities[j][i] * xi >= 0
///
/// and
///
///     equalities[j][0] + sum_i equalities[j][i] * xi == 0.


// The names of the four explicit ConvexPolytopeData are below.
// These occur in both xx_lift... and xx_region... But the order is slightly different.
// Combinations obey:
// "ah" and "B3" are always together
// "af" and "B1" are always together
// So we use just B1 and B3 to distinguish the pairs.
// "A unreflected", "B unreflected" and "slant" always appear together.
// "A reflected", "B reflected" and "strength" always appear together.
// We label these triples with just "reflected" and "unreflected".
//
// If this code is modified so that the characteristics of the polytopes
// changes in the future, the simplifications mentioned above may not hold.
//
// "A unreflected ∩ ah slant ∩ al frustrum ∩ B alcove ∩ B unreflected ∩ AF=B3"
// "A reflected ∩ ah strength ∩ al frustrum ∩ B alcove ∩ B reflected ∩ AF=B3"
// "A unreflected ∩ af slant ∩ al frustrum ∩ B alcove ∩ B unreflected ∩ AF=B1"
// "A reflected ∩ af strength ∩ al frustrum ∩ B alcove ∩ B reflected ∩ AF=B1"


#[derive(PartialEq)]
pub(crate) struct AlcoveDetails {
    reflection: Reflection,
    af: AF
}

impl AlcoveDetails {
    pub(crate) fn new(reflection: Reflection, af: AF) -> AlcoveDetails {
        AlcoveDetails { reflection, af }
    }
}

#[derive(PartialEq)]
pub(crate) enum Reflection {
    Reflected,
    Unreflected,
}

// B2 is also referenced in the Python code.
// But no data carries a "tag" B2, so that code is never used.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum AF {
    B1,
    B3,
}

static polys: Vec<Polytope<'static, 11>> = vec! [ ];

pub(crate) struct Polytope<'a, const NCOLS: usize> {
    pub(crate) ineqs: &'a [[i32; NCOLS]],
    pub(crate) eqs: Option<&'a[[i32; NCOLS]]>,
    pub(crate) alcove_details: Option<AlcoveDetails>,
}

impl<'a, const NCOLS: usize> Polytope<'a, NCOLS> {

    /// Tests whether the polytope contains `point`.
    pub(crate) fn has_element(&self, point: &Vec<f64>) -> bool {
        if ! self.ineqs
            .iter()
            .all(|ie| (-EPSILON <= ie[0] as f64 + point.iter().zip(&ie[1..]).map(|(p, i)| p * *i as f64).sum::<f64>())) {
                return false
            };
        if self.eqs.is_none() {
            return false
        }
        // This is never called due to the static data in the current implementation.
        self.eqs.unwrap()
            .iter()
            .all(|e| (e[0] as f64 + point.iter().zip(&e[1..]).map(|(p, i)| p * *i as f64).sum::<f64>() <= EPSILON))
    }

    #[allow(non_snake_case)]
    pub(crate) fn get_AF(&self) -> Option<AF> {        
        self.alcove_details.as_ref().map(|x| x.af)
    }
}

pub(crate) struct ConvexPolytopeData<'a, const MI:usize, const NI: usize, const ME:usize, const NE: usize> {
    pub(crate) inequalities: [[i32; MI]; NI],
    pub(crate) equalities: [[i32; ME]; NE],
    pub(crate) name: &'a str,
}

// /// The raw data of a union of convex polytopes.
// pub(crate) struct PolytopeData<'a, const MI:usize, const NI: usize, const ME:usize, const NE: usize, const NC: usize> {
//     pub(crate) convex_subpolytopes: [ConvexPolytopeData<'a, MI, NI, ME, NE>; NC],
// }

// TODO: In the original, this is not a class-instance method. Could be I think.

// fn polytope_has_element<const MI:usize, const NI: usize, const ME:usize, const NE: usize>
//     (polytope: ConvexPolytopeData<MI, NI, ME, NE>, point: &Vec<f64>) -> bool {
//     polytope.inequalities
//             .iter()
//             .all(|ie| (-EPSILON <= ie[0] as f64 + point.iter().zip(&ie[1..]).map(|(p, i)| p * *i as f64).sum::<f64>()))
//     //         &&
//     // polytope.equalities
//     //         .iter()
//     //         .all(|e| (e[0] + point.iter().zip(&e[1..]).map(|(p, i)| p * i).sum::<f64>() <= EPSILON))
// }

/// Describes those two-qubit programs accessible to a given sequence of XX-type interactions.
///
/// NOTE: Strengths are normalized so that CX corresponds to pi / 4, which differs from Qiskit's
///       conventions around RZX elsewhere.
struct XXPolytope {
    total_strength: f64,
    max_strength: f64,
    place_strength: f64,
}

impl XXPolytope {

    // Method add_strength appears in the original Python
    // But it is only called in the test suite.
    //    fn add_strength() {}
}


// Comment from polytopes.py:
// These globals house matrices, computed once-and-for-all, which project onto the Euclidean-nearest
// planes spanned by the planes/points/lines defined by the faces of any XX polytope.
//
// See `XXPolytope.nearest`.

// In the Python version, most of the following multidimensional arrays are computed when
// the file loads via linear algebra routines. This is not possible in Rust, so we create
// the arrays with literals.
// Fractional numbers are introduced by computing inverses.
// Many of the numbers in the Python version differ from the correct value by something near
// an ulp, about one part in 1^-16. Correcting these numbers makes literals easier to read
// and gives more accurate results when multiplying matrices by their inverses.
// Correcting the numbers was done by first setting `np.set_printoptions(precision=n)`,
// with n taking values depending on the array. Then doing,
// for example, `np.round(A3inv, decimals=14)` to correctly print the numbers with two to
// four digits. The repeating fractions are replaced by numbers with the correct digits
// up to 16 digits, as determined by, for example `1/3`.

static A: [[i32; 3]; 7] = [
        [1, -1, 0],  // a ≥ b
        [0, 1, -1],  // b ≥ c
        [0, 0, 1],  // c ≥ 0
        [-1, -1, 0],  // pi/2 ≥ a + b
        [-1, -1, -1],  // strength
        [1, -1, -1],  // slant
        [0, 0, -1],  // frustrum
 ];

// In Python the data in A is reorganized as A1  so that
// scipy can compute the pseudo-inverse of each 3-vector.
// The result is A1inv.

static A1: [[[f64; 3]; 1] ; 7] =
      [[[ 1., -1.,  0.]],

       [[ 0.,  1., -1.]],

       [[ 0.,  0.,  1.]],

       [[-1., -1.,  0.]],

       [[-1., -1., -1.]],

       [[ 1., -1., -1.]],

       [[ 0.,  0., -1.]]];

static A1INV: [[[f64; 1]; 3] ; 7] =
        [[[ 0.5             ],
        [-0.5             ],
        [ 0.              ]],

       [[ 0.              ],
        [ 0.5             ],
        [-0.5             ]],

       [[ 0.              ],
        [ 0.              ],
        [ 1.              ]],

       [[-0.5             ],
        [-0.5             ],
        [ 0.              ]],

       [[-0.3333333333333333],
        [-0.3333333333333333],
        [-0.3333333333333333]],

       [[ 0.3333333333333333],
        [-0.3333333333333333],
        [-0.3333333333333333]],

       [[ 0.              ],
        [ 0.              ],
        [-1.              ]]];

// A2 collects all pairs of rows in A
// A2inv collects the pseudo-inverse of each 2x3 matrix.

static A2: [[[f64; 3]; 2]; 21] =
      [[[ 1., -1.,  0.],
        [ 0.,  1., -1.]],

       [[ 1., -1.,  0.],
        [ 0.,  0.,  1.]],

       [[ 1., -1.,  0.],
        [-1., -1.,  0.]],

       [[ 1., -1.,  0.],
        [-1., -1., -1.]],

       [[ 1., -1.,  0.],
        [ 1., -1., -1.]],

       [[ 1., -1.,  0.],
        [ 0.,  0., -1.]],

       [[ 0.,  1., -1.],
        [ 0.,  0.,  1.]],

       [[ 0.,  1., -1.],
        [-1., -1.,  0.]],

       [[ 0.,  1., -1.],
        [-1., -1., -1.]],

       [[ 0.,  1., -1.],
        [ 1., -1., -1.]],

       [[ 0.,  1., -1.],
        [ 0.,  0., -1.]],

       [[ 0.,  0.,  1.],
        [-1., -1.,  0.]],

       [[ 0.,  0.,  1.],
        [-1., -1., -1.]],

       [[ 0.,  0.,  1.],
        [ 1., -1., -1.]],

       [[ 0.,  0.,  1.],
        [ 0.,  0., -1.]],

       [[-1., -1.,  0.],
        [-1., -1., -1.]],

       [[-1., -1.,  0.],
        [ 1., -1., -1.]],

       [[-1., -1.,  0.],
        [ 0.,  0., -1.]],

       [[-1., -1., -1.],
        [ 1., -1., -1.]],

       [[-1., -1., -1.],
        [ 0.,  0., -1.]],

       [[ 1., -1., -1.],
        [ 0.,  0., -1.]]];

// A3 collects all triples of rows in A
// A3inv collects the pseudo-inverse of each 3x3 matrix.

static A2INV: [[[f64; 2]; 3]; 21] =
    [[[ 0.6666666666666666,  0.3333333333333333],
        [-0.3333333333333333,  0.3333333333333333],
        [-0.3333333333333333, -0.6666666666666666]],

       [[ 0.5             ,  0.              ],
        [-0.5             ,  0.              ],
        [ 0.              ,  1.              ]],

       [[ 0.5             , -0.5             ],
        [-0.5             , -0.5             ],
        [ 0.              ,  0.              ]],

       [[ 0.5             , -0.3333333333333333],
        [-0.5             , -0.3333333333333333],
        [ 0.              , -0.3333333333333333]],

       [[ 0.5             ,  0.              ],
        [-0.5             , -0.              ],
        [ 1.              , -1.              ]],

       [[ 0.5             ,  0.              ],
        [-0.5             ,  0.              ],
        [ 0.              , -1.              ]],

       [[ 0.              ,  0.              ],
        [ 1.              ,  1.              ],
        [ 0.              ,  1.              ]],

       [[-0.3333333333333333, -0.6666666666666666],
        [ 0.3333333333333333, -0.3333333333333333],
        [-0.6666666666666666, -0.3333333333333333]],

       [[ 0.              , -0.3333333333333333],
        [ 0.5             , -0.3333333333333333],
        [-0.5             , -0.3333333333333333]],

       [[ 0.              ,  0.3333333333333333],
        [ 0.5             , -0.3333333333333333],
        [-0.5             , -0.3333333333333333]],

       [[ 0.              , -0.              ],
        [ 1.              , -1.              ],
        [ 0.              , -1.              ]],

       [[ 0.              , -0.5             ],
        [ 0.              , -0.5             ],
        [ 1.              ,  0.              ]],

       [[-0.5             , -0.5             ],
        [-0.5             , -0.5             ],
        [ 1.              ,  0.              ]],

       [[ 0.5             ,  0.5             ],
        [-0.5             , -0.5             ],
        [ 1.              ,  0.              ]],

       [[ 0.              ,  0.              ],
        [ 0.              ,  0.              ],
        [ 0.5             , -0.5             ]],

       [[-0.5             , -0.              ],
        [-0.5             , -0.              ],
        [ 1.              , -1.              ]],

       [[-0.5             ,  0.3333333333333333],
        [-0.5             , -0.3333333333333333],
        [ 0.              , -0.3333333333333333]],

       [[-0.5             ,  0.              ],
        [-0.5             ,  0.              ],
        [ 0.              , -1.              ]],

       [[-0.5             ,  0.5             ],
        [-0.25            , -0.25            ],
        [-0.25            , -0.25            ]],

       [[-0.5             ,  0.5             ],
        [-0.5             ,  0.5             ],
        [ 0.              , -1.              ]],

       [[ 0.5             , -0.5             ],
        [-0.5             ,  0.5             ],
        [ 0.              , -1.              ]]];


static A3: [[[f64; 3]; 3]; 35] =
      [[[ 1., -1.,  0.],
        [ 0.,  1., -1.],
        [ 0.,  0.,  1.]],

       [[ 1., -1.,  0.],
        [ 0.,  1., -1.],
        [-1., -1.,  0.]],

       [[ 1., -1.,  0.],
        [ 0.,  1., -1.],
        [-1., -1., -1.]],

       [[ 1., -1.,  0.],
        [ 0.,  1., -1.],
        [ 1., -1., -1.]],

       [[ 1., -1.,  0.],
        [ 0.,  1., -1.],
        [ 0.,  0., -1.]],

       [[ 1., -1.,  0.],
        [ 0.,  0.,  1.],
        [-1., -1.,  0.]],

       [[ 1., -1.,  0.],
        [ 0.,  0.,  1.],
        [-1., -1., -1.]],

       [[ 1., -1.,  0.],
        [ 0.,  0.,  1.],
        [ 1., -1., -1.]],

       [[ 1., -1.,  0.],
        [ 0.,  0.,  1.],
        [ 0.,  0., -1.]],

       [[ 1., -1.,  0.],
        [-1., -1.,  0.],
        [-1., -1., -1.]],

       [[ 1., -1.,  0.],
        [-1., -1.,  0.],
        [ 1., -1., -1.]],

       [[ 1., -1.,  0.],
        [-1., -1.,  0.],
        [ 0.,  0., -1.]],

       [[ 1., -1.,  0.],
        [-1., -1., -1.],
        [ 1., -1., -1.]],

       [[ 1., -1.,  0.],
        [-1., -1., -1.],
        [ 0.,  0., -1.]],

       [[ 1., -1.,  0.],
        [ 1., -1., -1.],
        [ 0.,  0., -1.]],

       [[ 0.,  1., -1.],
        [ 0.,  0.,  1.],
        [-1., -1.,  0.]],

       [[ 0.,  1., -1.],
        [ 0.,  0.,  1.],
        [-1., -1., -1.]],

       [[ 0.,  1., -1.],
        [ 0.,  0.,  1.],
        [ 1., -1., -1.]],

       [[ 0.,  1., -1.],
        [ 0.,  0.,  1.],
        [ 0.,  0., -1.]],

       [[ 0.,  1., -1.],
        [-1., -1.,  0.],
        [-1., -1., -1.]],

       [[ 0.,  1., -1.],
        [-1., -1.,  0.],
        [ 1., -1., -1.]],

       [[ 0.,  1., -1.],
        [-1., -1.,  0.],
        [ 0.,  0., -1.]],

       [[ 0.,  1., -1.],
        [-1., -1., -1.],
        [ 1., -1., -1.]],

       [[ 0.,  1., -1.],
        [-1., -1., -1.],
        [ 0.,  0., -1.]],

       [[ 0.,  1., -1.],
        [ 1., -1., -1.],
        [ 0.,  0., -1.]],

       [[ 0.,  0.,  1.],
        [-1., -1.,  0.],
        [-1., -1., -1.]],

       [[ 0.,  0.,  1.],
        [-1., -1.,  0.],
        [ 1., -1., -1.]],

       [[ 0.,  0.,  1.],
        [-1., -1.,  0.],
        [ 0.,  0., -1.]],

       [[ 0.,  0.,  1.],
        [-1., -1., -1.],
        [ 1., -1., -1.]],

       [[ 0.,  0.,  1.],
        [-1., -1., -1.],
        [ 0.,  0., -1.]],

       [[ 0.,  0.,  1.],
        [ 1., -1., -1.],
        [ 0.,  0., -1.]],

       [[-1., -1.,  0.],
        [-1., -1., -1.],
        [ 1., -1., -1.]],

       [[-1., -1.,  0.],
        [-1., -1., -1.],
        [ 0.,  0., -1.]],

       [[-1., -1.,  0.],
        [ 1., -1., -1.],
        [ 0.,  0., -1.]],

       [[-1., -1., -1.],
        [ 1., -1., -1.],
        [ 0.,  0., -1.]]];

static A3INV: [[[f64; 3]; 3]; 35] =
     [[[ 1.     ,  1.     ,  1.     ],
        [-0.     ,  1.     ,  1.     ],
        [-0.     , -0.     ,  1.     ]],

       [[ 0.5    ,  0.     , -0.5    ],
        [-0.5    , -0.     , -0.5    ],
        [-0.5    , -1.     , -0.5    ]],

       [[ 0.6666666666666666,  0.3333333333333333, -0.3333333333333333],
        [-0.3333333333333333,  0.3333333333333333, -0.3333333333333333],
        [-0.3333333333333333, -0.6666666666666666, -0.3333333333333333]],

       [[ 2.     ,  1.     , -1.     ],
        [ 1.     ,  1.     , -1.     ],
        [ 1.     ,  0.     , -1.     ]],

       [[ 1.     ,  1.     , -1.     ],
        [-0.     ,  1.     , -1.     ],
        [-0.     , -0.     , -1.     ]],

       [[ 0.5    ,  0.     , -0.5    ],
        [-0.5    ,  0.     , -0.5    ],
        [ 0.     ,  1.     ,  0.     ]],

       [[ 0.5    , -0.5    , -0.5    ],
        [-0.5    , -0.5    , -0.5    ],
        [-0.     ,  1.     ,  0.     ]],

       [[ 0.3333333333333333,  0.1666666666666666,  0.1666666666666666],
        [-0.3333333333333333, -0.1666666666666666, -0.1666666666666666],
        [ 0.3333333333333333,  0.6666666666666666, -0.3333333333333333]],

       [[ 0.5    ,  0.     ,  0.     ],
        [-0.5    ,  0.     ,  0.     ],
        [ 0.     ,  0.5    , -0.5    ]],

       [[ 0.5    , -0.5    ,  0.     ],
        [-0.5    , -0.5    ,  0.     ],
        [ 0.     ,  1.     , -1.     ]],

       [[ 0.5    , -0.5    , -0.     ],
        [-0.5    , -0.5    , -0.     ],
        [ 1.     , -0.     , -1.     ]],

       [[ 0.5    , -0.5    ,  0.     ],
        [-0.5    , -0.5    ,  0.     ],
        [ 0.     ,  0.     , -1.     ]],

       [[ 0.     , -0.5    ,  0.5    ],
        [-1.     , -0.5    ,  0.5    ],
        [ 1.     , -0.     , -1.     ]],

       [[ 0.5    , -0.5    ,  0.5    ],
        [-0.5    , -0.5    ,  0.5    ],
        [ 0.     ,  0.     , -1.     ]],

       [[ 0.3333333333333333,  0.1666666666666666, -0.1666666666666666],
        [-0.3333333333333333, -0.1666666666666666,  0.1666666666666666],
        [ 0.3333333333333333, -0.3333333333333333, -0.6666666666666666]],

       [[-1.     , -1.     , -1.     ],
        [ 1.     ,  1.     , -0.     ],
        [ 0.     ,  1.     , -0.     ]],

       [[-1.     , -2.     , -1.     ],
        [ 1.     ,  1.     , -0.     ],
        [-0.     ,  1.     , -0.     ]],

       [[ 1.     ,  2.     ,  1.     ],
        [ 1.     ,  1.     , -0.     ],
        [-0.     ,  1.     , -0.     ]],

       [[ 0.     ,  0.     ,  0.     ],
        [ 1.     ,  0.5    , -0.5    ],
        [ 0.     ,  0.5    , -0.5    ]],

       [[-1.     , -2.     ,  1.     ],
        [ 1.     ,  1.     , -1.     ],
        [ 0.     ,  1.     , -1.     ]],

       [[-0.3333333333333333, -0.6666666666666666,  0.3333333333333333],
        [ 0.3333333333333333, -0.3333333333333333, -0.3333333333333333],
        [-0.6666666666666666, -0.3333333333333333, -0.3333333333333333]],

       [[-1.     , -1.     ,  1.     ],
        [ 1.     , -0.     , -1.     ],
        [-0.     , -0.     , -1.     ]],

       [[ 0.     , -0.5    ,  0.5    ],
        [ 0.5    , -0.25   , -0.25   ],
        [-0.5    , -0.25   , -0.25   ]],

       [[-1.     , -1.     ,  2.     ],
        [ 1.     , -0.     , -1.     ],
        [-0.     ,  0.     , -1.     ]],

       [[ 1.     ,  1.     , -2.     ],
        [ 1.     ,  0.     , -1.     ],
        [-0.     ,  0.     , -1.     ]],

       [[-0.1666666666666666, -0.3333333333333333, -0.1666666666666666],
        [-0.1666666666666666, -0.3333333333333333, -0.1666666666666666],
        [ 0.6666666666666666,  0.3333333333333333, -0.3333333333333333]],

       [[ 0.5    , -0.5    ,  0.5    ],
        [-0.5    , -0.5    , -0.5    ],
        [ 1.     ,  0.     ,  0.     ]],

       [[ 0.     , -0.5    ,  0.     ],
        [ 0.     , -0.5    ,  0.     ],
        [ 0.5    ,  0.     , -0.5    ]],

       [[ 0.     , -0.5    ,  0.5    ],
        [-1.     , -0.5    , -0.5    ],
        [ 1.     ,  0.     , -0.     ]],

       [[-0.25   , -0.5    ,  0.25   ],
        [-0.25   , -0.5    ,  0.25   ],
        [ 0.5    , -0.     , -0.5    ]],

       [[ 0.25   ,  0.5    , -0.25   ],
        [-0.25   , -0.5    ,  0.25   ],
        [ 0.5    , -0.     , -0.5    ]],

       [[-0.     , -0.5    ,  0.5    ],
        [-1.     ,  0.5    , -0.5    ],
        [ 1.     , -1.     , -0.     ]],

       [[-0.3333333333333333, -0.1666666666666666,  0.1666666666666666],
        [-0.3333333333333333, -0.1666666666666666,  0.1666666666666666],
        [ 0.3333333333333333, -0.3333333333333333, -0.6666666666666666]],

       [[-0.5    ,  0.5    , -0.5    ],
        [-0.5    , -0.5    ,  0.5    ],
        [ 0.     ,  0.     , -1.     ]],

       [[-0.5    ,  0.5    ,  0.     ],
        [-0.5    , -0.5    ,  1.     ],
        [-0.     , -0.     , -1.     ]]];
