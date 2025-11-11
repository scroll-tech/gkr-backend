use super::{Expression, StructuralWitIn, WitIn};
use crate::{Fixed, Instance, WitnessId, combine_cumulative_either, monomial::Term};
use either::Either;
use ff_ext::ExtensionField;
use itertools::Itertools;
use p3::field::Field;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

impl WitIn {
    pub fn assign<E: ExtensionField>(&self, instance: &mut [E::BaseField], value: E::BaseField) {
        instance[self.id as usize] = value;
    }
}

impl StructuralWitIn {
    pub fn assign<E: ExtensionField>(&self, instance: &mut [E::BaseField], value: E::BaseField) {
        instance[self.id as usize] = value;
    }
}

pub fn eval_by_expr_constant<E: ExtensionField>(
    instance: &[Either<E::BaseField, E>],
    challenges: &[E],
    expr: &Expression<E>,
) -> Either<E::BaseField, E> {
    expr.evaluate_with_instance(
        &|_| unimplemented!(),
        &|_| unimplemented!(),
        &|_, _| unimplemented!(),
        &|i| instance[i.0],
        &|scalar| scalar,
        &|challenge_id, pow, scalar, offset| {
            // TODO cache challenge power to be acquired once for each power
            let challenge = challenges[challenge_id as usize];
            Either::Right(challenge.exp_u64(pow as u64) * scalar + offset)
        },
        &|a, b| combine_cumulative_either!(a, b, |a, b| a + b),
        &|a, b| combine_cumulative_either!(a, b, |a, b| a * b),
        &|x, a, b| {
            let ax = combine_cumulative_either!(a, x, |c1, c2| c1 * c2);
            // ax + b
            combine_cumulative_either!(ax, b, |c1, c2| c1 + c2)
        },
    )
}

pub fn eval_by_expr<E: ExtensionField>(
    witnesses: &[E],
    structural_witnesses: &[E],
    challenges: &[E],
    expr: &Expression<E>,
) -> E {
    eval_by_expr_with_fixed(&[], witnesses, structural_witnesses, challenges, expr)
}

/// Evaluates the expression using fixed values, witnesses, structural witnesses, and challenges.
/// We allow shorter fixed vectors, which are of the length 2^k and repeated cyclically. `fixed_len_minus_one` is the
/// length of the fixed vector minus one, which is used to wrap around the indices.
pub fn eval_by_expr_with_fixed<E: ExtensionField>(
    fixed: &[E],
    witnesses: &[E],
    structural_witnesses: &[E],
    challenges: &[E],
    expr: &Expression<E>,
) -> E {
    expr.evaluate::<E>(
        &|f| fixed[f.0],
        &|witness_id| witnesses[witness_id as usize],
        &|witness_id, _| structural_witnesses[witness_id as usize],
        &|scalar| {
            scalar
                .map_either(|scalar| E::from(scalar), |scalar| scalar)
                .into_inner()
        },
        &|challenge_id, pow, scalar, offset| {
            // TODO cache challenge power to be acquired once for each power
            let challenge = challenges[challenge_id as usize];
            challenge.exp_u64(pow as u64) * scalar + offset
        },
        &|a, b| a + b,
        &|a, b| a * b,
        &|x, a, b| a * x + b,
    )
}

pub fn eval_by_expr_with_instance<E: ExtensionField>(
    fixed: &[E],
    witnesses: &[E],
    structural_witnesses: &[E],
    instance: &[E],
    challenges: &[E],
    expr: &Expression<E>,
) -> Either<E::BaseField, E> {
    expr.evaluate_with_instance::<Either<_, _>>(
        &|f| Either::Right(fixed[f.0]),
        &|witness_id| Either::Right(witnesses[witness_id as usize]),
        &|witness_id, _| Either::Right(structural_witnesses[witness_id as usize]),
        &|i| Either::Right(instance[i.0]),
        &|scalar| scalar,
        &|challenge_id, pow, scalar, offset| {
            // TODO cache challenge power to be acquired once for each power
            let challenge = challenges[challenge_id as usize];
            Either::Right(challenge.exp_u64(pow as u64) * scalar + offset)
        },
        &|a, b| combine_cumulative_either!(a, b, |a, b| a + b),
        &|a, b| combine_cumulative_either!(a, b, |a, b| a * b),
        &|x, a, b| {
            let ax = combine_cumulative_either!(a, x, |c1, c2| c1 * c2);
            // ax + b
            combine_cumulative_either!(ax, b, |c1, c2| c1 + c2)
        },
    )
}

/// convert complex expression into monomial form to WitIn
/// orders WitIn ++ StructuralWitIn ++ Fixed
pub fn monomialize_expr_to_wit_terms<E: ExtensionField>(
    expr: &Expression<E>,
    num_witin: WitnessId,
    num_fixed: WitnessId,
    num_instance: usize,
) -> Vec<Term<Expression<E>, Expression<E>>> {
    let witid_offset = 0 as WitnessId;
    let fixed_offset = witid_offset + num_witin;
    let instance_offset = fixed_offset + num_fixed;
    let structural_witin_offset = instance_offset + num_instance as WitnessId;

    let monomial_terms_expr = expr.get_monomial_terms();
    monomial_terms_expr
        .into_iter()
        .map(
            |Term {
                 scalar,
                 mut product,
             }| {
                product.iter_mut().for_each(|t| match t {
                    Expression::WitIn(_) => (),
                    Expression::Fixed(Fixed(fixed_id)) => {
                        *t = Expression::WitIn(fixed_offset + (*fixed_id as u16));
                    }
                    Expression::Instance(Instance(instance_id)) => {
                        *t = Expression::WitIn(instance_offset + (*instance_id as u16));
                    }
                    Expression::StructuralWitIn(structural_wit_id, _) => {
                        *t = Expression::WitIn(structural_witin_offset + *structural_wit_id);
                    }
                    e => panic!("unknown monomial terms {:?}", e),
                });
                Term { scalar, product }
            },
        )
        .collect_vec()
}

/// convert complex expression into monomial form to WitIn
/// orders WitIn ++ StructuralWitIn ++ Fixed
pub fn expr_convert_to_witins<E: ExtensionField>(
    expr: &mut Expression<E>,
    num_witin: WitnessId,
    num_fixed: WitnessId,
    num_instance: usize,
) {
    let witid_offset = 0 as WitnessId;
    let fixed_offset = witid_offset + num_witin;
    let instance_offset = fixed_offset + num_fixed;
    let structural_witin_offset = instance_offset + num_instance as WitnessId;

    match expr {
        Expression::Fixed(fixed_id) => {
            *expr = Expression::WitIn(fixed_offset + (fixed_id.0 as u16))
        }
        Expression::WitIn(..) => (),
        Expression::StructuralWitIn(structural_wit_id, ..) => {
            *expr = Expression::WitIn(structural_witin_offset + *structural_wit_id)
        }
        Expression::Instance(i) => *expr = Expression::WitIn(instance_offset + (i.0 as u16)),
        Expression::InstanceScalar(..) => (),
        Expression::Constant(..) => (),
        Expression::Sum(a, b) => {
            expr_convert_to_witins(a, num_witin, num_fixed, num_instance);
            expr_convert_to_witins(b, num_witin, num_fixed, num_instance);
        }
        Expression::Product(a, b) => {
            expr_convert_to_witins(a, num_witin, num_fixed, num_instance);
            expr_convert_to_witins(b, num_witin, num_fixed, num_instance);
        }
        Expression::ScaledSum(x, a, b) => {
            expr_convert_to_witins(x, num_witin, num_fixed, num_instance);
            expr_convert_to_witins(a, num_witin, num_fixed, num_instance);
            expr_convert_to_witins(b, num_witin, num_fixed, num_instance);
        }
        Expression::Challenge(..) => (),
    }
}

pub const DagLoadWit: usize = 0;
pub const DagLoadScalar: usize = 1;
pub const DagAdd: usize = 2;
pub const DagMul: usize = 3;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct Node {
    pub op: u32,
    pub left_id: u32,
    pub right_id: u32,
    pub out: u32,
}

fn is_one<E: ExtensionField>(e: &Expression<E>) -> bool {
    matches!(e, Expression::Constant(v)
        if v.map_either(|b| b.is_one(), |x| x.is_one()).into_inner())
}

fn is_zero<E: ExtensionField>(e: &Expression<E>) -> bool {
    matches!(e, Expression::Constant(v)
        if v.map_either(|b| b.is_zero(), |x| x.is_zero()).into_inner())
}

pub fn expr_compression_to_dag<E: ExtensionField>(
    expr: &Expression<E>,
) -> (
    Vec<Node>,
    Vec<Instance>,
    Vec<Expression<E>>,
    Vec<Either<E::BaseField, E>>,
    u32,
    (usize, usize),
) {
    let mut constant_dedup = HashMap::new();
    let mut challenges_dedup = HashMap::new();
    let mut dag = vec![];
    let mut constant = vec![];
    let mut instance_scalar = vec![];
    let mut challenges = vec![];
    let mut stack_pos: u32 = 0;
    // traverse first time to collect offset
    let _ = expr_compression_to_dag_helper(
        &mut dag,
        &mut instance_scalar,
        0,
        &mut challenges,
        0,
        &mut constant,
        &mut challenges_dedup,
        &mut constant_dedup,
        &mut stack_pos,
        expr,
    );

    let challenge_offset = instance_scalar.len();
    let constant_offset = instance_scalar.len() + challenges.len();

    dag.truncate(0);
    constant.truncate(0);
    instance_scalar.truncate(0);
    challenges.truncate(0);
    challenges_dedup.clear();
    constant_dedup.clear();
    stack_pos = 0;
    let Some((max_degree, max_depth)) = expr_compression_to_dag_helper(
        &mut dag,
        &mut instance_scalar,
        challenge_offset,
        &mut challenges,
        constant_offset,
        &mut constant,
        &mut challenges_dedup,
        &mut constant_dedup,
        &mut stack_pos,
        expr,
    ) else {
        panic!("zero expression expr {expr}")
    };
    (
        dag,
        instance_scalar,
        challenges,
        constant,
        stack_pos,
        (max_degree, max_depth),
    )
}

fn expr_compression_to_dag_helper<E: ExtensionField>(
    dag: &mut Vec<Node>,
    instance_scalar: &mut Vec<Instance>,
    challenges_offset: usize,
    challenges: &mut Vec<Expression<E>>,
    constant_offset: usize,
    constant: &mut Vec<Either<E::BaseField, E>>,
    challenges_dedup: &mut HashMap<Expression<E>, u32>,
    constant_dedup: &mut HashMap<Either<E::BaseField, E>, u32>,
    stack_pos: &mut u32,
    expr: &Expression<E>,
) -> Option<(usize, usize)> {
    // (max_degree, max_depth)
    match expr {
        Expression::Fixed(_) => unimplemented!(),
        Expression::WitIn(wit_id) => {
            dag.push(Node {
                op: DagLoadWit as u32,
                left_id: *wit_id as u32,
                right_id: 0,
                out: *stack_pos,
            });
            *stack_pos += 1;
            Some((1, 1))
        }
        Expression::StructuralWitIn(_, ..) => unimplemented!(),
        Expression::Instance(_) => unimplemented!(),
        Expression::InstanceScalar(inst) => {
            instance_scalar.push(inst.clone());
            dag.push(Node {
                op: DagLoadScalar as u32,
                left_id: instance_scalar.len() as u32 - 1,
                right_id: 0,
                out: *stack_pos,
            });
            *stack_pos += 1;
            Some((0, 1))
        }
        Expression::Constant(value) => {
            // if zero, skip entirely
            let is_zero = value
                .map_either(|b| b.is_zero(), |e| e.is_zero())
                .into_inner();
            if is_zero {
                return None;
            }

            let constant_id = *constant_dedup.entry(value.clone()).or_insert_with(|| {
                constant.push(value.clone());
                (constant_offset + constant.len() - 1) as u32
            });

            dag.push(Node {
                op: DagLoadScalar as u32,
                left_id: constant_id,
                right_id: 0,
                out: *stack_pos,
            });
            *stack_pos += 1;
            Some((0, 1))
        }
        Expression::Sum(a, b) => {
            let lhs = expr_compression_to_dag_helper(
                dag,
                instance_scalar,
                challenges_offset,
                challenges,
                constant_offset,
                constant,
                challenges_dedup,
                constant_dedup,
                stack_pos,
                a,
            );
            let rhs = expr_compression_to_dag_helper(
                dag,
                instance_scalar,
                challenges_offset,
                challenges,
                constant_offset,
                constant,
                challenges_dedup,
                constant_dedup,
                stack_pos,
                b,
            );

            match (lhs, rhs) {
                (None, None) => None,                         // 0 + 0 = 0
                (Some(x), None) | (None, Some(x)) => Some(x), // a + 0 = a
                (Some((da, dep_a)), Some((db, dep_b))) => {
                    dag.push(Node {
                        op: DagAdd as u32,
                        left_id: *stack_pos - 2,
                        right_id: *stack_pos - 1,
                        out: *stack_pos - 2,
                    });
                    *stack_pos -= 1;
                    Some((da.max(db), dep_a.max(dep_b + 1))) // 1 comes from store result of `a`
                }
            }
        }
        Expression::Product(a, b) => {
            // ---- identity short-circuits BEFORE recursion (so we don't push junk) ----
            if is_zero(a) || is_zero(b) {
                // nothing pushed, caller treats None as 0
                None
            } else if is_one(a) {
                // 1 * b = b  (evaluate only b; it will land at the current top)
                expr_compression_to_dag_helper(
                    dag, instance_scalar, challenges_offset, challenges,
                    constant_offset, constant, challenges_dedup, constant_dedup,
                    stack_pos, b,
                )
            } else if is_one(b) {
                // a * 1 = a  (evaluate only a)
                expr_compression_to_dag_helper(
                    dag, instance_scalar, challenges_offset, challenges,
                    constant_offset, constant, challenges_dedup, constant_dedup,
                    stack_pos, a,
                )
            } else {
                // ---- general case: evaluate a, then b, emit MUL into (top-2), pop 1 ----
                let lhs = expr_compression_to_dag_helper(
                    dag, instance_scalar, challenges_offset, challenges,
                    constant_offset, constant, challenges_dedup, constant_dedup,
                    stack_pos, a,
                );
                let rhs = expr_compression_to_dag_helper(
                    dag, instance_scalar, challenges_offset, challenges,
                    constant_offset, constant, challenges_dedup, constant_dedup,
                    stack_pos, b,
                );

                match (lhs, rhs) {
                    (None, _) | (_, None) => None, // defensive (shouldn’t reach due to early zero)
                    (Some((da, dep_a)), Some((db, dep_b))) => {
                        dag.push(Node {
                            op: DagMul as u32,
                            left_id: *stack_pos - 2,
                            right_id: *stack_pos - 1,
                            out: *stack_pos - 2,       // overwrite lhs slot
                        });
                        *stack_pos -= 1;                // consume rhs
                        Some((da + db, dep_a.max(dep_b + 1)))
                    }
                }
            }
        }
        Expression::ScaledSum(x, a, b) => {
            // algebraic simplifications BEFORE recursion:
            // x*a + b =>
            //   if x==0 or a==0 -> 0 + b = b
            //   if x==1         -> a + b
            //   if a==1         -> x + b
            //   if b==0         -> x*a
            //
            // We’ll implement these in order, so we only evaluate what's needed.

            if is_zero(x) || is_zero(a) {
                // becomes b
                return expr_compression_to_dag_helper(
                    dag, instance_scalar, challenges_offset, challenges,
                    constant_offset, constant, challenges_dedup, constant_dedup,
                    stack_pos, b,
                );
            }

            if is_one(x) {
                // 1*a + b = a + b
                let lhs_a = expr_compression_to_dag_helper(
                    dag, instance_scalar, challenges_offset, challenges,
                    constant_offset, constant, challenges_dedup, constant_dedup,
                    stack_pos, a,
                );
                let rhs_b = expr_compression_to_dag_helper(
                    dag, instance_scalar, challenges_offset, challenges,
                    constant_offset, constant, challenges_dedup, constant_dedup,
                    stack_pos, b,
                );

                return match (lhs_a, rhs_b) {
                    (None, None) => None,
                    (Some(x), None) | (None, Some(x)) => Some(x),
                    (Some((da, dep_a)), Some((db, dep_b))) => {
                        dag.push(Node {
                            op: DagAdd as u32,
                            left_id: *stack_pos - 2,
                            right_id: *stack_pos - 1,
                            out: *stack_pos - 2,
                        });
                        *stack_pos -= 1;
                        Some((da.max(db), dep_a.max(dep_b + 1)))
                    }
                };
            }

            if is_one(a) {
                // x*1 + b = x + b
                let lhs_x = expr_compression_to_dag_helper(
                    dag, instance_scalar, challenges_offset, challenges,
                    constant_offset, constant, challenges_dedup, constant_dedup,
                    stack_pos, x,
                );
                let rhs_b = expr_compression_to_dag_helper(
                    dag, instance_scalar, challenges_offset, challenges,
                    constant_offset, constant, challenges_dedup, constant_dedup,
                    stack_pos, b,
                );

                return match (lhs_x, rhs_b) {
                    (None, None) => None,
                    (Some(x), None) | (None, Some(x)) => Some(x),
                    (Some((dx, dep_x)), Some((db, dep_b))) => {
                        dag.push(Node {
                            op: DagAdd as u32,
                            left_id: *stack_pos - 2,
                            right_id: *stack_pos - 1,
                            out: *stack_pos - 2,
                        });
                        *stack_pos -= 1;
                        Some((dx.max(db), dep_x.max(dep_b + 1)))
                    }
                };
            }

            if is_zero(b) {
                // general product without identities since x!=0, a!=0 here
                // x*a + 0 = x*a
                let lhs_x = expr_compression_to_dag_helper(
                    dag, instance_scalar, challenges_offset, challenges,
                    constant_offset, constant, challenges_dedup, constant_dedup,
                    stack_pos, x,
                );
                let lhs_a = expr_compression_to_dag_helper(
                    dag, instance_scalar, challenges_offset, challenges,
                    constant_offset, constant, challenges_dedup, constant_dedup,
                    stack_pos, a,
                );

                return match (lhs_x, lhs_a) {
                    (None, _) | (_, None) => None, // defensive
                    (Some((dx, dep_x)), Some((da, dep_a))) => {
                        dag.push(Node {
                            op: DagMul as u32,
                            left_id: *stack_pos - 2,
                            right_id: *stack_pos - 1,
                            out: *stack_pos - 2,
                        });
                        *stack_pos -= 1;
                        Some((dx + da, dep_x.max(dep_a + 1)))
                    }
                };
            }

            // General case: compute (x*a) then + b
            let lhs_x = expr_compression_to_dag_helper(
                dag, instance_scalar, challenges_offset, challenges,
                constant_offset, constant, challenges_dedup, constant_dedup,
                stack_pos, x,
            );
            let lhs_a = expr_compression_to_dag_helper(
                dag, instance_scalar, challenges_offset, challenges,
                constant_offset, constant, challenges_dedup, constant_dedup,
                stack_pos, a,
            );

            let mul = match (lhs_x, lhs_a) {
                (None, _) | (_, None) => None, // x or a simplified to 0 above; defensive
                (Some((dx, dep_x)), Some((da, dep_a))) => {
                    dag.push(Node {
                        op: DagMul as u32,
                        left_id: *stack_pos - 2,
                        right_id: *stack_pos - 1,
                        out: *stack_pos - 2,
                    });
                    *stack_pos -= 1;
                    Some((dx + da, dep_x.max(dep_a + 1)))
                }
            };

            let rhs_b = expr_compression_to_dag_helper(
                dag, instance_scalar, challenges_offset, challenges,
                constant_offset, constant, challenges_dedup, constant_dedup,
                stack_pos, b,
            );

            match (mul, rhs_b) {
                (None, None) => None,
                (Some(xa), None) | (None, Some(xa)) => Some(xa),
                (Some((dm, dep_m)), Some((db, dep_b))) => {
                    dag.push(Node {
                        op: DagAdd as u32,
                        left_id: *stack_pos - 2,
                        right_id: *stack_pos - 1,
                        out: *stack_pos - 2,
                    });
                    *stack_pos -= 1;
                    Some((dm.max(db), dep_m.max(dep_b + 1)))
                }
            }
        }
        c @ Expression::Challenge(_, _power, scalar, offset) => {
            if *scalar == E::ZERO && *offset == E::ZERO {
                return None
            }
            let challenge_id = *challenges_dedup.entry(c.clone()).or_insert_with(|| {
                challenges.push(c.clone());
                (challenges_offset + challenges.len() - 1) as u32
            });

            dag.push(Node {
                op: DagLoadScalar as u32,
                left_id: challenge_id,
                right_id: 0,
                out: *stack_pos,
            });
            *stack_pos += 1;
            Some((0, 1))
        }
    }
}

// trie
#[derive(Default)]
struct TrieNode {
    children: BTreeMap<u16, TrieNode>, // Sorted keys: commutative grouping
    scalar_indices: Vec<usize>,
}
pub fn build_factored_dag_commutative<E: ExtensionField>(
    terms: &[Term<Expression<E>, Expression<E>>],
    hint_shared_witin_lower_id: bool,
) -> (Vec<Node>, Vec<Expression<E>>, Option<u32>, u32) {
    let mut root = TrieNode::default();
    let mut scalars: Vec<Expression<E>> = Vec::new();

    // ---- Step 1: canonicalize products (commutative) ----
    for term in terms {
        let mut ids: Vec<u16> = term
            .product
            .iter()
            .filter_map(|e| match e {
                Expression::WitIn(id) => Some(*id),
                e => unimplemented!("unknown expression {e}"),
            })
            .collect();
        ids.sort(); // ensure a*b == b*a
        if !hint_shared_witin_lower_id {
            // witiness being shared will be made with larger id
            // so we build the prefix tree with larger id go first
            ids.reverse();
        }

        let mut cur = &mut root;
        for wid in ids {
            cur = cur.children.entry(wid).or_default();
        }

        let idx = scalars.len();
        scalars.push(term.scalar.clone());
        cur.scalar_indices.push(idx);
    }

    // ---- Step 2: emit DAG (stack semantics) ----
    let mut dag = Vec::new();
    let mut stack_top: u32 = 0;
    let mut max_stack_depth: u32 = 0;

    fn push(stack_top: &mut u32, max_depth: &mut u32) -> u32 {
        let out = *stack_top;
        *stack_top += 1;
        *max_depth = (*max_depth).max(*stack_top);
        out
    }

    fn pop2_push1(stack_top: &mut u32) -> (u32, u32, u32) {
        let left = *stack_top - 2;
        let right = *stack_top - 1;
        let out = left;
        *stack_top -= 1;
        (left, right, out)
    }

    fn emit<E: ExtensionField>(
        node: &TrieNode,
        dag: &mut Vec<Node>,
        stack_top: &mut u32,
        max_depth: &mut u32,
    ) -> Option<u32> {
        let mut acc_child: Option<u32> = None;

        // Recurse into children (witness factors)
        for (&wid, child) in &node.children {
            let child_out = emit::<E>(child, dag, stack_top, max_depth);

            // LOAD_WIT: push
            let out = push(stack_top, max_depth);
            dag.push(Node {
                op: DagLoadWit as u32,
                left_id: wid as u32,
                right_id: 0,
                out,
            });

            // If child exists, multiply with it
            if let Some(_) = child_out {
                let (left, right, out) = pop2_push1(stack_top);
                dag.push(Node {
                    op: DagMul as u32,
                    left_id: left,
                    right_id: right,
                    out,
                });
                acc_child = Some(match acc_child {
                    None => out,
                    Some(_) => {
                        let (l, r, out) = pop2_push1(stack_top);
                        dag.push(Node {
                            op: DagAdd as u32,
                            left_id: l,
                            right_id: r,
                            out,
                        });
                        out
                    }
                });
            } else {
                acc_child = Some(out);
            }
        }

        // Handle scalar accumulation at leaf
        let mut acc_scalar: Option<u32> = None;
        for &idx in &node.scalar_indices {
            let out = push(stack_top, max_depth);
            dag.push(Node {
                op: DagLoadScalar as u32,
                left_id: idx as u32,
                right_id: 0,
                out,
            });

            acc_scalar = Some(match acc_scalar {
                None => out,
                Some(_) => {
                    let (l, r, out) = pop2_push1(stack_top);
                    dag.push(Node {
                        op: DagAdd as u32,
                        left_id: l,
                        right_id: r,
                        out,
                    });
                    out
                }
            });
        }

        // Merge both child and scalar accumulations
        match (acc_scalar, acc_child) {
            (Some(_), Some(_)) => {
                let (l, r, out) = pop2_push1(stack_top);
                dag.push(Node {
                    op: DagAdd as u32,
                    left_id: l,
                    right_id: r,
                    out,
                });
                Some(out)
            }
            (Some(s), None) => Some(s),
            (None, Some(c)) => Some(c),
            (None, None) => None,
        }
    }

    let final_out = emit::<E>(&root, &mut dag, &mut stack_top, &mut max_stack_depth);
    (dag, scalars, final_out, max_stack_depth)
}
#[cfg(test)]
mod tests {
    use std::ops::Neg;
    use either::Either;
    use itertools::Itertools;
    use ff_ext::{BabyBearExt4, ExtensionField};
    use p3::babybear::BabyBear;
    use p3::field::FieldAlgebra;
    use crate::{power_sequence, Expression, Instance, ToExpr};
    use crate::utils::{build_factored_dag_commutative, expr_compression_to_dag, Node};

    type E = BabyBearExt4;
    type B = BabyBear;

    fn extract_num_add_mul<E: ExtensionField>(expr: &Expression<E>) -> (Vec<Node>, Vec<Instance>, Vec<Expression<E>>, Vec<Either<<E as ExtensionField>::BaseField, E>>, u32, (i32, i32), (usize, usize)) {
        let (
            dag,
            instance_scalar_expr,
            challenges_expr,
            constant_expr,
            stack_top,
            (max_degree, max_dag_depth),
        ) = expr_compression_to_dag(&expr);

        let mut num_add = 0;
        let mut num_mul = 0;

        for node in &dag {
            match node.op {
                0 => (), // skip wit index
                1 => (), // skip scalar index
                2 => {
                    num_add += 1;
                }
                3 => {
                    num_mul += 1;
                }
                op => panic!("unknown op {op}"),
            }
        }

        (
            dag,
            instance_scalar_expr,
            challenges_expr,
            constant_expr,
            stack_top,
            (num_add, num_mul),
            (max_degree, max_dag_depth),
        )
    }
    #[test]
    fn test_normal_expr_compression_to_dag_helper() {
        let a = Expression::<E>::WitIn(0);
        let b = Expression::<E>::WitIn(1);
        let s2 = Expression::<E>::Constant(Either::Left(B::from_canonical_u32(2)));
        let s3 = Expression::<E>::Constant(Either::Left(B::from_canonical_u32(3)));
        let s4 = Expression::<E>::Constant(Either::Left(B::from_canonical_u32(4)));

        let e: Expression<E> = s3.expr() * (s2.expr() * a.expr() * b.expr() + s4.expr());
        let (
            dag,
            instance_scalar_expr,
            challenges_expr,
            constant_expr,
            stack_top,
            (num_add, num_mul),
            (max_degree, max_dag_depth),
        ) = extract_num_add_mul(&e);

        assert_eq!(constant_expr.len(), 3);
        assert!(challenges_expr.is_empty());
        assert_eq!(num_add, 1);
        assert_eq!(num_mul, 3);
        assert_eq!(max_degree, 2);

    }

    #[test]
    fn test_challenge_expr_compression_to_dag_helper() {
        let a = Expression::<E>::WitIn(0);
        let b = Expression::<E>::WitIn(1);
        let c = Expression::<E>::Challenge(2, 1, E::ONE, E::ZERO);
        let alpha = Expression::<E>::Challenge(0, 1, E::ONE, E::ZERO);
        let pow_c = power_sequence(c);

        // alpha * (1 * a + c * b)
        // will be optimized as alpha * (a + c * b)
        let e: Expression<E> = vec![a.expr(), b.expr()].into_iter().zip(pow_c).map(|(e1, e2)| e1.expr()*e2.expr()).sum::<Expression<E>>();
        let e = e * alpha;
        let (
            dag,
            instance_scalar_expr,
            challenges_expr,
            constant_expr,
            stack_top,
            (num_add, num_mul),
            (max_degree, max_dag_depth),
        ) = extract_num_add_mul(&e);

        assert_eq!(constant_expr.len(), 0); // 1 was absorbed
        assert_eq!(challenges_expr.len(), 2);
        assert_eq!(num_add, 1);
        assert_eq!(num_mul, 2);
        assert_eq!(max_degree, 1);

    }

    #[test]
    fn test_build_factored_dag_commutative() {
        // w1 * (c2 * (2 + w0*c1 -1))
        let w0 = Expression::<E>::WitIn(0);
        let w1 = Expression::<E>::WitIn(1);
        let c1 = Expression::<E>::Challenge(0, 1, E::ONE, E::ZERO);
        let c2 = Expression::<E>::Challenge(2, 1, E::ONE, E::ZERO);
        let constant_2 = Expression::<E>::Constant(Either::Left(B::from_canonical_u32(2)));
        let constant_negative_1 = Expression::<E>::Constant(Either::Left(B::from_canonical_u32(1).neg()));

        let e: Expression<E> = w1.expr() * (c2.expr() * (constant_2.expr() + w0.expr() * c1.expr() - constant_negative_1.expr()));
        let e_monomials = e.get_monomial_terms();
        let (dag, coeffs, final_out, _)= build_factored_dag_commutative(&e_monomials);

        let mut num_add = 0;
        let mut num_mul = 0;

        for node in &dag {
            match node.op {
                0 => (), // skip wit index
                1 => (), // skip scalar index
                2 => {
                    num_add += 1;
                }
                3 => {
                    num_mul += 1;
                }
                op => panic!("unknown op {op}"),
            }
        }
        assert_eq!(num_add, 1);
        assert_eq!(num_mul, 3);
    }
}