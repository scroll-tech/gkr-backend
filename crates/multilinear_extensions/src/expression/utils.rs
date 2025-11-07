use super::{Expression, StructuralWitIn, WitIn};
use crate::{Fixed, Instance, WitnessId, combine_cumulative_either, monomial::Term};
use either::Either;
use ff_ext::ExtensionField;
use itertools::Itertools;

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

pub fn expr_compression_to_dag<E: ExtensionField>(
    expr: &Expression<E>,
) -> (
    Vec<u32>,
    Vec<Instance>,
    Vec<Expression<E>>,
    Vec<Either<E::BaseField, E>>,
    (usize, usize)
) {
    let mut dag = vec![];
    let mut constant = vec![];
    let mut instance_scalar = vec![];
    let mut challenges = vec![];
    // traverse first time to collect offset
    let _ = expr_compression_to_dag_helper(
        &mut dag,
        &mut instance_scalar,
        0,
        &mut challenges,
        0,
        &mut constant,
        expr,
    );

    let challenge_offset = instance_scalar.len();
    let constant_offset = instance_scalar.len() + challenges.len();

    dag.truncate(0);
    constant.truncate(0);
    instance_scalar.truncate(0);
    challenges.truncate(0);
    let (max_degree, max_depth) = expr_compression_to_dag_helper(
        &mut dag,
        &mut instance_scalar,
        challenge_offset,
        &mut challenges,
        constant_offset,
        &mut constant,
        expr,
    );
    (dag, instance_scalar, challenges, constant, (max_degree, max_depth))
}

fn expr_compression_to_dag_helper<E: ExtensionField>(
    dag: &mut Vec<u32>,
    instance_scalar: &mut Vec<Instance>,
    challenges_offset: usize,
    challenges: &mut Vec<Expression<E>>,
    constant_offset: usize,
    constant: &mut Vec<Either<E::BaseField, E>>,
    expr: &Expression<E>,
) -> (usize, usize) {
    // (max_degree, max_depth)
    match expr {
        Expression::Fixed(_) => unimplemented!(),
        Expression::WitIn(wit_id) => {
            dag.extend(vec![DagLoadWit as u32, *wit_id as u32]);
            (1, 1)
        }
        Expression::StructuralWitIn(_, ..) => unimplemented!(),
        Expression::Instance(_) => unimplemented!(),
        Expression::InstanceScalar(inst) => {
            instance_scalar.push(inst.clone());
            dag.extend(vec![DagLoadScalar as u32, instance_scalar.len() as u32 - 1]);
            (0, 1)
        }
        Expression::Constant(value) => {
            constant.push(value.clone());
            dag.extend(vec![
                DagLoadScalar as u32,
                (constant_offset + constant.len()) as u32 - 1,
            ]);
            (0, 1)
        }
        Expression::Sum(a, b) => {
            let (max_degree_a, max_depth_a) = expr_compression_to_dag_helper(
                dag,
                instance_scalar,
                challenges_offset,
                challenges,
                constant_offset,
                constant,
                a,
            );
            let (max_degree_b, max_depth_b) = expr_compression_to_dag_helper(
                dag,
                instance_scalar,
                challenges_offset,
                challenges,
                constant_offset,
                constant,
                b,
            );
            dag.extend(vec![DagAdd as u32]);
            (
                max_degree_a.max(max_degree_b),
                max_depth_a.max(max_depth_b + 1),
            ) // 1 comes from store result of `a`
        }
        Expression::Product(a, b) => {
            let (max_degree_a, max_depth_a) = expr_compression_to_dag_helper(
                dag,
                instance_scalar,
                challenges_offset,
                challenges,
                constant_offset,
                constant,
                a,
            );
            let (max_degree_b, max_depth_b) = expr_compression_to_dag_helper(
                dag,
                instance_scalar,
                challenges_offset,
                challenges,
                constant_offset,
                constant,
                b,
            );
            dag.extend(vec![DagMul as u32]);
            (
                max_degree_a + max_degree_b,
                max_depth_a.max(max_depth_b + 1),
            ) // 1 comes from store result of `a`
        }
        Expression::ScaledSum(x, a, b) => {
            let (max_degree_x, max_depth_x) = expr_compression_to_dag_helper(
                dag,
                instance_scalar,
                challenges_offset,
                challenges,
                constant_offset,
                constant,
                x,
            );
            let (max_degree_a, max_depth_a) = expr_compression_to_dag_helper(
                dag,
                instance_scalar,
                challenges_offset,
                challenges,
                constant_offset,
                constant,
                a,
            );
            let xa_degree = max_degree_x + max_degree_a;
            let ax_max_depth = max_depth_x.max(max_depth_a + 1);
            dag.extend(vec![DagMul as u32]);
            let (max_degree_b, max_depth_b) = expr_compression_to_dag_helper(
                dag,
                instance_scalar,
                challenges_offset,
                challenges,
                constant_offset,
                constant,
                b,
            );
            dag.extend(vec![DagAdd as u32]);
            (
                xa_degree.max(max_degree_b),
                (ax_max_depth).max(max_depth_b + 1),
            ) // 1 comes from store result of `ax`
        }
        c @ Expression::Challenge(..) => {
            challenges.push(c.clone());
            dag.extend(vec![
                DagLoadScalar as u32,
                (challenges_offset + challenges.len()) as u32 - 1,
            ]);
            (0, 1)
        }
    }
}
