use p3::symmetric::{CryptographicPermutation, Permutation};
use std::array::from_fn;
#[derive(Clone)]
pub struct Wrapper<P: Clone, const N: usize> {
    inner: P,
}

impl<P: Clone, const N: usize> Wrapper<P, N> {
    pub fn new(inner: P) -> Self {
        Self { inner }
    }
}

impl<const N: usize, T: Clone, P: Permutation<[T; N]> + Clone> Permutation<Vec<T>>
    for Wrapper<P, N>
{
    fn permute(&self, input: Vec<T>) -> Vec<T> {
        assert_eq!(input.len(), N, "Input vector must be of length {}", N);

        let mut array = from_fn(|i| input[i].clone());
        self.inner.permute_mut(&mut array);

        array.to_vec()
    }

    fn permute_mut(&self, input: &mut Vec<T>) {
        assert_eq!(input.len(), N, "Input vector must be of length {}", N);
        let mut array = from_fn(|i| input[i].clone());
        self.inner.permute_mut(&mut array);
        for i in 0..N {
            input[i] = array[i].clone();
        }
    }
}
impl<const N: usize, T: Clone, P: Permutation<[T; N]> + Clone> Permutation<[T; N]>
    for Wrapper<P, N>
{
    fn permute(&self, input: [T; N]) -> [T; N] {
        self.inner.permute(input)
    }

    fn permute_mut(&self, input: &mut [T; N]) {
        self.inner.permute_mut(input);
    }
}

impl<const N: usize, T: Clone, P: Permutation<[T; N]> + Clone> CryptographicPermutation<[T; N]>
    for Wrapper<P, N>
{
}
