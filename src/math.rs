use core::ops::{Mul, Add};

use fixed::traits::ToFixed;
use fixed::traits::FromFixed;
use crate::fixed_32::Fixed32;

pub trait WeightedSum: Sized {
    fn weighted_sum(values: &[Self], weights: &[Fixed32]) -> Self;
}

#[derive(Copy, Clone)]
pub struct Unit;

impl WeightedSum for Unit {
    fn weighted_sum(_: &[Self], _: &[Fixed32]) -> Self { Unit }
}

impl<T: Clone + Mul<Fixed32, Output = T> + Add<Output = T>> WeightedSum for T {
    #[inline(always)]
    fn weighted_sum(values: &[Self], weights: &[Fixed32]) -> Self {
        values[1..].iter().zip(weights[1..].iter()).fold(values[0].clone() * weights[0], |a, (x, w)| a + x.clone() * *w)
    }
}

pub trait Denormalize<T>: Sized {
    fn denormalize_to(self, scale: T) -> T;
    fn denormalize_array<const N: usize>(this: [Self; N], other: [T; N]) -> [T; N];
}

macro_rules! impl_denormalize {
    ($this:ty, $other:ty) => {
        impl Denormalize<$other> for $this {
            fn denormalize_to(self, scale: $other) -> $other {
                <$other>::from_fixed((self.wrapping_mul(scale.to_fixed::<$this>())).max(<$this>::ZERO).min((scale - 1).to_fixed::<$this>()))
            }

            fn denormalize_array<const N: usize>(this: [Self; N], other: [$other; N]) -> [$other; N] {
                let mut out = [0; N];
                (0..N).for_each(|i| out[i] = this[i].denormalize_to(other[i]));
                out
            }
        }
    };
}

impl_denormalize!(Fixed32, u8);
impl_denormalize!(Fixed32, u16);
impl_denormalize!(Fixed32, u32);
impl_denormalize!(Fixed32, u64);
impl_denormalize!(Fixed32, u128);
impl_denormalize!(Fixed32, usize);
