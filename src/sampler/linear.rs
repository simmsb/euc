use crate::fixed_32::Fixed32;

use super::*;
use core::{
    ops::{Add, Mul},
    marker::PhantomData,
};

use fixed::traits::ToFixed;
use fixed::traits::FromFixed;
#[cfg(feature = "micromath")]
use micromath_::F32Ext;

/// A sampler that uses nearest-neighbor sampling.
pub struct Linear<T, I = Fixed32>(T, PhantomData<I>);

impl<T, I> Linear<T, I> {
    /// Create a new
    pub fn new(texture: T) -> Self {
        Self(texture, PhantomData)
    }
}

impl<'a, T> Sampler<2> for Linear<T, Fixed32>
where
    T: Texture<2, Index = usize>,
    T::Texel: Mul<Fixed32, Output = T::Texel> + Add<Output = T::Texel> + ToFixed,
{
    type Index = Fixed32;

    type Sample = T::Texel;

    type Texture = T;

    #[inline(always)]
    fn raw_texture(&self) -> &Self::Texture { &self.0 }

    #[inline(always)]
    fn sample(&self, mut index: [Self::Index; 2]) -> Self::Sample {
        assert!(index[0] <= 1.0, "{:?}", index);
        assert!(index[1] <= 1.0, "{:?}", index);

        let size = self.raw_texture().size();
        let size_f32: [Fixed32; 2] = size.map(|e| e.to_fixed());
        // Index in texture coordinates
        let index_tex: [Fixed32; 2] = [index[0].frac() * size_f32[0], index[1].frac() * size_f32[1]];
        // Find texel sample coordinates
        let posi: [usize; 2] = index_tex.map(|e| usize::from_fixed(e.int()));
        // Find interpolation values
        let fract: [usize; 2] = index_tex.map(|e| usize::from_fixed(e.frac()));

        assert!(posi[0] < size[0], "pos: {:?}, sz: {:?}, idx: {:?}", posi, size, index);
        assert!(posi[1] < size[1], "pos: {:?}, sz: {:?}, idx: {:?}", posi, size, index);

        let t00: Self::Sample = self.raw_texture().read([(posi[0] + 0).min(size[0] - 1), (posi[1] + 0).min(size[1] - 1)]);
        let t10: Self::Sample = self.raw_texture().read([(posi[0] + 1).min(size[0] - 1), (posi[1] + 0).min(size[1] - 1)]);
        let t01: Self::Sample = self.raw_texture().read([(posi[0] + 0).min(size[0] - 1), (posi[1] + 1).min(size[1] - 1)]);
        let t11: Self::Sample = self.raw_texture().read([(posi[0] + 1).min(size[0] - 1), (posi[1] + 1).min(size[1] - 1)]);

        let t0 = t00 * (1 - fract[1]).to_fixed() + t01 * fract[1].to_fixed();
        let t1 = t10 * (1 - fract[1]).to_fixed() + t11 * fract[1].to_fixed();

        let t = t0 * (1 - fract[0]).to_fixed() + t1 * fract[0].to_fixed();

        t
    }

    #[inline(always)]
    unsafe fn sample_unchecked(&self, index: [Self::Index; 2]) -> Self::Sample {
        // TODO: Not this
        self.sample(index)
    }
}
