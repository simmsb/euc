pub type Fixed32 = fixed::FixedI32<16>;

#[derive(Clone, Copy)]
pub struct ViaFixed32(pub Fixed32);

impl From<Fixed32> for ViaFixed32 {
    fn from(inner: Fixed32) -> Self {
        Self(inner)
    }
}

impl From<ViaFixed32> for Fixed32 {
    fn from(inner: ViaFixed32) -> Self {
        inner.0
    }
}

impl vek::ops::Clamp<Fixed32> for ViaFixed32 {
    fn clamped(self, lower: Fixed32, upper: Fixed32) -> Self {
        Self(self.0.clamp(lower, upper))
    }
}

impl vek::ops::Clamp for ViaFixed32 {
    fn clamped(self, lower: Self, upper: Self) -> Self {
        Self(self.0.clamp(lower.0, upper.0))
    }
}
