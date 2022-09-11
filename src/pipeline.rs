use crate::{
    buffer::Buffer2d, math::WeightedSum, primitives::PrimitiveKind, rasterizer::Rasterizer,
    sampler::Linear, texture::Target, fixed_32::Fixed32,
};
use alloc::{collections::VecDeque, vec::Vec};
use core::{
    borrow::Borrow,
    cmp::Ordering,
    marker::PhantomData,
    ops::{Add, Mul, Range},
};

use fixed::traits::ToFixed;

#[cfg(feature = "micromath")]
use micromath_::F32Ext;

/// Defines how a [`Pipeline`] will interact with the depth target.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DepthMode {
    /// The test, if any, that occurs when comparing the depth of the new fragment with that of the current depth.
    pub test: Option<Ordering>,
    /// Whether the fragment's depth should be written to the depth target if the test was passed.
    pub write: bool,
}

impl DepthMode {
    pub const NONE: Self = Self {
        test: None,
        write: false,
    };

    pub const LESS_WRITE: Self = Self {
        test: Some(Ordering::Less),
        write: true,
    };

    pub const GREATER_WRITE: Self = Self {
        test: Some(Ordering::Greater),
        write: true,
    };

    pub const LESS_PASS: Self = Self {
        test: Some(Ordering::Less),
        write: false,
    };

    pub const GREATER_PASS: Self = Self {
        test: Some(Ordering::Greater),
        write: false,
    };
}

impl DepthMode {
    /// Determine whether the depth mode needs to interact with the depth target at all.
    pub fn uses_depth(&self) -> bool {
        self.test.is_some() || self.write
    }
}

/// Defines how a [`Pipeline`] will interact with the pixel target.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PixelMode {
    /// Whether the fragment's pixel should be written to the pixel target.
    pub write: bool,
}

impl PixelMode {
    pub const WRITE: Self = Self { write: true };

    pub const PASS: Self = Self { write: false };
}

impl Default for PixelMode {
    fn default() -> Self {
        Self::WRITE
    }
}

/// The handedness of the coordinate space used by a pipeline.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Handedness {
    /// Left-handed coordinate space (used by Vulkan and DirectX)
    Left,
    /// Right-handed coordinate space (used by OpenGL and Metal)
    Right,
}

/// The direction represented by +y in screen space.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum YAxisDirection {
    // +y points down towards the bottom of the screen (i.e: -y is up).
    Down,
    // +y points up towards the top of the screen (i.e: -y is down).
    Up,
}

/// The configuration of the coordinate system used by a pipeline.
pub struct CoordinateMode {
    pub handedness: Handedness,
    pub y_axis_direction: YAxisDirection,
    pub z_clip_range: Option<Range<Fixed32>>,
}

impl CoordinateMode {
    /// OpenGL-like coordinates (right-handed, y = up, -1 to 1 z clip range).
    pub const OPENGL: Self = Self {
        handedness: Handedness::Right,
        y_axis_direction: YAxisDirection::Up,
        z_clip_range: Some(Fixed32::unwrapped_from_str("-1")..Fixed32::unwrapped_from_str("1")),
    };

    /// Vulkan-like coordinates (left-handed, y = down, 0 to 1 z clip range).
    pub const VULKAN: Self = Self {
        handedness: Handedness::Left,
        y_axis_direction: YAxisDirection::Down,
        z_clip_range: Some(Fixed32::unwrapped_from_str("0")..Fixed32::unwrapped_from_str("1")),
    };

    /// Metal-like coordinates (right-handed, y = down, 0 to 1 z clip range).
    pub const METAL: Self = Self {
        handedness: Handedness::Right,
        y_axis_direction: YAxisDirection::Down,
        z_clip_range: Some(Fixed32::unwrapped_from_str("0")..Fixed32::unwrapped_from_str("1")),
    };

    /// DirectX-like coordinates (left-handed, y = up, 0 to 1 z clip range).
    pub const DIRECTX: Self = Self {
        handedness: Handedness::Left,
        y_axis_direction: YAxisDirection::Up,
        z_clip_range: Some(Fixed32::unwrapped_from_str("0")..Fixed32::unwrapped_from_str("1")),
    };

    pub fn without_z_clip(self) -> Self {
        Self {
            z_clip_range: None,
            ..self
        }
    }
}

impl Default for CoordinateMode {
    fn default() -> Self {
        Self::VULKAN
    }
}

/// Represents the high-level structure of a rendering pipeline.
///
/// Conventionally, uniform data is stores as state within the pipeline itself.
///
/// Additional methods such as [`Pipeline::depth_mode`], [Pipeline::`cull_mode`], etc. may be implemented to customize
/// the behaviour of the pipeline even further.
pub trait Pipeline: Sized {
    type Vertex;
    type VertexData: Clone + WeightedSum + Send + Sync;
    type Primitives: PrimitiveKind<Self::VertexData>;
    type Fragment: Clone + WeightedSum;
    type Pixel: Clone;

    /// Returns the [`PixelMode`] of this pipeline.
    #[inline(always)]
    fn pixel_mode(&self) -> PixelMode {
        PixelMode::default()
    }

    /// Returns the [`DepthMode`] of this pipeline.
    #[inline(always)]
    fn depth_mode(&self) -> DepthMode {
        DepthMode::NONE
    }

    /// Returns the [`CoordinateMode`] of this pipeline.
    #[inline(always)]
    fn coordinate_mode(&self) -> CoordinateMode {
        CoordinateMode::default()
    }

    /// Transforms a [`Pipeline::Vertex`] into homogeneous NDCs (Normalised Device Coordinates) for the vertex and a
    /// [`Pipeline::VertexData`] to be interpolated and passed to the fragment shader.
    ///
    /// This stage is executed at the beginning of pipeline execution.
    fn vertex_shader(&self, vertex: &Self::Vertex) -> ([Fixed32; 4], Self::VertexData);

    /// Turn a primitive into many primitives.
    ///
    /// This stage sits between the vertex shader and the fragment shader.
    #[inline(always)]
    fn geometry_shader<O>(
        &self,
        primitive: <Self::Primitives as PrimitiveKind<Self::VertexData>>::Primitive,
        mut output: O,
    ) where
        O: FnMut(<Self::Primitives as PrimitiveKind<Self::VertexData>>::Primitive),
    {
        output(primitive);
    }

    /// Transforms a [`Pipeline::VertexData`] into a fragment to be rendered to a pixel target.
    ///
    /// This stage is executed for every fragment generated by the rasterizer.
    fn fragment_shader(&self, vs_out: Self::VertexData) -> Self::Fragment;

    /// Blend an old fragment with a new fragment.
    ///
    /// This stage is executed after rasterization and defines how a fragment may be blended into an existing fragment
    /// from the pixel target.
    ///
    /// The default implementation simply returns the new fragment and ignores the old one. However, this may be used
    /// to implement techniques such as alpha blending.
    fn blend_shader(&self, old: Self::Pixel, new: Self::Fragment) -> Self::Pixel;

    /// Render a stream of vertices to given provided pixel target and depth target using the rasterizer.
    ///
    /// **Do not implement this method**
    fn render<S, V, P, D>(
        &self,
        vertices: S,
        rasterizer_config: <<Self::Primitives as PrimitiveKind<Self::VertexData>>::Rasterizer as Rasterizer>::Config,
        pixel: &mut P,
        depth: &mut D,
    ) where
        Self: Send + Sync,
        S: IntoIterator<Item = V>,
        V: Borrow<Self::Vertex>,
        P: Target<Texel = Self::Pixel> + Send + Sync,
        D: Target<Texel = Fixed32> + Send + Sync,
    {
        let target_size = match (self.pixel_mode().write, self.depth_mode().uses_depth()) {
            (false, false) => return, // No targets actually get written to, don't bother doing anything
            (true, false) => pixel.size(),
            (false, true) => depth.size(),
            (true, true) => {
                // Ensure that the pixel target and depth target are compatible
                assert_eq!(
                    pixel.size(),
                    depth.size(),
                    "Pixel target size is compatible with depth target size"
                );
                // Prefer
                pixel.size()
            }
        };

        // Produce an iterator over vertices (using the vertex shader and geometry shader to product them)
        let mut vert_outs = vertices
            .into_iter()
            .map(|v| self.vertex_shader(v.borrow()))
            .peekable();
        let mut vert_out_queue = VecDeque::new();
        let fetch_vertex = core::iter::from_fn(move || loop {
            match vert_out_queue.pop_front() {
                Some(v) => break Some(v),
                None if vert_outs.peek().is_none() => break None,
                None => {
                    let prim = Self::Primitives::collect_primitive(&mut vert_outs)?;
                    self.geometry_shader(prim, |prim| {
                        Self::Primitives::primitive_vertices(prim, |v| vert_out_queue.push_back(v))
                    });
                }
            }
        });

        #[cfg(not(feature = "par"))]
        let r = render_seq(
            self,
            fetch_vertex,
            rasterizer_config,
            target_size,
            pixel,
            depth,
        );
        r
    }
}

fn render_seq<Pipe, S, P, D>(
    pipeline: &Pipe,
    fetch_vertex: S,
    rasterizer_config: <<Pipe::Primitives as PrimitiveKind<Pipe::VertexData>>::Rasterizer as Rasterizer>::Config,
    tgt_size: [usize; 2],
    pixel: &mut P,
    depth: &mut D,
) where
    Pipe: Pipeline + Send + Sync,
    S: Iterator<Item = ([Fixed32; 4], Pipe::VertexData)>,
    P: Target<Texel = Pipe::Pixel> + Send + Sync,
    D: Target<Texel = Fixed32> + Send + Sync,
{
    // Safety: we have exclusive access to `pixel` and `depth`
    unsafe {
        render_inner(
            pipeline,
            fetch_vertex,
            rasterizer_config,
            ([0; 2], tgt_size),
            tgt_size,
            pixel,
            depth,
        )
    }
}

unsafe fn render_inner<Pipe, S, P, D>(
    pipeline: &Pipe,
    fetch_vertex: S,
    rasterizer_config: <<Pipe::Primitives as PrimitiveKind<Pipe::VertexData>>::Rasterizer as Rasterizer>::Config,
    (tgt_min, tgt_max): ([usize; 2], [usize; 2]),
    tgt_size: [usize; 2],
    pixel: &P,
    depth: &D,
) where
    Pipe: Pipeline + Send + Sync,
    S: Iterator<Item = ([Fixed32; 4], Pipe::VertexData)>,
    P: Target<Texel = Pipe::Pixel> + Send + Sync,
    D: Target<Texel = Fixed32> + Send + Sync,
{
    let write_pixels = pipeline.pixel_mode().write;
    let depth_mode = pipeline.depth_mode();
    for i in 0..2 {
        // Safety check
        if write_pixels {
            assert!(
                tgt_min[i] <= pixel.size()[i],
                "{}, {}, {}",
                i,
                tgt_min[i],
                pixel.size()[i]
            );
            assert!(
                tgt_max[i] <= pixel.size()[i],
                "{}, {}, {}",
                i,
                tgt_min[i],
                pixel.size()[i]
            );
        }
        if depth_mode.uses_depth() {
            assert!(
                tgt_min[i] <= depth.size()[i],
                "{}, {}, {}",
                i,
                tgt_min[i],
                depth.size()[i]
            );
            assert!(
                tgt_max[i] <= depth.size()[i],
                "{}, {}, {}",
                i,
                tgt_min[i],
                depth.size()[i]
            );
        }
    }

    let principal_x = depth.principal_axis() == 0;

    use crate::rasterizer::Blitter;

    struct BlitterImpl<'a, Pipe: Pipeline, P, D> {
        write_pixels: bool,
        depth_mode: DepthMode,

        tgt_min: [usize; 2],
        tgt_max: [usize; 2],
        tgt_size: [usize; 2],

        pipeline: &'a Pipe,
        pixel: &'a P,
        depth: &'a D,
        primitive_count: u64,
    }

    impl<'a, Pipe, P, D> Blitter<Pipe::VertexData> for BlitterImpl<'a, Pipe, P, D>
    where
        Pipe: Pipeline + Send + Sync,
        P: Target<Texel = Pipe::Pixel> + Send + Sync,
        D: Target<Texel = Fixed32> + Send + Sync,
    {
        fn target_size(&self) -> [usize; 2] {
            self.tgt_size
        }
        fn target_min(&self) -> [usize; 2] {
            self.tgt_min
        }
        fn target_max(&self) -> [usize; 2] {
            self.tgt_max
        }

        #[inline(always)]
        fn begin_primitive(&mut self) {
            self.primitive_count = self.primitive_count.wrapping_add(1);
        }

        #[inline(always)]
        unsafe fn test_fragment(&mut self, pos: [usize; 2], z: Fixed32) -> bool {
            if let Some(test) = self.depth_mode.test {
                let old_z = self.depth.read_exclusive_unchecked(pos);
                z.partial_cmp(&old_z) == Some(test)
            } else {
                true
            }
        }

        #[inline(always)]
        unsafe fn emit_fragment<F: FnMut([Fixed32; 2]) -> Pipe::VertexData>(
            &mut self,
            pos: [usize; 2],
            mut get_v_data: F,
            z: Fixed32,
        ) {
            if self.depth_mode.write {
                self.depth.write_exclusive_unchecked(pos, z);
            }

            if self.write_pixels {
                let frag = self
                    .pipeline
                    .fragment_shader(get_v_data([pos[0].to_fixed(), pos[1].to_fixed()]));
                let old_px = self.pixel.read_exclusive_unchecked(pos);
                let blended_px = self.pipeline.blend_shader(old_px, frag);
                self.pixel.write_exclusive_unchecked(pos, blended_px);
            }
        }
    }

    <Pipe::Primitives as PrimitiveKind<Pipe::VertexData>>::Rasterizer::default().rasterize(
        fetch_vertex,
        principal_x,
        pipeline.coordinate_mode(),
        rasterizer_config,
        BlitterImpl {
            write_pixels,
            depth_mode,

            tgt_size,
            tgt_min,
            tgt_max,

            pipeline,
            pixel,
            depth,
            primitive_count: 0,
        },
    );
}
