//! GPU renderer: wgpu 24 surface + single R8 glyph-atlas pipeline.
//! Text path: rustybuzz (HarfBuzz port) shapes → ab_glyph rasterizes into the
//! atlas → quads land in one vertex buffer → GPU draws. Per-line quad cache
//! keyed by (line, revision): scrolling repacks cached quads + a uniform
//! translate — no re-shaping, no I/O, no allocation growth.

use std::collections::HashMap;
use std::sync::Arc;

use ab_glyph::{Font as _, FontArc, Glyph, PxScale};
use rustybuzz::UnicodeBuffer;
use winit::window::Window;

use crate::syntax::{Kind, Syntax};

// ------------------------------------------------------------- palette ----
pub const BG: [f32; 4] = rgb(0x0a, 0x12, 0x0e);
pub const TEXT: [f32; 4] = rgb(0xd7, 0xe8, 0xdf);
pub const DIM: [f32; 4] = rgb(0x45, 0x60, 0x53);
pub const GREEN: [f32; 4] = rgb(0x22, 0xc5, 0x5e);
pub const COMMENT: [f32; 4] = rgb(0x5c, 0x7a, 0x69);
pub const STRING_C: [f32; 4] = rgb(0xa9, 0xdc, 0x76);
pub const NUMBER_C: [f32; 4] = rgb(0xab, 0x9d, 0xf2);
pub const KEYWORD_C: [f32; 4] = rgb(0xff, 0x8b, 0x75);
pub const FUNC_C: [f32; 4] = rgb(0x78, 0xdc, 0xe8);
pub const TYPE_C: [f32; 4] = rgb(0xf5, 0xc5, 0x42);
pub const PANEL: [f32; 4] = rgb(0x12, 0x21, 0x1a);
pub const CURSOR_C: [f32; 4] = rgb(0x22, 0xc5, 0x5e);
pub const DIAG_C: [f32; 4] = rgb(0xff, 0x8b, 0x75);
pub const SEARCH_C: [f32; 4] = rgb(0x1e, 0x3f, 0x30);

const fn rgb(r: u8, g: u8, b: u8) -> [f32; 4] {
    [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]
}

pub fn kind_color(k: Kind) -> [f32; 4] {
    match k {
        Kind::Text => TEXT,
        Kind::Comment => COMMENT,
        Kind::String => STRING_C,
        Kind::Number => NUMBER_C,
        Kind::Keyword => KEYWORD_C,
        Kind::Function => FUNC_C,
        Kind::Type => TYPE_C,
        Kind::Punct => TEXT,
    }
}

// ------------------------------------------------------------- vertices ---
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

pub fn quad(
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    u0: f32,
    v0: f32,
    u1: f32,
    v1: f32,
    c: [f32; 4],
) -> [Vertex; 6] {
    let (x0, y0, x1, y1) = (x, y, x + w, y + h);
    [
        Vertex { pos: [x0, y0], uv: [u0, v0], color: c },
        Vertex { pos: [x1, y0], uv: [u1, v0], color: c },
        Vertex { pos: [x1, y1], uv: [u1, v1], color: c },
        Vertex { pos: [x0, y0], uv: [u0, v0], color: c },
        Vertex { pos: [x1, y1], uv: [u1, v1], color: c },
        Vertex { pos: [x0, y1], uv: [u0, v1], color: c },
    ]
}

const WHITE_U: f32 = 0.5 / 2048.0; // center of texel (0,0) — the white pixel
const WHITE_V: f32 = 0.5 / 2048.0;

/// solid rect using the atlas' white texel
pub fn rect(x: f32, y: f32, w: f32, h: f32, c: [f32; 4]) -> [Vertex; 6] {
    quad(x, y, w, h, WHITE_U, WHITE_V, WHITE_U, WHITE_V, c)
}

const WGSL: &str = r#"
struct U { off: vec2<f32>, res: vec2<f32> };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VsIn {
  @location(0) pos: vec2<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) color: vec4<f32>,
};
struct VsOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec4<f32>,
};

@vertex fn vs_main(v: VsIn) -> VsOut {
  var out: VsOut;
  let p = (v.pos - u.off) / u.res;
  out.clip = vec4<f32>(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
  out.uv = v.uv;
  out.color = v.color;
  return out;
}

@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let a = textureSample(atlas, samp, in.uv).r;
  return vec4<f32>(in.color.rgb, in.color.a * a);
}
"#;

// ------------------------------------------------------------- atlas ------
#[derive(Clone)]
struct AtlasEntry {
    u0: f32,
    v0: f32,
    u1: f32,
    v1: f32,
    w: f32,
    h: f32,
    left: f32,
    top: f32,
}

struct Atlas {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    size: u32,
    sx: u32,
    sy: u32,
    sh: u32,
    cache: HashMap<(u16, u16), AtlasEntry>,
}

impl Atlas {
    fn create(device: &wgpu::Device, queue: &wgpu::Queue, size: u32) -> Self {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("glyph-atlas"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(1),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let view = tex.create_view(&Default::default());
        Self {
            tex,
            view,
            size,
            sx: 2,
            sy: 2,
            sh: 0,
            cache: HashMap::new(),
        }
    }

    fn put(
        &mut self,
        queue: &wgpu::Queue,
        bitmap: &[u8],
        bw: u32,
        bh: u32,
        left: f32,
        top: f32,
    ) -> Option<AtlasEntry> {
        if bw == 0 || bh == 0 {
            return None;
        }
        if self.sx + bw + 2 > self.size {
            self.sx = 2;
            self.sy += self.sh + 2;
            self.sh = 0;
        }
        if self.sy + bh + 2 > self.size {
            return None; // full — glyph falls back to next frame attempt
        }
        let (x, y) = (self.sx, self.sy);
        self.sx += bw + 2;
        self.sh = self.sh.max(bh);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            bitmap,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bw),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: bw,
                height: bh,
                depth_or_array_layers: 1,
            },
        );
        let s = self.size as f32;
        Some(AtlasEntry {
            u0: x as f32 / s,
            v0: y as f32 / s,
            u1: (x + bw) as f32 / s,
            v1: (y + bh) as f32 / s,
            w: bw as f32,
            h: bh as f32,
            left,
            top,
        })
    }
}

// ----------------------------------------------------------- metrics -----
pub struct Metrics {
    pub line_h: f32,
    /// monospace advance as an em fraction (0..1)
    pub advance: f32,
    pub base_px: f32,
    /// font units per em — rustybuzz emits advances in font units
    pub upem: f32,
}

/// Screen-space text drawn through the same glyph pipeline.
pub struct OverlayText {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub color_override: Option<[f32; 4]>,
}

// ----------------------------------------------------------- renderer -----
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    active: bool,
    pipeline: wgpu::RenderPipeline,
    ubuf: wgpu::Buffer,
    bind: wgpu::BindGroup,
    vbuf: wgpu::Buffer,
    vcap: usize,
    atlas: Atlas,
    font_ab: FontArc,
    face_bytes: Arc<Vec<u8>>,
    line_cache: HashMap<(usize, u64), Vec<Vertex>>,
    cached_rev: u64,
    pub metrics: Metrics,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    off: [f32; 2],
    res: [f32; 2],
}

fn load_font() -> Vec<u8> {
    for cand in [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ] {
        if let Ok(b) = std::fs::read(cand) {
            return b;
        }
    }
    panic!("no font found");
}

impl Renderer {
    pub fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            },
        ))
        .expect("adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("vortexedit"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: Vec::new(),
        };
        surface.configure(&device, &config);

        let bytes = Arc::new(load_font());
        let font_ab = FontArc::try_from_vec(bytes.as_ref().clone()).expect("ab_glyph font");
        let face = rustybuzz::Face::from_slice(bytes.as_ref(), 0).expect("rustybuzz face");
        let mut buf = UnicodeBuffer::new();
        buf.push_str("M");
        let shaped = rustybuzz::shape(&face, &[], buf);
        let upem = face.units_per_em() as f32;
        let probe = shaped
            .glyph_positions()
            .first()
            .map(|p| p.x_advance as f32 / upem)
            .filter(|a| a.is_finite() && *a > 0.05)
            .unwrap_or(0.6); // fraction of em per char (monospace)
        let metrics = Metrics {
            line_h: 1.55,
            advance: probe, // em fraction — multiply by px in layout
            base_px: 15.0,
            upem,
        };

        let atlas = Atlas::create(&device, &queue, 2048);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("text"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let smp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atlas-smp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("text-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: true,
                        },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("text-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&atlas.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&smp),
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("text-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("text"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2,
                        1 => Float32x2,
                        2 => Float32x4
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let vcap = 1 << 16;
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("verts"),
            size: (vcap * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            surface,
            config,
            active: true,
            pipeline,
            ubuf,
            bind,
            vbuf,
            vcap,
            atlas,
            font_ab,
            face_bytes: bytes,
            line_cache: HashMap::new(),
            cached_rev: 0,
            metrics,
        }
    }

    /// Drop the line quad cache (call when syntax state changes).
    pub fn invalidate(&mut self) {
        self.line_cache.clear();
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            self.active = false;
            return;
        }
        self.active = true;
        if self.config.width != w || self.config.height != h {
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    fn glyph(&mut self, id: u16, px: u16) -> Option<AtlasEntry> {
        let key = (id, px);
        if let Some(e) = self.atlas.cache.get(&key) {
            return Some(e.clone());
        }
        let scale = PxScale::from(px as f32);
        let glyph = Glyph {
            id: ab_glyph::GlyphId(id),
            scale,
            position: ab_glyph::point(0.0, 0.0),
        };
        let outlined = self.font_ab.outline_glyph(glyph)?;
        let bounds = outlined.px_bounds();
        let (bw, bh) = (bounds.width().ceil() as u32, bounds.height().ceil() as u32);
        if bw == 0 || bh == 0 {
            return None;
        }
        let mut bitmap = vec![0u8; (bw * bh) as usize];
        outlined.draw(|x, y, c| {
            let i = (y * bw + x) as usize;
            if i < bitmap.len() {
                bitmap[i] = (c * 255.0) as u8;
            }
        });
        let entry = self
            .atlas
            .put(&self.queue, &bitmap, bw, bh, bounds.min.x, bounds.min.y)?;
        self.atlas.cache.insert(key, entry.clone());
        Some(entry)
    }

    /// Build all vertices for one frame.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        &mut self,
        ed: &crate::buffer::Editor,
        syn: Option<&Syntax>,
        scroll: f64,
        scale: f64,
        caret_on: bool,
        search_hits: &[(usize, usize)],
        diagnostics: &[(usize, usize)],
        overlays: &[OverlayText],
        panels: &[(f32, f32, f32, f32, [f32; 4])],
        underlays: &[(f32, f32, f32, f32, [f32; 4])],
    ) -> Vec<Vertex> {
        let mut out: Vec<Vertex> = Vec::with_capacity(16384);
        let h = self.config.height as f32;
        let px = (self.metrics.base_px * scale as f32).round().max(8.0) as u16;
        let em_advance = self.metrics.advance * px as f32;
        let line_h = px as f32 * self.metrics.line_h;

        if self.cached_rev != ed.revision {
            self.line_cache.clear();
            self.cached_rev = ed.revision;
        }

        for &(x, y, ww, hh, c) in underlays {
            out.extend(rect(x, y, ww, hh, c));
        }

        let first_line = (scroll / line_h as f64).floor().max(0.0) as usize;
        let visible = ((h / line_h) as usize) + 2;
        let last_line = (first_line + visible).min(ed.line_count());

        let v_start = ed.line_range(first_line).map(|r| r.0).unwrap_or(0);
        let v_end = ed
            .line_range(last_line.saturating_sub(1))
            .map(|r| r.1)
            .unwrap_or(ed.rope.len_chars());
        let spans: Vec<(usize, usize, Kind)> = syn
            .map(|s| s.highlights(&ed.rope, v_start, v_end))
            .unwrap_or_default();

        for line in first_line..last_line {
            let Some((ls, le)) = ed.line_range(line) else { continue };
            let y = line as f32 * line_h;
            let cached = match self.line_cache.get(&(line, ed.revision)) {
                Some(v) => v.clone(),
                None => {
                    let text = ed.text_slice(ls, le);
                    let text = text.trim_end_matches(['\r', '\n']);
                    let v = self.shape_line(text, &spans, ls, 0.0, y, px);
                    self.line_cache.insert((line, ed.revision), v.clone());
                    out.extend(v);
                    continue;
                }
            };
            out.extend(cached);
        }

        // search hit underlines
        for &(a, b) in search_hits {
            let la = ed.rope.try_char_to_line(a).unwrap_or(0);
            if la < first_line || la >= last_line {
                continue;
            }
            let (ls, lend) = match ed.line_range(la) {
                Some(r) => r,
                None => continue,
            };
            let x0 = (a.max(ls) - ls) as f32 * em_advance;
            let x1 = (b.min(lend) - ls) as f32 * em_advance;
            out.extend(rect(
                x0,
                la as f32 * line_h + line_h - 2.0,
                (x1 - x0).max(2.0),
                2.0,
                GREEN,
            ));
        }
        // diagnostic underlines
        for &(a, b) in diagnostics {
            let la = ed.rope.try_char_to_line(a).unwrap_or(0);
            if la < first_line || la >= last_line {
                continue;
            }
            let (ls, lend) = match ed.line_range(la) {
                Some(r) => r,
                None => continue,
            };
            let x0 = (a.min(lend) - ls) as f32 * em_advance;
            let x1 = (b.min(lend) - ls) as f32 * em_advance;
            out.extend(rect(
                x0,
                la as f32 * line_h + line_h - 2.0,
                (x1 - x0).max(em_advance),
                2.0,
                DIAG_C,
            ));
        }

        // block cursors (GPU-side blink alpha)
        let alpha = if caret_on { 0.85 } else { 0.28 };
        for &c in &ed.cursors {
            let line = ed.rope.try_char_to_line(c).unwrap_or(0);
            if line < first_line || line >= last_line {
                continue;
            }
            let ls = ed.line_range(line).map(|r| r.0).unwrap_or(0);
            let x = (c - ls) as f32 * em_advance;
            let mut col = CURSOR_C;
            col[3] = alpha;
            out.extend(rect(
                x,
                line as f32 * line_h + 1.0,
                (em_advance * 0.9).max(2.0),
                line_h - 2.0,
                col,
            ));
        }

        for &(x, y, ww, hh, c) in panels {
            out.extend(rect(x, y, ww, hh, c));
        }

        for o in overlays {
            let mut y = o.y;
            let mut start_x = o.x;
            for (i, seg) in o.text.split('\n').enumerate() {
                if i > 0 {
                    y += line_h;
                }
                let v = self.shape_line(seg, &[], 0, start_x, y, px);
                let tinted: Vec<Vertex> = v
                    .into_iter()
                    .map(|mut vt| {
                        if let Some(c) = o.color_override {
                            vt.color = c;
                        }
                        vt
                    })
                    .collect();
                out.extend(tinted);
                start_x = o.x;
            }
        }

        out
    }

    fn shape_line(
        &mut self,
        text: &str,
        spans: &[(usize, usize, Kind)],
        char_base: usize,
        x: f32,
        y: f32,
        px: u16,
    ) -> Vec<Vertex> {
        if text.is_empty() {
            return Vec::new();
        }
        let face = rustybuzz::Face::from_slice(&self.face_bytes, 0).expect("face");
        let mut buf = UnicodeBuffer::new();
        buf.push_str(text);
        let shaped = rustybuzz::shape(&face, &[], buf);
        let infos = shaped.glyph_infos();
        let positions = shaped.glyph_positions();

        let color_for = |ci: usize| {
            let abs = char_base + ci;
            for (s, e, k) in spans {
                if abs >= *s && abs < *e {
                    return kind_color(*k);
                }
            }
            TEXT
        };

        let mut out = Vec::with_capacity(text.len() * 6);
        let mut pen = x;
        let mut prev_cluster = usize::MAX;
        let mut ci = 0usize;
        for (info, pos) in infos.iter().zip(positions.iter()) {
            let cluster = info.cluster as usize;
            if cluster != prev_cluster {
                // cluster = byte offset in line; recompute char index
                ci = text[..cluster.min(text.len())].chars().count();
                prev_cluster = cluster;
            }
            let entry = self.glyph(info.glyph_id as u16, px);
            if let Some(e) = entry {
                let gx = pen + e.left;
                let gy = y + e.top;
                out.extend(quad(
                    gx,
                    gy,
                    e.w,
                    e.h,
                    e.u0,
                    e.v0,
                    e.u1,
                    e.v1,
                    color_for(ci),
                ));
            }
            let px_adv = pos.x_advance as f32 / self.metrics.upem * px as f32;
            pen += if px_adv > 0.0 {
                px_adv
            } else {
                self.metrics.advance * px as f32
            };
        }
        out
    }

    pub fn render(&mut self, verts: &[Vertex], scroll: f32) {
        if !self.active || verts.is_empty() {
            return;
        }
        if verts.len() > self.vcap {
            let new_cap = verts.len().next_power_of_two();
            self.vbuf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("verts"),
                size: (new_cap * std::mem::size_of::<Vertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.vcap = new_cap;
        }
        self.queue
            .write_buffer(&self.vbuf, 0, bytemuck::cast_slice(verts));
        self.queue.write_buffer(
            &self.ubuf,
            0,
            bytemuck::bytes_of(&Uniforms {
                off: [0.0, scroll],
                res: [self.config.width as f32, self.config.height as f32],
            }),
        );

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                self.surface.configure(&self.device, &self.config);
                match self.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(_) => return,
                }
            }
            Err(_) => return,
        };
        let view = frame.texture.create_view(&Default::default());
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("text-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: BG[0] as f64,
                            g: BG[1] as f64,
                            b: BG[2] as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.set_vertex_buffer(0, self.vbuf.slice(..));
            pass.draw(0..verts.len() as u32, 0..1);
        }
        self.queue.submit(std::iter::once(enc.finish()));
        frame.present();
    }
}
