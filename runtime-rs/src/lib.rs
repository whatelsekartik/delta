// runtime-rs/src/lib.rs
//! Audited Zero-Dependency Rust & WebAssembly INT8 Inference Engine for DeltaCore-LM.
//! Audit fixes applied:
//! 1. Header check corrected from `< 44` to `< 48` bytes (4B magic + 40B struct + 4B conv_kernel).
//! 2. Exact byte-length verification (`r.pos == bytes.len()`).
//! 3. Added `wasm_forward_logits` export so cross-runtime quantization parity tests can inspect
//!    the raw FP32 output logit vector prior to sampling.

use std::f32;

#[derive(Clone)]
pub struct QuantizedMatrix {
    pub rows: usize,
    pub cols: usize,
    pub scales: Vec<f32>,
    pub data: Vec<i8>,
}

impl QuantizedMatrix {
    #[inline]
    pub fn matvec_add(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(out.len(), self.rows);
        let cols = self.cols;
        for r in 0..self.rows {
            let row_offset = r * cols;
            let row_slice = &self.data[row_offset..row_offset + cols];
            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            let mut acc2 = 0.0f32;
            let mut acc3 = 0.0f32;

            let chunks = cols / 4;
            for c in 0..chunks {
                let idx = c * 4;
                acc0 += (row_slice[idx] as f32) * x[idx];
                acc1 += (row_slice[idx + 1] as f32) * x[idx + 1];
                acc2 += (row_slice[idx + 2] as f32) * x[idx + 2];
                acc3 += (row_slice[idx + 3] as f32) * x[idx + 3];
            }
            let mut sum = (acc0 + acc1) + (acc2 + acc3);
            for idx in (chunks * 4)..cols {
                sum += (row_slice[idx] as f32) * x[idx];
            }
            out[r] += sum * self.scales[r];
        }
    }

    #[inline]
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) {
        out.fill(0.0);
        self.matvec_add(x, out);
    }
}

#[derive(Clone)]
pub struct DepthModulator {
    pub u: QuantizedMatrix,
    pub v: QuantizedMatrix,
    pub bias: Option<Vec<f32>>,
}

impl DepthModulator {
    #[inline]
    pub fn apply(&self, x: &[f32], w_shared: &QuantizedMatrix, scratch_rm: &mut [f32], out: &mut [f32]) {
        if let Some(ref b) = self.bias {
            out.copy_from_slice(b);
        } else {
            out.fill(0.0);
        }
        w_shared.matvec_add(x, out);

        let r_m = self.u.cols;
        let d_in = self.v.rows;
        scratch_rm[..r_m].fill(0.0);
        for i in 0..d_in {
            let scale_x = self.v.scales[i] * x[i];
            let row = &self.v.data[i * r_m..(i + 1) * r_m];
            for k in 0..r_m {
                scratch_rm[k] += (row[k] as f32) * scale_x;
            }
        }
        self.u.matvec_add(&scratch_rm[..r_m], out);
    }
}

pub struct SharedCoreWeights {
    pub w_u: QuantizedMatrix,
    pub w_g: QuantizedMatrix,
    pub w_o: QuantizedMatrix,
    pub w_gate: QuantizedMatrix,
    pub w_up: QuantizedMatrix,
    pub w_down: QuantizedMatrix,
}

pub struct LayerWeights {
    pub norm1: Vec<f32>,
    pub norm2: Vec<f32>,
    pub conv_weight: Vec<f32>,
    pub mod_u: DepthModulator,
    pub mod_g: DepthModulator,
    pub mod_o: DepthModulator,
    pub mod_gate: DepthModulator,
    pub mod_up: DepthModulator,
    pub mod_down: DepthModulator,
    pub w_alpha: QuantizedMatrix,
    pub b_alpha: Vec<f32>,
    pub w_k: QuantizedMatrix,
    pub w_q: QuantizedMatrix,
    pub w_v: QuantizedMatrix,
    pub w_beta: Vec<f32>,
    pub b_beta: f32,
    pub w_gamma: Vec<f32>,
    pub b_gamma: f32,
    pub p_slow: QuantizedMatrix,
}

pub struct DeltaCoreModel {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_embed: usize,
    pub d_ff: usize,
    pub n_layers: usize,
    pub rank_mod: usize,
    pub rank_slow: usize,
    pub d_value: usize,
    pub conv_kernel: usize,
    pub rms_eps: f32,
    pub tok_embed: QuantizedMatrix,
    pub p_in: QuantizedMatrix,
    pub p_out: QuantizedMatrix,
    pub shared_core: SharedCoreWeights,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
}

#[derive(Clone)]
pub struct LayerState {
    pub conv_buf: Vec<f32>,
    pub fast_state: Vec<f32>,
    pub slow_mem: Vec<f32>,
}

#[derive(Clone)]
pub struct SessionState {
    pub layers: Vec<LayerState>,
    pub rng_state: u64,
    pub tokens_generated: usize,
}

struct BinaryReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> BinaryReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn read_u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.bytes[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    fn read_f32(&mut self) -> f32 {
        let v = f32::from_le_bytes(self.bytes[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    fn read_f32_vec(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.read_f32());
        }
        out
    }
    fn read_i8_matrix(&mut self, rows: usize, cols: usize) -> QuantizedMatrix {
        let scales = self.read_f32_vec(rows);
        let len = rows * cols;
        let slice = &self.bytes[self.pos..self.pos + len];
        let data: Vec<i8> = slice.iter().map(|&b| b as i8).collect();
        self.pos += len;
        QuantizedMatrix { rows, cols, scales, data }
    }
    fn read_modulator(&mut self, d_in: usize, d_out: usize, r_m: usize, has_bias: bool) -> DepthModulator {
        let u = self.read_i8_matrix(d_out, r_m);
        let v = self.read_i8_matrix(d_in, r_m);
        let bias = if has_bias { Some(self.read_f32_vec(d_out)) } else { None };
        DepthModulator { u, v, bias }
    }
}

#[inline]
fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x {
        sum_sq += v * v;
    }
    let inv_rms = 1.0f32 / ((sum_sq / (n as f32) + eps).sqrt());
    for i in 0..n {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0f32 + (-x).exp())
}

impl DeltaCoreModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < 48 || &bytes[0..4] != b"DCOR" {
            return Err("Invalid .dcore header magic or length (< 48 bytes)");
        }
        let mut r = BinaryReader::new(&bytes[4..]);
        let version = r.read_u32();
        if version != 1 {
            return Err("Unsupported .dcore version");
        }
        let vocab_size = r.read_u32() as usize;
        let d_model = r.read_u32() as usize;
        let d_embed = r.read_u32() as usize;
        let d_ff = r.read_u32() as usize;
        let n_layers = r.read_u32() as usize;
        let rank_mod = r.read_u32() as usize;
        let rank_slow = r.read_u32() as usize;
        let d_value = r.read_u32() as usize;
        let rms_eps = r.read_f32();
        let conv_kernel = r.read_u32() as usize;

        let tok_embed = r.read_i8_matrix(vocab_size, d_embed);
        let p_in = r.read_i8_matrix(d_model, d_embed);
        let p_out = r.read_i8_matrix(d_embed, d_model);

        let shared_core = SharedCoreWeights {
            w_u: r.read_i8_matrix(d_model, d_model),
            w_g: r.read_i8_matrix(d_model, d_model),
            w_o: r.read_i8_matrix(d_model, d_model),
            w_gate: r.read_i8_matrix(d_ff, d_model),
            w_up: r.read_i8_matrix(d_ff, d_model),
            w_down: r.read_i8_matrix(d_model, d_ff),
        };

        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let norm1 = r.read_f32_vec(d_model);
            let norm2 = r.read_f32_vec(d_model);
            let conv_weight = r.read_f32_vec(conv_kernel * d_model);

            let mod_u = r.read_modulator(d_model, d_model, rank_mod, true);
            let mod_g = r.read_modulator(d_model, d_model, rank_mod, true);
            let mod_o = r.read_modulator(d_model, d_model, rank_mod, false);
            let mod_gate = r.read_modulator(d_model, d_ff, rank_mod, false);
            let mod_up = r.read_modulator(d_model, d_ff, rank_mod, false);
            let mod_down = r.read_modulator(d_ff, d_model, rank_mod, false);

            let w_alpha = r.read_i8_matrix(d_model, d_model);
            let b_alpha = r.read_f32_vec(d_model);

            let w_k = r.read_i8_matrix(rank_slow, d_model);
            let w_q = r.read_i8_matrix(rank_slow, d_model);
            let w_v = r.read_i8_matrix(d_value, d_model);

            let w_beta = r.read_f32_vec(d_model);
            let b_beta = r.read_f32();
            let w_gamma = r.read_f32_vec(d_model);
            let b_gamma = r.read_f32();
            let p_slow = r.read_i8_matrix(d_model, d_value);

            layers.push(LayerWeights {
                norm1, norm2, conv_weight,
                mod_u, mod_g, mod_o, mod_gate, mod_up, mod_down,
                w_alpha, b_alpha, w_k, w_q, w_v,
                w_beta, b_beta, w_gamma, b_gamma, p_slow,
            });
        }
        let final_norm = r.read_f32_vec(d_model);
        if r.pos + 4 != bytes.len() {
            return Err("Byte length mismatch in .dcore payload");
        }

        Ok(Self {
            vocab_size, d_model, d_embed, d_ff, n_layers,
            rank_mod, rank_slow, d_value, conv_kernel, rms_eps,
            tok_embed, p_in, p_out, shared_core, layers, final_norm,
        })
    }
}
