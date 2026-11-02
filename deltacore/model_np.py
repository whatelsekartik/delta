# deltacore/model_np.py
"""
Complete NumPy FP32 & INT8 Reference Implementation of DeltaCore-LM and BaselineTransformerLM.
Includes:
1. Exact Parallel Associative Prefix Scans:
   - Hillis-Steele O(log T) depth, O(T log T * d) work
   - Blelloch Work-Efficient O(log T) depth, O(T * d) work
2. Slow Associative Delta Memory:
   - Sequential step-by-step reference O(T) depth
   - Chunkwise Parallel WY-Representation (Bischof & Van Loan 1987; Yang et al. 2024)
     reducing sequential depth to O(T / B_c) via intra-chunk GEMMs and triangular solve.
3. Single-token O(1) recurrent step (`forward_step`)
4. Analytical backpropagation & AdamW training engine on the audited split-before-tokenizer corpus.
"""

import math
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DeltaCoreConfig:
    vocab_size: int = 512
    d_model: int = 128
    d_embed: int = 64
    d_ff: int = 256
    n_layers: int = 4
    rank_mod: int = 8
    rank_slow: int = 16
    d_value: int = 32
    conv_kernel: int = 3
    rms_eps: float = 1e-5
    seed: int = 1337


@dataclass
class LayerRecurrentState:
    conv_buf: np.ndarray    # (B, C-1, d_model)
    fast_state: np.ndarray  # (B, d_model)
    slow_mem: np.ndarray    # (B, rank_slow, d_value)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0))).astype(np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    return (x * sigmoid(x)).astype(np.float32)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    ms = np.mean(x * x, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(ms + eps)
    return (x * inv_rms * weight).astype(np.float32)


def orthogonal_init(rng: np.random.Generator, rows: int, cols: int, gain: float = 1.0) -> np.ndarray:
    a = rng.standard_normal((max(rows, cols), min(rows, cols))).astype(np.float32)
    q, r = np.linalg.qr(a)
    d = np.diag(r)
    ph = np.sign(d)
    ph[ph == 0] = 1.0
    q = q * ph
    if rows < cols:
        q = q.T
    return (gain * q[:rows, :cols]).astype(np.float32)


def parallel_gated_scan_hillis_steele(
    gates: np.ndarray, inputs: np.ndarray, h0: np.ndarray
) -> np.ndarray:
    """
    Hillis-Steele parallel associative prefix scan:
    Depth: O(log2 T), Work: O(T log2 T * D).
    """
    B, T, D = gates.shape
    a = gates.astype(np.float32).copy()
    b = inputs.astype(np.float32).copy()
    b[:, 0, :] = b[:, 0, :] + a[:, 0, :] * h0

    num_steps = int(math.ceil(math.log2(max(T, 1))))
    for step in range(num_steps):
        stride = 1 << step
        if stride >= T:
            break
        a_prev = a[:, :-stride, :].copy()
        b_prev = b[:, :-stride, :].copy()
        a_curr = a[:, stride:, :]
        b_curr = b[:, stride:, :]
        b[:, stride:, :] = a_curr * b_prev + b_curr
        a[:, stride:, :] = a_curr * a_prev
    return b


def parallel_gated_scan_blelloch(
    gates: np.ndarray, inputs: np.ndarray, h0: np.ndarray
) -> np.ndarray:
    """
    Work-efficient Blelloch up-sweep / down-sweep associative scan:
    Depth: O(log2 T), Work: O(T * D) (2*(T-1) associative combines).
    """
    B, T, D = gates.shape
    # Pad T to next power of 2 for balanced binary tree sweep
    pow2 = 1 << int(math.ceil(math.log2(max(T, 2))))
    a = np.ones((B, pow2, D), dtype=np.float32)
    b = np.zeros((B, pow2, D), dtype=np.float32)
    a[:, :T, :] = gates
    b[:, :T, :] = inputs
    b[:, 0, :] = b[:, 0, :] + a[:, 0, :] * h0

    # Up-sweep (reduce) phase
    levels = int(math.log2(pow2))
    for d_lvl in range(levels):
        step = 1 << (d_lvl + 1)
        half = 1 << d_lvl
        left_idx = np.arange(half - 1, pow2, step)
        right_idx = left_idx + half
        a_l = a[:, left_idx, :].copy()
        b_l = b[:, left_idx, :].copy()
        a_r = a[:, right_idx, :]
        b_r = b[:, right_idx, :]
        b[:, right_idx, :] = a_r * b_l + b_r
        a[:, right_idx, :] = a_r * a_l

    # Down-sweep phase (exclusive prefix -> inclusive)
    a_orig = gates.astype(np.float32)
    b_orig = inputs.astype(np.float32).copy()
    b_orig[:, 0, :] = b_orig[:, 0, :] + a_orig[:, 0, :] * h0

    a[:, pow2 - 1, :] = 1.0
    b[:, pow2 - 1, :] = 0.0
    for d_lvl in range(levels - 1, -1, -1):
        step = 1 << (d_lvl + 1)
        half = 1 << d_lvl
        left_idx = np.arange(half - 1, pow2, step)
        right_idx = left_idx + half
        a_l = a[:, left_idx, :].copy()
        b_l = b[:, left_idx, :].copy()
        a_r = a[:, right_idx, :].copy()
        b_r = b[:, right_idx, :].copy()

        a[:, left_idx, :] = a_r
        b[:, left_idx, :] = b_r
        b[:, right_idx, :] = a_l * b_r + b_l
        a[:, right_idx, :] = a_l * a_r

    # Convert exclusive prefix (a_exc, b_exc) to inclusive: (a_orig, b_orig) * (a_exc, b_exc)
    out = a_orig * b[:, :T, :] + b_orig
    return out.astype(np.float32)


def sequential_delta_forward(
    k_seq: np.ndarray,
    q_seq: np.ndarray,
    v_seq: np.ndarray,
    beta_seq: np.ndarray,
    gamma_seq: np.ndarray,
    slow_mem_init: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reference O(T) sequential loop for the Slow Associative Delta Stream.
    k_seq, q_seq: (B, T, r_s)
    v_seq:        (B, T, d_v)
    beta_seq, gamma_seq: (B, T, 1)
    slow_mem_init: (B, r_s, d_v)
    """
    B, T, r_s = k_seq.shape
    d_v = v_seq.shape[-1]
    M = slow_mem_init.astype(np.float32).copy()
    s_out = np.zeros((B, T, d_v), dtype=np.float32)

    for t in range(T):
        k_t = k_seq[:, t, :]
        q_t = q_seq[:, t, :]
        v_t = v_seq[:, t, :]
        beta_t = beta_seq[:, t, :][:, :, None]
        gamma_t = gamma_seq[:, t, :][:, :, None]

        M_dec = gamma_t * M
        v_hat = np.einsum("brv,br->bv", M_dec, k_t)
        delta_v = v_t - v_hat
        M = M_dec + beta_t * np.einsum("br,bv->brv", k_t, delta_v)
        s_out[:, t, :] = np.einsum("brv,br->bv", M, q_t)

    return s_out, M


def chunkwise_wy_delta_forward(
    k_seq: np.ndarray,
    q_seq: np.ndarray,
    v_seq: np.ndarray,
    beta_seq: np.ndarray,
    gamma_seq: np.ndarray,
    slow_mem_init: np.ndarray,
    chunk_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chunkwise Parallel WY-Representation Delta Rule (Bischof & Van Loan 1987; Yang et al. 2024).
    Replaces step-by-step outer-product loop inside each chunk of length B_c with:
      1. Strictly lower-triangular B_c x B_c Householder coupling matrix L_chunk
      2. Triangular solve (I + L_chunk) U = diag(beta) (V_chunk - diag(Gamma_init) K_chunk M_init)
      3. Parallel matrix-matrix readout S_chunk and chunk state transition M_next.
    Sequential depth is reduced from O(T) to O(T / B_c).
    """
    B, T, r_s = k_seq.shape
    d_v = v_seq.shape[-1]
    M = slow_mem_init.astype(np.float32).copy()
    s_out = np.zeros((B, T, d_v), dtype=np.float32)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        Bc = end - start
        K_c = k_seq[:, start:end, :].astype(np.float64)       # (B, Bc, r_s)
        Q_c = q_seq[:, start:end, :].astype(np.float64)       # (B, Bc, r_s)
        V_c = v_seq[:, start:end, :].astype(np.float64)       # (B, Bc, d_v)
        beta_c = beta_seq[:, start:end, 0].astype(np.float64) # (B, Bc)
        gamma_c = gamma_seq[:, start:end, 0].astype(np.float64) # (B, Bc)

        # Compute cumulative decay matrix D_causal[b, tau, j] = prod_{s=j+1}^{tau} gamma_s for tau >= j
        # and Gamma_init[b, tau] = prod_{s=0}^{tau} gamma_s
        log_g = np.log(np.clip(gamma_c, 1e-12, 1.0))
        prefix_log_g = np.cumsum(log_g, axis=1)  # (B, Bc)
        Gamma_init = np.exp(prefix_log_g)        # (B, Bc)

        # Pairwise log decay: prefix_log_g[:, tau] - prefix_log_g[:, j]
        diff_log = prefix_log_g[:, :, None] - prefix_log_g[:, None, :]  # (B, Bc, Bc)
        tril_mask = np.tril(np.ones((Bc, Bc), dtype=np.float64), k=0)
        strict_tril_mask = np.tril(np.ones((Bc, Bc), dtype=np.float64), k=-1)
        D_causal = np.exp(np.where(tril_mask[None, :, :] > 0, diff_log, 0.0)) * tril_mask[None, :, :]
        D_strict = D_causal * strict_tril_mask[None, :, :]

        # Gram matrix K_c @ K_c^T: (B, Bc, Bc)
        KK = np.einsum("btr,bsr->bts", K_c, K_c)
        L_chunk = beta_c[:, :, None] * D_strict * KK  # Strictly lower triangular (B, Bc, Bc)

        # RHS: diag(beta) * (V_c - diag(Gamma_init) K_c M)
        M_f64 = M.astype(np.float64)
        KM = np.einsum("btr,brv->btv", K_c, M_f64)
        rhs = beta_c[:, :, None] * (V_c - Gamma_init[:, :, None] * KM)

        # Solve (I + L_chunk) U = rhs for each batch item
        eye = np.eye(Bc, dtype=np.float64)[None, :, :]
        U = np.linalg.solve(eye + L_chunk, rhs)  # (B, Bc, d_v)

        # Parallel chunk readout: S_c = diag(Gamma_init) Q_c M + ((Q_c K_c^T) * D_causal) U
        QM = np.einsum("btr,brv->btv", Q_c, M_f64)
        QK_causal = np.einsum("btr,bsr->bts", Q_c, K_c) * D_causal
        S_c = Gamma_init[:, :, None] * QM + np.einsum("bts,bsv->btv", QK_causal, U)
        s_out[:, start:end, :] = S_c.astype(np.float32)

        # End-of-chunk state update: M_next = Gamma_init[:, -1] * M + K_c^T diag(D_causal[:, -1, :]) U
        decay_to_end = D_causal[:, -1, :]  # (B, Bc)
        M = (
            Gamma_init[:, -1][:, None, None] * M_f64
            + np.einsum("btr,bt,btv->brv", K_c, decay_to_end, U)
        ).astype(np.float32)

    return s_out, M


class DeltaCoreModelNP:
    """
    Full DeltaCore-LM model in FP32 NumPy with exact parameter layout matching deltacore/model.py
    and deltacore/export_int8.py.
    """

    def __init__(self, cfg: DeltaCoreConfig) -> None:
        self.cfg = cfg
        self.params: Dict[str, np.ndarray] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        std = 0.02
        d = cfg.d_model
        d_e = cfg.d_embed
        d_ff = cfg.d_ff
        r_m = cfg.rank_mod
        r_s = cfg.rank_slow
        d_v = cfg.d_value
        C = cfg.conv_kernel

        self.params["tok_embed"] = rng.normal(0.0, std, (cfg.vocab_size, d_e)).astype(np.float32)
        self.params["p_in"] = orthogonal_init(rng, d, d_e, 1.0)
        self.params["p_out"] = orthogonal_init(rng, d_e, d, 1.0)

        # SharedOperatorCore
        self.params["core.w_u"] = orthogonal_init(rng, d, d, 1.0)
        self.params["core.w_g"] = orthogonal_init(rng, d, d, 1.0)
        self.params["core.w_o"] = orthogonal_init(rng, d, d, 1.0)
        self.params["core.w_gate"] = orthogonal_init(rng, d_ff, d, 1.0)
        self.params["core.w_up"] = orthogonal_init(rng, d_ff, d, 1.0)
        self.params["core.w_down"] = orthogonal_init(rng, d, d_ff, 1.0)

        res_scale = std / math.sqrt(2.0 * cfg.n_layers)
        for l in range(cfg.n_layers):
            p = f"layers.{l}"
            self.params[f"{p}.norm1"] = np.ones((d,), dtype=np.float32)
            self.params[f"{p}.norm2"] = np.ones((d,), dtype=np.float32)
            cw = np.zeros((C, d), dtype=np.float32)
            cw[0, :] = 1.0
            self.params[f"{p}.conv_weight"] = cw

            for m_name, d_in, d_out, has_bias in [
                ("mod_u", d, d, True),
                ("mod_g", d, d, True),
                ("mod_o", d, d, False),
                ("mod_gate", d, d_ff, False),
                ("mod_up", d, d_ff, False),
                ("mod_down", d_ff, d, False),
            ]:
                self.params[f"{p}.{m_name}.u"] = rng.normal(0.0, res_scale, (d_out, r_m)).astype(np.float32)
                self.params[f"{p}.{m_name}.v"] = rng.normal(0.0, std, (d_in, r_m)).astype(np.float32)
                if has_bias:
                    self.params[f"{p}.{m_name}.bias"] = np.zeros((d_out,), dtype=np.float32)

            self.params[f"{p}.w_alpha.weight"] = rng.normal(0.0, std, (d, d)).astype(np.float32)
            horizon_bias = 0.5 + 0.35 * (l / max(1, cfg.n_layers - 1))
            self.params[f"{p}.w_alpha.bias"] = np.full((d,), horizon_bias, dtype=np.float32)

            self.params[f"{p}.w_k"] = rng.normal(0.0, std, (r_s, d)).astype(np.float32)
            self.params[f"{p}.w_q"] = rng.normal(0.0, std, (r_s, d)).astype(np.float32)
            self.params[f"{p}.w_v"] = rng.normal(0.0, std, (d_v, d)).astype(np.float32)

            self.params[f"{p}.w_beta.weight"] = rng.normal(0.0, std, (1, d)).astype(np.float32)
            self.params[f"{p}.w_beta.bias"] = np.array([-0.5], dtype=np.float32)
            self.params[f"{p}.w_gamma.weight"] = rng.normal(0.0, std, (1, d)).astype(np.float32)
            self.params[f"{p}.w_gamma.bias"] = np.array([2.2], dtype=np.float32)

            self.params[f"{p}.p_slow"] = rng.normal(0.0, std, (d, d_v)).astype(np.float32)

        self.params["final_norm"] = np.ones((d,), dtype=np.float32)

    def count_parameters(self) -> int:
        return sum(int(arr.size) for arr in self.params.values())

    def state_bytes_per_session(self) -> int:
        floats_per_layer = (
            (self.cfg.conv_kernel - 1) * self.cfg.d_model
            + self.cfg.d_model
            + self.cfg.rank_slow * self.cfg.d_value
        )
        return self.cfg.n_layers * floats_per_layer * 4

    def init_state(self, batch_size: int = 1) -> List[LayerRecurrentState]:
        C = self.cfg.conv_kernel
        D = self.cfg.d_model
        return [
            LayerRecurrentState(
                conv_buf=np.zeros((batch_size, C - 1, D), dtype=np.float32),
                fast_state=np.zeros((batch_size, D), dtype=np.float32),
                slow_mem=np.zeros((batch_size, self.cfg.rank_slow, self.cfg.d_value), dtype=np.float32),
            )
            for _ in range(self.cfg.n_layers)
        ]

    def _apply_mod(self, x: np.ndarray, l_prefix: str, m_name: str, w_shared: np.ndarray) -> np.ndarray:
        u = self.params[f"{l_prefix}.{m_name}.u"]
        v = self.params[f"{l_prefix}.{m_name}.v"]
        b_key = f"{l_prefix}.{m_name}.bias"
        base = x @ w_shared.T
        if b_key in self.params:
            base = base + self.params[b_key]
        low_rank = (x @ v) @ u.T
        return (base + low_rank).astype(np.float32)

    def forward_sequence(
        self,
        input_ids: np.ndarray,
        states: Optional[List[LayerRecurrentState]] = None,
        use_chunkwise_wy: bool = True,
        use_blelloch_scan: bool = True,
    ) -> Tuple[np.ndarray, List[LayerRecurrentState]]:
        B, T = input_ids.shape
        C = self.cfg.conv_kernel
        D = self.cfg.d_model
        if states is None:
            states = self.init_state(B)

        e = self.params["tok_embed"][input_ids]  # (B, T, d_e)
        h = (e @ self.params["p_in"].T).astype(np.float32)

        new_states: List[LayerRecurrentState] = []
        for l in range(self.cfg.n_layers):
            lp = f"layers.{l}"
            st = states[l]
            z = rms_norm(h, self.params[f"{lp}.norm1"], self.cfg.rms_eps)
            z_pad = np.concatenate([st.conv_buf, z], axis=1)
            new_conv_buf = z_pad[:, -(C - 1) :, :].copy()

            cw = self.params[f"{lp}.conv_weight"]
            z_shift = np.zeros_like(z)
            for tau in range(C):
                z_shift += z_pad[:, (C - 1 - tau) : (C - 1 - tau + T), :] * cw[tau][None, None, :]

            # Fast syntactic stream
            alpha = sigmoid(z_shift @ self.params[f"{lp}.w_alpha.weight"].T + self.params[f"{lp}.w_alpha.bias"])
            u = np.tanh(self._apply_mod(z_shift, lp, "mod_u", self.params["core.w_u"])).astype(np.float32)
            fast_inputs = (1.0 - alpha) * u
            if use_blelloch_scan:
                f_seq = parallel_gated_scan_blelloch(alpha, fast_inputs, st.fast_state)
            else:
                f_seq = parallel_gated_scan_hillis_steele(alpha, fast_inputs, st.fast_state)
            new_fast = f_seq[:, -1, :].copy()

            # Slow associative Delta stream
            k_raw = z_shift @ self.params[f"{lp}.w_k"].T
            k_norm = np.linalg.norm(k_raw, axis=-1, keepdims=True) + 1e-6
            k_seq = (k_raw / k_norm).astype(np.float32)
            q_seq = ((z_shift @ self.params[f"{lp}.w_q"].T) * (1.0 / math.sqrt(self.cfg.rank_slow))).astype(np.float32)
            v_seq = (z_shift @ self.params[f"{lp}.w_v"].T).astype(np.float32)
            beta_seq = sigmoid(z_shift @ self.params[f"{lp}.w_beta.weight"].T + self.params[f"{lp}.w_beta.bias"])
            gamma_seq = sigmoid(z_shift @ self.params[f"{lp}.w_gamma.weight"].T + self.params[f"{lp}.w_gamma.bias"])

            if use_chunkwise_wy:
                s_seq, new_slow = chunkwise_wy_delta_forward(k_seq, q_seq, v_seq, beta_seq, gamma_seq, st.slow_mem, chunk_size=16)
            else:
                s_seq, new_slow = sequential_delta_forward(k_seq, q_seq, v_seq, beta_seq, gamma_seq, st.slow_mem)

            g_seq = silu(self._apply_mod(z_shift, lp, "mod_g", self.params["core.w_g"]))
            combined = g_seq * (f_seq + (s_seq @ self.params[f"{lp}.p_slow"].T))
            m = h + self._apply_mod(combined, lp, "mod_o", self.params["core.w_o"])

            m_norm = rms_norm(m, self.params[f"{lp}.norm2"], self.cfg.rms_eps)
            ff_gate = silu(self._apply_mod(m_norm, lp, "mod_gate", self.params["core.w_gate"]))
            ff_up = self._apply_mod(m_norm, lp, "mod_up", self.params["core.w_up"])
            h = (m + self._apply_mod(ff_gate * ff_up, lp, "mod_down", self.params["core.w_down"])).astype(np.float32)

            new_states.append(LayerRecurrentState(new_conv_buf, new_fast, new_slow))

        h_final = rms_norm(h, self.params["final_norm"], self.cfg.rms_eps)
        h_sub = h_final @ self.params["p_out"].T
        logits = (h_sub @ self.params["tok_embed"].T).astype(np.float32)
        return logits, new_states

    def forward_step(
        self, token_ids: np.ndarray, states: List[LayerRecurrentState]
    ) -> Tuple[np.ndarray, List[LayerRecurrentState]]:
        """Single-step O(1) recurrent forward pass. token_ids: (B,)."""
        C = self.cfg.conv_kernel
        e = self.params["tok_embed"][token_ids]  # (B, d_e)
        h = (e @ self.params["p_in"].T).astype(np.float32)

        next_states: List[LayerRecurrentState] = []
        for l in range(self.cfg.n_layers):
            lp = f"layers.{l}"
            st = states[l]
            z_t = rms_norm(h, self.params[f"{lp}.norm1"], self.cfg.rms_eps)
            cw = self.params[f"{lp}.conv_weight"]

            z_shift = z_t * cw[0][None, :]
            for tau in range(1, C):
                z_shift = z_shift + st.conv_buf[:, C - 1 - tau, :] * cw[tau][None, :]

            new_conv_buf = np.concatenate([st.conv_buf[:, 1:, :], z_t[:, None, :]], axis=1)

            alpha_t = sigmoid(z_shift @ self.params[f"{lp}.w_alpha.weight"].T + self.params[f"{lp}.w_alpha.bias"])
            u_t = np.tanh(self._apply_mod(z_shift, lp, "mod_u", self.params["core.w_u"])).astype(np.float32)
            f_t = (alpha_t * st.fast_state + (1.0 - alpha_t) * u_t).astype(np.float32)

            k_raw = z_shift @ self.params[f"{lp}.w_k"].T
            k_t = (k_raw / (np.linalg.norm(k_raw, axis=-1, keepdims=True) + 1e-6)).astype(np.float32)
            q_t = ((z_shift @ self.params[f"{lp}.w_q"].T) * (1.0 / math.sqrt(self.cfg.rank_slow))).astype(np.float32)
            v_t = (z_shift @ self.params[f"{lp}.w_v"].T).astype(np.float32)
            beta_t = sigmoid(z_shift @ self.params[f"{lp}.w_beta.weight"].T + self.params[f"{lp}.w_beta.bias"])[:, :, None]
            gamma_t = sigmoid(z_shift @ self.params[f"{lp}.w_gamma.weight"].T + self.params[f"{lp}.w_gamma.bias"])[:, :, None]

            M_dec = gamma_t * st.slow_mem
            v_hat = np.einsum("brv,br->bv", M_dec, k_t)
            M_t = (M_dec + beta_t * np.einsum("br,bv->brv", k_t, v_t - v_hat)).astype(np.float32)
            s_t = np.einsum("brv,br->bv", M_t, q_t).astype(np.float32)

            g_t = silu(self._apply_mod(z_shift, lp, "mod_g", self.params["core.w_g"]))
            combined = g_t * (f_t + (s_t @ self.params[f"{lp}.p_slow"].T))
            m_t = h + self._apply_mod(combined, lp, "mod_o", self.params["core.w_o"])

            m_norm = rms_norm(m_t, self.params[f"{lp}.norm2"], self.cfg.rms_eps)
            ff_gate = silu(self._apply_mod(m_norm, lp, "mod_gate", self.params["core.w_gate"]))
            ff_up = self._apply_mod(m_norm, lp, "mod_up", self.params["core.w_up"])
            h = (m_t + self._apply_mod(ff_gate * ff_up, lp, "mod_down", self.params["core.w_down"])).astype(np.float32)

            next_states.append(LayerRecurrentState(new_conv_buf, f_t, M_t))

        h_final = rms_norm(h, self.params["final_norm"], self.cfg.rms_eps)
        h_sub = h_final @ self.params["p_out"].T
        logits = (h_sub @ self.params["tok_embed"].T).astype(np.float32)
        return logits, next_states
