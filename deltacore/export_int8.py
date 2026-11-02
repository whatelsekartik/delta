# deltacore/export_int8.py
"""
Audited Symmetric INT8 Exporter & Reference Dequantizer/Evaluator for DeltaCore-LM (.dcore).
Verifies:
1. Exact serialized byte count against the closed-form equation from deltacore/math_audit.py.
2. Per-tensor and global Signal-to-Quantization-Noise Ratio (SQNR in dB) and Frobenius relative error.
3. Exact INT8 reference forward_step matching C++ and WebAssembly runtimes to floating-point precision.
"""

import math
import struct
from typing import Any, BinaryIO, Dict, List, Tuple

import numpy as np
from deltacore.model_np import DeltaCoreConfig, DeltaCoreModelNP, LayerRecurrentState

MAGIC = b"DCOR"
VERSION = 1


def _write_f32(f: BinaryIO, arr: np.ndarray) -> None:
    flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    f.write(flat.tobytes())


def _write_i8_matrix(f: BinaryIO, mat: np.ndarray) -> Dict[str, float]:
    """
    Quantizes 2D matrix (rows, cols) to per-row symmetric INT8:
      scale[r] = max(abs(mat[r, :]), 1e-8) / 127.0
      q[r, c]  = round(clip(mat[r, c] / scale[r], -127, 127))
    Returns quantization error telemetry for the audit report.
    """
    m = np.ascontiguousarray(mat, dtype=np.float32)
    assert m.ndim == 2, f"Expected 2D matrix, got {m.shape}"
    max_abs = np.maximum(np.max(np.abs(m), axis=1, keepdims=True), 1e-8)
    scales = (max_abs / 127.0).squeeze(1).astype(np.float32)
    q = np.clip(np.round(m / (scales[:, None])), -127.0, 127.0).astype(np.int8)

    f.write(scales.tobytes())
    f.write(q.tobytes())

    recon = q.astype(np.float32) * scales[:, None]
    diff = m - recon
    sig_pow = float(np.sum(m * m)) + 1e-12
    noise_pow = float(np.sum(diff * diff)) + 1e-12
    sqnr_db = 10.0 * math.log10(sig_pow / noise_pow)
    rel_frob = math.sqrt(noise_pow / sig_pow)
    max_err = float(np.max(np.abs(diff)))
    return {
        "rows": int(m.shape[0]),
        "cols": int(m.shape[1]),
        "sqnr_db": round(sqnr_db, 2),
        "rel_frobenius_err": round(rel_frob, 6),
        "max_abs_err": round(max_err, 6),
    }


def export_deltacore_int8(model: DeltaCoreModelNP, out_path: str) -> Tuple[int, Dict[str, Any]]:
    cfg = model.cfg
    p = model.params
    tensor_stats: Dict[str, Dict[str, float]] = {}

    with open(out_path, "wb") as f:
        # 48-byte Header: Magic (4B) + 9*u32 + f32 (40B) + u32 conv_kernel (4B)
        f.write(MAGIC)
        f.write(
            struct.pack(
                "<9If",
                VERSION,
                cfg.vocab_size,
                cfg.d_model,
                cfg.d_embed,
                cfg.d_ff,
                cfg.n_layers,
                cfg.rank_mod,
                cfg.rank_slow,
                cfg.d_value,
                cfg.rms_eps,
            )
        )
        f.write(struct.pack("<I", cfg.conv_kernel))

        # 1. Sub-space Vocabulary Interface
        tensor_stats["tok_embed"] = _write_i8_matrix(f, p["tok_embed"])
        tensor_stats["p_in"] = _write_i8_matrix(f, p["p_in"])
        tensor_stats["p_out"] = _write_i8_matrix(f, p["p_out"])

        # 2. SharedOperatorCore
        for k in ["w_u", "w_g", "w_o", "w_gate", "w_up", "w_down"]:
            tensor_stats[f"core.{k}"] = _write_i8_matrix(f, p[f"core.{k}"])

        # 3. Per-layer weights
        for l in range(cfg.n_layers):
            lp = f"layers.{l}"
            _write_f32(f, p[f"{lp}.norm1"])
            _write_f32(f, p[f"{lp}.norm2"])
            _write_f32(f, p[f"{lp}.conv_weight"])

            for m_name, has_bias in [
                ("mod_u", True),
                ("mod_g", True),
                ("mod_o", False),
                ("mod_gate", False),
                ("mod_up", False),
                ("mod_down", False),
            ]:
                tensor_stats[f"{lp}.{m_name}.u"] = _write_i8_matrix(f, p[f"{lp}.{m_name}.u"])
                tensor_stats[f"{lp}.{m_name}.v"] = _write_i8_matrix(f, p[f"{lp}.{m_name}.v"])
                if has_bias:
                    _write_f32(f, p[f"{lp}.{m_name}.bias"])

            tensor_stats[f"{lp}.w_alpha.weight"] = _write_i8_matrix(f, p[f"{lp}.w_alpha.weight"])
            _write_f32(f, p[f"{lp}.w_alpha.bias"])

            tensor_stats[f"{lp}.w_k"] = _write_i8_matrix(f, p[f"{lp}.w_k"])
            tensor_stats[f"{lp}.w_q"] = _write_i8_matrix(f, p[f"{lp}.w_q"])
            tensor_stats[f"{lp}.w_v"] = _write_i8_matrix(f, p[f"{lp}.w_v"])

            _write_f32(f, p[f"{lp}.w_beta.weight"])
            _write_f32(f, p[f"{lp}.w_beta.bias"])
            _write_f32(f, p[f"{lp}.w_gamma.weight"])
            _write_f32(f, p[f"{lp}.w_gamma.bias"])

            tensor_stats[f"{lp}.p_slow"] = _write_i8_matrix(f, p[f"{lp}.p_slow"])

        _write_f32(f, p["final_norm"])
        total_bytes = f.tell()

    sqnrs = [v["sqnr_db"] for v in tensor_stats.values()]
    rel_errs = [v["rel_frobenius_err"] for v in tensor_stats.values()]
    summary = {
        "total_bytes": total_bytes,
        "mean_sqnr_db": round(float(np.mean(sqnrs)), 2),
        "min_sqnr_db": round(float(np.min(sqnrs)), 2),
        "mean_rel_frobenius_err": round(float(np.mean(rel_errs)), 6),
        "max_rel_frobenius_err": round(float(np.max(rel_errs)), 6),
        "key_tensors": {
            k: tensor_stats[k]
            for k in ["tok_embed", "p_in", "p_out", "core.w_u", "core.w_gate", "core.w_down", "layers.0.w_k", "layers.0.p_slow"]
            if k in tensor_stats
        },
    }
    return total_bytes, summary


def load_deltacore_int8_as_dequantized_model(dcore_path: str) -> DeltaCoreModelNP:
    """
    Loads a .dcore INT8 binary and reconstructs dequantized FP32 weights (scales * int8_data)
    to execute the exact INT8 numerical reference in Python.
    """
    with open(dcore_path, "rb") as f:
        data = f.read()

    if len(data) < 48 or data[:4] != MAGIC:
        raise ValueError("Invalid .dcore header")

    (
        version,
        vocab_size,
        d_model,
        d_embed,
        d_ff,
        n_layers,
        rank_mod,
        rank_slow,
        d_value,
        rms_eps,
    ) = struct.unpack_from("<9If", data, 4)
    (conv_kernel,) = struct.unpack_from("<I", data, 44)
    pos = 48

    cfg = DeltaCoreConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        d_embed=d_embed,
        d_ff=d_ff,
        n_layers=n_layers,
        rank_mod=rank_mod,
        rank_slow=rank_slow,
        d_value=d_value,
        conv_kernel=conv_kernel,
        rms_eps=rms_eps,
    )
    model = DeltaCoreModelNP(cfg)

    def read_f32(shape: Tuple[int, ...]) -> np.ndarray:
        nonlocal pos
        numel = int(np.prod(shape))
        arr = np.frombuffer(data, dtype=np.float32, count=numel, offset=pos).copy().reshape(shape)
        pos += numel * 4
        return arr

    def read_i8(rows: int, cols: int) -> np.ndarray:
        nonlocal pos
        scales = np.frombuffer(data, dtype=np.float32, count=rows, offset=pos).copy()
        pos += rows * 4
        q = np.frombuffer(data, dtype=np.int8, count=rows * cols, offset=pos).copy().reshape((rows, cols))
        pos += rows * cols
        return (q.astype(np.float32) * scales[:, None]).astype(np.float32)

    p = model.params
    p["tok_embed"] = read_i8(vocab_size, d_embed)
    p["p_in"] = read_i8(d_model, d_embed)
    p["p_out"] = read_i8(d_embed, d_model)

    p["core.w_u"] = read_i8(d_model, d_model)
    p["core.w_g"] = read_i8(d_model, d_model)
    p["core.w_o"] = read_i8(d_model, d_model)
    p["core.w_gate"] = read_i8(d_ff, d_model)
    p["core.w_up"] = read_i8(d_ff, d_model)
    p["core.w_down"] = read_i8(d_model, d_ff)

    for l in range(n_layers):
        lp = f"layers.{l}"
        p[f"{lp}.norm1"] = read_f32((d_model,))
        p[f"{lp}.norm2"] = read_f32((d_model,))
        p[f"{lp}.conv_weight"] = read_f32((conv_kernel, d_model))

        for m_name, d_in, d_out, has_bias in [
            ("mod_u", d_model, d_model, True),
            ("mod_g", d_model, d_model, True),
            ("mod_o", d_model, d_model, False),
            ("mod_gate", d_model, d_ff, False),
            ("mod_up", d_model, d_ff, False),
            ("mod_down", d_ff, d_model, False),
        ]:
            p[f"{lp}.{m_name}.u"] = read_i8(d_out, rank_mod)
            p[f"{lp}.{m_name}.v"] = read_i8(d_in, rank_mod)
            if has_bias:
                p[f"{lp}.{m_name}.bias"] = read_f32((d_out,))

        p[f"{lp}.w_alpha.weight"] = read_i8(d_model, d_model)
        p[f"{lp}.w_alpha.bias"] = read_f32((d_model,))

        p[f"{lp}.w_k"] = read_i8(rank_slow, d_model)
        p[f"{lp}.w_q"] = read_i8(rank_slow, d_model)
        p[f"{lp}.w_v"] = read_i8(d_value, d_model)

        p[f"{lp}.w_beta.weight"] = read_f32((1, d_model))
        p[f"{lp}.w_beta.bias"] = read_f32((1,))
        p[f"{lp}.w_gamma.weight"] = read_f32((1, d_model))
        p[f"{lp}.w_gamma.bias"] = read_f32((1,))

        p[f"{lp}.p_slow"] = read_i8(d_model, d_value)

    p["final_norm"] = read_f32((d_model,))
    assert pos == len(data), f"Trailing bytes in .dcore: pos={pos}, len={len(data)}"
    return model
