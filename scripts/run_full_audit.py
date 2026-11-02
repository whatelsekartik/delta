# scripts/run_full_audit.py
"""
Master Python Audit Pipeline (Phases 1, 2, 3, and 4):
1. Runs numerical equivalence & parameter count unit tests.
2. Runs mathematical & literature audit.
3. Builds deduplicated 360-document CC0-1.0 corpus, splits into Train/Val/Test BEFORE tokenizer training.
4. Trains ByteBPETokenizer exclusively on Train split (`artifacts/tokenizer.json`).
5. Trains DeltaCore-LM & BaselineTransformer from random init and evaluates held-out Test generalization.
6. Exports `artifacts/model.dcore` (deployed spec) and `artifacts/model_phase4_spec.dcore` (Phase-4 spec)
   and verifies exact byte counts.
7. Generates reference FP32 and Python INT8 logit vectors & greedy token sequences across 6 test prompts
   (`artifacts/parity_reference.json`) for cross-runtime C++ and WebAssembly verification.
"""

import json
import math
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deltacore.corpus_builder import partition_and_audit_corpus
from deltacore.export_int8 import (
    export_deltacore_int8,
    load_deltacore_int8_as_dequantized_model,
)
from deltacore.math_audit import run_mathematical_and_architectural_audit
from deltacore.model_np import DeltaCoreConfig, DeltaCoreModelNP
from deltacore.tokenizer import ByteBPETokenizer
from deltacore.trainer import softmax, train_deltacore_and_baseline
from tests.test_numerical import (
    test_chunkwise_wy_delta_equivalence,
    test_exact_parameter_and_state_audit_match,
    test_parallel_scans_equivalence,
    test_sequence_vs_recurrent_step_equivalence,
)


def compute_logit_error_metrics(logits_fp32: np.ndarray, logits_int8: np.ndarray) -> Dict[str, float]:
    diff = logits_fp32 - logits_int8
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))

    norm_a = float(np.linalg.norm(logits_fp32)) + 1e-12
    norm_b = float(np.linalg.norm(logits_int8)) + 1e-12
    cos_sim = float(np.dot(logits_fp32, logits_int8) / (norm_a * norm_b))

    p_fp32 = np.clip(softmax(logits_fp32), 1e-12, 1.0)
    p_int8 = np.clip(softmax(logits_int8), 1e-12, 1.0)
    kl_div = float(np.sum(p_fp32 * np.log(p_fp32 / p_int8)))

    top1_fp32 = int(np.argmax(logits_fp32))
    top1_int8 = int(np.argmax(logits_int8))
    top5_fp32 = set(np.argsort(logits_fp32)[-5:].tolist())
    top5_int8 = set(np.argsort(logits_int8)[-5:].tolist())
    top5_overlap = len(top5_fp32 & top5_int8) / 5.0

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "max_abs_err": round(max_abs, 6),
        "cosine_similarity": round(cos_sim, 6),
        "kl_divergence": round(kl_div, 6),
        "top1_match": 1.0 if top1_fp32 == top1_int8 else 0.0,
        "top5_overlap": round(top5_overlap, 4),
        "top1_token_fp32": top1_fp32,
        "top1_token_int8": top1_int8,
    }


def main() -> None:
    os.makedirs("artifacts", exist_ok=True)
    print("=== [1/6] Running Phase 3 Numerical Equivalence & Parameter Formula Tests ===")
    scan_res = test_parallel_scans_equivalence()
    wy_res = test_chunkwise_wy_delta_equivalence()
    step_res = test_sequence_vs_recurrent_step_equivalence()
    param_res = test_exact_parameter_and_state_audit_match()

    print("\n=== [2/6] Running Phase 1 Mathematical & Architectural Literature Audit ===")
    math_report = run_mathematical_and_architectural_audit()

    print("\n=== [3/6] Building Curated CC0-1.0 Corpus & Enforcing Split-Before-Tokenization ===")
    train_text, val_text, test_text, corpus_audit = partition_and_audit_corpus()
    print(
        f"Corpus Partitioned: Train={corpus_audit.train_documents} docs ({corpus_audit.train_bytes:,} B), "
        f"Val={corpus_audit.val_documents} docs ({corpus_audit.val_bytes:,} B), "
        f"Test={corpus_audit.test_documents} docs ({corpus_audit.test_bytes:,} B)"
    )
    print(
        f"Doc Overlap (Train/Val/Test): {corpus_audit.doc_overlap_train_val} / "
        f"{corpus_audit.doc_overlap_train_test} / {corpus_audit.doc_overlap_val_test} "
        f"| 5-gram Jaccard (Train/Val): {corpus_audit.word_5gram_jaccard_train_val:.5f} "
        f"(vs Legacy Leaked Corpus: {corpus_audit.legacy_corpus_5gram_jaccard:.4f})"
    )

    tok = ByteBPETokenizer()
    tok.train(train_text, target_vocab_size=512, split_name="train_only")
    tok.save("artifacts/tokenizer.json")

    train_ids = np.array(tok.encode(train_text), dtype=np.int64)
    val_ids = np.array(tok.encode(val_text), dtype=np.int64)
    test_ids = np.array(tok.encode(test_text), dtype=np.int64)
    print(
        f"Tokenizer trained exclusively on train_only (vocab={tok.vocab_size}) | "
        f"Tokens: train={len(train_ids):,}, val={len(val_ids):,}, test={len(test_ids):,}"
    )

    print("\n=== [4/6] Training DeltaCore-LM & Baseline Transformer from Random Initialization ===")
    deployed_cfg = DeltaCoreConfig(
        vocab_size=tok.vocab_size,
        d_model=128,
        d_embed=64,
        d_ff=256,
        n_layers=4,
        rank_mod=8,
        rank_slow=16,
        d_value=32,
        conv_kernel=3,
        seed=1337,
    )
    model_fp32 = DeltaCoreModelNP(deployed_cfg)
    training_report = train_deltacore_and_baseline(
        model_fp32,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        train_bytes_len=corpus_audit.train_bytes,
        test_bytes_len=corpus_audit.test_bytes,
        steps=360,
        batch_size=8,
        seq_len=64,
    )
    dc_tr = training_report["deltacore"]
    print(
        f"DeltaCore-LM Trained ({training_report['train_time_sec']}s): "
        f"Init Val Loss={dc_tr['init_val_loss']:.4f} (PPL {dc_tr['init_val_ppl']:.1f}) -> "
        f"Final Train Loss={dc_tr['final_train_loss']:.4f} | "
        f"Final Val Loss={dc_tr['final_val_loss']:.4f} (PPL {dc_tr['final_val_ppl']:.2f}) | "
        f"Final Held-Out Test Loss={dc_tr['final_test_loss']:.4f} (PPL {dc_tr['final_test_ppl']:.2f}, BPB {dc_tr['test_bits_per_byte']:.3f})"
    )

    print("\n=== [5/6] Exporting Symmetric INT8 (.dcore) Artifacts & Verifying Exact Byte Sizes ===")
    dcore_bytes, quant_summary_deployed = export_deltacore_int8(model_fp32, "artifacts/model.dcore")
    expected_dep_bytes = math_report["deployed_spec_metrics"]["dcore_int8_bytes"]
    assert dcore_bytes == expected_dep_bytes, f"Byte mismatch: {dcore_bytes} != {expected_dep_bytes}"

    # Also train & export Phase-4 spec (d=192, L=6) and verify exact byte match
    p4_cfg = DeltaCoreConfig(
        vocab_size=512, d_model=192, d_embed=96, d_ff=384, n_layers=6, rank_mod=16, rank_slow=16, d_value=48
    )
    model_p4 = DeltaCoreModelNP(p4_cfg)
    _ = train_deltacore_and_baseline(
        model_p4,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        train_bytes_len=corpus_audit.train_bytes,
        test_bytes_len=corpus_audit.test_bytes,
        steps=160,
        batch_size=8,
        seq_len=64,
    )
    p4_bytes, quant_summary_p4 = export_deltacore_int8(model_p4, "artifacts/model_phase4_spec.dcore")
    expected_p4_bytes = math_report["phase4_spec_metrics"]["dcore_int8_bytes"]
    assert p4_bytes == expected_p4_bytes, f"Phase-4 byte mismatch: {p4_bytes} != {expected_p4_bytes}"
    print(
        f"Verified .dcore Exports: Deployed={dcore_bytes:,} B ({dcore_bytes/1024:.2f} KiB, mean SQNR={quant_summary_deployed['mean_sqnr_db']} dB) | "
        f"Phase-4 Spec={p4_bytes:,} B ({p4_bytes/1024:.2f} KiB, mean SQNR={quant_summary_p4['mean_sqnr_db']} dB)"
    )

    print("\n=== [6/6] Running FP32 vs Python INT8 Logit & Sequence Parity Audit ===")
    model_int8_py = load_deltacore_int8_as_dequantized_model("artifacts/model.dcore")

    test_prompts = [
        "On the Xeon IceLake-SP processor clocked at 2.60 GHz",
        "Architectural analysis of Widrow-Hoff Delta Rule Network",
        "Question: Which cluster and port currently host vault-auth? Answer:",
        "Engineering note on WebAssembly linear memory during production audit",
        "When executing a matrix-vector product at batch size one",
        "Under sequence length 1024 and dimension 160, empirical perturbation",
    ]

    prompt_parity_cases: List[Dict[str, Any]] = []
    for p_idx, prompt in enumerate(test_prompts):
        p_ids = tok.encode(prompt)
        st_fp32 = model_fp32.init_state(1)
        st_int8 = model_int8_py.init_state(1)
        logits_fp32 = None
        logits_int8 = None
        for tid in p_ids:
            t_arr = np.array([tid], dtype=np.int64)
            logits_fp32, st_fp32 = model_fp32.forward_step(t_arr, st_fp32)
            logits_int8, st_int8 = model_int8_py.forward_step(t_arr, st_int8)

        l_fp32_1d = logits_fp32[0]
        l_int8_1d = logits_int8[0]
        metrics = compute_logit_error_metrics(l_fp32_1d, l_int8_1d)

        # Generate 16 greedy tokens from both FP32 and Python INT8
        gen_fp32 = list(p_ids)
        gen_int8 = list(p_ids)
        cur_l_fp32 = l_fp32_1d.copy()
        cur_l_int8 = l_int8_1d.copy()
        step_top1_matches = 0
        for _ in range(16):
            next_fp32 = int(np.argmax(cur_l_fp32))
            next_int8 = int(np.argmax(cur_l_int8))
            if next_fp32 == next_int8:
                step_top1_matches += 1
            gen_fp32.append(next_fp32)
            gen_int8.append(next_int8)
            cur_l_fp32_batch, st_fp32 = model_fp32.forward_step(np.array([next_fp32], dtype=np.int64), st_fp32)
            cur_l_int8_batch, st_int8 = model_int8_py.forward_step(np.array([next_int8], dtype=np.int64), st_int8)
            cur_l_fp32 = cur_l_fp32_batch[0]
            cur_l_int8 = cur_l_int8_batch[0]

        prompt_parity_cases.append({
            "case_id": f"PARITY-{p_idx+1:02d}",
            "prompt": prompt,
            "prompt_token_ids": p_ids,
            "fp32_vs_int8_metrics": metrics,
            "greedy_16tok_agreement_rate": round(step_top1_matches / 16.0, 4),
            "fp32_decoded": tok.decode(gen_fp32),
            "int8_py_decoded": tok.decode(gen_int8),
            "int8_py_generated_ids": gen_int8[len(p_ids):],
            "fp32_logits_sample16": [round(float(x), 6) for x in l_fp32_1d[:16]],
            "int8_py_logits_full": [round(float(x), 7) for x in l_int8_1d],
        })

    base_report = {
        "numerical_tests": {**scan_res, **wy_res, **step_res, **param_res},
        "math_and_literature_audit": math_report,
        "corpus_audit": asdict(corpus_audit),
        "training_audit": training_report,
        "quantization_export_audit": {
            "deployed_dcore": quant_summary_deployed,
            "phase4_spec_dcore": quant_summary_p4,
        },
        "parity_cases": prompt_parity_cases,
    }
    with open("artifacts/python_audit_stage.json", "w", encoding="utf-8") as f:
        json.dump(base_report, f, indent=2)
    print("\nSaved artifacts/python_audit_stage.json successfully.")


if __name__ == "__main__":
    main()
