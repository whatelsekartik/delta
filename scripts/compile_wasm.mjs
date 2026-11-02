// scripts/compile_wasm.mjs
/**
 * Generates `runtime-wasm/deltacore_runtime.wat` and compiles it to a native
 * WebAssembly binary `artifacts/deltacore_runtime.wasm` using WABT.
 *
 * Implements the complete DeltaCore-LM INT8 inference engine inside WebAssembly:
 * - Zero host imports (`WebAssembly.instantiate(wasmBytes, {})`)
 * - IEEE-754 Cody-Waite range-reduced degree-6 minimax `expf`, `tanhf`, `sigmoidf`, `siluf`
 * - 4x unrolled INT8 per-row symmetric quantized matrix-vector multiply (`matvec_i8_add`)
 * - Depth-modulated Shared Operator Core (`mod_apply`: W_shared*x + U_l*(V_l^T*x) + b_l)
 * - Fast syntactic diagonal gated recurrence + Slow associative low-rank Delta-rule matrix memory
 * - Free-list session allocator (`wasm_create_session` / `wasm_free_session`) with zero heap leak
 * - Exports:
 *   - `memory`
 *   - `wasm_alloc(size) -> ptr`
 *   - `wasm_init_model(ptr, len) -> status`
 *   - `wasm_create_session(seed_i64) -> session_ptr`
 *   - `wasm_free_session(session_ptr)`
 *   - `wasm_forward_logits(session_ptr, token_id, logits_out_ptr) -> status`
 *   - `wasm_step_and_sample(session_ptr, token_id, temperature) -> next_token_id`
 *   - `wasm_get_session_state_bytes() -> bytes`
 *   - `wasm_get_heap_top() -> ptr`
 */

import fs from "fs";
import path from "path";
import wabtModule from "wabt";

const watSource = `(module
  (memory (export "memory") 64 512)

  ;; Global state pointers and model hyperparameters
  (global $heap_top (mut i32) (i32.const 65536))
  (global $free_session_head (mut i32) (i32.const 0))
  (global $scratch_ptr (mut i32) (i32.const 0))

  (global $model_ptr (mut i32) (i32.const 0))
  (global $vocab_size (mut i32) (i32.const 0))
  (global $d_model (mut i32) (i32.const 0))
  (global $d_embed (mut i32) (i32.const 0))
  (global $d_ff (mut i32) (i32.const 0))
  (global $n_layers (mut i32) (i32.const 0))
  (global $rank_mod (mut i32) (i32.const 0))
  (global $rank_slow (mut i32) (i32.const 0))
  (global $d_value (mut i32) (i32.const 0))
  (global $rms_eps (mut f32) (f32.const 0.00001))
  (global $conv_kernel (mut i32) (i32.const 3))

  ;; Precomputed offsets inside .dcore buffer
  (global $off_tok_embed (mut i32) (i32.const 0))
  (global $off_p_in (mut i32) (i32.const 0))
  (global $off_p_out (mut i32) (i32.const 0))
  (global $off_core_wu (mut i32) (i32.const 0))
  (global $off_core_wg (mut i32) (i32.const 0))
  (global $off_core_wo (mut i32) (i32.const 0))
  (global $off_core_wgate (mut i32) (i32.const 0))
  (global $off_core_wup (mut i32) (i32.const 0))
  (global $off_core_wdown (mut i32) (i32.const 0))
  (global $off_layers_start (mut i32) (i32.const 0))
  (global $layer_stride_bytes (mut i32) (i32.const 0))
  (global $off_final_norm (mut i32) (i32.const 0))
  (global $session_state_bytes (mut i32) (i32.const 0))

  ;; ---------------------------------------------------------------------------
  ;; Bump allocator (16-byte aligned)
  ;; ---------------------------------------------------------------------------
  (func $wasm_alloc (export "wasm_alloc") (param $size i32) (result i32)
    (local $ptr i32)
    (local $aligned i32)
    (local $new_top i32)
    (local $pages_needed i32)
    (local $cur_pages i32)
    (local.set $ptr (global.get $heap_top))
    (local.set $aligned (i32.and (i32.add (local.get $size) (i32.const 15)) (i32.const -16)))
    (local.set $new_top (i32.add (local.get $ptr) (local.get $aligned)))
    (local.set $pages_needed (i32.shr_u (i32.add (local.get $new_top) (i32.const 65535)) (i32.const 16)))
    (local.set $cur_pages (memory.size))
    (if (i32.gt_u (local.get $pages_needed) (local.get $cur_pages))
      (then
        (drop (memory.grow (i32.sub (local.get $pages_needed) (local.get $cur_pages))))
      )
    )
    (global.set $heap_top (local.get $new_top))
    (local.get $ptr)
  )

  (func (export "wasm_get_heap_top") (result i32)
    (global.get $heap_top)
  )

  (func (export "wasm_get_session_state_bytes") (result i32)
    (global.get $session_state_bytes)
  )

  ;; ---------------------------------------------------------------------------
  ;; Fast IEEE-754 Cody-Waite range-reduced expf(x), sigmoidf(x), siluf(x), tanhf(x)
  ;; ---------------------------------------------------------------------------
  (func $expf (param $x f32) (result f32)
    (local $xc f32)
    (local $kf f32)
    (local $k i32)
    (local $r f32)
    (local $p f32)
    (local $scale f32)
    ;; Clamp x to [-87.0, 88.0]
    (local.set $xc (f32.min (f32.max (local.get $x) (f32.const -87.0)) (f32.const 88.0)))
    ;; kf = nearest(xc * log2(e))
    (local.set $kf (f32.nearest (f32.mul (local.get $xc) (f32.const 1.4426950408889634))))
    (local.set $k (i32.trunc_f32_s (local.get $kf)))
    ;; Cody-Waite range reduction: r = xc - kf * ln2_hi - kf * ln2_lo
    (local.set $r
      (f32.sub
        (f32.sub (local.get $xc) (f32.mul (local.get $kf) (f32.const 0.693145751953125)))
        (f32.mul (local.get $kf) (f32.const 0.000001428606765330187))
      )
    )
    ;; Degree-6 Horner polynomial on [-0.3466, 0.3466]
    (local.set $p (f32.add (f32.const 0.008333333767950535) (f32.mul (local.get $r) (f32.const 0.0013888889225199819))))
    (local.set $p (f32.add (f32.const 0.0416666679084301) (f32.mul (local.get $r) (local.get $p))))
    (local.set $p (f32.add (f32.const 0.1666666716337204) (f32.mul (local.get $r) (local.get $p))))
    (local.set $p (f32.add (f32.const 0.5) (f32.mul (local.get $r) (local.get $p))))
    (local.set $p (f32.add (f32.const 1.0) (f32.mul (local.get $r) (local.get $p))))
    (local.set $p (f32.add (f32.const 1.0) (f32.mul (local.get $r) (local.get $p))))
    ;; Construct 2^k via IEEE-754 exponent bits: ((k + 127) << 23)
    (local.set $scale (f32.reinterpret_i32 (i32.shl (i32.add (local.get $k) (i32.const 127)) (i32.const 23))))
    (f32.mul (local.get $p) (local.get $scale))
  )

  (func $sigmoidf (param $x f32) (result f32)
    (f32.div (f32.const 1.0) (f32.add (f32.const 1.0) (call $expf (f32.neg (local.get $x)))))
  )

  (func $siluf (param $x f32) (result f32)
    (f32.div (local.get $x) (f32.add (f32.const 1.0) (call $expf (f32.neg (local.get $x)))))
  )

  (func $tanhf (param $x f32) (result f32)
    (local $e2x f32)
    (if (result f32) (f32.gt (f32.abs (local.get $x)) (f32.const 12.0))
      (then (f32.copysign (f32.const 1.0) (local.get $x)))
      (else
        (local.set $e2x (call $expf (f32.mul (f32.const 2.0) (local.get $x))))
        (f32.div (f32.sub (local.get $e2x) (f32.const 1.0)) (f32.add (local.get $e2x) (f32.const 1.0)))
      )
    )
  )

  ;; ---------------------------------------------------------------------------
  ;; Helper: byte size of an INT8 QuantizedMatrix(rows, cols) = rows * (cols + 4)
  ;; ---------------------------------------------------------------------------
  (func $i8_mat_bytes (param $rows i32) (param $cols i32) (result i32)
    (i32.mul (local.get $rows) (i32.add (local.get $cols) (i32.const 4)))
  )

  ;; ---------------------------------------------------------------------------
  ;; Initialize Model from .dcore buffer in WASM memory
  ;; ---------------------------------------------------------------------------
  (func (export "wasm_init_model") (param $ptr i32) (param $len i32) (result i32)
    (local $d i32)
    (local $de i32)
    (local $dff i32)
    (local $L i32)
    (local $rm i32)
    (local $rs i32)
    (local $dv i32)
    (local $C i32)
    (local $V i32)
    (local $cur i32)
    (local $l_bytes i32)
    (local $floats_per_layer i32)

    ;; Check header length >= 48 and magic == "DCOR" (0x524f4344) and version == 1
    (if (i32.lt_u (local.get $len) (i32.const 48)) (then (return (i32.const -1))))
    (if (i32.ne (i32.load (local.get $ptr)) (i32.const 0x524f4344)) (then (return (i32.const -1))))
    (if (i32.ne (i32.load (i32.add (local.get $ptr) (i32.const 4))) (i32.const 1)) (then (return (i32.const -1))))

    (global.set $model_ptr (local.get $ptr))
    (local.set $V (i32.load (i32.add (local.get $ptr) (i32.const 8))))
    (local.set $d (i32.load (i32.add (local.get $ptr) (i32.const 12))))
    (local.set $de (i32.load (i32.add (local.get $ptr) (i32.const 16))))
    (local.set $dff (i32.load (i32.add (local.get $ptr) (i32.const 20))))
    (local.set $L (i32.load (i32.add (local.get $ptr) (i32.const 24))))
    (local.set $rm (i32.load (i32.add (local.get $ptr) (i32.const 28))))
    (local.set $rs (i32.load (i32.add (local.get $ptr) (i32.const 32))))
    (local.set $dv (i32.load (i32.add (local.get $ptr) (i32.const 36))))
    (global.set $rms_eps (f32.load (i32.add (local.get $ptr) (i32.const 40))))
    (local.set $C (i32.load (i32.add (local.get $ptr) (i32.const 44))))

    (global.set $vocab_size (local.get $V))
    (global.set $d_model (local.get $d))
    (global.set $d_embed (local.get $de))
    (global.set $d_ff (local.get $dff))
    (global.set $n_layers (local.get $L))
    (global.set $rank_mod (local.get $rm))
    (global.set $rank_slow (local.get $rs))
    (global.set $d_value (local.get $dv))
    (global.set $conv_kernel (local.get $C))

    ;; Compute tensor offsets
    (local.set $cur (i32.add (local.get $ptr) (i32.const 48)))
    (global.set $off_tok_embed (local.get $cur))
    (local.set $cur (i32.add (local.get $cur) (call $i8_mat_bytes (local.get $V) (local.get $de))))

    (global.set $off_p_in (local.get $cur))
    (local.set $cur (i32.add (local.get $cur) (call $i8_mat_bytes (local.get $d) (local.get $de))))

    (global.set $off_p_out (local.get $cur))
    (local.set $cur (i32.add (local.get $cur) (call $i8_mat_bytes (local.get $de) (local.get $d))))

    ;; Shared core
    (global.set $off_core_wu (local.get $cur))
    (local.set $cur (i32.add (local.get $cur) (call $i8_mat_bytes (local.get $d) (local.get $d))))

    (global.set $off_core_wg (local.get $cur))
    (local.set $cur (i32.add (local.get $cur) (call $i8_mat_bytes (local.get $d) (local.get $d))))

    (global.set $off_core_wo (local.get $cur))
    (local.set $cur (i32.add (local.get $cur) (call $i8_mat_bytes (local.get $d) (local.get $d))))

    (global.set $off_core_wgate (local.get $cur))
    (local.set $cur (i32.add (local.get $cur) (call $i8_mat_bytes (local.get $dff) (local.get $d))))

    (global.set $off_core_wup (local.get $cur))
    (local.set $cur (i32.add (local.get $cur) (call $i8_mat_bytes (local.get $dff) (local.get $d))))

    (global.set $off_core_wdown (local.get $cur))
    (local.set $cur (i32.add (local.get $cur) (call $i8_mat_bytes (local.get $d) (local.get $dff))))

    (global.set $off_layers_start (local.get $cur))

    ;; Per-layer byte stride
    (local.set $l_bytes
      (i32.add
        (i32.mul (i32.mul (i32.add (i32.const 2) (local.get $C)) (local.get $d)) (i32.const 4)) ;; norm1, norm2, conv
        (i32.add
          (i32.add
            ;; mod_u, mod_g (with bias) + mod_o (no bias)
            (i32.add
              (i32.mul (i32.const 6) (call $i8_mat_bytes (local.get $d) (local.get $rm)))
              (i32.mul (i32.const 8) (local.get $d))
            )
            ;; mod_gate, mod_up, mod_down (3 * (i8(dff,rm) + i8(d,rm)))
            (i32.mul
              (i32.const 3)
              (i32.add
                (call $i8_mat_bytes (local.get $dff) (local.get $rm))
                (call $i8_mat_bytes (local.get $d) (local.get $rm))
              )
            )
          )
          (i32.add
            ;; w_alpha + b_alpha
            (i32.add (call $i8_mat_bytes (local.get $d) (local.get $d)) (i32.mul (local.get $d) (i32.const 4)))
            (i32.add
              ;; w_k, w_q, w_v
              (i32.add
                (i32.mul (i32.const 2) (call $i8_mat_bytes (local.get $rs) (local.get $d)))
                (call $i8_mat_bytes (local.get $dv) (local.get $d))
              )
              ;; w_beta, b_beta, w_gamma, b_gamma + p_slow
              (i32.add
                (i32.mul (i32.const 2) (i32.mul (i32.add (local.get $d) (i32.const 1)) (i32.const 4)))
                (call $i8_mat_bytes (local.get $d) (local.get $dv))
              )
            )
          )
        )
      )
    )
    (global.set $layer_stride_bytes (local.get $l_bytes))
    (global.set $off_final_norm (i32.add (local.get $cur) (i32.mul (local.get $L) (local.get $l_bytes))))

    ;; Verify total file length matches expected end offset
    (if
      (i32.ne
        (i32.add (global.get $off_final_norm) (i32.mul (local.get $d) (i32.const 4)))
        (i32.add (local.get $ptr) (local.get $len))
      )
      (then (return (i32.const -2)))
    )

    ;; Compute session state bytes: L * ((C-1)*d + d + rs*dv) * 4
    (local.set $floats_per_layer
      (i32.add
        (i32.mul (local.get $C) (local.get $d))
        (i32.mul (local.get $rs) (local.get $dv))
      )
    )
    (global.set $session_state_bytes (i32.mul (i32.mul (local.get $L) (local.get $floats_per_layer)) (i32.const 4)))

    ;; Allocate reusable scratch workspace (64 KiB) if not yet allocated
    (if (i32.eqz (global.get $scratch_ptr))
      (then (global.set $scratch_ptr (call $wasm_alloc (i32.const 65536))))
    )
    (i32.const 0)
  )

  ;; ---------------------------------------------------------------------------
  ;; Session Management: Layout of SessionState block:
  ;;   offset 0..8:   rng_state (i64)
  ;;   offset 8..12:  tokens_generated (i32)
  ;;   offset 12..16: next_free_ptr (i32)
  ;;   offset 16..:   layer states (session_state_bytes)
  ;; ---------------------------------------------------------------------------
  (func (export "wasm_create_session") (param $seed i64) (result i32)
    (local $ptr i32)
    (local $total_bytes i32)
    (local.set $total_bytes (i32.add (i32.const 16) (global.get $session_state_bytes)))

    (if (i32.ne (global.get $free_session_head) (i32.const 0))
      (then
        (local.set $ptr (global.get $free_session_head))
        (global.set $free_session_head (i32.load (i32.add (local.get $ptr) (i32.const 12))))
      )
      (else
        (local.set $ptr (call $wasm_alloc (local.get $total_bytes)))
      )
    )
    ;; Zero-fill session block
    (memory.fill (local.get $ptr) (i32.const 0) (local.get $total_bytes))
    (if (i64.le_u (local.get $seed) (i64.const 0))
      (then (local.set $seed (i64.const 1)))
    )
    (i64.store (local.get $ptr) (local.get $seed))
    (local.get $ptr)
  )

  (func (export "wasm_free_session") (param $session_ptr i32)
    (if (i32.ne (local.get $session_ptr) (i32.const 0))
      (then
        (i32.store (i32.add (local.get $session_ptr) (i32.const 12)) (global.get $free_session_head))
        (global.set $free_session_head (local.get $session_ptr))
      )
    )
  )

  ;; ---------------------------------------------------------------------------
  ;; Matrix-Vector Kernels
  ;; ---------------------------------------------------------------------------
  ;; QuantizedMatrix at $mat_ptr: [rows * f32 scales][rows * cols * i8 weights]
  ;; Computes out[r] += scale[r] * sum_c (data[r, c] * x[c])
  (func $matvec_i8_add (param $mat_ptr i32) (param $rows i32) (param $cols i32) (param $x_ptr i32) (param $out_ptr i32)
    (local $r i32)
    (local $c i32)
    (local $data_base i32)
    (local $row_ptr i32)
    (local $acc0 f32)
    (local $acc1 f32)
    (local $acc2 f32)
    (local $acc3 f32)
    (local $out_elem_ptr i32)

    (local.set $data_base (i32.add (local.get $mat_ptr) (i32.shl (local.get $rows) (i32.const 2))))
    (local.set $r (i32.const 0))
    (block $break_r
      (loop $loop_r
        (br_if $break_r (i32.ge_u (local.get $r) (local.get $rows)))
        (local.set $row_ptr (i32.add (local.get $data_base) (i32.mul (local.get $r) (local.get $cols))))
        (local.set $acc0 (f32.const 0.0))
        (local.set $acc1 (f32.const 0.0))
        (local.set $acc2 (f32.const 0.0))
        (local.set $acc3 (f32.const 0.0))
        (local.set $c (i32.const 0))
        (block $break_c
          (loop $loop_c
            (br_if $break_c (i32.ge_u (local.get $c) (local.get $cols)))
            (local.set $acc0
              (f32.add
                (local.get $acc0)
                (f32.mul
                  (f32.convert_i32_s (i32.load8_s (i32.add (local.get $row_ptr) (local.get $c))))
                  (f32.load (i32.add (local.get $x_ptr) (i32.shl (local.get $c) (i32.const 2))))
                )
              )
            )
            (local.set $acc1
              (f32.add
                (local.get $acc1)
                (f32.mul
                  (f32.convert_i32_s (i32.load8_s (i32.add (local.get $row_ptr) (i32.add (local.get $c) (i32.const 1)))))
                  (f32.load (i32.add (local.get $x_ptr) (i32.shl (i32.add (local.get $c) (i32.const 1)) (i32.const 2))))
                )
              )
            )
            (local.set $acc2
              (f32.add
                (local.get $acc2)
                (f32.mul
                  (f32.convert_i32_s (i32.load8_s (i32.add (local.get $row_ptr) (i32.add (local.get $c) (i32.const 2)))))
                  (f32.load (i32.add (local.get $x_ptr) (i32.shl (i32.add (local.get $c) (i32.const 2)) (i32.const 2))))
                )
              )
            )
            (local.set $acc3
              (f32.add
                (local.get $acc3)
                (f32.mul
                  (f32.convert_i32_s (i32.load8_s (i32.add (local.get $row_ptr) (i32.add (local.get $c) (i32.const 3)))))
                  (f32.load (i32.add (local.get $x_ptr) (i32.shl (i32.add (local.get $c) (i32.const 3)) (i32.const 2))))
                )
              )
            )
            (local.set $c (i32.add (local.get $c) (i32.const 4)))
            (br $loop_c)
          )
        )
        (local.set $out_elem_ptr (i32.add (local.get $out_ptr) (i32.shl (local.get $r) (i32.const 2))))
        (f32.store
          (local.get $out_elem_ptr)
          (f32.add
            (f32.load (local.get $out_elem_ptr))
            (f32.mul
              (f32.add (f32.add (local.get $acc0) (local.get $acc1)) (f32.add (local.get $acc2) (local.get $acc3)))
              (f32.load (i32.add (local.get $mat_ptr) (i32.shl (local.get $r) (i32.const 2))))
            )
          )
        )
        (local.set $r (i32.add (local.get $r) (i32.const 1)))
        (br $loop_r)
      )
    )
  )

  (func $matvec_i8 (param $mat_ptr i32) (param $rows i32) (param $cols i32) (param $x_ptr i32) (param $out_ptr i32)
    (memory.fill (local.get $out_ptr) (i32.const 0) (i32.shl (local.get $rows) (i32.const 2)))
    (call $matvec_i8_add (local.get $mat_ptr) (local.get $rows) (local.get $cols) (local.get $x_ptr) (local.get $out_ptr))
  )

  ;; DepthModulator apply:
  ;; Computes out = bias + W_shared * x + U * (V^T * x), returns next cursor after modulator in .dcore
  (func $mod_apply
    (param $cur i32) (param $d_in i32) (param $d_out i32) (param $rm i32) (param $has_bias i32)
    (param $w_shared_ptr i32) (param $x_ptr i32) (param $scratch_rm_ptr i32) (param $out_ptr i32)
    (result i32)
    (local $u_ptr i32)
    (local $v_ptr i32)
    (local $b_ptr i32)
    (local $next_cur i32)
    (local $v_data_base i32)
    (local $i i32)
    (local $k i32)
    (local $scale_x f32)
    (local $row_ptr i32)
    (local $sk_ptr i32)

    (local.set $u_ptr (local.get $cur))
    (local.set $v_ptr (i32.add (local.get $u_ptr) (call $i8_mat_bytes (local.get $d_out) (local.get $rm))))
    (local.set $b_ptr (i32.add (local.get $v_ptr) (call $i8_mat_bytes (local.get $d_in) (local.get $rm))))

    (if (local.get $has_bias)
      (then
        (memory.copy (local.get $out_ptr) (local.get $b_ptr) (i32.shl (local.get $d_out) (i32.const 2)))
        (local.set $next_cur (i32.add (local.get $b_ptr) (i32.shl (local.get $d_out) (i32.const 2))))
      )
      (else
        (memory.fill (local.get $out_ptr) (i32.const 0) (i32.shl (local.get $d_out) (i32.const 2)))
        (local.set $next_cur (local.get $b_ptr))
      )
    )

    ;; out += W_shared * x
    (call $matvec_i8_add (local.get $w_shared_ptr) (local.get $d_out) (local.get $d_in) (local.get $x_ptr) (local.get $out_ptr))

    ;; scratch_rm = V^T * x where V is stored as (d_in, rm)
    (memory.fill (local.get $scratch_rm_ptr) (i32.const 0) (i32.shl (local.get $rm) (i32.const 2)))
    (local.set $v_data_base (i32.add (local.get $v_ptr) (i32.shl (local.get $d_in) (i32.const 2))))
    (local.set $i (i32.const 0))
    (block $break_i
      (loop $loop_i
        (br_if $break_i (i32.ge_u (local.get $i) (local.get $d_in)))
        (local.set $scale_x
          (f32.mul
            (f32.load (i32.add (local.get $v_ptr) (i32.shl (local.get $i) (i32.const 2))))
            (f32.load (i32.add (local.get $x_ptr) (i32.shl (local.get $i) (i32.const 2))))
          )
        )
        (local.set $row_ptr (i32.add (local.get $v_data_base) (i32.mul (local.get $i) (local.get $rm))))
        (local.set $k (i32.const 0))
        (block $break_k
          (loop $loop_k
            (br_if $break_k (i32.ge_u (local.get $k) (local.get $rm)))
            (local.set $sk_ptr (i32.add (local.get $scratch_rm_ptr) (i32.shl (local.get $k) (i32.const 2))))
            (f32.store
              (local.get $sk_ptr)
              (f32.add
                (f32.load (local.get $sk_ptr))
                (f32.mul (f32.convert_i32_s (i32.load8_s (i32.add (local.get $row_ptr) (local.get $k)))) (local.get $scale_x))
              )
            )
            (local.set $k (i32.add (local.get $k) (i32.const 1)))
            (br $loop_k)
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop_i)
      )
    )

    ;; out += U * scratch_rm
    (call $matvec_i8_add (local.get $u_ptr) (local.get $d_out) (local.get $rm) (local.get $scratch_rm_ptr) (local.get $out_ptr))
    (local.get $next_cur)
  )

  ;; RMSNorm: out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
  (func $rms_norm (param $x_ptr i32) (param $w_ptr i32) (param $n i32) (param $eps f32) (param $out_ptr i32)
    (local $i i32)
    (local $sum_sq f32)
    (local $v f32)
    (local $inv_rms f32)
    (local.set $sum_sq (f32.const 0.0))
    (local.set $i (i32.const 0))
    (block $brk
      (loop $lp
        (br_if $brk (i32.ge_u (local.get $i) (local.get $n)))
        (local.set $v (f32.load (i32.add (local.get $x_ptr) (i32.shl (local.get $i) (i32.const 2)))))
        (local.set $sum_sq (f32.add (local.get $sum_sq) (f32.mul (local.get $v) (local.get $v))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $lp)
      )
    )
    (local.set $inv_rms
      (f32.div
        (f32.const 1.0)
        (f32.sqrt (f32.add (f32.div (local.get $sum_sq) (f32.convert_i32_u (local.get $n))) (local.get $eps)))
      )
    )
    (local.set $i (i32.const 0))
    (block $brk2
      (loop $lp2
        (br_if $brk2 (i32.ge_u (local.get $i) (local.get $n)))
        (f32.store
          (i32.add (local.get $out_ptr) (i32.shl (local.get $i) (i32.const 2)))
          (f32.mul
            (f32.mul
              (f32.load (i32.add (local.get $x_ptr) (i32.shl (local.get $i) (i32.const 2))))
              (local.get $inv_rms)
            )
            (f32.load (i32.add (local.get $w_ptr) (i32.shl (local.get $i) (i32.const 2))))
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $lp2)
      )
    )
  )

  ;; ---------------------------------------------------------------------------
  ;; Full Single-Token Forward Step: wasm_forward_logits
  ;; ---------------------------------------------------------------------------
  (func $wasm_forward_logits (export "wasm_forward_logits")
    (param $session_ptr i32) (param $token_id i32) (param $logits_out_ptr i32) (result i32)
    (local $d i32) (local $de i32) (local $dff i32) (local $L i32)
    (local $rm i32) (local $rs i32) (local $dv i32) (local $C i32) (local $V i32)
    (local $scr i32)
    (local $e_ptr i32) (local $h_ptr i32) (local $z_ptr i32) (local $z_shift_ptr i32)
    (local $scratch_rm_ptr i32) (local $u_vec_ptr i32) (local $g_vec_ptr i32) (local $alpha_vec_ptr i32)
    (local $k_vec_ptr i32) (local $q_vec_ptr i32) (local $v_vec_ptr i32) (local $s_vec_ptr i32)
    (local $combined_ptr i32) (local $o_proj_ptr i32) (local $ff_gate_ptr i32) (local $ff_up_ptr i32)
    (local $emb_scale f32) (local $emb_row_ptr i32)
    (local $i i32) (local $l_idx i32) (local $tau i32) (local $j i32) (local $r i32)
    (local $cur i32) (local $st_ptr i32)
    (local $conv_buf_ptr i32) (local $fast_state_ptr i32) (local $slow_mem_ptr i32)
    (local $norm1_ptr i32) (local $norm2_ptr i32) (local $conv_w_ptr i32)
    (local $w_alpha_ptr i32) (local $b_alpha_ptr i32)
    (local $w_k_ptr i32) (local $w_q_ptr i32) (local $w_v_ptr i32)
    (local $w_beta_ptr i32) (local $b_beta f32) (local $w_gamma_ptr i32) (local $b_gamma f32)
    (local $p_slow_ptr i32)
    (local $a_val f32) (local $u_val f32) (local $f_old f32)
    (local $k_norm_sq f32) (local $inv_k_norm f32) (local $inv_sqrt_rs f32)
    (local $beta_dot f32) (local $gamma_dot f32) (local $beta f32) (local $gamma f32)
    (local $v_hat_j f32) (local $delta_j f32) (local $readout_j f32) (local $m_dec f32) (local $m_new f32)
    (local $m_idx_ptr i32)

    (local.set $V (global.get $vocab_size))
    (local.set $d (global.get $d_model))
    (local.set $de (global.get $d_embed))
    (local.set $dff (global.get $d_ff))
    (local.set $L (global.get $n_layers))
    (local.set $rm (global.get $rank_mod))
    (local.set $rs (global.get $rank_slow))
    (local.set $dv (global.get $d_value))
    (local.set $C (global.get $conv_kernel))

    ;; Partition scratch workspace (each slot 2048 bytes = 512 floats)
    (local.set $scr (global.get $scratch_ptr))
    (local.set $e_ptr (local.get $scr))
    (local.set $h_ptr (i32.add (local.get $scr) (i32.const 2048)))
    (local.set $z_ptr (i32.add (local.get $scr) (i32.const 4096)))
    (local.set $z_shift_ptr (i32.add (local.get $scr) (i32.const 6144)))
    (local.set $scratch_rm_ptr (i32.add (local.get $scr) (i32.const 8192)))
    (local.set $u_vec_ptr (i32.add (local.get $scr) (i32.const 10240)))
    (local.set $g_vec_ptr (i32.add (local.get $scr) (i32.const 12288)))
    (local.set $alpha_vec_ptr (i32.add (local.get $scr) (i32.const 14336)))
    (local.set $k_vec_ptr (i32.add (local.get $scr) (i32.const 16384)))
    (local.set $q_vec_ptr (i32.add (local.get $scr) (i32.const 18432)))
    (local.set $v_vec_ptr (i32.add (local.get $scr) (i32.const 20480)))
    (local.set $s_vec_ptr (i32.add (local.get $scr) (i32.const 22528)))
    (local.set $combined_ptr (i32.add (local.get $scr) (i32.const 24576)))
    (local.set $o_proj_ptr (i32.add (local.get $scr) (i32.const 26624)))
    (local.set $ff_gate_ptr (i32.add (local.get $scr) (i32.const 28672)))
    (local.set $ff_up_ptr (i32.add (local.get $scr) (i32.const 32768)))

    ;; 1. Token embedding lookup + P_in projection
    (local.set $emb_scale
      (f32.load (i32.add (global.get $off_tok_embed) (i32.shl (local.get $token_id) (i32.const 2))))
    )
    (local.set $emb_row_ptr
      (i32.add
        (i32.add (global.get $off_tok_embed) (i32.shl (local.get $V) (i32.const 2)))
        (i32.mul (local.get $token_id) (local.get $de))
      )
    )
    (local.set $i (i32.const 0))
    (block $brk_emb
      (loop $lp_emb
        (br_if $brk_emb (i32.ge_u (local.get $i) (local.get $de)))
        (f32.store
          (i32.add (local.get $e_ptr) (i32.shl (local.get $i) (i32.const 2)))
          (f32.mul
            (f32.convert_i32_s (i32.load8_s (i32.add (local.get $emb_row_ptr) (local.get $i))))
            (local.get $emb_scale)
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $lp_emb)
      )
    )
    (call $matvec_i8 (global.get $off_p_in) (local.get $d) (local.get $de) (local.get $e_ptr) (local.get $h_ptr))

    ;; 2. Iterate through all L layers
    (local.set $cur (global.get $off_layers_start))
    (local.set $st_ptr (i32.add (local.get $session_ptr) (i32.const 16)))
    (local.set $l_idx (i32.const 0))

    (block $brk_layer
      (loop $lp_layer
        (br_if $brk_layer (i32.ge_u (local.get $l_idx) (local.get $L)))

        ;; Session pointers for layer l_idx
        (local.set $conv_buf_ptr (local.get $st_ptr))
        (local.set $fast_state_ptr
          (i32.add (local.get $conv_buf_ptr) (i32.shl (i32.mul (i32.sub (local.get $C) (i32.const 1)) (local.get $d)) (i32.const 2)))
        )
        (local.set $slow_mem_ptr
          (i32.add (local.get $fast_state_ptr) (i32.shl (local.get $d) (i32.const 2)))
        )
        (local.set $st_ptr
          (i32.add (local.get $slow_mem_ptr) (i32.shl (i32.mul (local.get $rs) (local.get $dv)) (i32.const 2)))
        )

        ;; Layer weights: norm1, norm2, conv_weight
        (local.set $norm1_ptr (local.get $cur))
        (local.set $norm2_ptr (i32.add (local.get $norm1_ptr) (i32.shl (local.get $d) (i32.const 2))))
        (local.set $conv_w_ptr (i32.add (local.get $norm2_ptr) (i32.shl (local.get $d) (i32.const 2))))
        (local.set $cur (i32.add (local.get $conv_w_ptr) (i32.shl (i32.mul (local.get $C) (local.get $d)) (i32.const 2))))

        ;; RMSNorm1
        (call $rms_norm (local.get $h_ptr) (local.get $norm1_ptr) (local.get $d) (global.get $rms_eps) (local.get $z_ptr))

        ;; Causal micro-conv (C=3)
        (local.set $i (i32.const 0))
        (block $brk_c0
          (loop $lp_c0
            (br_if $brk_c0 (i32.ge_u (local.get $i) (local.get $d)))
            (f32.store
              (i32.add (local.get $z_shift_ptr) (i32.shl (local.get $i) (i32.const 2)))
              (f32.mul
                (f32.load (i32.add (local.get $z_ptr) (i32.shl (local.get $i) (i32.const 2))))
                (f32.load (i32.add (local.get $conv_w_ptr) (i32.shl (local.get $i) (i32.const 2))))
              )
            )
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $lp_c0)
          )
        )
        (local.set $tau (i32.const 1))
        (block $brk_tau
          (loop $lp_tau
            (br_if $brk_tau (i32.ge_u (local.get $tau) (local.get $C)))
            (local.set $i (i32.const 0))
            (block $brk_ci
              (loop $lp_ci
                (br_if $brk_ci (i32.ge_u (local.get $i) (local.get $d)))
                (f32.store
                  (i32.add (local.get $z_shift_ptr) (i32.shl (local.get $i) (i32.const 2)))
                  (f32.add
                    (f32.load (i32.add (local.get $z_shift_ptr) (i32.shl (local.get $i) (i32.const 2))))
                    (f32.mul
                      (f32.load
                        (i32.add
                          (local.get $conv_buf_ptr)
                          (i32.shl
                            (i32.add (i32.mul (i32.sub (i32.sub (local.get $C) (i32.const 1)) (local.get $tau)) (local.get $d)) (local.get $i))
                            (i32.const 2)
                          )
                        )
                      )
                      (f32.load
                        (i32.add
                          (local.get $conv_w_ptr)
                          (i32.shl (i32.add (i32.mul (local.get $tau) (local.get $d)) (local.get $i)) (i32.const 2))
                        )
                      )
                    )
                  )
                )
                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                (br $lp_ci)
              )
            )
            (local.set $tau (i32.add (local.get $tau) (i32.const 1)))
            (br $lp_tau)
          )
        )

        ;; Shift conv_buf: copy slot 1..C-1 down to 0..C-2, then store z at C-2
        (if (i32.gt_u (local.get $C) (i32.const 2))
          (then
            (memory.copy
              (local.get $conv_buf_ptr)
              (i32.add (local.get $conv_buf_ptr) (i32.shl (local.get $d) (i32.const 2)))
              (i32.shl (i32.mul (i32.sub (local.get $C) (i32.const 2)) (local.get $d)) (i32.const 2))
            )
          )
        )
        (memory.copy
          (i32.add (local.get $conv_buf_ptr) (i32.shl (i32.mul (i32.sub (local.get $C) (i32.const 2)) (local.get $d)) (i32.const 2)))
          (local.get $z_ptr)
          (i32.shl (local.get $d) (i32.const 2))
        )

        ;; Apply mod_u and mod_g, save mod_o / mod_gate / mod_up / mod_down cursors
        (local.set $cur
          (call $mod_apply
            (local.get $cur) (local.get $d) (local.get $d) (local.get $rm) (i32.const 1)
            (global.get $off_core_wu) (local.get $z_shift_ptr) (local.get $scratch_rm_ptr) (local.get $u_vec_ptr)
          )
        )
        (local.set $cur
          (call $mod_apply
            (local.get $cur) (local.get $d) (local.get $d) (local.get $rm) (i32.const 1)
            (global.get $off_core_wg) (local.get $z_shift_ptr) (local.get $scratch_rm_ptr) (local.get $g_vec_ptr)
          )
        )
        ;; Save cursor to mod_o in local $norm1_ptr (reusing local) and advance past mod_o, mod_gate, mod_up, mod_down
        (local.set $norm1_ptr (local.get $cur))
        (local.set $cur
          (i32.add
            (local.get $cur)
            (i32.add
              (i32.mul (i32.const 2) (call $i8_mat_bytes (local.get $d) (local.get $rm)))
              (i32.mul
                (i32.const 3)
                (i32.add
                  (call $i8_mat_bytes (local.get $dff) (local.get $rm))
                  (call $i8_mat_bytes (local.get $d) (local.get $rm))
                )
              )
            )
          )
        )

        ;; Fast Syntactic Stream: w_alpha, b_alpha
        (local.set $w_alpha_ptr (local.get $cur))
        (local.set $b_alpha_ptr (i32.add (local.get $w_alpha_ptr) (call $i8_mat_bytes (local.get $d) (local.get $d))))
        (local.set $cur (i32.add (local.get $b_alpha_ptr) (i32.shl (local.get $d) (i32.const 2))))

        (memory.copy (local.get $alpha_vec_ptr) (local.get $b_alpha_ptr) (i32.shl (local.get $d) (i32.const 2)))
        (call $matvec_i8_add (local.get $w_alpha_ptr) (local.get $d) (local.get $d) (local.get $z_shift_ptr) (local.get $alpha_vec_ptr))

        (local.set $i (i32.const 0))
        (block $brk_fast
          (loop $lp_fast
            (br_if $brk_fast (i32.ge_u (local.get $i) (local.get $d)))
            (local.set $a_val (call $sigmoidf (f32.load (i32.add (local.get $alpha_vec_ptr) (i32.shl (local.get $i) (i32.const 2))))))
            (local.set $u_val (call $tanhf (f32.load (i32.add (local.get $u_vec_ptr) (i32.shl (local.get $i) (i32.const 2))))))
            (local.set $f_old (f32.load (i32.add (local.get $fast_state_ptr) (i32.shl (local.get $i) (i32.const 2)))))
            (f32.store
              (i32.add (local.get $fast_state_ptr) (i32.shl (local.get $i) (i32.const 2)))
              (f32.add
                (f32.mul (local.get $a_val) (local.get $f_old))
                (f32.mul (f32.sub (f32.const 1.0) (local.get $a_val)) (local.get $u_val))
              )
            )
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $lp_fast)
          )
        )

        ;; Slow Associative Delta Stream: w_k, w_q, w_v
        (local.set $w_k_ptr (local.get $cur))
        (local.set $w_q_ptr (i32.add (local.get $w_k_ptr) (call $i8_mat_bytes (local.get $rs) (local.get $d))))
        (local.set $w_v_ptr (i32.add (local.get $w_q_ptr) (call $i8_mat_bytes (local.get $rs) (local.get $d))))
        (local.set $cur (i32.add (local.get $w_v_ptr) (call $i8_mat_bytes (local.get $dv) (local.get $d))))

        (call $matvec_i8 (local.get $w_k_ptr) (local.get $rs) (local.get $d) (local.get $z_shift_ptr) (local.get $k_vec_ptr))
        (local.set $k_norm_sq (f32.const 0.0))
        (local.set $i (i32.const 0))
        (block $brk_kn
          (loop $lp_kn
            (br_if $brk_kn (i32.ge_u (local.get $i) (local.get $rs)))
            (local.set $a_val (f32.load (i32.add (local.get $k_vec_ptr) (i32.shl (local.get $i) (i32.const 2)))))
            (local.set $k_norm_sq (f32.add (local.get $k_norm_sq) (f32.mul (local.get $a_val) (local.get $a_val))))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $lp_kn)
          )
        )
        (local.set $inv_k_norm (f32.div (f32.const 1.0) (f32.add (f32.sqrt (local.get $k_norm_sq)) (f32.const 0.000001))))
        (local.set $i (i32.const 0))
        (block $brk_kns
          (loop $lp_kns
            (br_if $brk_kns (i32.ge_u (local.get $i) (local.get $rs)))
            (f32.store
              (i32.add (local.get $k_vec_ptr) (i32.shl (local.get $i) (i32.const 2)))
              (f32.mul (f32.load (i32.add (local.get $k_vec_ptr) (i32.shl (local.get $i) (i32.const 2)))) (local.get $inv_k_norm))
            )
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $lp_kns)
          )
        )

        (call $matvec_i8 (local.get $w_q_ptr) (local.get $rs) (local.get $d) (local.get $z_shift_ptr) (local.get $q_vec_ptr))
        (local.set $inv_sqrt_rs (f32.div (f32.const 1.0) (f32.sqrt (f32.convert_i32_u (local.get $rs)))))
        (local.set $i (i32.const 0))
        (block $brk_qs
          (loop $lp_qs
            (br_if $brk_qs (i32.ge_u (local.get $i) (local.get $rs)))
            (f32.store
              (i32.add (local.get $q_vec_ptr) (i32.shl (local.get $i) (i32.const 2)))
              (f32.mul (f32.load (i32.add (local.get $q_vec_ptr) (i32.shl (local.get $i) (i32.const 2)))) (local.get $inv_sqrt_rs))
            )
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $lp_qs)
          )
        )

        (call $matvec_i8 (local.get $w_v_ptr) (local.get $dv) (local.get $d) (local.get $z_shift_ptr) (local.get $v_vec_ptr))

        ;; w_beta, b_beta, w_gamma, b_gamma
        (local.set $w_beta_ptr (local.get $cur))
        (local.set $b_beta (f32.load (i32.add (local.get $w_beta_ptr) (i32.shl (local.get $d) (i32.const 2)))))
        (local.set $w_gamma_ptr (i32.add (local.get $w_beta_ptr) (i32.shl (i32.add (local.get $d) (i32.const 1)) (i32.const 2))))
        (local.set $b_gamma (f32.load (i32.add (local.get $w_gamma_ptr) (i32.shl (local.get $d) (i32.const 2)))))
        (local.set $p_slow_ptr (i32.add (local.get $w_gamma_ptr) (i32.shl (i32.add (local.get $d) (i32.const 1)) (i32.const 2))))
        (local.set $cur (i32.add (local.get $p_slow_ptr) (call $i8_mat_bytes (local.get $d) (local.get $dv))))

        (local.set $beta_dot (local.get $b_beta))
        (local.set $gamma_dot (local.get $b_gamma))
        (local.set $i (i32.const 0))
        (block $brk_bg
          (loop $lp_bg
            (br_if $brk_bg (i32.ge_u (local.get $i) (local.get $d)))
            (local.set $a_val (f32.load (i32.add (local.get $z_shift_ptr) (i32.shl (local.get $i) (i32.const 2)))))
            (local.set $beta_dot
              (f32.add (local.get $beta_dot) (f32.mul (f32.load (i32.add (local.get $w_beta_ptr) (i32.shl (local.get $i) (i32.const 2)))) (local.get $a_val)))
            )
            (local.set $gamma_dot
              (f32.add (local.get $gamma_dot) (f32.mul (f32.load (i32.add (local.get $w_gamma_ptr) (i32.shl (local.get $i) (i32.const 2)))) (local.get $a_val)))
            )
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $lp_bg)
          )
        )
        (local.set $beta (call $sigmoidf (local.get $beta_dot)))
        (local.set $gamma (call $sigmoidf (local.get $gamma_dot)))

        ;; Slow Memory Delta Rule Update (M_t in R^{rs x dv})
        (local.set $j (i32.const 0))
        (block $brk_j
          (loop $lp_j
            (br_if $brk_j (i32.ge_u (local.get $j) (local.get $dv)))
            (local.set $v_hat_j (f32.const 0.0))
            (local.set $r (i32.const 0))
            (block $brk_r1
              (loop $lp_r1
                (br_if $brk_r1 (i32.ge_u (local.get $r) (local.get $rs)))
                (local.set $m_idx_ptr
                  (i32.add (local.get $slow_mem_ptr) (i32.shl (i32.add (i32.mul (local.get $r) (local.get $dv)) (local.get $j)) (i32.const 2)))
                )
                (local.set $m_dec (f32.mul (local.get $gamma) (f32.load (local.get $m_idx_ptr))))
                (f32.store (local.get $m_idx_ptr) (local.get $m_dec))
                (local.set $v_hat_j
                  (f32.add
                    (local.get $v_hat_j)
                    (f32.mul (local.get $m_dec) (f32.load (i32.add (local.get $k_vec_ptr) (i32.shl (local.get $r) (i32.const 2)))))
                  )
                )
                (local.set $r (i32.add (local.get $r) (i32.const 1)))
                (br $lp_r1)
              )
            )
            (local.set $delta_j
              (f32.mul
                (local.get $beta)
                (f32.sub (f32.load (i32.add (local.get $v_vec_ptr) (i32.shl (local.get $j) (i32.const 2)))) (local.get $v_hat_j))
              )
            )
            (local.set $readout_j (f32.const 0.0))
            (local.set $r (i32.const 0))
            (block $brk_r2
              (loop $lp_r2
                (br_if $brk_r2 (i32.ge_u (local.get $r) (local.get $rs)))
                (local.set $m_idx_ptr
                  (i32.add (local.get $slow_mem_ptr) (i32.shl (i32.add (i32.mul (local.get $r) (local.get $dv)) (local.get $j)) (i32.const 2)))
                )
                (local.set $m_new
                  (f32.add
                    (f32.load (local.get $m_idx_ptr))
                    (f32.mul (f32.load (i32.add (local.get $k_vec_ptr) (i32.shl (local.get $r) (i32.const 2)))) (local.get $delta_j))
                  )
                )
                (f32.store (local.get $m_idx_ptr) (local.get $m_new))
                (local.set $readout_j
                  (f32.add
                    (local.get $readout_j)
                    (f32.mul (local.get $m_new) (f32.load (i32.add (local.get $q_vec_ptr) (i32.shl (local.get $r) (i32.const 2)))))
                  )
                )
                (local.set $r (i32.add (local.get $r) (i32.const 1)))
                (br $lp_r2)
              )
            )
            (f32.store (i32.add (local.get $s_vec_ptr) (i32.shl (local.get $j) (i32.const 2))) (local.get $readout_j))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $lp_j)
          )
        )

        ;; Dual-Timescale Fusion: combined = silu(g) * (fast_state + p_slow * s_vec)
        (memory.copy (local.get $combined_ptr) (local.get $fast_state_ptr) (i32.shl (local.get $d) (i32.const 2)))
        (call $matvec_i8_add (local.get $p_slow_ptr) (local.get $d) (local.get $dv) (local.get $s_vec_ptr) (local.get $combined_ptr))
        (local.set $i (i32.const 0))
        (block $brk_comb
          (loop $lp_comb
            (br_if $brk_comb (i32.ge_u (local.get $i) (local.get $d)))
            (f32.store
              (i32.add (local.get $combined_ptr) (i32.shl (local.get $i) (i32.const 2)))
              (f32.mul
                (f32.load (i32.add (local.get $combined_ptr) (i32.shl (local.get $i) (i32.const 2))))
                (call $siluf (f32.load (i32.add (local.get $g_vec_ptr) (i32.shl (local.get $i) (i32.const 2)))))
              )
            )
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $lp_comb)
          )
        )

        ;; Apply mod_o, mod_gate, mod_up, mod_down using saved cursor in $norm1_ptr
        (local.set $norm1_ptr
          (call $mod_apply
            (local.get $norm1_ptr) (local.get $d) (local.get $d) (local.get $rm) (i32.const 0)
            (global.get $off_core_wo) (local.get $combined_ptr) (local.get $scratch_rm_ptr) (local.get $o_proj_ptr)
          )
        )
        (local.set $i (i32.const 0))
        (block $brk_res1
          (loop $lp_res1
            (br_if $brk_res1 (i32.ge_u (local.get $i) (local.get $d)))
            (f32.store
              (i32.add (local.get $h_ptr) (i32.shl (local.get $i) (i32.const 2)))
              (f32.add
                (f32.load (i32.add (local.get $h_ptr) (i32.shl (local.get $i) (i32.const 2))))
                (f32.load (i32.add (local.get $o_proj_ptr) (i32.shl (local.get $i) (i32.const 2))))
              )
            )
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $lp_res1)
          )
        )

        ;; SwiGLU Channel Mixer
        (call $rms_norm (local.get $h_ptr) (local.get $norm2_ptr) (local.get $d) (global.get $rms_eps) (local.get $z_ptr))
        (local.set $norm1_ptr
          (call $mod_apply
            (local.get $norm1_ptr) (local.get $d) (local.get $dff) (local.get $rm) (i32.const 0)
            (global.get $off_core_wgate) (local.get $z_ptr) (local.get $scratch_rm_ptr) (local.get $ff_gate_ptr)
          )
        )
        (local.set $norm1_ptr
          (call $mod_apply
            (local.get $norm1_ptr) (local.get $d) (local.get $dff) (local.get $rm) (i32.const 0)
            (global.get $off_core_wup) (local.get $z_ptr) (local.get $scratch_rm_ptr) (local.get $ff_up_ptr)
          )
        )
        (local.set $i (i32.const 0))
        (block $brk_ff
          (loop $lp_ff
            (br_if $brk_ff (i32.ge_u (local.get $i) (local.get $dff)))
            (f32.store
              (i32.add (local.get $ff_gate_ptr) (i32.shl (local.get $i) (i32.const 2)))
              (f32.mul
                (call $siluf (f32.load (i32.add (local.get $ff_gate_ptr) (i32.shl (local.get $i) (i32.const 2)))))
                (f32.load (i32.add (local.get $ff_up_ptr) (i32.shl (local.get $i) (i32.const 2))))
              )
            )
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $lp_ff)
          )
        )
        (drop
          (call $mod_apply
            (local.get $norm1_ptr) (local.get $dff) (local.get $d) (local.get $rm) (i32.const 0)
            (global.get $off_core_wdown) (local.get $ff_gate_ptr) (local.get $scratch_rm_ptr) (local.get $o_proj_ptr)
          )
        )
        (local.set $i (i32.const 0))
        (block $brk_res2
          (loop $lp_res2
            (br_if $brk_res2 (i32.ge_u (local.get $i) (local.get $d)))
            (f32.store
              (i32.add (local.get $h_ptr) (i32.shl (local.get $i) (i32.const 2)))
              (f32.add
                (f32.load (i32.add (local.get $h_ptr) (i32.shl (local.get $i) (i32.const 2))))
                (f32.load (i32.add (local.get $o_proj_ptr) (i32.shl (local.get $i) (i32.const 2))))
              )
            )
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $lp_res2)
          )
        )

        (local.set $l_idx (i32.add (local.get $l_idx) (i32.const 1)))
        (br $lp_layer)
      )
    )

    ;; Final Norm + P_out + Tied Vocabulary Logits
    (call $rms_norm (local.get $h_ptr) (global.get $off_final_norm) (local.get $d) (global.get $rms_eps) (local.get $z_ptr))
    (call $matvec_i8 (global.get $off_p_out) (local.get $de) (local.get $d) (local.get $z_ptr) (local.get $e_ptr))
    (call $matvec_i8 (global.get $off_tok_embed) (local.get $V) (local.get $de) (local.get $e_ptr) (local.get $logits_out_ptr))

    ;; Increment tokens_generated counter in session header
    (i32.store
      (i32.add (local.get $session_ptr) (i32.const 8))
      (i32.add (i32.load (i32.add (local.get $session_ptr) (i32.const 8))) (i32.const 1))
    )
    (i32.const 0)
  )

  ;; ---------------------------------------------------------------------------
  ;; Forward Step + Temperature / xorshift64* Sampling
  ;; ---------------------------------------------------------------------------
  (func (export "wasm_step_and_sample")
    (param $session_ptr i32) (param $input_token i32) (param $temperature f32) (result i32)
    (local $V i32)
    (local $logits_ptr i32)
    (local $probs_ptr i32)
    (local $i i32)
    (local $best_idx i32)
    (local $best_val f32)
    (local $val f32)
    (local $inv_temp f32)
    (local $sum f32)
    (local $rng i64)
    (local $rand_u f32)
    (local $thresh f32)
    (local $cdf f32)

    (local.set $V (global.get $vocab_size))
    (local.set $logits_ptr (i32.add (global.get $scratch_ptr) (i32.const 36864)))
    (local.set $probs_ptr (i32.add (global.get $scratch_ptr) (i32.const 49152)))

    (drop (call $wasm_forward_logits (local.get $session_ptr) (local.get $input_token) (local.get $logits_ptr)))

    ;; Find max logit (also serves greedy argmax when temperature <= 0.0001)
    (local.set $best_idx (i32.const 0))
    (local.set $best_val (f32.const -1000000000.0))
    (local.set $i (i32.const 0))
    (block $brk_max
      (loop $lp_max
        (br_if $brk_max (i32.ge_u (local.get $i) (local.get $V)))
        (local.set $val (f32.load (i32.add (local.get $logits_ptr) (i32.shl (local.get $i) (i32.const 2)))))
        (if (f32.gt (local.get $val) (local.get $best_val))
          (then
            (local.set $best_val (local.get $val))
            (local.set $best_idx (local.get $i))
          )
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $lp_max)
      )
    )
    (if (f32.le (local.get $temperature) (f32.const 0.0001))
      (then (return (local.get $best_idx)))
    )

    (local.set $inv_temp (f32.div (f32.const 1.0) (local.get $temperature)))
    (local.set $sum (f32.const 0.0))
    (local.set $i (i32.const 0))
    (block $brk_p
      (loop $lp_p
        (br_if $brk_p (i32.ge_u (local.get $i) (local.get $V)))
        (local.set $val
          (call $expf
            (f32.mul
              (f32.sub (f32.load (i32.add (local.get $logits_ptr) (i32.shl (local.get $i) (i32.const 2)))) (local.get $best_val))
              (local.get $inv_temp)
            )
          )
        )
        (f32.store (i32.add (local.get $probs_ptr) (i32.shl (local.get $i) (i32.const 2))) (local.get $val))
        (local.set $sum (f32.add (local.get $sum) (local.get $val)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $lp_p)
      )
    )

    ;; xorshift64* PRNG
    (local.set $rng (i64.load (local.get $session_ptr)))
    (local.set $rng (i64.xor (local.get $rng) (i64.shr_u (local.get $rng) (i64.const 12))))
    (local.set $rng (i64.xor (local.get $rng) (i64.shl (local.get $rng) (i64.const 25))))
    (local.set $rng (i64.xor (local.get $rng) (i64.shr_u (local.get $rng) (i64.const 27))))
    (i64.store (local.get $session_ptr) (local.get $rng))

    (local.set $rand_u
      (f32.div
        (f32.convert_i64_u (i64.shr_u (i64.mul (local.get $rng) (i64.const 2685821657736338717)) (i64.const 40)))
        (f32.const 16777216.0)
      )
    )
    (local.set $thresh (f32.mul (local.get $rand_u) (local.get $sum)))
    (local.set $cdf (f32.const 0.0))
    (local.set $i (i32.const 0))
    (block $brk_s
      (loop $lp_s
        (br_if $brk_s (i32.ge_u (local.get $i) (local.get $V)))
        (local.set $cdf (f32.add (local.get $cdf) (f32.load (i32.add (local.get $probs_ptr) (i32.shl (local.get $i) (i32.const 2))))))
        (if (f32.ge (local.get $cdf) (local.get $thresh))
          (then (return (local.get $i)))
        )
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $lp_s)
      )
    )
    (i32.sub (local.get $V) (i32.const 1))
  )
)
`;

async function main() {
  fs.mkdirSync("runtime-wasm", { recursive: true });
  fs.mkdirSync("artifacts", { recursive: true });
  const watPath = path.join("runtime-wasm", "deltacore_runtime.wat");
  const wasmPath = path.join("artifacts", "deltacore_runtime.wasm");

  fs.writeFileSync(watPath, watSource, "utf8");
  const wabt = await wabtModule();
  const parsed = wabt.parseWat(watPath, watSource, {
    mutable_globals: true,
    bulk_memory: true,
  });
  parsed.validate();
  const { buffer } = parsed.toBinary({ write_debug_names: false });
  fs.writeFileSync(wasmPath, Buffer.from(buffer));
  console.log(`[WASM Compiler] Compiled ${watPath} -> ${wasmPath} (${buffer.byteLength} bytes)`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
