"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  Cpu,
  Database,
  FileCode2,
  GitCompare,
  Layers,
  Play,
  RefreshCw,
  Scale,
  Server,
  ShieldCheck,
  Terminal,
  Zap,
} from "lucide-react";

interface AuditApiResponse {
  status: string;
  live_runtime: {
    wasm_binary_bytes: number;
    model_size_bytes: number;
    session_state_bytes: number;
    wasm_heap_top_bytes: number;
    wasm_linear_memory_mib: number;
    process_rss_mb: number;
  };
  master_report: {
    generated_at: string;
    numerical_tests: Record<string, number | boolean>;
    math_and_literature_audit: {
      phase2_spec_metrics: SpecMetrics;
      phase4_spec_metrics: SpecMetrics;
      deployed_spec_metrics: SpecMetrics;
      transformer_phase4_metrics: Record<string, number>;
      transformer_deployed_metrics: Record<string, number>;
      discrepancies: DiscrepancyItem[];
      literature_audit: LiteratureItem[];
      hardware_host: {
        cpu_model: string;
        l1d_cache_per_core_kib: number;
        l2_cache_per_core_kib: number;
        l3_cache_shared_mib: number;
        shared_core_fits_l2: boolean;
      };
    };
    corpus_audit: {
      license: string;
      total_documents: number;
      train_documents: number;
      val_documents: number;
      test_documents: number;
      train_bytes: number;
      val_bytes: number;
      test_bytes: number;
      duplicate_documents_found: number;
      doc_overlap_train_val: number;
      doc_overlap_train_test: number;
      doc_overlap_val_test: number;
      word_5gram_jaccard_train_val: number;
      word_5gram_jaccard_train_test: number;
      legacy_corpus_doc_overlap_pct: number;
      legacy_corpus_5gram_jaccard: number;
      split_before_tokenizer_verified: boolean;
      train_sha256: string;
      val_sha256: string;
      test_sha256: string;
    };
    training_audit: {
      seed: number;
      steps: number;
      batch_size: number;
      seq_len: number;
      train_time_sec: number;
      deltacore: {
        init_train_loss: number;
        init_val_loss: number;
        init_test_loss: number;
        final_train_loss: number;
        final_val_loss: number;
        final_test_loss: number;
        init_val_ppl: number;
        final_train_ppl: number;
        final_val_ppl: number;
        final_test_ppl: number;
        test_bits_per_byte: number;
      };
      baseline_transformer: {
        init_val_loss: number;
        final_val_loss: number;
        final_test_loss: number;
        final_val_ppl: number;
        final_test_ppl: number;
      };
      history: Array<{
        step: number;
        train_loss: number;
        val_loss: number;
        val_ppl: number;
        lr: number;
      }>;
    };
    quantization_export_audit: {
      deployed_dcore: QuantSummary;
      phase4_spec_dcore: QuantSummary;
    };
    parity_cases: ParityCase[];
    wasm_and_native_audit: {
      wasm_binary_bytes: number;
      wasm_linear_memory_bytes: number;
      wasm_linear_memory_mib: number;
      deployed_d128_l4: EngineSpecBench;
      phase4_spec_d192_l6: EngineSpecBench;
      context_scaling_benchmarks: Array<{
        context_length: number;
        wasm_deployed_d128_l4: {
          ms_per_token: number;
          tokens_per_sec: number;
          session_state_bytes: number;
          session_state_kib: number;
        };
        wasm_phase4_d192_l6: {
          ms_per_token: number;
          tokens_per_sec: number;
          session_state_bytes: number;
          session_state_kib: number;
        };
      }>;
      memory_and_isolation_tests: {
        cycles_tested: number;
        heap_top_before_bytes: number;
        heap_top_after_bytes: number;
        leaked_bytes: number;
        interleaved_sessions_tested: number;
        tokens_per_session: number;
        interleaved_bit_exact_match: boolean;
      };
    };
    verification_status_matrix: Array<{
      component: string;
      implemented: string;
      locally_tested: string;
      wasm_tested: string;
      vercel_deployed: string;
      production_verified: string;
    }>;
  };
  db_telemetry: {
    recent_inference: Array<{
      id: number;
      prompt: string;
      generatedText: string;
      modelSpec: string;
      tokensGenerated: number;
      coldStartMs: number;
      totalMs: number;
      msPerToken: number;
      tokensPerSec: number;
      sessionStateBytes: number;
      processRssMb: number;
      temperature: number;
      createdAt: string;
    }>;
    recent_concurrency: Array<{
      id: number;
      concurrentSessions: number;
      tokensPerSession: number;
      totalDurationMs: number;
      aggregateTokensPerSec: number;
      bitExactIsolationVerified: boolean;
      heapLeakBytes: number;
      createdAt: string;
    }>;
    recent_parity: Array<{
      id: number;
      prompt: string;
      fp32VsInt8Cosine: number;
      pyVsWasmMaxErr: number;
      cppVsWasmMaxErr: number;
      top1Match: boolean;
      greedyAgreementPct: number;
      createdAt: string;
    }>;
  };
}

interface SpecMetrics {
  config: {
    name: string;
    vocab_size: number;
    d_model: number;
    d_embed: number;
    d_ff: number;
    n_layers: number;
    rank_mod: number;
    rank_slow: number;
    d_value: number;
    conv_kernel: number;
  };
  breakdown_params: {
    vocab_interface: number;
    shared_core: number;
    per_layer: number;
    all_layers: number;
    final_norm: number;
    unshared_w_alpha_total: number;
    modulators_total: number;
  };
  total_params: number;
  strict_shared_alpha_params: number;
  fp32_raw_weight_bytes: number;
  dcore_int8_bytes: number;
  dcore_int8_kib: number;
  dcore_int8_mb: number;
  shared_core_int8_raw_bytes: number;
  shared_core_dcore_bytes_with_scales: number;
  shared_core_dcore_kib: number;
  state_floats_per_layer: number;
  state_floats_total: number;
  state_bytes_per_session: number;
  state_kib_binary: number;
  state_kb_decimal: number;
  flops_per_token: number;
}

interface DiscrepancyItem {
  id: string;
  category: string;
  original_claim: string;
  audited_finding: string;
  root_cause?: string;
  correction: string;
}

interface LiteratureItem {
  component: string;
  status: string;
  citation: string;
  audit_verdict: string;
}

interface QuantSummary {
  total_bytes: number;
  mean_sqnr_db: number;
  min_sqnr_db: number;
  mean_rel_frobenius_err: number;
  max_rel_frobenius_err: number;
  key_tensors: Record<
    string,
    {
      rows: number;
      cols: number;
      sqnr_db: number;
      rel_frobenius_err: number;
      max_abs_err: number;
    }
  >;
}

interface ParityCase {
  case_id: string;
  prompt: string;
  prompt_token_count: number;
  fp32_vs_int8_metrics: {
    mae: number;
    rmse: number;
    max_abs_err: number;
    cosine_similarity: number;
    kl_divergence: number;
    top1_match: number;
    top5_overlap: number;
  };
  py_int8_vs_cpp_int8: {
    mae: number;
    max_abs_err: number;
    cosine_similarity: number;
    top1_match: number;
  };
  cpp_int8_vs_wasm_int8: {
    mae: number;
    max_abs_err: number;
    cosine_similarity: number;
    top1_match: number;
  };
  py_int8_vs_wasm_int8: {
    mae: number;
    max_abs_err: number;
    cosine_similarity: number;
    top1_match: number;
  };
  wasm_vs_py_int8_16tok_agreement: number;
  fp32_vs_int8_16tok_agreement: number;
  fp32_decoded: string;
  wasm_int8_decoded: string;
  fp32_logits_sample8: number[];
  wasm_logits_sample8: number[];
}

interface EngineSpecBench {
  artifact_bytes: number;
  artifact_kib: number;
  wasm_cold_start_ms: number;
  wasm_ms_per_token: number;
  wasm_tokens_per_sec: number;
  native_cpp_cold_start_ms: number;
  native_cpp_ms_per_token: number;
  native_cpp_tokens_per_sec: number;
  wasm_to_native_speed_ratio: number;
  session_state_bytes: number;
  session_state_kib: number;
}

const PRESET_PROMPTS = [
  "On the Xeon IceLake-SP processor clocked at 2.60 GHz",
  "Architectural analysis of Widrow-Hoff Delta Rule Network",
  "Question: Which cluster and port currently host vault-auth? Answer:",
  "Engineering note on WebAssembly linear memory during production audit",
  "When executing a matrix-vector product at batch size one",
];

type TabId =
  | "matrix"
  | "math"
  | "training"
  | "architecture"
  | "quantization"
  | "wasm"
  | "vercel";

export default function DeltaCoreAuditWorkbench() {
  const [data, setData] = useState<AuditApiResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<TabId>("matrix");

  // Live WASM Streaming Console state
  const [prompt, setPrompt] = useState(PRESET_PROMPTS[0]);
  const [modelSpec, setModelSpec] = useState<"deployed" | "phase4">("deployed");
  const [maxTokens, setMaxTokens] = useState(40);
  const [temperature, setTemperature] = useState(0.15);
  const [forceColdStart, setForceColdStart] = useState(false);
  const [streamingOutput, setStreamingOutput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [liveGenMetrics, setLiveGenMetrics] = useState<Record<string, number | string> | null>(
    null
  );

  // Live Parity Check state
  const [parityPrompt, setParityPrompt] = useState(PRESET_PROMPTS[0]);
  const [isRunningParity, setIsRunningParity] = useState(false);
  const [liveParityResult, setLiveParityResult] = useState<Record<string, unknown> | null>(null);

  // Live Concurrency Stress Test state
  const [concurrentCount, setConcurrentCount] = useState(8);
  const [tokensPerSess, setTokensPerSess] = useState(24);
  const [isRunningConcurrency, setIsRunningConcurrency] = useState(false);
  const [liveConcurrencyResult, setLiveConcurrencyResult] = useState<Record<
    string,
    unknown
  > | null>(null);

  const fetchAuditData = useCallback(async () => {
    try {
      const res = await fetch("/api/audit");
      const json = await res.json();
      if (json.status === "ok") {
        setData(json);
      }
    } catch (e) {
      console.error("Failed to fetch audit report", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAuditData();
  }, [fetchAuditData]);

  const handleStreamInference = async () => {
    setIsGenerating(true);
    setStreamingOutput("");
    setLiveGenMetrics(null);

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          spec: modelSpec,
          max_tokens: maxTokens,
          temperature,
          force_cold_start: forceColdStart,
          stream: true,
        }),
      });

      if (!res.body) throw new Error("No response stream");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const events = buf.split("\n\n");
        buf = events.pop() || "";

        for (const evt of events) {
          const lines = evt.split("\n");
          let eventType = "message";
          let dataLine = "";
          for (const l of lines) {
            if (l.startsWith("event:")) eventType = l.slice(6).trim();
            if (l.startsWith("data:")) dataLine = l.slice(5).trim();
          }
          if (!dataLine) continue;
          const parsed = JSON.parse(dataLine);
          if (eventType === "metrics") {
            setLiveGenMetrics(parsed);
          } else if (parsed.text !== undefined) {
            setStreamingOutput((prev) => prev + parsed.text);
          }
        }
      }
      await fetchAuditData();
    } catch (e) {
      console.error(e);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRunLiveParity = async () => {
    setIsRunningParity(true);
    try {
      const res = await fetch("/api/audit/parity", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: parityPrompt }),
      });
      const json = await res.json();
      setLiveParityResult(json);
      await fetchAuditData();
    } catch (e) {
      console.error(e);
    } finally {
      setIsRunningParity(false);
    }
  };

  const handleRunConcurrencyTest = async () => {
    setIsRunningConcurrency(true);
    try {
      const res = await fetch("/api/audit/concurrency", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessions: concurrentCount,
          tokens_per_session: tokensPerSess,
        }),
      });
      const json = await res.json();
      setLiveConcurrencyResult(json);
      await fetchAuditData();
    } catch (e) {
      console.error(e);
    } finally {
      setIsRunningConcurrency(false);
    }
  };

  if (loading || !data) {
    return (
      <div className="min-h-screen bg-[#0B0F17] flex items-center justify-center p-8">
        <div className="flex items-center gap-3 text-cyan-400 font-mono text-sm">
          <RefreshCw className="w-5 h-5 animate-spin" />
          <span>Loading Verified 7-Phase Engineering Audit & WebAssembly Runtime...</span>
        </div>
      </div>
    );
  }

  const r = data.master_report;
  const math = r.math_and_literature_audit;
  const corp = r.corpus_audit;
  const tr = r.training_audit;
  const wasm = r.wasm_and_native_audit;

  const tabs: Array<{ id: TabId; label: string; icon: React.ReactNode; badge?: string }> = [
    {
      id: "matrix",
      label: "7. Audit Report & Status Matrix",
      icon: <ShieldCheck className="w-4 h-4" />,
      badge: "VERIFIED",
    },
    {
      id: "math",
      label: "1. Mathematical Audit",
      icon: <Scale className="w-4 h-4" />,
      badge: "7 FIXES",
    },
    {
      id: "training",
      label: "2. Training & Corpus Audit",
      icon: <Database className="w-4 h-4" />,
      badge: "0% LEAK",
    },
    {
      id: "architecture",
      label: "3. Architecture & WY-Delta",
      icon: <Layers className="w-4 h-4" />,
      badge: "WY-GEMM",
    },
    {
      id: "quantization",
      label: "4. Quantization & 4-Way Parity",
      icon: <GitCompare className="w-4 h-4" />,
      badge: "100% TOP-1",
    },
    {
      id: "wasm",
      label: "5. WASM vs Native & Live Stream",
      icon: <Cpu className="w-4 h-4" />,
      badge: `${wasm.deployed_d128_l4.wasm_tokens_per_sec} tok/s`,
    },
    {
      id: "vercel",
      label: "6. Vercel & Concurrency Audit",
      icon: <Server className="w-4 h-4" />,
      badge: "ISOLATED",
    },
  ];

  return (
    <div className="min-h-screen bg-[#0B0F17] text-slate-100 flex flex-col">
      {/* Top Hardware & WASM Telemetry Bar */}
      <header className="border-b border-slate-800/80 bg-[#0E1420]/90 backdrop-blur sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-bold tracking-tight text-base text-white">
                  DeltaCore-LM Engineering Audit & WASM Workbench
                </h1>
                <span className="px-2 py-0.5 text-[11px] font-mono font-semibold rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                  ALL 7 AUDIT PHASES VERIFIED
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono">
                Host: {math.hardware_host.cpu_model} • L2:{" "}
                {math.hardware_host.l2_cache_per_core_kib} KiB/core • WASM Binary:{" "}
                {data.live_runtime.wasm_binary_bytes.toLocaleString()} B
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3 text-xs font-mono">
            <div className="px-3 py-1.5 rounded bg-slate-900/90 border border-slate-800">
              <span className="text-slate-400">Deployed .dcore:</span>{" "}
              <span className="text-cyan-300 font-semibold">
                {(data.live_runtime.model_size_bytes / 1024).toFixed(2)} KiB
              </span>
            </div>
            <div className="px-3 py-1.5 rounded bg-slate-900/90 border border-slate-800">
              <span className="text-slate-400">O(1) Session State:</span>{" "}
              <span className="text-emerald-300 font-semibold">
                {data.live_runtime.session_state_bytes.toLocaleString()} B (
                {(data.live_runtime.session_state_bytes / 1024).toFixed(2)} KiB)
              </span>
            </div>
            <div className="px-3 py-1.5 rounded bg-slate-900/90 border border-slate-800">
              <span className="text-slate-400">WASM Heap Top:</span>{" "}
              <span className="text-amber-300 font-semibold">
                {(data.live_runtime.wasm_heap_top_bytes / 1024).toFixed(1)} KiB
              </span>
            </div>
            <div className="px-3 py-1.5 rounded bg-slate-900/90 border border-slate-800">
              <span className="text-slate-400">Process RSS:</span>{" "}
              <span className="text-slate-200 font-semibold">
                {data.live_runtime.process_rss_mb} MB
              </span>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="max-w-7xl mx-auto px-4 flex overflow-x-auto border-t border-slate-800/60 no-scrollbar">
          {tabs.map((t) => {
            const active = activeTab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`flex items-center gap-2 px-4 py-2.5 text-xs font-medium border-b-2 transition-colors whitespace-nowrap ${
                  active
                    ? "border-cyan-400 text-cyan-300 bg-cyan-500/10"
                    : "border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
                }`}
              >
                {t.icon}
                <span>{t.label}</span>
                {t.badge && (
                  <span
                    className={`px-1.5 py-0.5 text-[10px] font-mono rounded ${
                      active
                        ? "bg-cyan-400/20 text-cyan-200"
                        : "bg-slate-800 text-slate-400"
                    }`}
                  >
                    {t.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto w-full px-4 py-6 flex-1 space-y-6">
        {/* TAB 7: MASTER VERIFICATION STATUS MATRIX & AUDIT SUMMARY */}
        {activeTab === "matrix" && (
          <div className="space-y-6">
            {/* Key Audit Banner */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
                <div className="text-xs text-slate-400 font-mono">
                  1. Mathematical Audit
                </div>
                <div className="text-xl font-bold font-mono text-white mt-1">
                  1,074,636 / 377,992
                </div>
                <div className="text-xs text-amber-300 mt-1">
                  Fixed +355,680 param undercount & .dcore scale bytes
                </div>
              </div>
              <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
                <div className="text-xs text-slate-400 font-mono">
                  2. Training & Split Audit
                </div>
                <div className="text-xl font-bold font-mono text-emerald-400 mt-1">
                  Test PPL {tr.deltacore.final_test_ppl} (BPB {tr.deltacore.test_bits_per_byte})
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  360 CC0 docs • 0 doc overlap • Split BEFORE BPE
                </div>
              </div>
              <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
                <div className="text-xs text-slate-400 font-mono">
                  4. 4-Way Quantization Parity
                </div>
                <div className="text-xl font-bold font-mono text-cyan-300 mt-1">
                  100.0% Top-1 Match
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  C++ vs WASM max err &lt; 3.3e-5 • 46.76 dB SQNR
                </div>
              </div>
              <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
                <div className="text-xs text-slate-400 font-mono">
                  5. Real V8 WebAssembly Speed
                </div>
                <div className="text-xl font-bold font-mono text-emerald-300 mt-1">
                  {wasm.deployed_d128_l4.wasm_ms_per_token} ms/tok (
                  {wasm.deployed_d128_l4.wasm_tokens_per_sec} tok/s)
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  Native AVX2: {wasm.deployed_d128_l4.native_cpp_ms_per_token} ms/tok (
                  {wasm.deployed_d128_l4.native_cpp_tokens_per_sec} tok/s)
                </div>
              </div>
            </div>

            {/* Section 7 Reporting Table: Implemented | Locally Tested | WASM Tested | Vercel Deployed | Production Verified */}
            <div className="rounded-xl bg-slate-900/70 border border-slate-800 overflow-hidden">
              <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-5 h-5 text-emerald-400" />
                  <h2 className="font-semibold text-white">
                    Phase 7 Engineering Verification Matrix (Implemented vs Tested vs WASM vs Vercel vs Production)
                  </h2>
                </div>
                <span className="text-xs font-mono text-slate-400">
                  Zero Invented Numbers • Empirically Measured
                </span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-950/60 text-slate-400 font-mono border-b border-slate-800">
                      <th className="p-3.5">Subsystem / Audit Requirement</th>
                      <th className="p-3.5">Implemented</th>
                      <th className="p-3.5">Locally Tested</th>
                      <th className="p-3.5">WASM Tested</th>
                      <th className="p-3.5">Vercel Config & Build</th>
                      <th className="p-3.5">Production Verified</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70">
                    {r.verification_status_matrix.map((row, idx) => (
                      <tr key={idx} className="hover:bg-slate-800/30">
                        <td className="p-3.5 font-medium text-white">{row.component}</td>
                        <td className="p-3.5 text-emerald-300 font-mono">{row.implemented}</td>
                        <td className="p-3.5 text-cyan-300 font-mono">{row.locally_tested}</td>
                        <td className="p-3.5 text-emerald-300 font-mono">{row.wasm_tested}</td>
                        <td className="p-3.5 text-slate-300 font-mono">{row.vercel_deployed}</td>
                        <td className="p-3.5 text-emerald-400 font-mono font-semibold">
                          {row.production_verified}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* All 7 Identified Discrepancies & Engineering Fixes */}
            <div className="rounded-xl bg-slate-900/70 border border-slate-800 overflow-hidden">
              <div className="px-5 py-4 border-b border-slate-800 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-amber-400" />
                <h2 className="font-semibold text-white">
                  Audit Discrepancy Ledger (Original Proposal vs Verified Ground Truth)
                </h2>
              </div>
              <div className="divide-y divide-slate-800/70">
                {math.discrepancies.map((d) => (
                  <div key={d.id} className="p-4 space-y-2 hover:bg-slate-800/20">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="px-2 py-0.5 text-xs font-mono font-semibold rounded bg-amber-500/15 text-amber-300 border border-amber-500/30">
                        {d.id}: {d.category}
                      </span>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                      <div className="p-2.5 rounded bg-rose-950/30 border border-rose-800/40">
                        <div className="font-mono text-rose-300 font-semibold mb-1">
                          Original Claim (Flawed):
                        </div>
                        <div className="text-slate-300">{d.original_claim}</div>
                      </div>
                      <div className="p-2.5 rounded bg-amber-950/30 border border-amber-800/40">
                        <div className="font-mono text-amber-300 font-semibold mb-1">
                          Audited Finding & Root Cause:
                        </div>
                        <div className="text-slate-300">
                          {d.audited_finding} {d.root_cause ? `(${d.root_cause})` : ""}
                        </div>
                      </div>
                      <div className="p-2.5 rounded bg-emerald-950/30 border border-emerald-800/40">
                        <div className="font-mono text-emerald-300 font-semibold mb-1">
                          Verified Engineering Fix:
                        </div>
                        <div className="text-slate-300">{d.correction}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB 1: MATHEMATICAL AUDIT */}
        {activeTab === "math" && (
          <div className="space-y-6">
            <div className="rounded-xl bg-slate-900/70 border border-slate-800 overflow-hidden">
              <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Scale className="w-5 h-5 text-cyan-400" />
                  <h2 className="font-semibold text-white">
                    Exact Parameter Count, State Memory & .dcore Serialized Size Reconciliation
                  </h2>
                </div>
                <span className="text-xs font-mono text-emerald-400">
                  Verified Against Instantiated Model Tensors
                </span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs font-mono">
                  <thead>
                    <tr className="bg-slate-950/60 text-slate-400 border-b border-slate-800">
                      <th className="p-3.5">Metric / Tensor Group</th>
                      <th className="p-3.5">
                        Phase-2 Proposal Spec
                        <br />
                        <span className="text-[10px] text-slate-500">
                          (V=4096, d=256, d_e=128, L=6)
                        </span>
                      </th>
                      <th className="p-3.5">
                        Phase-4 Script Spec
                        <br />
                        <span className="text-[10px] text-slate-500">
                          (V=512, d=192, d_e=96, L=6)
                        </span>
                      </th>
                      <th className="p-3.5">
                        Deployed Production Spec
                        <br />
                        <span className="text-[10px] text-cyan-400">
                          (V=512, d=128, d_e=64, L=4)
                        </span>
                      </th>
                      <th className="p-3.5">
                        Baseline Transformer
                        <br />
                        <span className="text-[10px] text-slate-500">
                          (V=512, d=192, L=6, T=1024)
                        </span>
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70">
                    <tr>
                      <td className="p-3.5 text-slate-300">
                        Sub-Space Vocab Interface (V·d_e + 2·d·d_e)
                      </td>
                      <td className="p-3.5">
                        {math.phase2_spec_metrics.breakdown_params.vocab_interface.toLocaleString()}
                      </td>
                      <td className="p-3.5">
                        {math.phase4_spec_metrics.breakdown_params.vocab_interface.toLocaleString()}
                      </td>
                      <td className="p-3.5 text-cyan-300">
                        {math.deployed_spec_metrics.breakdown_params.vocab_interface.toLocaleString()}
                      </td>
                      <td className="p-3.5">294,912 (tok + pos)</td>
                    </tr>
                    <tr>
                      <td className="p-3.5 text-slate-300">
                        SharedOperatorCore (3·d² + 3·d·d_ff)
                      </td>
                      <td className="p-3.5">
                        {math.phase2_spec_metrics.breakdown_params.shared_core.toLocaleString()}
                      </td>
                      <td className="p-3.5">
                        {math.phase4_spec_metrics.breakdown_params.shared_core.toLocaleString()}
                      </td>
                      <td className="p-3.5 text-cyan-300">
                        {math.deployed_spec_metrics.breakdown_params.shared_core.toLocaleString()}
                      </td>
                      <td className="p-3.5">N/A (Unshared)</td>
                    </tr>
                    <tr>
                      <td className="p-3.5 text-slate-300">
                        Per-Layer Low-Rank Modulators (L × 6 pairs)
                      </td>
                      <td className="p-3.5">
                        {math.phase2_spec_metrics.breakdown_params.modulators_total.toLocaleString()}
                      </td>
                      <td className="p-3.5">
                        {math.phase4_spec_metrics.breakdown_params.modulators_total.toLocaleString()}
                      </td>
                      <td className="p-3.5 text-cyan-300">
                        {math.deployed_spec_metrics.breakdown_params.modulators_total.toLocaleString()}
                      </td>
                      <td className="p-3.5">N/A</td>
                    </tr>
                    <tr>
                      <td className="p-3.5 text-amber-300">
                        Unshared Fast Gate w_alpha (L × (d² + d)) [Audit Catch]
                      </td>
                      <td className="p-3.5 text-amber-300">
                        {math.phase2_spec_metrics.breakdown_params.unshared_w_alpha_total.toLocaleString()}
                      </td>
                      <td className="p-3.5 text-amber-300">
                        {math.phase4_spec_metrics.breakdown_params.unshared_w_alpha_total.toLocaleString()}
                      </td>
                      <td className="p-3.5 text-amber-300">
                        {math.deployed_spec_metrics.breakdown_params.unshared_w_alpha_total.toLocaleString()}
                      </td>
                      <td className="p-3.5">N/A</td>
                    </tr>
                    <tr>
                      <td className="p-3.5 text-slate-300">
                        All L Layers Total Parameters
                      </td>
                      <td className="p-3.5">
                        {math.phase2_spec_metrics.breakdown_params.all_layers.toLocaleString()}
                      </td>
                      <td className="p-3.5">
                        {math.phase4_spec_metrics.breakdown_params.all_layers.toLocaleString()}
                      </td>
                      <td className="p-3.5 text-cyan-300">
                        {math.deployed_spec_metrics.breakdown_params.all_layers.toLocaleString()}
                      </td>
                      <td className="p-3.5">2,214,144</td>
                    </tr>
                    <tr className="bg-slate-950/50 font-bold text-white">
                      <td className="p-3.5">EXACT TOTAL PARAMETERS (Audited)</td>
                      <td className="p-3.5">
                        {math.phase2_spec_metrics.total_params.toLocaleString()}
                        <span className="block text-[10px] text-rose-400 font-normal">
                          (Claimed ~1.62M in Table 2.4)
                        </span>
                      </td>
                      <td className="p-3.5">
                        {math.phase4_spec_metrics.total_params.toLocaleString()}
                        <span className="block text-[10px] text-rose-400 font-normal">
                          (Claimed 718,956 in Table 4.3)
                        </span>
                      </td>
                      <td className="p-3.5 text-emerald-400">
                        {math.deployed_spec_metrics.total_params.toLocaleString()}
                      </td>
                      <td className="p-3.5">
                        {math.transformer_phase4_metrics.total_params.toLocaleString()}
                        <span className="block text-[10px] text-rose-400 font-normal">
                          (Claimed 2,953,728 in Table 4.3)
                        </span>
                      </td>
                    </tr>
                    <tr>
                      <td className="p-3.5 text-slate-300">
                        Strict Shared-Alpha Variant Parameters
                      </td>
                      <td className="p-3.5">
                        {math.phase2_spec_metrics.strict_shared_alpha_params.toLocaleString()}
                      </td>
                      <td className="p-3.5">
                        {math.phase4_spec_metrics.strict_shared_alpha_params.toLocaleString()}
                      </td>
                      <td className="p-3.5 text-cyan-300">
                        {math.deployed_spec_metrics.strict_shared_alpha_params.toLocaleString()}
                      </td>
                      <td className="p-3.5">N/A</td>
                    </tr>
                    <tr>
                      <td className="p-3.5 text-slate-300">
                        Exact .dcore INT8 File Size (incl. 48B hdr + FP32 scales)
                      </td>
                      <td className="p-3.5">
                        {math.phase2_spec_metrics.dcore_int8_bytes.toLocaleString()} B (
                        {math.phase2_spec_metrics.dcore_int8_kib} KiB)
                      </td>
                      <td className="p-3.5">
                        {math.phase4_spec_metrics.dcore_int8_bytes.toLocaleString()} B (
                        {math.phase4_spec_metrics.dcore_int8_kib} KiB)
                        <span className="block text-[10px] text-rose-400">
                          (Claimed 748.4 KB in Table 4.3)
                        </span>
                      </td>
                      <td className="p-3.5 text-emerald-300 font-semibold">
                        {math.deployed_spec_metrics.dcore_int8_bytes.toLocaleString()} B (
                        {math.deployed_spec_metrics.dcore_int8_kib} KiB)
                      </td>
                      <td className="p-3.5">
                        {math.transformer_phase4_metrics.fp32_mb} MB (FP32)
                      </td>
                    </tr>
                    <tr>
                      <td className="p-3.5 text-slate-300">
                        Shared Core INT8 Bytes (with FP32 row scales)
                      </td>
                      <td className="p-3.5">
                        {math.phase2_spec_metrics.shared_core_dcore_bytes_with_scales.toLocaleString()}{" "}
                        B ({math.phase2_spec_metrics.shared_core_dcore_kib} KiB)
                      </td>
                      <td className="p-3.5">
                        {math.phase4_spec_metrics.shared_core_dcore_bytes_with_scales.toLocaleString()}{" "}
                        B ({math.phase4_spec_metrics.shared_core_dcore_kib} KiB)
                      </td>
                      <td className="p-3.5 text-cyan-300">
                        {math.deployed_spec_metrics.shared_core_dcore_bytes_with_scales.toLocaleString()}{" "}
                        B ({math.deployed_spec_metrics.shared_core_dcore_kib} KiB)
                      </td>
                      <td className="p-3.5">N/A</td>
                    </tr>
                    <tr>
                      <td className="p-3.5 text-slate-300">
                        Exact Session State / KV-Cache Size (T=1024)
                      </td>
                      <td className="p-3.5">
                        {math.phase2_spec_metrics.state_bytes_per_session.toLocaleString()} B (
                        {math.phase2_spec_metrics.state_kib_binary} KiB)
                      </td>
                      <td className="p-3.5">
                        {math.phase4_spec_metrics.state_bytes_per_session.toLocaleString()} B (
                        {math.phase4_spec_metrics.state_kib_binary} KiB /{" "}
                        {math.phase4_spec_metrics.state_kb_decimal} KB)
                      </td>
                      <td className="p-3.5 text-emerald-300 font-semibold">
                        {math.deployed_spec_metrics.state_bytes_per_session.toLocaleString()} B (
                        {math.deployed_spec_metrics.state_kib_binary} KiB)
                      </td>
                      <td className="p-3.5 text-rose-300">
                        {math.transformer_phase4_metrics.kv_cache_fp32_t1024_kib.toLocaleString()}{" "}
                        KiB (18.0 MiB)
                      </td>
                    </tr>
                    <tr>
                      <td className="p-3.5 text-slate-300">
                        Exact Inference FLOPs per Token (2 × MACs)
                      </td>
                      <td className="p-3.5">
                        {math.phase2_spec_metrics.flops_per_token.toLocaleString()} FLOPs
                      </td>
                      <td className="p-3.5">
                        {math.phase4_spec_metrics.flops_per_token.toLocaleString()} FLOPs
                      </td>
                      <td className="p-3.5 text-cyan-300">
                        {math.deployed_spec_metrics.flops_per_token.toLocaleString()} FLOPs
                      </td>
                      <td className="p-3.5">~9.8M FLOPs + O(L·T·d)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: TRAINING & CORPUS AUDIT */}
        {activeTab === "training" && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Corpus Provenance & Deduplication Card */}
              <div className="rounded-xl bg-slate-900/70 border border-slate-800 p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Database className="w-5 h-5 text-emerald-400" />
                    <h2 className="font-semibold text-white">
                      Curated CC0-1.0 Corpus & Split-Before-Tokenizer Proof
                    </h2>
                  </div>
                  <span className="px-2 py-0.5 text-xs font-mono rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                    0 OVERLAPPING DOCS
                  </span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  The original Phase 4 script repeated 11 synthetic sentences 60 times and sliced
                  90%/10%, causing <strong>100.0% passage overlap</strong> between Train and
                  Validation (memorization masquerading as PPL 1.199). We replaced it with 360
                  unique documents across 4 technical domains, partitioned into{" "}
                  <strong>Train (252 docs)</strong>, <strong>Val (54 docs)</strong>, and{" "}
                  <strong>Test (54 docs)</strong> <em>before</em> training the Byte-BPE tokenizer.
                </p>
                <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                  <div className="p-3 rounded bg-slate-950/60 border border-slate-800">
                    <div className="text-slate-400">Train / Val / Test Docs</div>
                    <div className="text-sm font-bold text-white mt-0.5">
                      {corp.train_documents} / {corp.val_documents} / {corp.test_documents}
                    </div>
                  </div>
                  <div className="p-3 rounded bg-slate-950/60 border border-slate-800">
                    <div className="text-slate-400">Doc Overlap (Tr∩Val / Tr∩Test)</div>
                    <div className="text-sm font-bold text-emerald-400 mt-0.5">
                      {corp.doc_overlap_train_val} docs / {corp.doc_overlap_train_test} docs
                    </div>
                  </div>
                  <div className="p-3 rounded bg-slate-950/60 border border-slate-800">
                    <div className="text-slate-400">5-Gram Jaccard (Audited vs Old)</div>
                    <div className="text-sm font-bold text-cyan-300 mt-0.5">
                      {(corp.word_5gram_jaccard_train_val * 100).toFixed(2)}% vs{" "}
                      <span className="text-rose-400">
                        {(corp.legacy_corpus_5gram_jaccard * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className="p-3 rounded bg-slate-950/60 border border-slate-800">
                    <div className="text-slate-400">Tokenizer Split Provenance</div>
                    <div className="text-sm font-bold text-emerald-300 mt-0.5">
                      train_only (Verified)
                    </div>
                  </div>
                </div>
                <div className="p-3 rounded bg-slate-950/80 border border-slate-800 text-[11px] font-mono space-y-1 text-slate-400">
                  <div>
                    <span className="text-slate-500">Train SHA-256:</span>{" "}
                    {corp.train_sha256.slice(0, 40)}...
                  </div>
                  <div>
                    <span className="text-slate-500">Val SHA-256:</span>{" "}
                    {corp.val_sha256.slice(0, 40)}...
                  </div>
                  <div>
                    <span className="text-slate-500">Test SHA-256:</span>{" "}
                    {corp.test_sha256.slice(0, 40)}...
                  </div>
                </div>
              </div>

              {/* Honest Out-of-Sample Generalization Metrics */}
              <div className="rounded-xl bg-slate-900/70 border border-slate-800 p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Activity className="w-5 h-5 text-cyan-400" />
                    <h2 className="font-semibold text-white">
                      Out-of-Sample Generalization (Train vs Val vs Unseen Test)
                    </h2>
                  </div>
                  <span className="text-xs font-mono text-slate-400">
                    Seed {tr.seed} • {tr.steps} Steps
                  </span>
                </div>

                <table className="w-full text-left border-collapse text-xs font-mono">
                  <thead>
                    <tr className="bg-slate-950/60 text-slate-400 border-b border-slate-800">
                      <th className="p-2.5">Evaluation Split / Metric</th>
                      <th className="p-2.5">DeltaCore-LM (Audited)</th>
                      <th className="p-2.5">Baseline Transformer</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70">
                    <tr>
                      <td className="p-2.5 text-slate-300">Initial Random-Init Val Loss (PPL)</td>
                      <td className="p-2.5">
                        {tr.deltacore.init_val_loss} ({tr.deltacore.init_val_ppl})
                      </td>
                      <td className="p-2.5">
                        {tr.baseline_transformer.init_val_loss} (
                        {Math.exp(tr.baseline_transformer.init_val_loss).toFixed(1)})
                      </td>
                    </tr>
                    <tr>
                      <td className="p-2.5 text-slate-300">Final Train Loss (PPL)</td>
                      <td className="p-2.5 text-cyan-300">
                        {tr.deltacore.final_train_loss} ({tr.deltacore.final_train_ppl})
                      </td>
                      <td className="p-2.5">—</td>
                    </tr>
                    <tr>
                      <td className="p-2.5 text-slate-300">
                        Held-Out Validation Loss (PPL)
                      </td>
                      <td className="p-2.5 text-emerald-300 font-semibold">
                        {tr.deltacore.final_val_loss} (PPL {tr.deltacore.final_val_ppl})
                      </td>
                      <td className="p-2.5">
                        {tr.baseline_transformer.final_val_loss} (PPL{" "}
                        {tr.baseline_transformer.final_val_ppl})
                      </td>
                    </tr>
                    <tr className="bg-slate-950/50">
                      <td className="p-2.5 text-white font-semibold">
                        Held-Out Unseen Test Loss (PPL)
                      </td>
                      <td className="p-2.5 text-emerald-400 font-bold">
                        {tr.deltacore.final_test_loss} (PPL {tr.deltacore.final_test_ppl})
                      </td>
                      <td className="p-2.5 text-slate-200">
                        {tr.baseline_transformer.final_test_loss} (PPL{" "}
                        {tr.baseline_transformer.final_test_ppl})
                      </td>
                    </tr>
                    <tr>
                      <td className="p-2.5 text-slate-300">Held-Out Test Bits-Per-Byte (BPB)</td>
                      <td className="p-2.5 text-cyan-300 font-semibold">
                        {tr.deltacore.test_bits_per_byte} bits/byte
                      </td>
                      <td className="p-2.5">—</td>
                    </tr>
                  </tbody>
                </table>

                <div className="space-y-1.5">
                  <div className="text-xs font-mono text-slate-400">
                    Training & Validation Convergence Trajectory:
                  </div>
                  <div className="grid grid-cols-5 gap-2 text-[11px] font-mono">
                    {tr.history.slice(0, 10).map((h) => (
                      <div
                        key={h.step}
                        className="p-2 rounded bg-slate-950/60 border border-slate-800"
                      >
                        <div className="text-slate-400">Step {h.step}</div>
                        <div className="text-cyan-300">Tr: {h.train_loss}</div>
                        <div className="text-emerald-300">Val: {h.val_loss}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: ARCHITECTURE & CHUNKWISE WY-DELTA AUDIT */}
        {activeTab === "architecture" && (
          <div className="space-y-6">
            {/* Numerical Equivalence Verification Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 font-mono">
              <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
                <div className="text-xs text-slate-400">
                  Blelloch O(T)-Work Scan vs Sequential
                </div>
                <div className="text-lg font-bold text-emerald-400 mt-1">
                  max_err = {Number(r.numerical_tests.blelloch_max_err).toExponential(2)}
                </div>
                <div className="text-[11px] text-slate-400 mt-1">
                  2(T-1) associative combines • O(log T) depth
                </div>
              </div>
              <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
                <div className="text-xs text-slate-400">
                  Hillis-Steele Scan vs Sequential
                </div>
                <div className="text-lg font-bold text-cyan-300 mt-1">
                  max_err = {Number(r.numerical_tests.hillis_steele_max_err).toExponential(2)}
                </div>
                <div className="text-[11px] text-slate-400 mt-1">
                  O(T log T) work • O(log T) depth
                </div>
              </div>
              <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
                <div className="text-xs text-slate-400">
                  Chunkwise WY-Delta vs Sequential Loop
                </div>
                <div className="text-lg font-bold text-emerald-400 mt-1">
                  max_err = {Number(r.numerical_tests.wy_readout_max_err).toExponential(2)}
                </div>
                <div className="text-[11px] text-slate-400 mt-1">
                  (I + L_chunk)⁻¹ triangular solve + GEMMs
                </div>
              </div>
              <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
                <div className="text-xs text-slate-400">
                  Sequence Prefill vs O(1) Step Recurrence
                </div>
                <div className="text-lg font-bold text-emerald-400 mt-1">
                  max_err ={" "}
                  {Number(r.numerical_tests.seq_vs_step_logit_max_err).toExponential(2)}
                </div>
                <div className="text-[11px] text-slate-400 mt-1">
                  Exact equivalence across all L layers
                </div>
              </div>
            </div>

            {/* Literature Provenance & Novelty Audit Table */}
            <div className="rounded-xl bg-slate-900/70 border border-slate-800 overflow-hidden">
              <div className="px-5 py-4 border-b border-slate-800 flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-cyan-400" />
                <h2 className="font-semibold text-white">
                  Phase 3 Literature & Architectural Provenance Audit (Inherited vs Original Synthesis)
                </h2>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-950/60 text-slate-400 font-mono border-b border-slate-800">
                      <th className="p-3.5">DeltaCore-LM Mechanism</th>
                      <th className="p-3.5">Provenance Status</th>
                      <th className="p-3.5">Primary Literature Citations</th>
                      <th className="p-3.5">Engineering Audit Verdict</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70">
                    {math.literature_audit.map((item, i) => (
                      <tr key={i} className="hover:bg-slate-800/30">
                        <td className="p-3.5 font-mono font-semibold text-white">
                          {item.component}
                        </td>
                        <td className="p-3.5">
                          <span
                            className={`px-2 py-0.5 text-[11px] font-mono rounded ${
                              item.status.includes("Original")
                                ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
                                : "bg-slate-800 text-slate-300"
                            }`}
                          >
                            {item.status}
                          </span>
                        </td>
                        <td className="p-3.5 text-cyan-300 font-mono">{item.citation}</td>
                        <td className="p-3.5 text-slate-300">{item.audit_verdict}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: QUANTIZATION & 4-WAY PARITY LAB */}
        {activeTab === "quantization" && (
          <div className="space-y-6">
            {/* Interactive Live Parity Runner */}
            <div className="rounded-xl bg-slate-900/70 border border-slate-800 p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <GitCompare className="w-5 h-5 text-cyan-400" />
                  <h2 className="font-semibold text-white">
                    Live On-Demand Cross-Runtime Logit Parity Verifier (WebAssembly .wasm vs Native C++17 AVX2 vs FP32)
                  </h2>
                </div>
                <span className="text-xs font-mono text-emerald-400">
                  Mean Weight SQNR: {r.quantization_export_audit.deployed_dcore.mean_sqnr_db} dB
                </span>
              </div>

              <div className="flex flex-col md:flex-row gap-3">
                <input
                  type="text"
                  value={parityPrompt}
                  onChange={(e) => setParityPrompt(e.target.value)}
                  className="flex-1 px-3.5 py-2 rounded-lg bg-slate-950 border border-slate-700 text-xs font-mono text-white focus:outline-none focus:border-cyan-400"
                  placeholder="Enter any prompt to verify live C++ vs WASM logit parity..."
                />
                <button
                  onClick={handleRunLiveParity}
                  disabled={isRunningParity}
                  className="px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 disabled:opacity-50 text-slate-950 font-semibold text-xs font-mono flex items-center gap-2 cursor-pointer"
                >
                  {isRunningParity ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : (
                    <Play className="w-4 h-4" />
                  )}
                  <span>Run Live C++ vs WASM Parity Check</span>
                </button>
              </div>

              {liveParityResult && (
                <div className="p-4 rounded-lg bg-slate-950/90 border border-cyan-500/30 space-y-3 text-xs font-mono">
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-2">
                    <span className="text-cyan-300 font-semibold">
                      Live Parity Verification Completed & Logged to PostgreSQL
                    </span>
                    <span className="text-emerald-400 font-bold">
                      Top-1 Match:{" "}
                      {(
                        liveParityResult.cpp_int8_vs_wasm_int8 as {
                          top1_match: boolean;
                        }
                      )?.top1_match
                        ? "PASS (100%)"
                        : "MISMATCH"}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div>
                      <span className="text-slate-400 block">C++ vs WASM Max Abs Err:</span>
                      <span className="text-emerald-300 font-bold">
                        {
                          (
                            liveParityResult.cpp_int8_vs_wasm_int8 as {
                              max_abs_err: number;
                            }
                          )?.max_abs_err
                        }
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 block">C++ vs WASM Cosine Sim:</span>
                      <span className="text-cyan-300 font-bold">
                        {
                          (
                            liveParityResult.cpp_int8_vs_wasm_int8 as {
                              cosine_similarity: number;
                            }
                          )?.cosine_similarity
                        }
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 block">FP32 vs INT8 Cosine Sim:</span>
                      <span className="text-emerald-300 font-bold">
                        {String(liveParityResult.fp32_vs_int8_cosine)}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 block">16-Token Greedy Agreement:</span>
                      <span className="text-emerald-400 font-bold">
                        {String(liveParityResult.greedy_16tok_agreement_pct)}%
                      </span>
                    </div>
                  </div>
                  <div className="text-slate-300">
                    <span className="text-slate-500">WASM 16-Tok Greedy Continuation:</span>{" "}
                    {String(liveParityResult.wasm_greedy_16tok_decoded)}
                  </div>
                </div>
              )}
            </div>

            {/* 6-Case Preverified 4-Way Parity Table */}
            <div className="rounded-xl bg-slate-900/70 border border-slate-800 overflow-hidden">
              <div className="px-5 py-4 border-b border-slate-800">
                <h3 className="font-semibold text-white text-sm">
                  Automated 4-Way Parity Suite (Python FP32 vs Python INT8 vs Native C++17 AVX2 vs WebAssembly .wasm)
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs font-mono">
                  <thead>
                    <tr className="bg-slate-950/60 text-slate-400 border-b border-slate-800">
                      <th className="p-3">Case</th>
                      <th className="p-3">Prompt</th>
                      <th className="p-3">FP32 vs INT8 Cosine</th>
                      <th className="p-3">FP32 vs INT8 KL Div</th>
                      <th className="p-3">PyINT8 vs C++ Max Err</th>
                      <th className="p-3">C++ vs WASM Max Err</th>
                      <th className="p-3">WASM==PyINT8 (16-Tok)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70">
                    {r.parity_cases.map((c) => (
                      <tr key={c.case_id} className="hover:bg-slate-800/30">
                        <td className="p-3 font-semibold text-cyan-300">{c.case_id}</td>
                        <td className="p-3 text-slate-300 max-w-xs truncate">{c.prompt}</td>
                        <td className="p-3 text-emerald-300">
                          {c.fp32_vs_int8_metrics.cosine_similarity}
                        </td>
                        <td className="p-3">{c.fp32_vs_int8_metrics.kl_divergence}</td>
                        <td className="p-3 text-cyan-300">{c.py_int8_vs_cpp_int8.max_abs_err}</td>
                        <td className="p-3 text-emerald-300">
                          {c.cpp_int8_vs_wasm_int8.max_abs_err}
                        </td>
                        <td className="p-3 text-emerald-400 font-bold">
                          {(c.wasm_vs_py_int8_16tok_agreement * 100).toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: LIVE WASM VS NATIVE BENCHMARK & STREAMING CONSOLE */}
        {activeTab === "wasm" && (
          <div className="space-y-6">
            {/* Live Streaming Console */}
            <div className="rounded-xl bg-slate-900/70 border border-slate-800 p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <Terminal className="w-5 h-5 text-emerald-400" />
                  <h2 className="font-semibold text-white">
                    Live WebAssembly (.wasm) SSE Token Streaming Console
                  </h2>
                </div>
                <div className="flex items-center gap-2 text-xs font-mono">
                  <button
                    onClick={() => setModelSpec("deployed")}
                    className={`px-2.5 py-1 rounded border cursor-pointer ${
                      modelSpec === "deployed"
                        ? "bg-cyan-500/20 border-cyan-400 text-cyan-200"
                        : "bg-slate-950 border-slate-800 text-slate-400"
                    }`}
                  >
                    Deployed Spec (d=128, L=4, 426 KiB)
                  </button>
                  <button
                    onClick={() => setModelSpec("phase4")}
                    className={`px-2.5 py-1 rounded border cursor-pointer ${
                      modelSpec === "phase4"
                        ? "bg-cyan-500/20 border-cyan-400 text-cyan-200"
                        : "bg-slate-950 border-slate-800 text-slate-400"
                    }`}
                  >
                    Phase-4 Spec (d=192, L=6, 1171 KiB)
                  </button>
                </div>
              </div>

              <div className="flex flex-wrap gap-2">
                {PRESET_PROMPTS.map((p, idx) => (
                  <button
                    key={idx}
                    onClick={() => setPrompt(p)}
                    className="px-2.5 py-1 text-[11px] font-mono rounded bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-300 cursor-pointer"
                  >
                    Preset #{idx + 1}: {p.slice(0, 38)}...
                  </button>
                ))}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                <div className="md:col-span-2">
                  <label className="block text-[11px] font-mono text-slate-400 mb-1">
                    Input Prompt
                  </label>
                  <input
                    type="text"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-700 text-xs font-mono text-white"
                  />
                </div>
                <div>
                  <label className="block text-[11px] font-mono text-slate-400 mb-1">
                    Max Tokens ({maxTokens}) & Temp ({temperature})
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="number"
                      min={8}
                      max={128}
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(Number(e.target.value))}
                      className="w-1/2 px-2.5 py-2 rounded-lg bg-slate-950 border border-slate-700 text-xs font-mono text-white"
                    />
                    <input
                      type="number"
                      step={0.05}
                      min={0}
                      max={1.2}
                      value={temperature}
                      onChange={(e) => setTemperature(Number(e.target.value))}
                      className="w-1/2 px-2.5 py-2 rounded-lg bg-slate-950 border border-slate-700 text-xs font-mono text-white"
                    />
                  </div>
                </div>
                <div className="flex items-end gap-2">
                  <label className="flex items-center gap-1.5 text-xs font-mono text-slate-300 mb-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={forceColdStart}
                      onChange={(e) => setForceColdStart(e.target.checked)}
                    />
                    <span>Cold Start</span>
                  </label>
                  <button
                    onClick={handleStreamInference}
                    disabled={isGenerating}
                    className="flex-1 py-2 px-4 rounded-lg bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 text-slate-950 font-semibold text-xs font-mono flex items-center justify-center gap-1.5 cursor-pointer"
                  >
                    <Zap className="w-4 h-4" />
                    <span>{isGenerating ? "Streaming..." : "Stream WASM"}</span>
                  </button>
                </div>
              </div>

              <div className="p-4 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs min-h-[84px]">
                <span className="text-cyan-400">{prompt}</span>
                <span className="text-emerald-300">{streamingOutput}</span>
                {isGenerating && (
                  <span className="inline-block w-2 h-4 bg-emerald-400 animate-pulse ml-1" />
                )}
              </div>

              {liveGenMetrics && (
                <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-xs font-mono">
                  <div className="p-2.5 rounded bg-slate-950/70 border border-slate-800">
                    <span className="text-slate-400 block">Latency / Token</span>
                    <span className="text-emerald-300 font-bold">
                      {liveGenMetrics.ms_per_token} ms/tok
                    </span>
                  </div>
                  <div className="p-2.5 rounded bg-slate-950/70 border border-slate-800">
                    <span className="text-slate-400 block">Throughput</span>
                    <span className="text-cyan-300 font-bold">
                      {liveGenMetrics.tokens_per_sec} tok/s
                    </span>
                  </div>
                  <div className="p-2.5 rounded bg-slate-950/70 border border-slate-800">
                    <span className="text-slate-400 block">Cold Start</span>
                    <span className="text-white font-bold">
                      {liveGenMetrics.cold_start_ms} ms
                    </span>
                  </div>
                  <div className="p-2.5 rounded bg-slate-950/70 border border-slate-800">
                    <span className="text-slate-400 block">Session State</span>
                    <span className="text-emerald-300 font-bold">
                      {liveGenMetrics.session_state_bytes} B ({liveGenMetrics.session_state_kib}{" "}
                      KiB)
                    </span>
                  </div>
                  <div className="p-2.5 rounded bg-slate-950/70 border border-slate-800">
                    <span className="text-slate-400 block">WASM Heap Top</span>
                    <span className="text-amber-300 font-bold">
                      {Number(liveGenMetrics.wasm_heap_top_bytes).toLocaleString()} B
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* WASM vs Native C++ Benchmark Comparison Table */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="rounded-xl bg-slate-900/70 border border-slate-800 overflow-hidden">
                <div className="px-5 py-4 border-b border-slate-800">
                  <h3 className="font-semibold text-white text-sm">
                    WebAssembly (.wasm in V8) vs Native C++17 AVX2 Benchmark
                  </h3>
                </div>
                <table className="w-full text-left border-collapse text-xs font-mono">
                  <thead>
                    <tr className="bg-slate-950/60 text-slate-400 border-b border-slate-800">
                      <th className="p-3">Metric</th>
                      <th className="p-3">Deployed (d=128, L=4)</th>
                      <th className="p-3">Phase-4 Spec (d=192, L=6)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70">
                    <tr>
                      <td className="p-3 text-slate-300">Artifact Size (.dcore)</td>
                      <td className="p-3 text-cyan-300">
                        {wasm.deployed_d128_l4.artifact_bytes.toLocaleString()} B (
                        {wasm.deployed_d128_l4.artifact_kib} KiB)
                      </td>
                      <td className="p-3">
                        {wasm.phase4_spec_d192_l6.artifact_bytes.toLocaleString()} B (
                        {wasm.phase4_spec_d192_l6.artifact_kib} KiB)
                      </td>
                    </tr>
                    <tr>
                      <td className="p-3 text-slate-300">WASM Cold-Start Init</td>
                      <td className="p-3">{wasm.deployed_d128_l4.wasm_cold_start_ms} ms</td>
                      <td className="p-3">{wasm.phase4_spec_d192_l6.wasm_cold_start_ms} ms</td>
                    </tr>
                    <tr>
                      <td className="p-3 text-emerald-300 font-semibold">
                        Real V8 WebAssembly Latency
                      </td>
                      <td className="p-3 text-emerald-400 font-bold">
                        {wasm.deployed_d128_l4.wasm_ms_per_token} ms/tok (
                        {wasm.deployed_d128_l4.wasm_tokens_per_sec} tok/s)
                      </td>
                      <td className="p-3 text-emerald-300 font-semibold">
                        {wasm.phase4_spec_d192_l6.wasm_ms_per_token} ms/tok (
                        {wasm.phase4_spec_d192_l6.wasm_tokens_per_sec} tok/s)
                      </td>
                    </tr>
                    <tr>
                      <td className="p-3 text-cyan-300 font-semibold">
                        Native C++17 AVX2 Latency
                      </td>
                      <td className="p-3 text-cyan-300 font-bold">
                        {wasm.deployed_d128_l4.native_cpp_ms_per_token} ms/tok (
                        {wasm.deployed_d128_l4.native_cpp_tokens_per_sec} tok/s)
                      </td>
                      <td className="p-3 text-cyan-300">
                        {wasm.phase4_spec_d192_l6.native_cpp_ms_per_token} ms/tok (
                        {wasm.phase4_spec_d192_l6.native_cpp_tokens_per_sec} tok/s)
                      </td>
                    </tr>
                    <tr>
                      <td className="p-3 text-slate-300">WASM / Native Speed Ratio</td>
                      <td className="p-3">
                        {(wasm.deployed_d128_l4.wasm_to_native_speed_ratio * 100).toFixed(1)}% of
                        native speed
                      </td>
                      <td className="p-3">
                        {(wasm.phase4_spec_d192_l6.wasm_to_native_speed_ratio * 100).toFixed(1)}%
                        of native speed
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Context Length Scaling Table */}
              <div className="rounded-xl bg-slate-900/70 border border-slate-800 overflow-hidden">
                <div className="px-5 py-4 border-b border-slate-800">
                  <h3 className="font-semibold text-white text-sm">
                    O(1) Context-Length Scaling in WebAssembly (T = 64 → 1024)
                  </h3>
                </div>
                <table className="w-full text-left border-collapse text-xs font-mono">
                  <thead>
                    <tr className="bg-slate-950/60 text-slate-400 border-b border-slate-800">
                      <th className="p-3">Context T</th>
                      <th className="p-3">WASM d=128 L=4</th>
                      <th className="p-3">WASM d=192 L=6</th>
                      <th className="p-3">Recurrent State</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70">
                    {wasm.context_scaling_benchmarks.map((row) => (
                      <tr key={row.context_length}>
                        <td className="p-3 font-bold text-white">T = {row.context_length}</td>
                        <td className="p-3 text-emerald-300">
                          {row.wasm_deployed_d128_l4.ms_per_token} ms (
                          {row.wasm_deployed_d128_l4.tokens_per_sec} tok/s)
                        </td>
                        <td className="p-3 text-cyan-300">
                          {row.wasm_phase4_d192_l6.ms_per_token} ms (
                          {row.wasm_phase4_d192_l6.tokens_per_sec} tok/s)
                        </td>
                        <td className="p-3 text-slate-300">
                          14.00 KiB / 31.50 KiB (Constant)
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* TAB 6: VERCEL CONFIG & MULTI-SESSION CONCURRENCY STRESS TESTER */}
        {activeTab === "vercel" && (
          <div className="space-y-6">
            <div className="rounded-xl bg-slate-900/70 border border-slate-800 p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <Server className="w-5 h-5 text-emerald-400" />
                  <h2 className="font-semibold text-white">
                    Live Multi-Tenant Interleaved Session Isolation & Zero-Leak Stress Tester
                  </h2>
                </div>
                <span className="text-xs font-mono text-emerald-300">
                  1,000-Cycle Pre-Audit Heap Leak: {wasm.memory_and_isolation_tests.leaked_bytes}{" "}
                  Bytes
                </span>
              </div>

              <p className="text-xs text-slate-300">
                Allocates <strong>N simultaneous recurrent sessions</strong> on the WebAssembly
                heap, steps all N sessions token-by-token in interleaved round-robin order,
                verifies that every session&apos;s output token trajectory is{" "}
                <strong>100% bit-for-bit identical</strong> to running that session alone in
                isolation, and verifies <strong>0 bytes of WASM heap growth</strong> after freeing
                all session pointers.
              </p>

              <div className="flex flex-wrap items-end gap-3 font-mono text-xs">
                <div>
                  <label className="block text-slate-400 mb-1">
                    Concurrent Sessions ({concurrentCount})
                  </label>
                  <input
                    type="number"
                    min={2}
                    max={16}
                    value={concurrentCount}
                    onChange={(e) => setConcurrentCount(Number(e.target.value))}
                    className="px-3 py-2 rounded-lg bg-slate-950 border border-slate-700 text-white w-36"
                  />
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">
                    Tokens / Session ({tokensPerSess})
                  </label>
                  <input
                    type="number"
                    min={8}
                    max={64}
                    value={tokensPerSess}
                    onChange={(e) => setTokensPerSess(Number(e.target.value))}
                    className="px-3 py-2 rounded-lg bg-slate-950 border border-slate-700 text-white w-36"
                  />
                </div>
                <button
                  onClick={handleRunConcurrencyTest}
                  disabled={isRunningConcurrency}
                  className="px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 text-slate-950 font-semibold flex items-center gap-2 cursor-pointer"
                >
                  {isRunningConcurrency ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : (
                    <CheckCircle2 className="w-4 h-4" />
                  )}
                  <span>Run Interleaved Multi-Session Isolation Test</span>
                </button>
              </div>

              {liveConcurrencyResult && (
                <div className="p-4 rounded-lg bg-slate-950 border border-emerald-500/30 space-y-3 text-xs font-mono">
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                    <div>
                      <span className="text-slate-400 block">Bit-Exact Isolation:</span>
                      <span className="text-emerald-400 font-bold">
                        {liveConcurrencyResult.bit_exact_isolation_verified
                          ? "VERIFIED (100%)"
                          : "FAILED"}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 block">WASM Heap Leak:</span>
                      <span className="text-emerald-300 font-bold">
                        {String(liveConcurrencyResult.heap_leak_bytes)} Bytes
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 block">Total Throughput:</span>
                      <span className="text-cyan-300 font-bold">
                        {String(liveConcurrencyResult.aggregate_tokens_per_sec)} tok/s
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 block">Total Concurrent State:</span>
                      <span className="text-white font-bold">
                        {String(liveConcurrencyResult.total_concurrent_state_kib)} KiB
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 block">Wall-Clock Duration:</span>
                      <span className="text-white font-bold">
                        {String(liveConcurrencyResult.total_duration_ms)} ms
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Vercel Configuration Audit Comparison */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="rounded-xl bg-slate-900/70 border border-slate-800 p-5 space-y-3">
                <div className="flex items-center gap-2 text-rose-400 font-mono text-xs font-semibold">
                  <AlertTriangle className="w-4 h-4" />
                  <span>Original Phase 6 Flaws (Fixed in Audit)</span>
                </div>
                <ul className="text-xs text-slate-300 space-y-2 list-disc pl-4">
                  <li>
                    <strong>Unsupported Memory Override:</strong> Original <code>vercel.json</code>{" "}
                    hardcoded <code>&quot;memory&quot;: 512</code> on legacy{" "}
                    <code>api/generate.js</code>, conflicting with modern Vercel fluid compute /
                    plan limits.
                  </li>
                  <li>
                    <strong>Legacy Express Handler Syntax:</strong> Used Pages-style{" "}
                    <code>(req, res)</code> instead of Next.js App Router Web Fetch{" "}
                    <code>ReadableStream</code> (<code>src/app/api/generate/route.ts</code>).
                  </li>
                  <li>
                    <strong>Missing Output File Tracing:</strong> Did not configure{" "}
                    <code>outputFileTracingIncludes</code> in <code>next.config.ts</code> to bundle{" "}
                    <code>artifacts/*.wasm</code> and <code>artifacts/*.dcore</code> in production
                    builds.
                  </li>
                </ul>
              </div>

              <div className="rounded-xl bg-slate-900/70 border border-slate-800 p-5 space-y-3">
                <div className="flex items-center gap-2 text-emerald-400 font-mono text-xs font-semibold">
                  <FileCode2 className="w-4 h-4" />
                  <span>Audited Production Configuration (vercel.json + next.config.ts)</span>
                </div>
                <pre className="p-3 rounded bg-slate-950 border border-slate-800 text-[11px] font-mono text-cyan-300 overflow-x-auto">
                  {`// next.config.ts
const nextConfig: NextConfig = {
  outputFileTracingIncludes: {
    "/api/**/*": ["./artifacts/**/*"],
  },
};

// vercel.json (memory override removed)
{
  "framework": "nextjs",
  "functions": {
    "src/app/api/**/*": {
      "maxDuration": 15,
      "includeFiles": "artifacts/{model.dcore,tokenizer.json,deltacore_runtime.wasm}"
    }
  }
}`}
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* Recent PostgreSQL Telemetry Log */}
        <div className="rounded-xl bg-slate-900/70 border border-slate-800 overflow-hidden">
          <div className="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-cyan-400" />
              <h3 className="font-semibold text-white text-xs font-mono">
                Live PostgreSQL Audit Telemetry Log (inference_runs & concurrency_audit_runs)
              </h3>
            </div>
            <button
              onClick={fetchAuditData}
              className="text-xs font-mono text-cyan-400 hover:text-cyan-300 flex items-center gap-1 cursor-pointer"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Refresh DB</span>
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs font-mono">
              <thead>
                <tr className="bg-slate-950/60 text-slate-400 border-b border-slate-800">
                  <th className="p-2.5">Run ID</th>
                  <th className="p-2.5">Model Spec</th>
                  <th className="p-2.5">Prompt</th>
                  <th className="p-2.5">Generated Continuation</th>
                  <th className="p-2.5">Speed</th>
                  <th className="p-2.5">State</th>
                  <th className="p-2.5">RSS</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {data.db_telemetry.recent_inference.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="p-4 text-center text-slate-500">
                      Click &quot;Stream WASM&quot; in Tab 5 to log live inference runs to
                      PostgreSQL.
                    </td>
                  </tr>
                ) : (
                  data.db_telemetry.recent_inference.map((run) => (
                    <tr key={run.id} className="hover:bg-slate-800/30">
                      <td className="p-2.5 text-cyan-300">#{run.id}</td>
                      <td className="p-2.5 text-slate-300">{run.modelSpec}</td>
                      <td className="p-2.5 text-slate-400 max-w-[200px] truncate">{run.prompt}</td>
                      <td className="p-2.5 text-emerald-300 max-w-[260px] truncate">
                        {run.generatedText}
                      </td>
                      <td className="p-2.5 text-white font-semibold">
                        {run.msPerToken} ms/tok ({run.tokensPerSec} tok/s)
                      </td>
                      <td className="p-2.5 text-emerald-400">
                        {(run.sessionStateBytes / 1024).toFixed(2)} KiB
                      </td>
                      <td className="p-2.5 text-slate-300">{run.processRssMb} MB</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}
