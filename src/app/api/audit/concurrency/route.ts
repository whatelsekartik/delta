import { NextRequest, NextResponse } from "next/server";
import { performance } from "perf_hooks";
import { db } from "@/db";
import { concurrencyAuditRuns } from "@/db/schema";
import {
  decodeTokens,
  encodeBPE,
  ensureAuditTables,
  getWasmEngine,
} from "@/lib/deltacore-wasm";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const DEFAULT_PROMPTS = [
  "On the Xeon IceLake-SP processor clocked at 2.60 GHz",
  "Architectural analysis of Widrow-Hoff Delta Rule Network",
  "Question: Which cluster and port currently host vault-auth? Answer:",
  "Engineering note on WebAssembly linear memory during production audit",
  "When executing a matrix-vector product at batch size one",
  "Under sequence length 1024 and dimension 160, empirical perturbation",
  "Telemetry record: microservice ledger-sync booted in cluster oregon-us2",
  "On the Graviton3 Neoverse-V1 processor clocked at 2.60 GHz with SVE-256",
];

export async function POST(req: NextRequest) {
  try {
    await ensureAuditTables();
    const body = await req.json().catch(() => ({}));
    const numSessions = Math.min(Math.max(parseInt(String(body.sessions || "8"), 10) || 8, 2), 16);
    const tokensPerSession = Math.min(Math.max(parseInt(String(body.tokens_per_session || "24"), 10) || 24, 4), 64);

    const engine = await getWasmEngine("deployed", false);
    const { exports, tokenizer, sessionStateBytes } = engine;

    // Warm up free-list pool for numSessions slots so we can verify 0 heap leak after release
    const warmupPtrs: number[] = [];
    for (let i = 0; i < numSessions; i++) {
      warmupPtrs.push(exports.wasm_create_session(BigInt(i + 1)));
    }
    for (const p of warmupPtrs) {
      exports.wasm_free_session(p);
    }
    const heapBefore = exports.wasm_get_heap_top();

    // 1. Run each session sequentially in isolation as ground truth
    const isolatedOutputs: number[][] = [];
    const promptTokenLists: number[][] = [];
    for (let s = 0; s < numSessions; s++) {
      const prompt = DEFAULT_PROMPTS[s % DEFAULT_PROMPTS.length];
      const pIds = encodeBPE(prompt, tokenizer);
      promptTokenLists.push(pIds);
      const seed = BigInt(1000 + s * 97);

      const ptr = exports.wasm_create_session(seed);
      try {
        let tok = pIds[0] || 1;
        for (let i = 0; i < pIds.length; i++) {
          tok = exports.wasm_step_and_sample(ptr, pIds[i], 0.25);
        }
        const gen: number[] = [];
        for (let step = 0; step < tokensPerSession; step++) {
          gen.push(tok);
          tok = exports.wasm_step_and_sample(ptr, tok, 0.25);
        }
        isolatedOutputs.push(gen);
      } finally {
        exports.wasm_free_session(ptr);
      }
    }

    // 2. Allocate all N sessions simultaneously and interleave token steps round-robin
    const t0 = performance.now();
    const activePtrs: number[] = [];
    const currentTok: number[] = new Array(numSessions).fill(1);
    const interleavedOutputs: number[][] = Array.from({ length: numSessions }, () => []);

    try {
      for (let s = 0; s < numSessions; s++) {
        const seed = BigInt(1000 + s * 97);
        const ptr = exports.wasm_create_session(seed);
        activePtrs.push(ptr);
      }

      // Interleaved prefill
      const maxPromptLen = Math.max(...promptTokenLists.map((p) => p.length));
      for (let step = 0; step < maxPromptLen; step++) {
        for (let s = 0; s < numSessions; s++) {
          const pIds = promptTokenLists[s];
          if (step < pIds.length) {
            currentTok[s] = exports.wasm_step_and_sample(activePtrs[s], pIds[step], 0.25);
          }
        }
      }

      // Interleaved autoregressive generation
      for (let step = 0; step < tokensPerSession; step++) {
        for (let s = 0; s < numSessions; s++) {
          interleavedOutputs[s].push(currentTok[s]);
          currentTok[s] = exports.wasm_step_and_sample(activePtrs[s], currentTok[s], 0.25);
        }
      }
    } finally {
      for (const ptr of activePtrs) {
        exports.wasm_free_session(ptr);
      }
    }

    const totalDurationMs = Number((performance.now() - t0).toFixed(3));
    const heapAfter = exports.wasm_get_heap_top();
    const heapLeakBytes = heapAfter - heapBefore;

    const totalTokens = numSessions * tokensPerSession;
    const aggregateTokensPerSec = Number(((totalTokens * 1000) / Math.max(0.001, totalDurationMs)).toFixed(1));

    let allBitExact = true;
    const sessionDetails = [];
    for (let s = 0; s < numSessions; s++) {
      const match = JSON.stringify(isolatedOutputs[s]) === JSON.stringify(interleavedOutputs[s]);
      if (!match) allBitExact = false;
      sessionDetails.push({
        session_index: s + 1,
        wasm_session_ptr: activePtrs[s],
        seed: 1000 + s * 97,
        prompt: DEFAULT_PROMPTS[s % DEFAULT_PROMPTS.length],
        generated_preview: decodeTokens(interleavedOutputs[s], tokenizer),
        isolated_vs_interleaved_bit_exact: match,
        state_bytes: sessionStateBytes,
      });
    }

    await db.insert(concurrencyAuditRuns).values({
      concurrentSessions: numSessions,
      tokensPerSession,
      totalDurationMs,
      aggregateTokensPerSec,
      bitExactIsolationVerified: allBitExact,
      heapLeakBytes,
      sessionDetailsJson: JSON.stringify(sessionDetails),
    });

    return NextResponse.json({
      status: "verified",
      concurrent_sessions: numSessions,
      tokens_per_session: tokensPerSession,
      total_tokens_generated: totalTokens,
      total_duration_ms: totalDurationMs,
      aggregate_tokens_per_sec: aggregateTokensPerSec,
      total_concurrent_state_kib: Number(((numSessions * sessionStateBytes) / 1024).toFixed(2)),
      bit_exact_isolation_verified: allBitExact,
      heap_before_bytes: heapBefore,
      heap_after_bytes: heapAfter,
      heap_leak_bytes: heapLeakBytes,
      sessions: sessionDetails,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Concurrency audit error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
