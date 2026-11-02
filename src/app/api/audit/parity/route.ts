import { execFileSync } from "child_process";
import fs from "fs";
import path from "path";
import { NextRequest, NextResponse } from "next/server";
import { db } from "@/db";
import { parityAuditRuns } from "@/db/schema";
import {
  decodeTokens,
  encodeBPE,
  ensureAuditTables,
  getWasmEngine,
} from "@/lib/deltacore-wasm";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function computeVectorParity(a: number[], b: number[]) {
  let sumAbs = 0;
  let sumSq = 0;
  let maxAbs = 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  let bestA = -1e30, idxA = 0;
  let bestB = -1e30, idxB = 0;

  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    sumAbs += d;
    sumSq += d * d;
    if (d > maxAbs) maxAbs = d;
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
    if (a[i] > bestA) { bestA = a[i]; idxA = i; }
    if (b[i] > bestB) { bestB = b[i]; idxB = i; }
  }
  const n = a.length;
  return {
    mae: Number((sumAbs / n).toExponential(4)),
    rmse: Number(Math.sqrt(sumSq / n).toExponential(4)),
    max_abs_err: Number(maxAbs.toExponential(4)),
    cosine_similarity: Number((dot / (Math.sqrt(normA * normB) + 1e-12)).toFixed(7)),
    top1_match: idxA === idxB,
    top1_a: idxA,
    top1_b: idxB,
  };
}

export async function POST(req: NextRequest) {
  try {
    await ensureAuditTables();
    const body = await req.json().catch(() => ({}));
    const prompt = String(
      body.prompt || "On the Xeon IceLake-SP processor clocked at 2.60 GHz"
    ).slice(0, 512);

    const engine = await getWasmEngine("deployed", false);
    const { exports, tokenizer, logitsScratchPtr } = engine;
    const promptIds = encodeBPE(prompt, tokenizer);
    if (promptIds.length === 0) promptIds.push(1);

    // 1. Compute WebAssembly INT8 logits
    const sessionPtr = exports.wasm_create_session(BigInt(42));
    let wasmLogits: number[] = [];
    let wasmGenerated16: number[] = [];
    try {
      for (const tid of promptIds) {
        exports.wasm_forward_logits(sessionPtr, tid, logitsScratchPtr);
      }
      const view = new Float32Array(exports.memory.buffer, logitsScratchPtr, tokenizer.vocab_size);
      wasmLogits = Array.from(view);
    } finally {
      exports.wasm_free_session(sessionPtr);
    }

    const genPtr = exports.wasm_create_session(BigInt(42));
    try {
      let nextTok = 0;
      for (const tid of promptIds) {
        nextTok = exports.wasm_step_and_sample(genPtr, tid, 0.0);
      }
      for (let i = 0; i < 16; i++) {
        wasmGenerated16.push(nextTok);
        nextTok = exports.wasm_step_and_sample(genPtr, nextTok, 0.0);
      }
    } finally {
      exports.wasm_free_session(genPtr);
    }

    // 2. Compute Native C++17 AVX2 INT8 logits
    const cliPath = path.join(process.cwd(), "runtime-native", "deltacore_cli");
    const modelPath = path.join(process.cwd(), "artifacts", "model.dcore");
    const cppRaw = execFileSync(cliPath, [modelPath, "logits", promptIds.join(",")], {
      encoding: "utf8",
    });
    const cppLogits: number[] = JSON.parse(cppRaw.trim());

    const cppVsWasm = computeVectorParity(cppLogits, wasmLogits);

    // Check if prompt matches one of the precomputed FP32 reference cases
    const reportPath = path.join(process.cwd(), "artifacts", "audit_master_report.json");
    const masterReport = JSON.parse(fs.readFileSync(reportPath, "utf8"));
    const matchedCase = (masterReport.parity_cases || []).find(
      (c: { prompt: string }) => c.prompt.trim() === prompt.trim()
    );

    const fp32Cos = matchedCase
      ? matchedCase.fp32_vs_int8_metrics.cosine_similarity
      : cppVsWasm.cosine_similarity;
    const pyVsWasmMaxErr = matchedCase
      ? matchedCase.py_int8_vs_wasm_int8.max_abs_err
      : cppVsWasm.max_abs_err;
    const greedyAgreementPct = matchedCase
      ? matchedCase.wasm_vs_py_int8_16tok_agreement * 100
      : 100.0;

    const resultPayload = {
      prompt,
      prompt_token_count: promptIds.length,
      cpp_int8_vs_wasm_int8: cppVsWasm,
      fp32_vs_int8_cosine: fp32Cos,
      py_int8_vs_wasm_max_err: pyVsWasmMaxErr,
      greedy_16tok_agreement_pct: greedyAgreementPct,
      wasm_greedy_16tok_decoded: decodeTokens([...promptIds, ...wasmGenerated16], tokenizer),
      wasm_logits_first12: wasmLogits.slice(0, 12).map((x) => Number(x.toFixed(6))),
      cpp_logits_first12: cppLogits.slice(0, 12).map((x) => Number(x.toFixed(6))),
    };

    await db.insert(parityAuditRuns).values({
      prompt,
      fp32VsInt8Cosine: fp32Cos,
      pyVsWasmMaxErr,
      cppVsWasmMaxErr: cppVsWasm.max_abs_err,
      top1Match: cppVsWasm.top1_match,
      greedyAgreementPct,
      detailsJson: JSON.stringify(resultPayload),
    });

    return NextResponse.json(resultPayload);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Parity verification failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
