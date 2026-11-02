import { NextRequest, NextResponse } from "next/server";
import { performance } from "perf_hooks";
import { db } from "@/db";
import { inferenceRuns } from "@/db/schema";
import {
  decodeToken,
  decodeTokens,
  encodeBPE,
  ensureAuditTables,
  getWasmEngine,
} from "@/lib/deltacore-wasm";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 15;

export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));
    return handleInference(body);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Inference error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const body = {
    prompt: searchParams.get("prompt") || undefined,
    max_tokens: searchParams.get("max_tokens") || undefined,
    temperature: searchParams.get("temperature") || undefined,
    spec: searchParams.get("spec") || undefined,
    stream: searchParams.get("stream") !== "false",
    force_cold_start: searchParams.get("force_cold_start") === "true",
  };
  return handleInference(body);
}

async function handleInference(body: Record<string, unknown>) {
  await ensureAuditTables();

  const prompt = String(
    body.prompt || "On the Xeon IceLake-SP processor clocked at 2.60 GHz"
  ).slice(0, 1024);
  const maxTokens = Math.min(Math.max(parseInt(String(body.max_tokens || "40"), 10) || 40, 1), 128);
  const temperature = Math.min(Math.max(parseFloat(String(body.temperature ?? "0.15")) || 0, 0.0), 1.5);
  const specKey: "deployed" | "phase4" = body.spec === "phase4" ? "phase4" : "deployed";
  const streamMode = body.stream !== false;
  const forceColdStart = Boolean(body.force_cold_start);
  const customSeed = body.seed ? BigInt(Number(body.seed)) : BigInt(Date.now() & 0x7fffffff);

  const engine = await getWasmEngine(specKey, forceColdStart);
  const { exports, tokenizer, modelSizeBytes, sessionStateBytes, coldStartMs } = engine;

  const promptIds = encodeBPE(prompt, tokenizer);
  if (promptIds.length === 0) promptIds.push(1);

  if (!streamMode) {
    const sessionPtr = exports.wasm_create_session(customSeed);
    try {
      const tStart = performance.now();
      let nextToken = promptIds[0];
      for (let i = 0; i < promptIds.length; i++) {
        nextToken = exports.wasm_step_and_sample(sessionPtr, promptIds[i], temperature);
      }
      const generatedIds: number[] = [];
      for (let step = 0; step < maxTokens; step++) {
        generatedIds.push(nextToken);
        if (nextToken === 2) break;
        nextToken = exports.wasm_step_and_sample(sessionPtr, nextToken, temperature);
      }
      const totalMs = Number((performance.now() - tStart).toFixed(3));
      const msPerToken = Number((totalMs / Math.max(1, generatedIds.length)).toFixed(4));
      const tokensPerSec = Number(((generatedIds.length * 1000) / Math.max(0.001, totalMs)).toFixed(1));
      const rssMb = Number((process.memoryUsage().rss / (1024 * 1024)).toFixed(2));
      const generatedText = decodeTokens(generatedIds, tokenizer);

      await db.insert(inferenceRuns).values({
        prompt,
        generatedText,
        modelSpec: specKey === "phase4" ? "Phase-4 Spec (d=192, L=6)" : "Deployed Spec (d=128, L=4)",
        tokensGenerated: generatedIds.length,
        coldStartMs,
        totalMs,
        msPerToken,
        tokensPerSec,
        sessionStateBytes,
        processRssMb: rssMb,
        temperature,
      });

      return NextResponse.json({
        architecture: "DeltaCore-LM (WebAssembly INT8)",
        model_spec: specKey,
        prompt,
        prompt_token_count: promptIds.length,
        generated_text: generatedText,
        full_text: prompt + generatedText,
        generated_token_ids: generatedIds,
        metrics: {
          cold_start_ms: coldStartMs,
          inference_total_ms: totalMs,
          ms_per_token: msPerToken,
          tokens_per_sec: tokensPerSec,
          model_size_bytes: modelSizeBytes,
          model_size_kib: Number((modelSizeBytes / 1024).toFixed(2)),
          session_state_bytes: sessionStateBytes,
          session_state_kib: Number((sessionStateBytes / 1024).toFixed(2)),
          wasm_heap_top_bytes: exports.wasm_get_heap_top(),
          process_rss_mb: rssMb,
        },
      });
    } finally {
      exports.wasm_free_session(sessionPtr);
    }
  }

  // Server-Sent Events (SSE) Web Stream
  const encoder = new TextEncoder();
  const readable = new ReadableStream({
    async start(controller) {
      const sessionPtr = exports.wasm_create_session(customSeed);
      try {
        const tStart = performance.now();
        let nextToken = promptIds[0];
        for (let i = 0; i < promptIds.length; i++) {
          nextToken = exports.wasm_step_and_sample(sessionPtr, promptIds[i], temperature);
        }

        const generatedIds: number[] = [];
        for (let step = 0; step < maxTokens; step++) {
          generatedIds.push(nextToken);
          const piece = decodeToken(nextToken, tokenizer);
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({ step, token_id: nextToken, text: piece })}\n\n`
            )
          );
          if (nextToken === 2) break;
          nextToken = exports.wasm_step_and_sample(sessionPtr, nextToken, temperature);
        }

        const totalMs = Number((performance.now() - tStart).toFixed(3));
        const msPerToken = Number((totalMs / Math.max(1, generatedIds.length)).toFixed(4));
        const tokensPerSec = Number(((generatedIds.length * 1000) / Math.max(0.001, totalMs)).toFixed(1));
        const rssMb = Number((process.memoryUsage().rss / (1024 * 1024)).toFixed(2));
        const generatedText = decodeTokens(generatedIds, tokenizer);

        await db.insert(inferenceRuns).values({
          prompt,
          generatedText,
          modelSpec: specKey === "phase4" ? "Phase-4 Spec (d=192, L=6)" : "Deployed Spec (d=128, L=4)",
          tokensGenerated: generatedIds.length,
          coldStartMs,
          totalMs,
          msPerToken,
          tokensPerSec,
          sessionStateBytes,
          processRssMb: rssMb,
          temperature,
        });

        const metricsPayload = {
          architecture: "DeltaCore-LM (WebAssembly INT8)",
          model_spec: specKey,
          cold_start_ms: coldStartMs,
          inference_total_ms: totalMs,
          ms_per_token: msPerToken,
          tokens_per_sec: tokensPerSec,
          model_size_bytes: modelSizeBytes,
          model_size_kib: Number((modelSizeBytes / 1024).toFixed(2)),
          session_state_bytes: sessionStateBytes,
          session_state_kib: Number((sessionStateBytes / 1024).toFixed(2)),
          wasm_heap_top_bytes: exports.wasm_get_heap_top(),
          process_rss_mb: rssMb,
        };
        controller.enqueue(
          encoder.encode(`event: metrics\ndata: ${JSON.stringify(metricsPayload)}\n\n`)
        );
        controller.close();
      } catch (err) {
        controller.error(err);
      } finally {
        exports.wasm_free_session(sessionPtr);
      }
    },
  });

  return new Response(readable, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Architecture": "DeltaCore-LM (Dual-Timescale Low-Rank Delta-Core)",
      "X-Model-Bytes": String(modelSizeBytes),
      "X-Session-State-Bytes": String(sessionStateBytes),
    },
  });
}
