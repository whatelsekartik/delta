import fs from "fs";
import path from "path";
import { performance } from "perf_hooks";
import { pool } from "@/db";

export interface WasmExports {
  memory: WebAssembly.Memory;
  wasm_alloc: (size: number) => number;
  wasm_init_model: (ptr: number, len: number) => number;
  wasm_create_session: (seed: bigint) => number;
  wasm_free_session: (sessionPtr: number) => void;
  wasm_forward_logits: (sessionPtr: number, tokenId: number, logitsOutPtr: number) => number;
  wasm_step_and_sample: (sessionPtr: number, inputToken: number, temperature: number) => number;
  wasm_get_session_state_bytes: () => number;
  wasm_get_heap_top: () => number;
}

export interface TokenizerData {
  vocab_size: number;
  trained_on_split: string;
  train_corpus_sha256: string;
  train_corpus_bytes: number;
  merges: [number, number, number][];
  vocab: Record<string, number[]>;
}

export interface LoadedEngine {
  specKey: "deployed" | "phase4";
  exports: WasmExports;
  tokenizer: TokenizerData;
  modelSizeBytes: number;
  wasmBinaryBytes: number;
  sessionStateBytes: number;
  coldStartMs: number;
  logitsScratchPtr: number;
}

const engineCache: Partial<Record<"deployed" | "phase4", LoadedEngine>> = {};
let tablesInitialized = false;

export async function ensureAuditTables(): Promise<void> {
  if (tablesInitialized) return;
  await pool.query(`
    CREATE TABLE IF NOT EXISTS inference_runs (
      id SERIAL PRIMARY KEY,
      prompt TEXT NOT NULL,
      generated_text TEXT NOT NULL,
      model_spec TEXT NOT NULL,
      tokens_generated INTEGER NOT NULL,
      cold_start_ms DOUBLE PRECISION NOT NULL,
      total_ms DOUBLE PRECISION NOT NULL,
      ms_per_token DOUBLE PRECISION NOT NULL,
      tokens_per_sec DOUBLE PRECISION NOT NULL,
      session_state_bytes INTEGER NOT NULL,
      process_rss_mb DOUBLE PRECISION NOT NULL,
      temperature DOUBLE PRECISION NOT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS concurrency_audit_runs (
      id SERIAL PRIMARY KEY,
      concurrent_sessions INTEGER NOT NULL,
      tokens_per_session INTEGER NOT NULL,
      total_duration_ms DOUBLE PRECISION NOT NULL,
      aggregate_tokens_per_sec DOUBLE PRECISION NOT NULL,
      bit_exact_isolation_verified BOOLEAN NOT NULL,
      heap_leak_bytes INTEGER NOT NULL,
      session_details_json TEXT NOT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS parity_audit_runs (
      id SERIAL PRIMARY KEY,
      prompt TEXT NOT NULL,
      fp32_vs_int8_cosine DOUBLE PRECISION NOT NULL,
      py_vs_wasm_max_err DOUBLE PRECISION NOT NULL,
      cpp_vs_wasm_max_err DOUBLE PRECISION NOT NULL,
      top1_match BOOLEAN NOT NULL,
      greedy_agreement_pct DOUBLE PRECISION NOT NULL,
      details_json TEXT NOT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT NOW()
    );
  `);
  tablesInitialized = true;
}

export async function getWasmEngine(
  specKey: "deployed" | "phase4" = "deployed",
  forceColdStart = false
): Promise<LoadedEngine> {
  if (!forceColdStart && engineCache[specKey]) {
    return { ...engineCache[specKey]!, coldStartMs: 0.0 };
  }

  const t0 = performance.now();
  const artifactsDir = path.join(process.cwd(), "artifacts");
  const dcoreFile = specKey === "phase4" ? "model_phase4_spec.dcore" : "model.dcore";

  const wasmBytes = fs.readFileSync(path.join(artifactsDir, "deltacore_runtime.wasm"));
  const modelBytes = fs.readFileSync(path.join(artifactsDir, dcoreFile));
  const tokenizerData: TokenizerData = JSON.parse(
    fs.readFileSync(path.join(artifactsDir, "tokenizer.json"), "utf8")
  );

  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const exports = instance.exports as unknown as WasmExports;

  const modelPtr = exports.wasm_alloc(modelBytes.length);
  const wasmMem = new Uint8Array(exports.memory.buffer, modelPtr, modelBytes.length);
  wasmMem.set(modelBytes);

  const status = exports.wasm_init_model(modelPtr, modelBytes.length);
  if (status !== 0) {
    throw new Error(`Failed to initialize DeltaCore-LM in WASM (status=${status})`);
  }

  const logitsScratchPtr = exports.wasm_alloc(tokenizerData.vocab_size * 4);
  const sessionStateBytes = exports.wasm_get_session_state_bytes();
  const coldStartMs = Number((performance.now() - t0).toFixed(3));

  const engine: LoadedEngine = {
    specKey,
    exports,
    tokenizer: tokenizerData,
    modelSizeBytes: modelBytes.length,
    wasmBinaryBytes: wasmBytes.length,
    sessionStateBytes,
    coldStartMs,
    logitsScratchPtr,
  };

  engineCache[specKey] = engine;
  return engine;
}

export function encodeBPE(text: string, tokenizer: TokenizerData): number[] {
  const NUM_SPECIAL = 4;
  const raw = Buffer.from(text, "utf8");
  let ids = Array.from(raw, (b) => b + NUM_SPECIAL);
  for (const [a, b, newId] of tokenizer.merges) {
    if (ids.length < 2) break;
    const next: number[] = [];
    let i = 0;
    while (i < ids.length) {
      if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) {
        next.push(newId);
        i += 2;
      } else {
        next.push(ids[i]);
        i += 1;
      }
    }
    ids = next;
  }
  return ids;
}

export function decodeToken(tokenId: number, tokenizer: TokenizerData): string {
  if (tokenId < 4) return "";
  const byteArr = tokenizer.vocab[String(tokenId)] || [];
  return Buffer.from(byteArr).toString("utf8");
}

export function decodeTokens(tokenIds: number[], tokenizer: TokenizerData): string {
  const bytes: number[] = [];
  for (const tid of tokenIds) {
    if (tid < 4) continue;
    const arr = tokenizer.vocab[String(tid)] || [];
    for (const b of arr) bytes.push(b);
  }
  return Buffer.from(bytes).toString("utf8");
}
