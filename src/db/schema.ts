import {
  boolean,
  doublePrecision,
  integer,
  pgTable,
  serial,
  text,
  timestamp,
} from "drizzle-orm/pg-core";

export const inferenceRuns = pgTable("inference_runs", {
  id: serial("id").primaryKey(),
  prompt: text("prompt").notNull(),
  generatedText: text("generated_text").notNull(),
  modelSpec: text("model_spec").notNull(),
  tokensGenerated: integer("tokens_generated").notNull(),
  coldStartMs: doublePrecision("cold_start_ms").notNull(),
  totalMs: doublePrecision("total_ms").notNull(),
  msPerToken: doublePrecision("ms_per_token").notNull(),
  tokensPerSec: doublePrecision("tokens_per_sec").notNull(),
  sessionStateBytes: integer("session_state_bytes").notNull(),
  processRssMb: doublePrecision("process_rss_mb").notNull(),
  temperature: doublePrecision("temperature").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const concurrencyAuditRuns = pgTable("concurrency_audit_runs", {
  id: serial("id").primaryKey(),
  concurrentSessions: integer("concurrent_sessions").notNull(),
  tokensPerSession: integer("tokens_per_session").notNull(),
  totalDurationMs: doublePrecision("total_duration_ms").notNull(),
  aggregateTokensPerSec: doublePrecision("aggregate_tokens_per_sec").notNull(),
  bitExactIsolationVerified: boolean("bit_exact_isolation_verified").notNull(),
  heapLeakBytes: integer("heap_leak_bytes").notNull(),
  sessionDetailsJson: text("session_details_json").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const parityAuditRuns = pgTable("parity_audit_runs", {
  id: serial("id").primaryKey(),
  prompt: text("prompt").notNull(),
  fp32VsInt8Cosine: doublePrecision("fp32_vs_int8_cosine").notNull(),
  pyVsWasmMaxErr: doublePrecision("py_vs_wasm_max_err").notNull(),
  cppVsWasmMaxErr: doublePrecision("cpp_vs_wasm_max_err").notNull(),
  top1Match: boolean("top1_match").notNull(),
  greedyAgreementPct: doublePrecision("greedy_agreement_pct").notNull(),
  detailsJson: text("details_json").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});
