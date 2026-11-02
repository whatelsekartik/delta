import fs from "fs";
import path from "path";
import { desc } from "drizzle-orm";
import { NextResponse } from "next/server";
import { db } from "@/db";
import { concurrencyAuditRuns, inferenceRuns, parityAuditRuns } from "@/db/schema";
import { ensureAuditTables, getWasmEngine } from "@/lib/deltacore-wasm";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    await ensureAuditTables();
    const reportPath = path.join(process.cwd(), "artifacts", "audit_master_report.json");
    const masterReport = JSON.parse(fs.readFileSync(reportPath, "utf8"));

    const engine = await getWasmEngine("deployed", false);
    const recentInference = await db
      .select()
      .from(inferenceRuns)
      .orderBy(desc(inferenceRuns.id))
      .limit(15);

    const recentConcurrency = await db
      .select()
      .from(concurrencyAuditRuns)
      .orderBy(desc(concurrencyAuditRuns.id))
      .limit(10);

    const recentParity = await db
      .select()
      .from(parityAuditRuns)
      .orderBy(desc(parityAuditRuns.id))
      .limit(10);

    return NextResponse.json({
      status: "ok",
      live_runtime: {
        wasm_binary_bytes: engine.wasmBinaryBytes,
        model_size_bytes: engine.modelSizeBytes,
        session_state_bytes: engine.sessionStateBytes,
        wasm_heap_top_bytes: engine.exports.wasm_get_heap_top(),
        wasm_linear_memory_mib: Number(
          (engine.exports.memory.buffer.byteLength / (1024 * 1024)).toFixed(2)
        ),
        process_rss_mb: Number((process.memoryUsage().rss / (1024 * 1024)).toFixed(2)),
      },
      master_report: masterReport,
      db_telemetry: {
        recent_inference: recentInference,
        recent_concurrency: recentConcurrency,
        recent_parity: recentParity,
      },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Failed to load audit report";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
