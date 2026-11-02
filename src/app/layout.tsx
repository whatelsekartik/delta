import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "DeltaCore-LM — 7-Phase Engineering Audit & WebAssembly CPU Inference Workbench",
  description:
    "Rigorous mathematical, training, architectural, INT8 quantization, WebAssembly, and Vercel serverless verification workbench for DeltaCore-LM.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#0B0F17] text-slate-100 antialiased selection:bg-cyan-500/30 selection:text-cyan-200">
        {children}
      </body>
    </html>
  );
}
