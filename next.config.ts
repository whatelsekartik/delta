import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  outputFileTracingIncludes: {
    "/api/**/*": ["./artifacts/**/*"],
  },
};

export default nextConfig;
