import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  // Images use unoptimized mode for static export
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
