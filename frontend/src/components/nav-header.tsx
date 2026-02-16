"use client";

import { useCallback } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useSystemStatus } from "@/hooks/use-system-status";
import { ThemeToggle } from "./theme-toggle";
import { Badge } from "@/components/ui/badge";
import { Activity, Cpu } from "lucide-react";

export function NavHeader() {
  const { data: status } = useSystemStatus();
  const pathname = usePathname();
  const router = useRouter();

  const handleTitleClick = useCallback(() => {
    if (pathname === "/") {
      window.dispatchEvent(new CustomEvent("resetPrediction"));
    } else {
      router.push("/");
    }
  }, [pathname, router]);

  return (
    <header className="sticky top-0 z-50 border-b border-border/70 bg-background/80 backdrop-blur-xl supports-[backdrop-filter]:bg-background/62">
      <div className="mx-auto flex h-16 max-w-[1360px] items-center justify-between px-4 md:px-6">
        <button
          onClick={handleTitleClick}
          className="flex items-baseline gap-2 transition-opacity hover:opacity-75"
          aria-label="Go to home page"
        >
          <h1 className="text-[1.35rem] font-semibold tracking-tight text-foreground">
            AlphaFold 3
            {" "}
            <span className="ml-1.5 text-[0.8rem] font-medium tracking-normal text-muted-foreground">
              MLX
            </span>
          </h1>
          <span className="hidden text-xs text-muted-foreground/90 lg:inline">
            Local structure prediction
          </span>
        </button>

        <div className="flex items-center gap-2.5">
          {status && (
            <div className="hidden items-center gap-2 md:flex">
              <Badge
                variant={status.model_loaded ? "default" : "secondary"}
                className="h-7 gap-1.5 rounded-full px-2.5 text-[11px] font-medium"
              >
                <Activity className="h-3.5 w-3.5" aria-hidden />
                {status.model_loading
                  ? "Loading model..."
                  : status.model_loaded
                    ? "Model ready"
                    : "Model not loaded"}
              </Badge>
              <Badge
                variant="outline"
                className="h-7 gap-1.5 rounded-full border-border/80 bg-card px-2.5 text-[11px] font-medium"
              >
                <Cpu className="h-3.5 w-3.5" aria-hidden />
                {status.chip_family} {status.memory_gb}GB
              </Badge>
            </div>
          )}
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
