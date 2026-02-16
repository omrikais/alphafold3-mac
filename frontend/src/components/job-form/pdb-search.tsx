"use client";

import { useCallback, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Loader2, Search } from "lucide-react";
import { fetchPdb } from "@/lib/api";
import type { StructureParseResult } from "@/lib/types";

interface PdbSearchProps {
  onResult: (result: StructureParseResult) => void;
  disabled?: boolean;
}

export function PdbSearch({ onResult, disabled }: PdbSearchProps) {
  const [pdbId, setPdbId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isValid = /^[a-zA-Z0-9]{4}$/.test(pdbId);

  const handleFetch = useCallback(async () => {
    if (!isValid || loading) return;
    setError(null);
    setLoading(true);
    try {
      const result = await fetchPdb(pdbId);
      onResult(result);
      setPdbId("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [pdbId, isValid, loading, onResult]);

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-1.5">
        <Input
          value={pdbId}
          onChange={(e) => {
            setPdbId(e.target.value.slice(0, 4));
            setError(null);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && isValid && !loading) handleFetch();
          }}
          placeholder="PDB ID"
          maxLength={4}
          className="h-8 w-[5.5rem] rounded-md font-mono text-xs uppercase"
          disabled={disabled || loading}
        />
        <Button
          variant="outline"
          size="icon-sm"
          className="h-8 w-8 shrink-0 rounded-md"
          onClick={handleFetch}
          disabled={!isValid || loading || disabled}
          aria-label="Fetch PDB structure"
        >
          {loading ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Search className="h-3.5 w-3.5" />
          )}
        </Button>
      </div>
      {error && (
        <p className="max-w-[12rem] text-[0.7rem] leading-tight text-destructive">
          {error}
        </p>
      )}
    </div>
  );
}
