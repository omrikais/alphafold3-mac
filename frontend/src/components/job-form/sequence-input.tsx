"use client";

import { Textarea } from "@/components/ui/textarea";
import type { EntityType } from "@/lib/types";
import { PROTEIN_CHARS, DNA_CHARS, RNA_CHARS } from "@/lib/constants";

interface Props {
  value: string;
  onChange: (value: string) => void;
  entityType: EntityType;
}

function getPlaceholder(type: EntityType): string {
  switch (type) {
    case "proteinChain":
      return "Paste amino acid sequence (e.g., MKTAYIAKQRQISFVK...)";
    case "rnaSequence":
      return "Paste RNA sequence (e.g., AUGCCCGAA...)";
    case "dnaSequence":
      return "Paste DNA sequence (e.g., ATGCCCGAA...)";
    default:
      return "";
  }
}

function getValidChars(type: EntityType): Set<string> | null {
  switch (type) {
    case "proteinChain":
      return PROTEIN_CHARS;
    case "rnaSequence":
      return RNA_CHARS;
    case "dnaSequence":
      return DNA_CHARS;
    default:
      return null;
  }
}

function cleanSequence(text: string): string {
  // Detect FASTA: strip header lines and whitespace
  const lines = text.split("\n");
  const seqLines = lines
    .filter((l) => !l.startsWith(">"))
    .map((l) => l.replace(/\s/g, "").toUpperCase());
  return seqLines.join("");
}

export function SequenceInput({ value, onChange, entityType }: Props) {
  const validChars = getValidChars(entityType);
  const clean = value.toUpperCase().replace(/\s/g, "");
  const invalidCount = validChars
    ? [...clean].filter((ch) => !validChars.has(ch)).length
    : 0;

  return (
    <div className="space-y-1.5">
      <Textarea
        value={value}
        onChange={(e) => onChange(cleanSequence(e.target.value))}
        placeholder={getPlaceholder(entityType)}
        wrap="soft"
        className="min-h-[96px] resize-y overflow-x-hidden whitespace-pre-wrap break-all font-mono text-xs leading-5"
        spellCheck={false}
      />
      <div className="flex items-center gap-3 text-xs text-muted-foreground">
        <span>{clean.length} residues</span>
        {invalidCount > 0 && (
          <span className="text-destructive">
            {invalidCount} invalid character{invalidCount > 1 ? "s" : ""}
          </span>
        )}
      </div>
    </div>
  );
}
