"use client";

import { memo, useState } from "react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Check, Trash2 } from "lucide-react";
import type { Modification, PtmType } from "@/lib/types";

interface ResidueCellProps {
  char: string;
  position: number;
  modification?: Modification;
  applicablePtms: PtmType[];
  onAdd: (ptmType: string) => void;
  onRemove: () => void;
  onChange: (ptmType: string) => void;
}

export const ResidueCell = memo(function ResidueCell({
  char,
  position,
  modification,
  applicablePtms,
  onAdd,
  onRemove,
  onChange,
}: ResidueCellProps) {
  const [customCode, setCustomCode] = useState("");
  const [showCustom, setShowCustom] = useState(false);
  const isModified = !!modification;

  return (
    <DropdownMenu onOpenChange={(open) => { if (!open) setShowCustom(false); }}>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          title={`${char}${position}${isModified ? ` (${modification.type})` : ""}`}
          className={`flex h-[22px] w-[18px] items-center justify-center rounded-[3px] font-mono text-xs transition-colors ${
            isModified
              ? "bg-primary/15 font-semibold ring-1 ring-primary/30"
              : "hover:bg-secondary"
          }`}
        >
          {char}
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="min-w-[180px]">
        {isModified ? (
          <>
            <DropdownMenuItem disabled className="text-xs opacity-70">
              <Check className="size-3.5" />
              {modification.type}
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            {applicablePtms
              .filter((ptm) => ptm.code !== modification.type)
              .map((ptm) => (
                <DropdownMenuItem
                  key={ptm.code}
                  className="text-xs"
                  onSelect={() => onChange(ptm.code)}
                >
                  {ptm.name}
                </DropdownMenuItem>
              ))}
            <DropdownMenuSeparator />
            <DropdownMenuItem
              variant="destructive"
              className="text-xs"
              onSelect={onRemove}
            >
              <Trash2 className="size-3.5" />
              Remove
            </DropdownMenuItem>
          </>
        ) : (
          <>
            {applicablePtms.map((ptm) => (
              <DropdownMenuItem
                key={ptm.code}
                className="text-xs"
                onSelect={() => onAdd(ptm.code)}
              >
                {ptm.name}
              </DropdownMenuItem>
            ))}
            {applicablePtms.length > 0 && <DropdownMenuSeparator />}
            {showCustom ? (
              <div className="flex items-center gap-1 px-2 py-1">
                <Input
                  autoFocus
                  placeholder="CCD code"
                  value={customCode}
                  onChange={(e) =>
                    setCustomCode(e.target.value.toUpperCase().slice(0, 5))
                  }
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && customCode.length > 0) {
                      const code = customCode.startsWith("CCD_")
                        ? customCode
                        : `CCD_${customCode}`;
                      onAdd(code);
                      setCustomCode("");
                      setShowCustom(false);
                    }
                  }}
                  className="h-6 w-20 px-1.5 font-mono text-xs"
                />
                <button
                  type="button"
                  className="rounded px-1.5 py-0.5 text-[10px] font-medium text-primary hover:bg-primary/10"
                  onClick={() => {
                    if (customCode.length > 0) {
                      const code = customCode.startsWith("CCD_")
                        ? customCode
                        : `CCD_${customCode}`;
                      onAdd(code);
                      setCustomCode("");
                      setShowCustom(false);
                    }
                  }}
                >
                  Add
                </button>
              </div>
            ) : (
              <DropdownMenuItem
                className="text-xs"
                onSelect={(e) => {
                  e.preventDefault();
                  setShowCustom(true);
                }}
              >
                Custom CCD code...
              </DropdownMenuItem>
            )}
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
},
(prev, next) =>
  prev.char === next.char &&
  prev.position === next.position &&
  prev.modification?.type === next.modification?.type,
);
