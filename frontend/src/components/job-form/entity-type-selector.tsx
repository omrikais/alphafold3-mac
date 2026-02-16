"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ENTITY_TYPE_OPTIONS } from "@/lib/constants";
import type { EntityType } from "@/lib/types";

interface Props {
  value: EntityType;
  onChange: (value: EntityType) => void;
}

export function EntityTypeSelector({ value, onChange }: Props) {
  return (
    <Select value={value} onValueChange={(v) => onChange(v as EntityType)}>
      <SelectTrigger className="h-8 w-[152px] text-sm font-medium">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {ENTITY_TYPE_OPTIONS.map((opt) => (
          <SelectItem key={opt.value} value={opt.value} className="text-sm">
            {opt.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
