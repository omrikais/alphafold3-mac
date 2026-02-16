"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { COMMON_IONS } from "@/lib/constants";

interface Props {
  value: string;
  onChange: (value: string) => void;
}

export function IonSelector({ value, onChange }: Props) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="h-9 text-sm">
        <SelectValue placeholder="Select ion type" />
      </SelectTrigger>
      <SelectContent>
        {COMMON_IONS.map((ion) => (
          <SelectItem key={ion.code} value={ion.code} className="text-sm">
            {ion.name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
