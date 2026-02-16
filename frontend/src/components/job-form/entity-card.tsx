"use client";

import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { GripVertical, Trash2 } from "lucide-react";
import type { EntityType, FormEntity } from "@/lib/types";
import { EntityTypeSelector } from "./entity-type-selector";
import { SequenceInput } from "./sequence-input";
import { LigandInput } from "./ligand-input";
import { IonSelector } from "./ion-selector";
import { PTMWidget } from "./ptm-widget";

interface Props {
  entity: FormEntity;
  index: number;
  onChange: (entity: FormEntity) => void;
  onRemove: () => void;
}

export function EntityCard({ entity, index, onChange, onRemove }: Props) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: entity.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  const update = (partial: Partial<FormEntity>) =>
    onChange({ ...entity, ...partial });

  const handleTypeChange = (type: EntityType) => {
    // Reset type-specific fields when switching
    update({
      type,
      sequence: "",
      ccdCode: "",
      smiles: "",
      modifications: [],
    });
  };

  const needsSequence =
    entity.type === "proteinChain" ||
    entity.type === "rnaSequence" ||
    entity.type === "dnaSequence";

  return (
    <Card
      ref={setNodeRef}
      style={style}
      className="border-border/80 bg-background/65 p-3.5 transition-colors"
    >
      <div className="flex items-start gap-2.5">
        {/* Drag handle */}
        <button
          className="mt-1 rounded-md p-1 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
          {...attributes}
          {...listeners}
          aria-label="Reorder entity"
        >
          <GripVertical className="h-4 w-4" />
        </button>

        <div className="flex-1 space-y-3.5">
          {/* Header row: type selector + copies + remove */}
          <div className="flex items-center gap-2">
            <span className="w-5 text-xs font-semibold text-muted-foreground">
              {index + 1}
            </span>
            <EntityTypeSelector
              value={entity.type}
              onChange={handleTypeChange}
            />
            <div className="ml-auto flex items-center gap-1.5">
              <Label className="text-xs font-medium text-muted-foreground">
                Copies
              </Label>
              <Input
                type="number"
                min={1}
                max={10}
                value={entity.copies}
                onChange={(e) =>
                  update({ copies: Math.max(1, parseInt(e.target.value) || 1) })
                }
                className="h-8 w-16 text-center text-xs tabular-nums"
              />
              <Button
                variant="ghost"
                size="icon-sm"
                className="h-8 w-8 rounded-md text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                onClick={onRemove}
                aria-label="Remove entity"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>

          {/* Type-specific input */}
          {needsSequence && (
            <SequenceInput
              value={entity.sequence}
              onChange={(seq) => update({ sequence: seq })}
              entityType={entity.type}
            />
          )}

          {entity.type === "proteinChain" && entity.sequence.length > 0 && (
            <PTMWidget
              sequence={entity.sequence}
              modifications={entity.modifications || []}
              onModificationsChange={(mods) => update({ modifications: mods })}
            />
          )}

          {entity.type === "ligand" && (
            <LigandInput
              ccdCode={entity.ccdCode || ""}
              smiles={entity.smiles || ""}
              onCcdChange={(v) => update({ ccdCode: v })}
              onSmilesChange={(v) => update({ smiles: v })}
            />
          )}

          {entity.type === "ion" && (
            <IonSelector
              value={entity.ccdCode || ""}
              onChange={(v) => update({ ccdCode: v })}
            />
          )}
        </div>
      </div>
    </Card>
  );
}
