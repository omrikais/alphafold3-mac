"use client";

import { useCallback, useId, useRef } from "react";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import {
  SortableContext,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
  arrayMove,
} from "@dnd-kit/sortable";
import { Button } from "@/components/ui/button";
import { Plus } from "lucide-react";
import type { FormEntity } from "@/lib/types";
import { EntityCard } from "./entity-card";

interface Props {
  entities: FormEntity[];
  onChange: (entities: FormEntity[]) => void;
}

export function EntityBuilder({ entities, onChange }: Props) {
  const idPrefix = useId();
  const counter = useRef(0);
  const nextEntityId = () => `${idPrefix}-eb-${counter.current++}`;

  const createEntity = (): FormEntity => ({
    id: nextEntityId(),
    type: "proteinChain",
    sequence: "",
    copies: 1,
  });
  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  );

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const { active, over } = event;
      if (over && active.id !== over.id) {
        const oldIndex = entities.findIndex((e) => e.id === active.id);
        const newIndex = entities.findIndex((e) => e.id === over.id);
        onChange(arrayMove(entities, oldIndex, newIndex));
      }
    },
    [entities, onChange],
  );

  const addEntity = () => onChange([...entities, createEntity()]);

  const updateEntity = (index: number, entity: FormEntity) => {
    const next = [...entities];
    next[index] = entity;
    onChange(next);
  };

  const removeEntity = (index: number) => {
    if (entities.length <= 1) return; // keep at least one
    onChange(entities.filter((_, i) => i !== index));
  };

  return (
    <section className="space-y-3">
      <div className="space-y-1">
        <h3 className="text-sm font-semibold tracking-tight">Entities</h3>
        <p className="text-xs text-muted-foreground">
          Add proteins, nucleic acids, ligands, and ions in the target complex.
        </p>
      </div>
      <DndContext
        id={idPrefix}
        sensors={sensors}
        collisionDetection={closestCenter}
        onDragEnd={handleDragEnd}
      >
        <SortableContext
          items={entities.map((e) => e.id)}
          strategy={verticalListSortingStrategy}
        >
          {entities.map((entity, i) => (
            <EntityCard
              key={entity.id}
              entity={entity}
              index={i}
              onChange={(e) => updateEntity(i, e)}
              onRemove={() => removeEntity(i)}
            />
          ))}
        </SortableContext>
      </DndContext>

      <Button
        variant="outline"
        size="sm"
        className="h-9 w-full gap-1.5 rounded-md text-sm font-medium"
        onClick={addEntity}
      >
        <Plus className="h-3.5 w-3.5" />
        Add entity
      </Button>
    </section>
  );
}
