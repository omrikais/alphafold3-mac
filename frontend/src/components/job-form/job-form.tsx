"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Switch } from "@/components/ui/switch";
import { AlertCircle, Eye, Loader2, Upload, Shuffle } from "lucide-react";
import type { FormEntity, JobSubmission, StructureParseResult } from "@/lib/types";
import type { RestraintConfig, GuidanceConfig } from "@/lib/restraints";
import { parseInputJson } from "@/lib/parse-input-json";
import { getJob, parseStructureFile } from "@/lib/api";
import { useSubmitJob } from "@/hooks/use-jobs";
import { useSystemStatus } from "@/hooks/use-system-status";
import { EntityBuilder } from "./entity-builder";
import { PreviewModal } from "./preview-modal";
import { RestraintEditor } from "@/components/restraint-editor";
import { PdbSearch } from "./pdb-search";

function generateJobName(): string {
  const adjectives = ["alpha", "beta", "gamma", "delta", "omega", "sigma"];
  const nouns = ["fold", "helix", "sheet", "loop", "turn", "bridge"];
  const adj = adjectives[Math.floor(Math.random() * adjectives.length)];
  const noun = nouns[Math.floor(Math.random() * nouns.length)];
  const num = Math.floor(Math.random() * 100);
  return `${adj}-${noun}-${num}`;
}

function entitiesToSequences(entities: FormEntity[]): Record<string, unknown>[] {
  return entities.map((entity) => {
    switch (entity.type) {
      case "proteinChain": {
        const pc: Record<string, unknown> = {
          sequence: entity.sequence,
          count: entity.copies,
        };
        if (entity.modifications?.length) {
          pc.modifications = entity.modifications.map((m) => ({
            ptmType: m.type,
            ptmPosition: m.position,
          }));
        }
        return { proteinChain: pc };
      }
      case "rnaSequence":
        return {
          rnaSequence: {
            sequence: entity.sequence,
            count: entity.copies,
          },
        };
      case "dnaSequence":
        return {
          dnaSequence: {
            sequence: entity.sequence,
            count: entity.copies,
          },
        };
      case "ligand":
        return {
          ligand: {
            ...(entity.smiles
              ? { smiles: entity.smiles }
              : { ligand: `CCD_${entity.ccdCode}` }),
            count: entity.copies,
          },
        };
      case "ion":
        return {
          ion: {
            ion: entity.ccdCode || "MG",
            count: entity.copies,
          },
        };
      default:
        return {};
    }
  });
}

export function JobForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const submitMutation = useSubmitJob();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const idPrefix = useId();
  const entityCounter = useRef(1);
  const nextEntityId = useCallback(() => {
    const id = `${idPrefix}-entity-${entityCounter.current}`;
    entityCounter.current += 1;
    return id;
  }, [idPrefix]);

  const [jobName, setJobName] = useState(() => generateJobName());
  const [seed, setSeed] = useState(42);
  const [autoSeed, setAutoSeed] = useState(false);
  const [entities, setEntities] = useState<FormEntity[]>(() => [
    {
      id: `${idPrefix}-entity-0`,
      type: "proteinChain",
      sequence: "",
      copies: 1,
    },
  ]);
  const { data: status } = useSystemStatus();
  const [runPipeline, setRunPipeline] = useState<boolean | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [restraints, setRestraints] = useState<RestraintConfig>({});
  const [guidance, setGuidance] = useState<GuidanceConfig>({});
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isParsingFile, setIsParsingFile] = useState(false);
  const [importWarnings, setImportWarnings] = useState<string[]>([]);
  const initialJobNameRef = useRef(jobName);

  // Sync initial value from server default once status loads
  useEffect(() => {
    if (runPipeline === null && status != null) {
      setRunPipeline(status.run_data_pipeline);
    }
  }, [runPipeline, status]);

  // Reset form when header title is clicked on the home page
  useEffect(() => {
    const handleReset = () => {
      const newName = generateJobName();
      setJobName(newName);
      initialJobNameRef.current = newName;
      setSeed(42);
      setAutoSeed(false);
      entityCounter.current = 1;
      setEntities([
        {
          id: `${idPrefix}-entity-0`,
          type: "proteinChain",
          sequence: "",
          copies: 1,
        },
      ]);
      setRunPipeline(status?.run_data_pipeline ?? null);
      setShowPreview(false);
      setRestraints({});
      setGuidance({});
      setLoadError(null);
      setImportWarnings([]);
    };
    window.addEventListener("resetPrediction", handleReset);
    return () => window.removeEventListener("resetPrediction", handleReset);
  }, [idPrefix, status]);

  // ── Load job from URL param (?loadJob=<id>) ──────────────────────────────
  const loadJobId = searchParams.get("loadJob");
  const requestTokenRef = useRef(0);
  const loadingJobRef = useRef<string | null>(null);

  useEffect(() => {
    if (!loadJobId) {
      loadingJobRef.current = null;
      return;
    }
    // Strict Mode double-invocation guard
    if (loadJobId === loadingJobRef.current) return;
    loadingJobRef.current = loadJobId;

    const token = ++requestTokenRef.current;
    setLoadError(null);

    const clearParam = () => {
      const current = new URLSearchParams(window.location.search);
      current.delete("loadJob");
      const qs = current.toString();
      router.replace(qs ? `/?${qs}` : "/", { scroll: false });
    };

    const hasDraftChanges =
      jobName.trim() !== initialJobNameRef.current.trim() ||
      seed !== 42 ||
      autoSeed ||
      entities.some((entity) => {
        if (entity.type === "proteinChain") {
          return (
            entity.sequence.trim().length > 0 ||
            entity.copies !== 1 ||
            (entity.modifications?.length ?? 0) > 0
          );
        }
        if (entity.type === "rnaSequence" || entity.type === "dnaSequence") {
          return entity.sequence.trim().length > 0 || entity.copies !== 1;
        }
        if (entity.type === "ligand") {
          return Boolean(entity.ccdCode || entity.smiles) || entity.copies !== 1;
        }
        if (entity.type === "ion") {
          return Boolean(entity.ccdCode) || entity.copies !== 1;
        }
        return false;
      }) ||
      (runPipeline != null &&
        status != null &&
        runPipeline !== status.run_data_pipeline) ||
      (restraints.distance?.length ?? 0) +
        (restraints.contact?.length ?? 0) +
        (restraints.repulsive?.length ?? 0) >
        0 ||
      Object.keys(guidance).length > 0;

    if (hasDraftChanges) {
      const shouldReplace = window.confirm(
        "Loading this job will replace your current draft. Continue?",
      );
      if (!shouldReplace) {
        clearParam();
        return;
      }
    }

    getJob(loadJobId)
      .then((jobDetail) => {
        if (token !== requestTokenRef.current) return;

        const parsed = parseInputJson(jobDetail.input_json, nextEntityId);
        if (!parsed) {
          setLoadError("Could not parse job input");
          clearParam();
          return;
        }

        setJobName(parsed.jobName ? `${parsed.jobName} (copy)` : generateJobName());
        setAutoSeed(false);
        if (parsed.seed != null) setSeed(parsed.seed);
        setEntities(parsed.entities);
        setRunPipeline(jobDetail.run_data_pipeline);
        setRestraints(parsed.restraints);
        setGuidance(parsed.guidance);
        clearParam();
      })
      .catch((err) => {
        if (token !== requestTokenRef.current) return;
        setLoadError(`Failed to load job: ${err instanceof Error ? err.message : String(err)}`);
        clearParam();
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadJobId, nextEntityId, router]);

  const hasRestraints =
    (restraints.distance?.length ?? 0) +
    (restraints.contact?.length ?? 0) +
    (restraints.repulsive?.length ?? 0) > 0;

  const handleSubmit = useCallback((useCache: boolean) => {
    const actualSeed = autoSeed ? Math.floor(Math.random() * 2 ** 31) : seed;
    const submission: JobSubmission = {
      name: jobName,
      modelSeeds: [actualSeed],
      sequences: entitiesToSequences(entities),
      dialect: "alphafoldserver",
      version: 1,
      ...(runPipeline != null ? { runDataPipeline: runPipeline } : {}),
      useCache,
      ...(hasRestraints ? { restraints: restraints as unknown as Record<string, unknown> } : {}),
      ...(hasRestraints ? { guidance: guidance as unknown as Record<string, unknown> } : {}),
    };
    submitMutation.mutate(submission, {
      onSuccess: (result) => {
        setShowPreview(false);
        router.push(`/job?id=${result.id}`);
      },
    });
  }, [jobName, seed, autoSeed, entities, runPipeline, restraints, guidance, hasRestraints, submitMutation, router]);

  const applyStructureResult = useCallback(
    (result: StructureParseResult) => {
      const parsed = parseInputJson(result, nextEntityId);
      if (!parsed) return;
      if (parsed.jobName) setJobName(parsed.jobName);
      if (parsed.seed != null) setSeed(parsed.seed);
      setEntities(parsed.entities);
      setRestraints(parsed.restraints);
      setGuidance(parsed.guidance);
      setImportWarnings(result.warnings ?? []);
      setLoadError(null);
    },
    [nextEntityId],
  );

  const handlePdbResult = useCallback(
    (result: StructureParseResult) => {
      applyStructureResult(result);
    },
    [applyStructureResult],
  );

  const handleUploadFile = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      const ext = file.name.toLowerCase().split(".").pop() ?? "";

      if (ext === "json") {
        // Existing JSON path
        const reader = new FileReader();
        reader.onload = (ev) => {
          try {
            const data = JSON.parse(ev.target?.result as string);
            const parsed = parseInputJson(data, nextEntityId);
            if (!parsed) return;
            if (parsed.jobName) setJobName(parsed.jobName);
            if (parsed.seed != null) setSeed(parsed.seed);
            setEntities(parsed.entities);
            if (parsed.runDataPipeline !== undefined) {
              setRunPipeline(parsed.runDataPipeline);
            }
            setRestraints(parsed.restraints);
            setGuidance(parsed.guidance);
            setImportWarnings([]);
          } catch {
            // ignore parse errors
          }
        };
        reader.readAsText(file);
      } else if (["pdb", "cif", "mmcif"].includes(ext)) {
        // Structure file → backend parse
        setIsParsingFile(true);
        setImportWarnings([]);
        try {
          const result = await parseStructureFile(file);
          applyStructureResult(result);
        } catch (err) {
          setLoadError(
            `Failed to parse structure: ${err instanceof Error ? err.message : String(err)}`,
          );
        } finally {
          setIsParsingFile(false);
        }
      }

      // Reset so the same file can be re-uploaded
      e.target.value = "";
    },
    [nextEntityId, applyStructureResult],
  );

  const canSubmit =
    entities.length > 0 &&
    entities.every((e) => {
      if (
        e.type === "proteinChain" ||
        e.type === "rnaSequence" ||
        e.type === "dnaSequence"
      ) {
        return e.sequence.length > 0;
      }
      if (e.type === "ligand") return !!(e.ccdCode || e.smiles);
      if (e.type === "ion") return !!e.ccdCode;
      return false;
    });

  return (
    <>
      {loadError && (
        <div className="flex items-center gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2.5">
          <AlertCircle className="h-4 w-4 shrink-0 text-destructive" />
          <p className="text-sm text-destructive">{loadError}</p>
        </div>
      )}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Prediction setup</CardTitle>
              <CardDescription>
                Prepare job metadata, seed strategy, and molecular entities.
              </CardDescription>
            </div>
            <div className="flex items-center gap-1.5">
              <PdbSearch onResult={handlePdbResult} disabled={isParsingFile} />
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="icon-sm"
                    className="h-8 w-8 rounded-md"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isParsingFile}
                    aria-label="Upload input file"
                  >
                    {isParsingFile ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <Upload className="h-3.5 w-3.5" />
                    )}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Upload input file</TooltipContent>
              </Tooltip>
              <input
                ref={fileInputRef}
                type="file"
                accept=".json,.pdb,.cif,.mmcif"
                className="hidden"
                onChange={handleUploadFile}
              />
            </div>
          </div>
        </CardHeader>
        {importWarnings.length > 0 && (
          <div className="mx-6 mb-1 flex flex-col gap-1 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2.5">
            {importWarnings.map((w, i) => (
              <p key={i} className="text-xs text-amber-700 dark:text-amber-400">
                {w}
              </p>
            ))}
          </div>
        )}
        <CardContent className="space-y-5">
          <section className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_250px]">
            <div className="space-y-1.5">
              <Label
                htmlFor="job-name"
                className="text-[0.78rem] font-semibold uppercase tracking-wide text-muted-foreground"
              >
                Job name
              </Label>
              <div className="flex items-center gap-2">
                <Input
                  id="job-name"
                  value={jobName}
                  onChange={(e) => setJobName(e.target.value)}
                  className="h-9"
                />
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      size="icon-sm"
                      className="h-9 w-9 shrink-0 rounded-md"
                      onClick={() => setJobName(generateJobName())}
                      aria-label="Generate random job name"
                    >
                      <Shuffle className="h-3.5 w-3.5" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Generate random name</TooltipContent>
                </Tooltip>
              </div>
            </div>

            <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-2">
              <div className="space-y-1.5">
                <Label
                  htmlFor="job-seed"
                  className="text-[0.78rem] font-semibold uppercase tracking-wide text-muted-foreground"
                >
                  Seed
                </Label>
                {autoSeed ? (
                  <div className="flex h-9 items-center rounded-md border border-border bg-secondary/70 px-3 text-sm text-muted-foreground">
                    Randomized at submit time
                  </div>
                ) : (
                  <Input
                    id="job-seed"
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
                    className="h-9 font-mono"
                  />
                )}
              </div>
              <Button
                variant="outline"
                size="sm"
                className="h-9 text-xs font-semibold uppercase tracking-wide"
                onClick={() => setAutoSeed(!autoSeed)}
              >
                {autoSeed ? "Manual seed" : "Auto seed"}
              </Button>
            </div>
          </section>

          <section className="af-panel bg-secondary/35 px-3.5 py-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-sm font-semibold">MSA & template search</p>
                <p className="text-xs text-muted-foreground">
                  Uses HMMER/databases for stronger predictions with longer runtime.
                </p>
              </div>
              <div className="flex items-center gap-2.5">
                <Switch checked={runPipeline ?? false} onCheckedChange={setRunPipeline} />
                <span className="text-xs font-medium text-muted-foreground">
                  {runPipeline ? "Enabled" : "Disabled"}
                </span>
              </div>
            </div>
            <Tooltip>
              <TooltipTrigger asChild>
                <p className="mt-2 cursor-help text-xs text-muted-foreground">
                  Hover for behavior details
                </p>
              </TooltipTrigger>
              <TooltipContent className="max-w-[260px]">
                Run HMMER to search for homologous sequences and templates.
                Improves accuracy but adds 5-10 minutes. Disable for quick local tests.
              </TooltipContent>
            </Tooltip>
          </section>

          <EntityBuilder entities={entities} onChange={setEntities} />

          <RestraintEditor
            restraints={restraints}
            guidance={guidance}
            entities={entities}
            onRestraintsChange={setRestraints}
            onGuidanceChange={setGuidance}
          />

          <div className="flex justify-end">
            <Button
              onClick={() => setShowPreview(true)}
              disabled={!canSubmit}
              className="h-10 gap-1.5 rounded-md px-4 text-sm font-semibold"
            >
              <Eye className="h-3.5 w-3.5" />
              Preview & submit
            </Button>
          </div>
        </CardContent>
      </Card>

      <PreviewModal
        open={showPreview}
        onOpenChange={setShowPreview}
        jobName={jobName}
        seed={autoSeed ? -1 : seed}
        runPipeline={runPipeline ?? false}
        entities={entities}
        onSubmit={handleSubmit}
        isSubmitting={submitMutation.isPending}
      />
    </>
  );
}
