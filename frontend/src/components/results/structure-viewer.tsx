"use client";

import { useEffect, useRef, useCallback } from "react";
import type { RestraintSatisfaction } from "@/lib/restraints";

// Mol* CSS is imported in structure-viewer-dynamic.tsx (the static wrapper)
// to ensure it's included in the page's CSS bundle for static exports.

interface Props {
  cifUrl: string;
  satisfaction?: RestraintSatisfaction | null;
}

/** Tag used to identify restraint measurement nodes in the Mol* state tree. */
const RESTRAINT_TAG = "restraint-viz";

/**
 * Mol* 3D molecular viewer wrapper with restraint visualization.
 *
 * Uses createPluginUI + renderReact18 (the recommended embedding API) with
 * hierarchy.applyPreset for reliable structure loading.
 *
 * When `satisfaction` is provided, draws color-coded distance lines between
 * restrained atom pairs (green = satisfied, red = unsatisfied).
 *
 * Loaded via next/dynamic (ssr: false) since Mol* depends on WebGL/DOM.
 */
export function StructureViewerInner({ cifUrl, satisfaction }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const pluginRef = useRef<any>(null);
  const satisfactionRef = useRef(satisfaction);
  satisfactionRef.current = satisfaction;

  /**
   * Remove all restraint measurement lines from the viewer state tree.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const clearRestraints = useCallback(async (plugin: any) => {
    try {
      const toDelete: string[] = [];
      plugin.state.data.cells.forEach((cell: { transform?: { ref?: string; tags?: string[] } }) => {
        if (cell?.transform?.tags?.includes(RESTRAINT_TAG)) {
          toDelete.push(cell.transform.ref!);
        }
      });
      if (toDelete.length > 0) {
        const update = plugin.state.data.build();
        for (const ref of toDelete) {
          update.delete(ref);
        }
        await update.commit();
      }
    } catch (err) {
      console.warn("[StructureViewer] Failed to clear restraint lines:", err);
    }
  }, []);

  /**
   * Draw restraint distance lines on the loaded structure using Mol*
   * measurement manager. Color-codes satisfied (green) vs unsatisfied (red).
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const drawRestraints = useCallback(async (plugin: any, sat: RestraintSatisfaction) => {
    try {
      // Dynamically import Mol* modules needed for atom selection
      const { MolScriptBuilder: MS } = await import(
        "molstar/lib/mol-script/language/builder.js"
      );
      const { StructureElement } = await import(
        "molstar/lib/mol-model/structure.js"
      );

      // Get the loaded structure from the plugin hierarchy
      const structures = plugin.managers.structure.hierarchy.current.structures;
      const structCell = structures?.[0]?.cell;
      const structure = structCell?.obj?.data;
      if (!structure) return;

      /**
       * Build an atom Loci for a specific chain:residue:atom triple.
       */
      function getAtomLoci(chainId: string, seqId: number, atomName: string) {
        const expr = MS.struct.generator.atomGroups({
          "chain-test": MS.core.rel.eq([MS.ammp("auth_asym_id"), chainId]),
          "residue-test": MS.core.rel.eq([MS.ammp("auth_seq_id"), seqId]),
          "atom-test": MS.core.rel.eq([MS.ammp("auth_atom_id"), atomName]),
        });
        return StructureElement.Loci.fromExpression(structure, expr);
      }

      // Draw distance restraints
      for (const d of sat.distance ?? []) {
        const lociA = getAtomLoci(d.chain_i, d.residue_i, d.atom_i);
        const lociB = getAtomLoci(d.chain_j, d.residue_j, d.atom_j);
        if (StructureElement.Loci.isEmpty(lociA) || StructureElement.Loci.isEmpty(lociB)) continue;

        const color = d.satisfied ? 0x22cc22 : 0xcc2222;
        await plugin.managers.structure.measurement.addDistance(lociA, lociB, {
          reprTags: [RESTRAINT_TAG],
          selectionTags: [RESTRAINT_TAG],
          visualParams: {
            linesColor: color as never,
            linesSize: 2,
            customText: `${d.actual_distance.toFixed(1)}A`,
          },
        });
      }

      // Draw contact restraints (CA-CA between source and closest candidate)
      for (const c of sat.contact ?? []) {
        const lociA = getAtomLoci(c.chain_i, c.residue_i, "CA");
        const lociB = getAtomLoci(
          c.closest_candidate_chain,
          c.closest_candidate_residue,
          "CA",
        );
        if (StructureElement.Loci.isEmpty(lociA) || StructureElement.Loci.isEmpty(lociB)) continue;

        const color = c.satisfied ? 0x22cc22 : 0xcc2222;
        await plugin.managers.structure.measurement.addDistance(lociA, lociB, {
          reprTags: [RESTRAINT_TAG],
          selectionTags: [RESTRAINT_TAG],
          visualParams: {
            linesColor: color as never,
            linesSize: 2,
            dashLength: 0.3,
            customText: `${c.actual_distance.toFixed(1)}A`,
          },
        });
      }

      // Draw repulsive restraints (CA-CA, dashed)
      for (const r of sat.repulsive ?? []) {
        const lociA = getAtomLoci(r.chain_i, r.residue_i, "CA");
        const lociB = getAtomLoci(r.chain_j, r.residue_j, "CA");
        if (StructureElement.Loci.isEmpty(lociA) || StructureElement.Loci.isEmpty(lociB)) continue;

        const color = r.satisfied ? 0x22cc22 : 0xcc2222;
        await plugin.managers.structure.measurement.addDistance(lociA, lociB, {
          reprTags: [RESTRAINT_TAG],
          selectionTags: [RESTRAINT_TAG],
          visualParams: {
            linesColor: color as never,
            linesSize: 2,
            dashLength: 0.5,
            customText: `${r.actual_distance.toFixed(1)}A`,
          },
        });
      }
    } catch (err) {
      console.warn("[StructureViewer] Failed to draw restraint lines:", err);
    }
  }, []);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const loadStructure = useCallback(async (plugin: any, url: string) => {
    try {
      await plugin.clear();

      const apiKey = process.env.NEXT_PUBLIC_API_KEY || "";
      const resp = await fetch(url, apiKey ? { headers: { Authorization: `Bearer ${apiKey}` } } : {});
      if (!resp.ok) {
        console.warn("[StructureViewer] CIF fetch failed:", resp.status);
        return;
      }
      const cifText = await resp.text();

      const data = await plugin.builders.data.rawData(
        { data: cifText, label: "structure.cif" },
        { state: { isGhost: true } },
      );
      const trajectory = await plugin.builders.structure.parseTrajectory(
        data,
        "mmcif",
      );

      await plugin.builders.structure.hierarchy.applyPreset(
        trajectory,
        "default",
      );

      // Draw restraint annotations after structure loads
      const sat = satisfactionRef.current;
      if (sat) {
        await drawRestraints(plugin, sat);
      }
    } catch (err) {
      console.warn("[StructureViewer] Failed to load structure:", err);
    }
  }, [drawRestraints]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    let disposed = false;

    async function init(container: HTMLDivElement) {
      const { createPluginUI } = await import(
        "molstar/lib/mol-plugin-ui/index.js"
      );
      const { renderReact18 } = await import(
        "molstar/lib/mol-plugin-ui/react18.js"
      );
      const { DefaultPluginUISpec } = await import(
        "molstar/lib/mol-plugin-ui/spec.js"
      );

      if (disposed) return;

      const spec = DefaultPluginUISpec();

      const plugin = await createPluginUI({
        target: container,
        render: renderReact18,
        spec: {
          ...spec,
          layout: {
            initial: {
              isExpanded: false,
              showControls: false,
              regionState: {
                top: "hidden",
                bottom: "hidden",
                left: "hidden",
                right: "hidden",
              },
            },
          },
          components: {
            remoteState: "none",
          },
        },
      });

      if (disposed) {
        plugin.dispose();
        return;
      }

      // Apply a neutral background matching our container
      try {
        if (plugin.canvas3d) {
          const renderer = plugin.canvas3d.props.renderer;
          // Mol* Color is a branded number
          const bgColor = 0x111111 as typeof renderer.backgroundColor;
          plugin.canvas3d.setProps({
            renderer: { ...renderer, backgroundColor: bgColor },
          });
        }
      } catch {
        // Non-critical visual preference
      }

      pluginRef.current = plugin;

      if (cifUrl) {
        loadStructure(plugin, cifUrl);
      }
    }

    init(el);

    return () => {
      disposed = true;
      const plugin = pluginRef.current;
      if (plugin) {
        plugin.dispose();
        pluginRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const plugin = pluginRef.current;
    if (plugin && cifUrl) {
      loadStructure(plugin, cifUrl);
    }
  }, [cifUrl, loadStructure]);

  // Update restraint lines when satisfaction data changes
  useEffect(() => {
    const plugin = pluginRef.current;
    if (!plugin) return;

    (async () => {
      await clearRestraints(plugin);
      if (satisfaction) {
        await drawRestraints(plugin, satisfaction);
      }
    })();
  }, [satisfaction, clearRestraints, drawRestraints]);

  return (
    <section className="af-panel af-panel-strong p-2.5">
      <div className="flex items-center justify-between px-1.5 pb-2.5">
        <h2 className="af-panel-header">3D structure</h2>
        <span className="af-panel-subtitle">Interactive molecular viewer</span>
      </div>
      <div
        className="overflow-hidden rounded-xl border border-border/80"
        style={{ background: "var(--viewer-bg)" }}
      >
        <div
          ref={containerRef}
          className="relative w-full"
          style={{ height: "clamp(380px, 52vh, 620px)" }}
        />
      </div>
    </section>
  );
}
