"use client";

import { useEffect, useReducer, useRef } from "react";
import { connectJobProgress } from "@/lib/ws";
import type { WSMessage } from "@/lib/types";

export interface ProgressState {
  stage: string | null;
  percentComplete: number;
  diffusionStep: number | null;
  diffusionTotal: number | null;
  recyclingIteration: number | null;
  recyclingTotal: number | null;
  terminal: boolean;
  terminalType: "completed" | "failed" | "cancelled" | null;
  error: string | null;
}

const INITIAL: ProgressState = {
  stage: null,
  percentComplete: 0,
  diffusionStep: null,
  diffusionTotal: null,
  recyclingIteration: null,
  recyclingTotal: null,
  terminal: false,
  terminalType: null,
  error: null,
};

type ProgressAction =
  | { type: "reset" }
  | { type: "message"; message: WSMessage };

function progressReducer(state: ProgressState, action: ProgressAction): ProgressState {
  if (action.type === "reset") {
    return INITIAL;
  }

  const msg = action.message;
  switch (msg.type) {
    case "stage_change":
      return {
        ...state,
        stage: msg.stage ?? state.stage,
        percentComplete: msg.percent_complete ?? state.percentComplete,
      };
    case "progress":
      return {
        ...state,
        stage: msg.stage ?? state.stage,
        percentComplete: msg.percent_complete ?? state.percentComplete,
        diffusionStep: msg.diffusion_step ?? state.diffusionStep,
        diffusionTotal: msg.diffusion_total ?? state.diffusionTotal,
        recyclingIteration: msg.recycling_iteration ?? state.recyclingIteration,
        recyclingTotal: msg.recycling_total ?? state.recyclingTotal,
      };
    case "completed":
      return {
        ...state,
        percentComplete: 100,
        terminal: true,
        terminalType: "completed",
      };
    case "failed":
      return {
        ...state,
        terminal: true,
        terminalType: "failed",
        error: msg.error ?? null,
      };
    case "cancelled":
      return {
        ...state,
        terminal: true,
        terminalType: "cancelled",
      };
    default:
      return state;
  }
}

export function useJobProgress(jobId: string | null): ProgressState {
  const [state, dispatch] = useReducer(progressReducer, INITIAL);
  const connRef = useRef<{ close: () => void } | null>(null);

  useEffect(() => {
    dispatch({ type: "reset" });
    if (!jobId) return;

    const conn = connectJobProgress(jobId, (msg: WSMessage) => {
      dispatch({ type: "message", message: msg });
    });

    connRef.current = conn;

    return () => {
      conn.close();
    };
  }, [jobId]);

  return state;
}
