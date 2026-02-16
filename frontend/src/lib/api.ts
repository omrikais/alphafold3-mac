/** API client for all REST endpoints. */

import type {
  CacheCheckResult,
  ConfidenceResult,
  JobCreated,
  JobDetail,
  JobSubmission,
  PaginatedJobs,
  SampleConfidence,
  StructureParseResult,
  SystemStatus,
  ValidationResult,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "";
const BASE = `${API_URL}/api`;
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";

/** Append ?token= to a URL for auth on plain <a href> downloads. */
function withToken(url: string): string {
  if (!API_KEY) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}token=${encodeURIComponent(API_KEY)}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (API_KEY) headers["Authorization"] = `Bearer ${API_KEY}`;
  const res = await fetch(`${BASE}${path}`, {
    headers,
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ---- System ----

export function getSystemStatus(): Promise<SystemStatus> {
  return request("/system/status");
}

// ---- Jobs ----

export function submitJob(submission: JobSubmission): Promise<JobCreated> {
  return request("/jobs", {
    method: "POST",
    body: JSON.stringify(submission),
  });
}

export function listJobs(params?: {
  status?: string;
  search?: string;
  page?: number;
  page_size?: number;
}): Promise<PaginatedJobs> {
  const qs = new URLSearchParams();
  if (params?.status) qs.set("status", params.status);
  if (params?.search) qs.set("search", params.search);
  if (params?.page) qs.set("page", String(params.page));
  if (params?.page_size) qs.set("page_size", String(params.page_size));
  const query = qs.toString();
  return request(`/jobs${query ? `?${query}` : ""}`);
}

export function getJob(jobId: string): Promise<JobDetail> {
  return request(`/jobs/${jobId}`);
}

export function cancelJob(jobId: string): Promise<{ status: string }> {
  return request(`/jobs/${jobId}/cancel`, { method: "POST" });
}

export function deleteJob(jobId: string): Promise<{ status: string }> {
  return request(`/jobs/${jobId}`, { method: "DELETE" });
}

// ---- Results ----

export function getResults(jobId: string): Promise<ConfidenceResult> {
  return request(`/jobs/${jobId}/results`);
}

export function getStructureUrl(jobId: string, rank: number): string {
  return withToken(`${BASE}/jobs/${jobId}/results/structure/${rank}`);
}

export function getSampleConfidence(
  jobId: string,
  sampleIndex: number,
): Promise<SampleConfidence> {
  return request(`/jobs/${jobId}/results/confidence/${sampleIndex}`);
}

export function getDownloadAllUrl(jobId: string): string {
  return withToken(`${BASE}/jobs/${jobId}/results/download`);
}

export function getConfidenceJsonUrl(jobId: string): string {
  return withToken(`${BASE}/jobs/${jobId}/results/confidence-json`);
}

export function openJobDirectory(jobId: string): Promise<{ status: string; path: string }> {
  return request(`/jobs/${jobId}/results/open-directory`, { method: "POST" });
}

// ---- Cache ----

export function checkCache(sequences: Record<string, unknown>[]): Promise<CacheCheckResult> {
  return request("/cache/check", {
    method: "POST",
    body: JSON.stringify({ sequences }),
  });
}

// ---- Structure ----

async function requestForm<T>(path: string, formData: FormData): Promise<T> {
  const headers: Record<string, string> = {};
  if (API_KEY) headers["Authorization"] = `Bearer ${API_KEY}`;
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers,
    body: formData,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function parseStructureFile(file: File): Promise<StructureParseResult> {
  const formData = new FormData();
  formData.append("file", file);
  return requestForm("/structure/parse", formData);
}

export function fetchPdb(pdbId: string): Promise<StructureParseResult> {
  return request(`/structure/fetch/${pdbId}`);
}

// ---- Validation ----

export function validateInput(body: {
  sequences: Record<string, unknown>[];
  name?: string;
  modelSeeds?: number[];
}): Promise<ValidationResult> {
  return request("/validate", {
    method: "POST",
    body: JSON.stringify(body),
  });
}
