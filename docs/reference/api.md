# API Reference

This document covers all public endpoints exposed by the FastAPI backend.

Base URL (default): `http://127.0.0.1:8642`

- REST prefix: `/api`
- Progress WebSocket: `/api/jobs/{job_id}/progress`

## Authentication

Authentication is optional and controlled by server startup config.

- If API key authentication is enabled (`--api-key` or `AF3_API_KEY`), send:
  - `Authorization: Bearer <token>` for HTTP requests, or
  - `?token=<token>` query parameter for WebSocket.
- If API key authentication is disabled, no token is required.

## Error Handling

- Validation failures in request schemas return FastAPI `422` responses.
- Domain-level validation in `/api/validate` is returned in the response body (`valid=false`) instead of raising an HTTP error.
- Common status codes: `400`, `403`, `404`, `409`, `422`, `429`, `500`, `502`, `504`.

## Jobs Endpoints

### `POST /api/jobs`

Submit a new prediction job.

- Request body: `JobSubmission`
  - Includes job metadata, sequences/entities, seeds, optional restraints, and runtime overrides.
- Response: `JobCreated`
  - Includes job ID and initial status.
- Errors:
  - `429` when queue capacity is exceeded.

### `GET /api/jobs`

List jobs with filters and pagination.

- Query parameters:
  - `status` (optional)
  - `search` (optional)
  - `page` (>= 1)
  - `page_size` (1-100)
- Response: `PaginatedJobs`
  - Includes list of `JobSummary` items and pagination metadata.

### `GET /api/jobs/{job_id}`

Get full details for one job.

- Response: `JobDetail`
- Errors:
  - `404` when job is not found.

### `POST /api/jobs/{job_id}/cancel`

Cancel a pending or running job.

- Response: `{ "status": "cancelled" }`
- Errors:
  - `404` if job does not exist.
  - `400` if job is already terminal.
  - `409` if cancellation races with terminal completion.

### `DELETE /api/jobs/{job_id}`

Delete a terminal job (`completed`, `failed`, `cancelled`).

- Response: `{ "status": "deleted" }`
- Errors:
  - `404` if job does not exist.
  - `400` if job is still pending/running.

## Results Endpoints

### `GET /api/jobs/{job_id}/results`

Return aggregate confidence output for a completed job.

- Response: `ConfidenceResult`
  - Top-level metrics, sample list, best sample index, complex flag.
- Errors:
  - `404` missing job or confidence file.
  - `400` job not completed.
  - `500` confidence file cannot be parsed.

### `GET /api/jobs/{job_id}/results/confidence/{sample_index}`

Return per-sample confidence details.

- Response: `SampleConfidence`
  - Includes per-sample pLDDT/PAE arrays and optional restraint satisfaction details.
- Errors:
  - `404` missing job or sample.

### `GET /api/jobs/{job_id}/results/confidence-json`

Download raw confidence JSON file.

- Response: `application/json` file response.
- Errors:
  - `404` missing job or confidence file.
  - `400` job not completed.

### `GET /api/jobs/{job_id}/results/download`

Download ZIP archive of job outputs.

- Response: ZIP stream (`application/zip`)
- Errors:
  - `404` missing job or output directory.
  - `400` job not completed.

### `POST /api/jobs/{job_id}/results/open-directory`

Open output directory in Finder.

- Response: `{ "status": "opened" }`
- Security limits:
  - Localhost only request origin.
  - macOS only.
- Errors:
  - `400` when unsupported platform.
  - `403` when not local request.
  - `404` missing job or directory.

### `GET /api/jobs/{job_id}/results/structure/{rank}`

Download one ranked structure as mmCIF.

- Response: `chemical/x-mmcif` file response.
- Errors:
  - `404` missing job or structure file.
  - `400` job not completed.

## Validation and Cache

### `POST /api/validate`

Run input validation without starting a job.

- Request body: `ValidationRequest`
- Response: `ValidationResult`
  - `valid` boolean
  - error list
  - estimated memory and sequence stats

### `POST /api/cache/check`

Check if MSA/template data is already cached for input sequences.

- Request body: `CacheCheckRequest`
- Response: `CacheCheckResponse`
  - `cached` boolean
  - optional cache key, timestamp, and size

## System

### `GET /api/system/status`

Return backend readiness and runtime environment info.

- Response: `SystemStatus`
  - Model loaded state
  - Hardware summary
  - Queue size
  - Version/build metadata
  - Pipeline mode flag

## Structure Import

### `POST /api/structure/parse`

Parse an uploaded structure file (`.pdb`, `.cif`, `.mmcif`) into supported input JSON.

- Request: multipart/form-data with file field
- Response: `StructureParseResult`
- Errors:
  - `422` on unsupported extension or parse failure.

### `GET /api/structure/fetch/{pdb_id}`

Fetch a structure by PDB ID from RCSB and convert to input JSON.

- Response: `StructureParseResult`
- Errors:
  - `422` malformed PDB ID or value errors.
  - `404` PDB ID not found upstream.
  - `504` upstream timeout.
  - `502` upstream service or network failure.

## WebSocket Progress Stream

### `WS /api/jobs/{job_id}/progress`

Real-time job progress stream used by the web UI.

- Message payload: serialized progress objects (`type`, stage/progress fields, terminal events).
- Idle keepalive: periodic ping payloads.
- Authentication:
  - If API key is enabled, `token` query parameter is required.
- Origin policy:
  - Origin must match host, localhost, or configured CORS origins.
- Close/error codes:
  - `4001` invalid or missing token.
  - `4003` origin rejected.
  - `4004` unknown job.

## Minimal `curl` Examples

```bash
# System status
curl http://127.0.0.1:8642/api/system/status

# Submit job
curl -X POST http://127.0.0.1:8642/api/jobs \
  -H 'Content-Type: application/json' \
  -d @examples/desi1_monomer.json

# Validate without enqueueing
curl -X POST http://127.0.0.1:8642/api/validate \
  -H 'Content-Type: application/json' \
  -d @examples/desi1_monomer.json
```
