// Tiny typed fetch wrapper around the kiosk backend.
// Mirrors the Pydantic models in src/storymesh/kiosk/models.py.

export type JobStatusName = "queued" | "running" | "completed" | "failed";

export interface JobStatus {
  run_id: string;
  status: JobStatusName;
  title: string | null;
  stage: string | null;
  stage_index: number;
  total_stages: number;
  started_at: number | null;
  queue_position: number | null;
  prompt_style: string;
}

export interface GalleryItem {
  run_id: string;
  title: string;
  cover_url: string;
  completed_at: number;
}

export interface PromptStyleOption {
  id: string;
  name: string;
  description: string;
  is_recommended: boolean;
}

export interface SubmitResponse {
  run_id: string;
  queue_position: number;
}

export interface RunSynopsis {
  run_id: string;
  title: string;
  synopsis: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    let detail: string = resp.statusText;
    try {
      const body = await resp.json();
      detail = body?.detail ?? body?.message ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return resp.json() as Promise<T>;
}

export const api = {
  promptStyles: () => request<PromptStyleOption[]>("/api/prompt-styles"),
  jobs: () => request<JobStatus[]>("/api/jobs"),
  gallery: () => request<GalleryItem[]>("/api/gallery"),
  runSynopsis: (runId: string) =>
    request<RunSynopsis>(`/api/run/${encodeURIComponent(runId)}/synopsis`),
  submit: (body: { prompt: string; email: string; prompt_style: string }) =>
    request<SubmitResponse>("/api/submit", { method: "POST", body: JSON.stringify(body) }),
};
