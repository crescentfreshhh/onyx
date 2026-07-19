import type { CatalogModel, Job, JobSettings, Preset, Preview, FileEntry, MediaInfo, StageModel, SystemInfo } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init);
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      detail = (await resp.json()).detail ?? detail;
    } catch {
      /* not json */
    }
    throw new Error(detail);
  }
  return resp.status === 204 ? (undefined as T) : resp.json();
}

export const api = {
  jobs: () => request<Job[]>("/api/jobs"),
  createJob: (input_path: string, settings: JobSettings) =>
    request<Job>("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input_path, settings }),
    }),
  cancelJob: (id: number) => request<void>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  deleteJob: (id: number) => request<void>(`/api/jobs/${id}`, { method: "DELETE" }),
  files: (path: string) =>
    request<{ path: string; entries: FileEntry[] }>(`/api/files?path=${encodeURIComponent(path)}`),
  mediaInfo: (path: string) => request<MediaInfo>(`/api/media/info?path=${encodeURIComponent(path)}`),
  presets: () => request<Preset[]>("/api/presets"),
  savePreset: (name: string, settings: JobSettings) =>
    request<void>("/api/presets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, settings }),
    }),
  deletePreset: (id: number) => request<void>(`/api/presets/${id}`, { method: "DELETE" }),
  models: () => request<Record<string, StageModel[]>>("/api/models"),
  modelCatalog: () => request<CatalogModel[]>("/api/models/catalog"),
  downloadModel: (id: string) =>
    request<void>(`/api/models/${encodeURIComponent(id)}/download`, { method: "POST" }),
  convertModel: (id: string) =>
    request<void>(`/api/models/${encodeURIComponent(id)}/convert`, { method: "POST" }),
  importModel: (url: string) =>
    request<{ id: string }>("/api/models/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    }),
  system: () => request<SystemInfo>("/api/system"),
  streamUrl: (path: string) => `/api/files/stream?path=${encodeURIComponent(path)}`,
  createPreview: (input_path: string, settings: JobSettings, start_seconds: number) =>
    request<{ id: string }>("/api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input_path, settings, start_seconds }),
    }),
  previewStatus: (id: string) => request<Preview>(`/api/preview/${id}`),
  deletePreview: (id: string) => request<void>(`/api/preview/${id}`, { method: "DELETE" }),
  previewUrl: (id: string, side: "original" | "processed") => `/api/preview/${id}/${side}`,
};
