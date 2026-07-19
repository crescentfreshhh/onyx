export interface JobSettings {
  deinterlace: { enabled: boolean; engine: string };
  enhance: { enabled: boolean; model: string; scale: number };
  interpolate: { enabled: boolean; model: string; fps: number };
  grain: { enabled: boolean; amount: number };
  encode: { codec: string; quality: number; container: string; audio: string };
}

export interface Job {
  id: number;
  input_path: string;
  output_path: string;
  settings: JobSettings;
  status: "queued" | "running" | "completed" | "failed" | "canceled";
  progress: number;
  fps: number | null;
  eta_seconds: number | null;
  error: string | null;
  created_at: number;
}

export interface Preset {
  id: number;
  name: string;
  settings: Partial<JobSettings>;
  builtin: boolean;
}

export interface FileEntry {
  name: string;
  type: "dir" | "file";
  size?: number;
}

export interface MediaInfo {
  width: number;
  height: number;
  fps: number;
  duration: number;
  codec: string;
  interlaced: boolean;
  size_bytes: number;
}

export interface StageModel {
  id: string;
  name: string;
  engine: string;
}

export interface CatalogModel {
  id: string;
  name: string;
  stage: string;
  engine: string;
  scale: number;
  license: string;
  status: "installed" | "downloading" | "available" | "failed" | "missing";
  progress?: number;
  error?: string;
}

export interface SystemInfo {
  version: string;
  gpu: string | null;
  ffmpeg: boolean;
}

export const defaultSettings = (): JobSettings => ({
  deinterlace: { enabled: false, engine: "bwdif" },
  enhance: { enabled: false, model: "lanczos", scale: 2 },
  interpolate: { enabled: false, model: "dup", fps: 60 },
  grain: { enabled: false, amount: 4 },
  encode: { codec: "libx264", quality: 18, container: "mkv", audio: "copy" },
});
