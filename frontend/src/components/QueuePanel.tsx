import { api } from "../api";
import type { Job } from "../types";

interface Props {
  jobs: Job[];
  onChanged: () => void;
  onError: (message: string) => void;
}

function summarize(job: Job): string {
  const s = job.settings;
  const parts: string[] = [];
  if (s.deinterlace?.enabled) parts.push("deinterlace");
  if (s.enhance?.enabled) {
    const model = s.enhance.model.replace(/^(custom:|import:)/, "");
    parts.push(`${s.enhance.scale}× ${model}`);
  }
  if (s.interpolate?.enabled) parts.push(`→${s.interpolate.fps}fps`);
  if (s.grain?.enabled) parts.push("grain");
  parts.push(s.encode?.codec ?? "");
  return parts.filter(Boolean).join(" · ");
}

function fmtEta(seconds: number | null): string {
  if (seconds == null) return "";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m}m ${s}s left` : `${s}s left`;
}

export function QueuePanel({ jobs, onChanged, onError }: Props) {
  const act = (fn: () => Promise<void>) => fn().then(onChanged).catch((e) => onError(String(e.message ?? e)));

  return (
    <div className="queue">
      <div className="queue-header">
        Queue
        <span className="count">
          {jobs.filter((j) => j.status === "queued" || j.status === "running").length} pending
        </span>
      </div>
      <div className="rows">
        {jobs.length === 0 && <div className="empty">Queue is empty — configure filters and add a video.</div>}
        {jobs.map((job) => (
          <div className="qrow" key={job.id}>
            <div>
              <div className="name" title={job.input_path}>
                {job.input_path.split("/").pop()}
              </div>
              <div className="sub" title={job.error ?? undefined}>
                {job.status === "failed" ? job.error : summarize(job)}
              </div>
            </div>
            <div>
              <span className={`status-chip status-${job.status}`}>{job.status}</span>
            </div>
            <div>
              {(job.status === "running" || job.status === "completed") && (
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${Math.round(job.progress * 100)}%` }} />
                </div>
              )}
            </div>
            <div className="meta">
              {job.status === "running" && (
                <>
                  {Math.round(job.progress * 100)}%{job.fps ? ` · ${job.fps} fps` : ""}
                  <br />
                  {fmtEta(job.eta_seconds)}
                </>
              )}
            </div>
            <div className="row-actions">
              {(job.status === "queued" || job.status === "running") && (
                <button className="danger-text" onClick={() => act(() => api.cancelJob(job.id))}>
                  Cancel
                </button>
              )}
              {job.status !== "queued" && job.status !== "running" && (
                <button onClick={() => act(() => api.deleteJob(job.id))}>Remove</button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
