import { useEffect, useState } from "react";
import { api } from "../api";
import type { CatalogModel } from "../types";

interface Props {
  onClose: () => void;
  onChanged: () => void;
}

const STATUS_LABEL: Record<CatalogModel["status"], string> = {
  installed: "Installed",
  downloading: "Downloading…",
  available: "Not downloaded",
  failed: "Failed",
  missing: "Missing file",
};

export function ModelManager({ onClose, onChanged }: Props) {
  const [models, setModels] = useState<CatalogModel[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const refresh = () =>
      api
        .modelCatalog()
        .then((catalog) => {
          if (!cancelled) setModels(catalog);
        })
        .catch((e) => setError(String(e.message ?? e)));
    refresh();
    const timer = window.setInterval(refresh, 1500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const download = (id: string) =>
    api
      .downloadModel(id)
      .then(onChanged)
      .catch((e) => setError(String(e.message ?? e)));

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          AI Models
          <span className="path">stored in /config/models — drop custom .onnx files there</span>
        </div>
        <div className="file-list">
          {error && <div style={{ padding: 16, color: "var(--danger)" }}>{error}</div>}
          {models.length === 0 && !error && (
            <div style={{ padding: 16, color: "var(--text-dim)" }}>No models in catalog.</div>
          )}
          {models.map((model) => (
            <div className="file-entry" key={model.id} style={{ cursor: "default" }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div>{model.name}</div>
                <div style={{ color: "var(--text-dim)", fontSize: 11 }}>
                  {model.stage} · {model.scale}× · {model.license}
                  {model.status === "failed" && model.error && (
                    <span style={{ color: "var(--danger)" }}> — {model.error}</span>
                  )}
                </div>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
                {model.status === "downloading" ? (
                  <div className="progress-track" style={{ width: 90 }}>
                    <div
                      className="progress-fill"
                      style={{ width: `${Math.round((model.progress ?? 0) * 100)}%` }}
                    />
                  </div>
                ) : (
                  <span
                    className={`status-chip ${model.status === "installed" ? "status-completed" : model.status === "failed" ? "status-failed" : "status-queued"}`}
                  >
                    {STATUS_LABEL[model.status]}
                  </span>
                )}
                {(model.status === "available" || model.status === "failed") && (
                  <button onClick={() => download(model.id)}>Download</button>
                )}
              </div>
            </div>
          ))}
        </div>
        <div className="modal-footer">
          <button onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
