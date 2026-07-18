import { useEffect, useState } from "react";
import { api } from "../api";
import type { FileEntry } from "../types";

interface Props {
  onSelect: (path: string) => void;
  onClose: () => void;
}

function fmtSize(bytes?: number): string {
  if (!bytes) return "";
  if (bytes > 1 << 30) return `${(bytes / (1 << 30)).toFixed(2)} GiB`;
  return `${(bytes / (1 << 20)).toFixed(1)} MiB`;
}

export function FileBrowser({ onSelect, onClose }: Props) {
  const [path, setPath] = useState("");
  const [entries, setEntries] = useState<FileEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .files(path)
      .then((r) => {
        setEntries(r.entries);
        setError(null);
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, [path]);

  const up = () => setPath(path.split("/").slice(0, -1).join("/"));

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          Open Video
          <span className="path">/input/{path}</span>
        </div>
        <div className="file-list">
          {path && (
            <div className="file-entry" onClick={up}>
              📁 ..
            </div>
          )}
          {error && <div className="empty" style={{ padding: 16 }}>{error}</div>}
          {!error && entries.length === 0 && (
            <div className="empty" style={{ padding: 16, color: "var(--text-dim)" }}>
              No videos found in this folder.
            </div>
          )}
          {entries.map((entry) => (
            <div
              className="file-entry"
              key={entry.name}
              onClick={() =>
                entry.type === "dir"
                  ? setPath(path ? `${path}/${entry.name}` : entry.name)
                  : onSelect(path ? `${path}/${entry.name}` : entry.name)
              }
            >
              {entry.type === "dir" ? "📁" : "🎬"} {entry.name}
              <span className="size">{fmtSize(entry.size)}</span>
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
