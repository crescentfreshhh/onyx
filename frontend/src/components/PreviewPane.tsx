import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { MediaInfo, Preview } from "../types";

interface Props {
  file: string | null;
  files: string[];
  info: MediaInfo | null;
  preview: Preview | null;
  onClosePreview: () => void;
  onSelectFile: (path: string) => void;
  onRemoveFile: (path: string) => void;
}

function fmtDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${m}:${String(s).padStart(2, "0")}`;
}

function fmtSize(bytes: number): string {
  if (bytes > 1 << 30) return `${(bytes / (1 << 30)).toFixed(2)} GiB`;
  return `${(bytes / (1 << 20)).toFixed(1)} MiB`;
}

function CompareView({ previewId }: { previewId: string }) {
  const [position, setPosition] = useState(50);
  const originalRef = useRef<HTMLVideoElement>(null);
  const processedRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const original = originalRef.current;
    const processed = processedRef.current;
    if (!original || !processed) return;
    const sync = () => {
      if (Math.abs(original.currentTime - processed.currentTime) > 0.08) {
        processed.currentTime = original.currentTime;
      }
    };
    original.addEventListener("timeupdate", sync);
    return () => original.removeEventListener("timeupdate", sync);
  }, [previewId]);

  return (
    <div className="compare">
      <div className="compare-stage">
        <video
          ref={processedRef}
          src={api.previewUrl(previewId, "processed")}
          muted
          loop
          autoPlay
          playsInline
        />
        <div className="compare-overlay" style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}>
          <video
            ref={originalRef}
            src={api.previewUrl(previewId, "original")}
            muted
            loop
            autoPlay
            playsInline
          />
        </div>
        <div className="compare-divider" style={{ left: `${position}%` }} />
        <span className="compare-label left">Original</span>
        <span className="compare-label right">Enhanced</span>
      </div>
      <input
        className="compare-slider"
        type="range"
        min={0}
        max={100}
        value={position}
        onChange={(e) => setPosition(Number(e.target.value))}
      />
    </div>
  );
}

export function PreviewPane({ file, files, info, preview, onClosePreview, onSelectFile, onRemoveFile }: Props) {
  return (
    <div className="preview">
      {files.length > 0 && (
        <div className="file-tabs">
          {files.map((path) => (
            <div
              key={path}
              className={path === file ? "file-tab active" : "file-tab"}
              onClick={() => path !== file && onSelectFile(path)}
              title={path}
            >
              <span>{path.split("/").pop()}</span>
              <button
                className="tab-close"
                onClick={(e) => {
                  e.stopPropagation();
                  onRemoveFile(path);
                }}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
      <div className="stage">
        {preview?.status === "ready" ? (
          <>
            <CompareView previewId={preview.id} />
            <button className="close-preview" onClick={onClosePreview}>
              ✕ Close preview
            </button>
          </>
        ) : preview?.status === "rendering" ? (
          <div className="placeholder">
            <div style={{ fontSize: 32, opacity: 0.4 }}>⟳</div>
            <div>Rendering preview…</div>
          </div>
        ) : file ? (
          <video key={file} src={api.streamUrl(file)} controls />
        ) : (
          <div className="placeholder">
            <div style={{ fontSize: 40, opacity: 0.3 }}>▶</div>
            <div>Open a video from your input folder to get started</div>
          </div>
        )}
      </div>
      <div className="infobar">
        {file && <span title={file}><b>{file.split("/").pop()}</b></span>}
        {info && (
          <>
            <span>
              <b>{info.width}×{info.height}</b> source
            </span>
            <span>
              <b>{info.fps}</b> fps
            </span>
            <span>
              <b>{info.codec.toUpperCase()}</b>
            </span>
            <span>
              <b>{fmtDuration(info.duration)}</b>
            </span>
            <span>
              <b>{fmtSize(info.size_bytes)}</b>
            </span>
            {info.interlaced && <span style={{ color: "var(--warn)" }}>Interlaced</span>}
          </>
        )}
        {!file && <span>No video loaded</span>}
      </div>
    </div>
  );
}
