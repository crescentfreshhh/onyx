import { api } from "../api";
import type { MediaInfo } from "../types";

interface Props {
  file: string | null;
  info: MediaInfo | null;
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

export function PreviewPane({ file, info }: Props) {
  return (
    <div className="preview">
      <div className="stage">
        {file ? (
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
