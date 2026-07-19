import type { SystemInfo } from "../types";

interface Props {
  system: SystemInfo | null;
  onOpen: () => void;
  onModels: () => void;
}

export function TopBar({ system, onOpen, onModels }: Props) {
  return (
    <div className="topbar">
      <div className="logo">
        ONY<span>X</span>
      </div>
      <div className="version">v{system?.version ?? "…"}</div>
      <button onClick={onOpen}>Open Video…</button>
      <button onClick={onModels}>Models…</button>
      <div className="spacer" />
      <div className="gpu" title={system?.gpu_error ?? undefined}>
        {system?.gpu ?? (system?.gpu_error ? "No GPU — hover for why" : "No GPU detected")}
        {system && !system.ffmpeg && " · ffmpeg missing!"}
      </div>
    </div>
  );
}
