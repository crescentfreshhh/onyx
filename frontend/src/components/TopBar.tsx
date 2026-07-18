import type { SystemInfo } from "../types";

interface Props {
  system: SystemInfo | null;
  onOpen: () => void;
}

export function TopBar({ system, onOpen }: Props) {
  return (
    <div className="topbar">
      <div className="logo">
        ONY<span>X</span>
      </div>
      <div className="version">v{system?.version ?? "…"}</div>
      <button onClick={onOpen}>Open Video…</button>
      <div className="spacer" />
      <div className="gpu">
        {system?.gpu ?? "No GPU detected"}
        {system && !system.ffmpeg && " · ffmpeg missing!"}
      </div>
    </div>
  );
}
