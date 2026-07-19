import { useState } from "react";
import type { JobSettings, Preset, StageModel } from "../types";
import { defaultSettings } from "../types";

interface Props {
  settings: JobSettings;
  onChange: (settings: JobSettings) => void;
  presets: Preset[];
  models: Record<string, StageModel[]>;
  onSavePreset: (name: string) => void;
  onAddToQueue: () => void;
  onQueueAll: () => void;
  fileCount: number;
  onPreview: (startSeconds: number) => void;
  previewBusy: boolean;
  canQueue: boolean;
}

function ModelHint({ models, selected }: { models: StageModel[]; selected: string }) {
  const model = models.find((m) => m.id === selected);
  if (!model?.description) return null;
  return (
    <div className="model-hint">
      {model.best_for && <span className="best-for">{model.best_for}</span>}
      {model.description}
    </div>
  );
}

interface SectionProps {
  title: string;
  enabled: boolean;
  onToggle: (on: boolean) => void;
  children: React.ReactNode;
}

function Section({ title, enabled, onToggle, children }: SectionProps) {
  const [open, setOpen] = useState(true);
  return (
    <div className="section">
      <div className="section-header" onClick={() => setOpen(!open)}>
        <label className="toggle" onClick={(e) => e.stopPropagation()}>
          <input type="checkbox" checked={enabled} onChange={(e) => onToggle(e.target.checked)} />
          <span className="track" />
        </label>
        <span className="title">{title}</span>
        <span style={{ color: "var(--text-dim)" }}>{open ? "▾" : "▸"}</span>
      </div>
      {open && enabled && <div className="section-body">{children}</div>}
    </div>
  );
}

export function FilterPanel({ settings, onChange, presets, models, onSavePreset, onAddToQueue, onQueueAll, fileCount, onPreview, previewBusy, canQueue }: Props) {
  const [previewStart, setPreviewStart] = useState(0);
  const set = (patch: Partial<JobSettings>) => onChange({ ...settings, ...patch });

  const applyPreset = (id: string) => {
    const preset = presets.find((p) => String(p.id) === id);
    if (!preset) return;
    const base = defaultSettings();
    onChange({
      deinterlace: { ...base.deinterlace, ...preset.settings.deinterlace },
      enhance: { ...base.enhance, ...preset.settings.enhance },
      interpolate: { ...base.interpolate, ...preset.settings.interpolate },
      grain: { ...base.grain, ...preset.settings.grain },
      encode: { ...base.encode, ...preset.settings.encode },
    });
  };

  return (
    <div className="filters">
      <div className="preset-row">
        <select defaultValue="" onChange={(e) => applyPreset(e.target.value)}>
          <option value="" disabled>
            Apply preset…
          </option>
          {presets.map((p) => (
            <option key={p.id} value={p.id}>
              {p.builtin ? "★ " : ""}{p.name}
            </option>
          ))}
        </select>
        <button
          onClick={() => {
            const name = window.prompt("Preset name:");
            if (name?.trim()) onSavePreset(name.trim());
          }}
        >
          Save
        </button>
      </div>

      <Section
        title="Deinterlace"
        enabled={settings.deinterlace.enabled}
        onToggle={(enabled) => set({ deinterlace: { ...settings.deinterlace, enabled } })}
      >
        <div className="field">
          <label>Engine</label>
          <select
            value={settings.deinterlace.engine}
            onChange={(e) => set({ deinterlace: { ...settings.deinterlace, engine: e.target.value } })}
          >
            {(models.deinterlace ?? []).map((m) => (
              <option key={m.id} value={m.id}>{m.name}</option>
            ))}
          </select>
        </div>
        <ModelHint models={models.deinterlace ?? []} selected={settings.deinterlace.engine} />
      </Section>

      <Section
        title="Enhance & Upscale"
        enabled={settings.enhance.enabled}
        onToggle={(enabled) => set({ enhance: { ...settings.enhance, enabled } })}
      >
        <div className="field">
          <label>Model</label>
          <select
            value={settings.enhance.model}
            onChange={(e) => set({ enhance: { ...settings.enhance, model: e.target.value } })}
          >
            {(models.enhance ?? []).map((m) => (
              <option key={m.id} value={m.id}>{m.name}</option>
            ))}
          </select>
        </div>
        <ModelHint models={models.enhance ?? []} selected={settings.enhance.model} />
        <div className="field">
          <label>Scale</label>
          <select
            value={settings.enhance.scale}
            onChange={(e) => set({ enhance: { ...settings.enhance, scale: Number(e.target.value) } })}
          >
            <option value={1}>1× (enhance only)</option>
            <option value={2}>2×</option>
            <option value={4}>4×</option>
          </select>
        </div>
      </Section>

      <Section
        title="Frame Interpolation"
        enabled={settings.interpolate.enabled}
        onToggle={(enabled) => set({ interpolate: { ...settings.interpolate, enabled } })}
      >
        <div className="field">
          <label>Model</label>
          <select
            value={settings.interpolate.model}
            onChange={(e) => set({ interpolate: { ...settings.interpolate, model: e.target.value } })}
          >
            {(models.interpolate ?? []).map((m) => (
              <option key={m.id} value={m.id}>{m.name}</option>
            ))}
          </select>
        </div>
        <ModelHint models={models.interpolate ?? []} selected={settings.interpolate.model} />
        <div className="field">
          <label>Target FPS</label>
          <input
            type="number"
            min={1}
            max={480}
            step="any"
            value={settings.interpolate.fps}
            onChange={(e) => {
              const fps = Number(e.target.value);
              if (Number.isFinite(fps)) set({ interpolate: { ...settings.interpolate, fps } });
            }}
          />
        </div>
        <div className="field">
          <label>Scene detect</label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={settings.interpolate.scene_detect}
              onChange={(e) =>
                set({ interpolate: { ...settings.interpolate, scene_detect: e.target.checked } })
              }
            />
            <span className="track" />
          </label>
          <span className="model-hint" style={{ flex: 1 }}>
            Skip interpolating across hard cuts (AI models only)
          </span>
        </div>
        <div className="field">
          <label />
          <div className="fps-presets">
            {[23.976, 24, 25, 29.97, 30, 50, 59.94, 60, 120].map((fps) => (
              <button
                key={fps}
                className={settings.interpolate.fps === fps ? "chip active" : "chip"}
                onClick={() => set({ interpolate: { ...settings.interpolate, fps } })}
              >
                {fps}
              </button>
            ))}
          </div>
        </div>
      </Section>

      <Section
        title="Film Grain"
        enabled={settings.grain.enabled}
        onToggle={(enabled) => set({ grain: { ...settings.grain, enabled } })}
      >
        <div className="field">
          <label>Amount</label>
          <input
            type="range"
            min={0}
            max={20}
            step={1}
            value={settings.grain.amount}
            onChange={(e) => set({ grain: { ...settings.grain, amount: Number(e.target.value) } })}
          />
          <span className="value">{settings.grain.amount}</span>
        </div>
        <div className="model-hint">
          Re-applies natural grain after AI cleaning — recommended for film sources,
          since upscale models tend to smear original grain.
        </div>
      </Section>

      <div className="section">
        <div className="section-header" style={{ cursor: "default" }}>
          <span className="title">Output</span>
        </div>
        <div className="section-body">
          <div className="field">
            <label>Encoder</label>
            <select
              value={settings.encode.codec}
              onChange={(e) => set({ encode: { ...settings.encode, codec: e.target.value } })}
            >
              <option value="libx264">H.264 (CPU)</option>
              <option value="libx265">H.265 (CPU)</option>
              <option value="h264_nvenc">H.264 (NVENC)</option>
              <option value="hevc_nvenc">H.265 (NVENC)</option>
            </select>
          </div>
          <div className="field">
            <label>Quality</label>
            <input
              type="range"
              min={10}
              max={35}
              step={1}
              value={settings.encode.quality}
              onChange={(e) => set({ encode: { ...settings.encode, quality: Number(e.target.value) } })}
            />
            <span className="value">{settings.encode.quality}</span>
          </div>
          <div className="field">
            <label>Container</label>
            <select
              value={settings.encode.container}
              onChange={(e) => set({ encode: { ...settings.encode, container: e.target.value } })}
            >
              <option value="mkv">MKV</option>
              <option value="mp4">MP4</option>
            </select>
          </div>
          <div className="field">
            <label>Audio</label>
            <select
              value={settings.encode.audio}
              onChange={(e) => set({ encode: { ...settings.encode, audio: e.target.value } })}
            >
              <option value="copy">Passthrough</option>
              <option value="aac">AAC 192k</option>
            </select>
          </div>
        </div>
      </div>

      <div className="actions">
        <div className="field">
          <label>Preview at (s)</label>
          <input
            type="number"
            min={0}
            step="any"
            value={previewStart}
            onChange={(e) => {
              const start = Number(e.target.value);
              if (Number.isFinite(start) && start >= 0) setPreviewStart(start);
            }}
          />
        </div>
        <button disabled={!canQueue || previewBusy} onClick={() => onPreview(previewStart)}>
          {previewBusy ? "Rendering preview…" : "Preview 5s"}
        </button>
        <button className="primary" disabled={!canQueue} onClick={onAddToQueue}>
          Add to Queue
        </button>
        {fileCount > 1 && (
          <button className="primary" onClick={onQueueAll}>
            Queue All {fileCount} Files
          </button>
        )}
      </div>
    </div>
  );
}
