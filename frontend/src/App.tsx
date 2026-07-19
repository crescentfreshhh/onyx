import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import type { Job, JobSettings, MediaInfo, Preset, StageModel, SystemInfo } from "./types";
import { defaultSettings } from "./types";
import { TopBar } from "./components/TopBar";
import { PreviewPane } from "./components/PreviewPane";
import { FilterPanel } from "./components/FilterPanel";
import { QueuePanel } from "./components/QueuePanel";
import { FileBrowser } from "./components/FileBrowser";
import { ModelManager } from "./components/ModelManager";

export default function App() {
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [models, setModels] = useState<Record<string, StageModel[]>>({});
  const [settings, setSettings] = useState<JobSettings>(defaultSettings());
  const [file, setFile] = useState<string | null>(null);
  const [info, setInfo] = useState<MediaInfo | null>(null);
  const [browsing, setBrowsing] = useState(false);
  const [managingModels, setManagingModels] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const showError = useCallback((message: string) => {
    setError(message);
    window.setTimeout(() => setError(null), 5000);
  }, []);

  const refreshJobs = useCallback(() => {
    api.jobs().then(setJobs).catch(() => {});
  }, []);

  useEffect(() => {
    api.system().then(setSystem).catch(() => {});
    api.presets().then(setPresets).catch(() => {});
    api.models().then(setModels).catch(() => {});
    refreshJobs();
    const timer = window.setInterval(refreshJobs, 1500);
    return () => window.clearInterval(timer);
  }, [refreshJobs]);

  const openFile = (path: string) => {
    setBrowsing(false);
    setFile(path);
    setInfo(null);
    api
      .mediaInfo(path)
      .then((mediaInfo) => {
        setInfo(mediaInfo);
        if (mediaInfo.interlaced) {
          setSettings((s) => ({ ...s, deinterlace: { ...s.deinterlace, enabled: true } }));
        }
      })
      .catch((e) => showError(String(e.message ?? e)));
  };

  const addToQueue = () => {
    if (!file) return;
    api
      .createJob(file, settings)
      .then(refreshJobs)
      .catch((e) => showError(String(e.message ?? e)));
  };

  const savePreset = (name: string) => {
    api
      .savePreset(name, settings)
      .then(() => api.presets().then(setPresets))
      .catch((e) => showError(String(e.message ?? e)));
  };

  return (
    <div className="app">
      <TopBar
        system={system}
        onOpen={() => setBrowsing(true)}
        onModels={() => setManagingModels(true)}
      />
      <PreviewPane file={file} info={info} />
      <FilterPanel
        settings={settings}
        onChange={setSettings}
        presets={presets}
        models={models}
        onSavePreset={savePreset}
        onAddToQueue={addToQueue}
        canQueue={file != null}
      />
      <QueuePanel jobs={jobs} onChanged={refreshJobs} onError={showError} />
      {browsing && <FileBrowser onSelect={openFile} onClose={() => setBrowsing(false)} />}
      {managingModels && (
        <ModelManager
          onClose={() => {
            setManagingModels(false);
            api.models().then(setModels).catch(() => {});
          }}
          onChanged={() => api.models().then(setModels).catch(() => {})}
        />
      )}
      {error && <div className="error-toast">{error}</div>}
    </div>
  );
}
