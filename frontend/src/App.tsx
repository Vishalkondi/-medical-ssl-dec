import { useCallback, useEffect, useState } from "react";
import { ApiError, getModelStatus, predict } from "./api";
import Header, { type Tab, type Theme } from "./components/Header";
import { InfoIcon } from "./components/Icons";
import ModelCard from "./components/ModelCard";
import ResultPanel, { type Phase } from "./components/ResultPanel";
import UploadPanel from "./components/UploadPanel";
import type { ModelStatus, PredictionResponse } from "./types";
import { validateFile } from "./validation";

const THEME_KEY = "cxr-ssl-theme";

function readTheme(): Theme {
  try {
    const t = localStorage.getItem(THEME_KEY);
    return t === "light" || t === "dark" ? t : "system";
  } catch {
    return "system";
  }
}

export default function App() {
  const [theme, setTheme] = useState<Theme>(readTheme);
  const [tab, setTab] = useState<Tab>("analyze");
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);

  const [phase, setPhase] = useState<Phase>("idle");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<{ code: string; message: string } | null>(null);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      /* storage unavailable: theme still applies for this session */
    }
  }, [theme]);

  const refreshStatus = useCallback(() => {
    getModelStatus()
      .then((s) => {
        setStatus(s);
        setStatusError(null);
      })
      .catch((e: ApiError) => setStatusError(e.message));
  }, []);

  useEffect(() => {
    refreshStatus();
    const id = window.setInterval(refreshStatus, 15000);
    return () => window.clearInterval(id);
  }, [refreshStatus]);

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  const maxMb = status?.max_upload_mb ?? 10;

  const onFile = (f: File) => {
    const problem = validateFile(f, maxMb);
    setResult(null);
    setError(null);
    setPhase("idle");
    if (problem) {
      setFileError(problem);
      return;
    }
    setFileError(null);
    setFile(f);
    setDims(null);
    setPreviewUrl(URL.createObjectURL(f));
  };

  const onClear = () => {
    setFile(null);
    setPreviewUrl(null);
    setDims(null);
    setFileError(null);
    setResult(null);
    setError(null);
    setPhase("idle");
  };

  const onAnalyze = async () => {
    if (!file) return;
    setPhase("loading");
    setError(null);
    try {
      setResult(await predict(file));
      setPhase("done");
    } catch (e) {
      const err = e instanceof ApiError ? e : new ApiError("UNKNOWN", "Unexpected error.");
      setError({ code: err.code, message: err.message });
      setPhase("error");
      if (err.code === "BACKEND_UNREACHABLE") setStatusError(err.message);
    }
  };

  return (
    <div className="app">
      <Header tab={tab} onTab={setTab} theme={theme} onTheme={setTheme} status={status} statusError={statusError} />

      <div className="notice" role="note">
        <InfoIcon width={18} height={18} />
        <p>
          <strong>Research prototype, not a medical device.</strong> Outputs describe dataset-defined radiographic
          findings, not a diagnosis. Do not upload images that identify a patient.
        </p>
      </div>

      <main className="main">
        {tab === "analyze" ? (
          <div className="grid">
            <UploadPanel
              file={file}
              previewUrl={previewUrl}
              dims={dims}
              fileError={fileError}
              maxMb={maxMb}
              busy={phase === "loading"}
              canAnalyze={!!file && phase !== "loading"}
              onFile={onFile}
              onClear={onClear}
              onDims={setDims}
              onAnalyze={onAnalyze}
            />
            <ResultPanel
              phase={phase}
              result={result}
              error={error}
              status={status}
              statusError={statusError}
              hasFile={!!file}
            />
          </div>
        ) : (
          <ModelCard status={status} />
        )}
      </main>

      <footer className="footer">
        Semi-supervised learning with deep embedded clustering · academic project · images are processed in memory and
        not stored
      </footer>
    </div>
  );
}
