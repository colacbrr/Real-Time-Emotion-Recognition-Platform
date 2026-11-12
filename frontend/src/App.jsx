import { useEffect, useMemo, useRef, useState } from "react";

const formatScoreList = (scores) => {
  if (!scores || typeof scores !== "object") return "-";
  return Object.entries(scores)
    .sort((a, b) => b[1] - a[1])
    .map(([label, value]) => `${label}: ${Number(value).toFixed(3)}`)
    .join(" | ");
};

export default function App() {
  const apiBase = useMemo(
    () => (import.meta.env.VITE_ASPICV_API_BASE || "http://localhost:8000").replace(/\/$/, ""),
    []
  );
  const [serverReady, setServerReady] = useState(null);
  const [progress, setProgress] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [dataset, setDataset] = useState(null);
  const [paths, setPaths] = useState(null);
  const [statusError, setStatusError] = useState(null);

  const [imagePath, setImagePath] = useState("");
  const [imageBase64, setImageBase64] = useState("");
  const [previewUrl, setPreviewUrl] = useState("");
  const [previewName, setPreviewName] = useState("No image selected");
  const previewUrlRef = useRef(null);

  const [predictStatus, setPredictStatus] = useState("idle");
  const [predictError, setPredictError] = useState(null);
  const [predictResult, setPredictResult] = useState(null);

  const [compareStatus, setCompareStatus] = useState("idle");
  const [compareError, setCompareError] = useState(null);
  const [compareResults, setCompareResults] = useState([]);

  const [reports, setReports] = useState(null);
  const [activityLog, setActivityLog] = useState([]);

  const addLog = (entry) => {
    setActivityLog((current) => [entry, ...current].slice(0, 8));
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      setImageBase64("");
      setPreviewUrl("");
      setPreviewName("No image selected");
      return;
    }

    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
    }

    const objectUrl = URL.createObjectURL(file);
    previewUrlRef.current = objectUrl;
    setPreviewUrl(objectUrl);
    setPreviewName(file.name);
    setImagePath("");

    const reader = new FileReader();
    reader.onload = () => {
      const result = typeof reader.result === "string" ? reader.result : "";
      setImageBase64(result);
    };
    reader.readAsDataURL(file);
  };

  const buildPayload = () => {
    const payload = {};
    if (imageBase64) {
      payload.image_base64 = imageBase64;
    } else if (imagePath) {
      payload.image_path = imagePath;
    }
    if (selectedModel) {
      payload.model_name = selectedModel;
    }
    return payload;
  };

  const ensureInput = () => {
    if (!imageBase64 && !imagePath) {
      setPredictError("Select a file or enter a local path first.");
      return false;
    }
    return true;
  };

  const runPredict = () => {
    if (!ensureInput()) return;
    setPredictStatus("loading");
    setPredictError(null);
    setPredictResult(null);

    fetch(`${apiBase}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.error) {
          throw new Error(data.error);
        }
        setPredictResult(data);
        setPredictStatus("ready");
        addLog({
          type: "predict",
          label: data.label,
          model: data.model,
          latency: data.latency_ms,
          time: new Date().toLocaleTimeString(),
        });
      })
      .catch((err) => {
        setPredictError(err.message);
        setPredictStatus("error");
      });
  };

  const runCompare = () => {
    if (!ensureInput()) return;
    setCompareStatus("loading");
    setCompareError(null);
    setCompareResults([]);

    fetch(`${apiBase}/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.error) {
          throw new Error(data.error);
        }
        setCompareResults(data.results || []);
        setCompareStatus("ready");
        addLog({
          type: "compare",
          modelCount: (data.results || []).length,
          latency: data.latency_ms,
          time: new Date().toLocaleTimeString(),
        });
      })
      .catch((err) => {
        setCompareError(err.message);
        setCompareStatus("error");
      });
  };

  useEffect(() => {
    let active = true;
    fetch(`${apiBase}/status`)
      .then((res) => res.json())
      .then((data) => {
        if (!active) return;
        setServerReady(data.ready);
        setProgress(data.progress || null);
        setMetrics(data.metrics || null);
        setDataset(data.dataset || null);
        setPaths(data.paths || null);
        setModels(Array.isArray(data.models) ? data.models : []);
        if (Array.isArray(data.models) && data.models.length > 0) {
          setSelectedModel(data.models[0]);
        }
      })
      .catch((err) => {
        if (!active) return;
        setStatusError(err.message);
        setServerReady(false);
      });
    return () => {
      active = false;
    };
  }, [apiBase]);

  useEffect(() => {
    const source = new EventSource(`${apiBase}/events`);
    source.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setProgress(data);
        setServerReady(data.ready);
      } catch {
        // Ignore malformed events.
      }
    };
    source.onerror = () => {
      source.close();
    };
    return () => {
      source.close();
    };
  }, [apiBase]);

  useEffect(() => {
    let active = true;
    fetch(`${apiBase}/reports`)
      .then((res) => res.json())
      .then((data) => {
        if (!active) return;
        setReports(data);
      })
      .catch(() => {
        if (!active) return;
        setReports(null);
      });
    return () => {
      active = false;
    };
  }, [apiBase]);

  useEffect(() => {
    return () => {
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    };
  }, []);

  return (
    <div className="app-shell">
      <div className="glow-layer" aria-hidden="true" />
      <header className="hero">
        <div>
          <div className="brand">ASPICV / Emotion Ops</div>
          <h1>Facial expression classification with classic models and CNNs.</h1>
          <p>
            A focused command center for testing HOG and LBP pipelines while keeping
            reports, datasets, and model snapshots in sync.
          </p>
        </div>
        <div className="status-card">
          <div className="status-row">
            <span className="status-label">Server</span>
            <span
              className={`status-pill ${
                serverReady === true ? "ready" : serverReady === false ? "offline" : "pending"
              }`}
            >
              {serverReady === null && "checking"}
              {serverReady === true && "online"}
              {serverReady === false && "offline"}
            </span>
          </div>
          {progress && progress.stage !== "ready" && (
            <div className="progress-card">
              <div className="progress-head">
                <span>{progress.message || "Loading"}</span>
                <span>{progress.percent || 0}%</span>
              </div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progress.percent || 0}%` }} />
              </div>
            </div>
          )}
          <div className="status-note">API: {apiBase}</div>
          {metrics && metrics.model_load_time_sec !== null && (
            <div className="status-note">Model load: {metrics.model_load_time_sec}s</div>
          )}
          {metrics && typeof metrics.avg_query_ms === "number" && (
            <div className="status-note">Avg latency: {metrics.avg_query_ms} ms</div>
          )}
          {dataset && (
            <div className="status-note">
              Dataset: train {dataset.train_count} / val {dataset.val_count} / predict {dataset.predict_count}
            </div>
          )}
          {statusError && <div className="status-error">{statusError}</div>}
        </div>
      </header>

      <main>
        <section className="panel health-panel">
          <div className="panel-header">
            <h2>System health</h2>
            <p>Model availability and dataset status.</p>
            <p>Model availability and dataset status.</p>
          </div>
          <div className="health-grid">
            <div className="health-card">
              <div className="health-title">Models</div>
              <div className="health-value">{models.length > 0 ? models.length : 0}</div>
              <div className="health-meta">
                <div>{models.length > 0 ? models.join(", ") : "No models loaded"}</div>
                <div>
                  Load time:{" "}
                  {metrics && metrics.model_load_time_sec !== null
                    ? `${metrics.model_load_time_sec}s`
                    : "-"}
                </div>
              </div>
            </div>
            <div className="health-card">
              <div className="health-title">Dataset</div>
              <div className="health-value">
                {dataset ? dataset.train_count + dataset.val_count + dataset.predict_count : 0}
              </div>
              <div className="health-meta">
                {dataset ? (
                  <>
                    <div>
                      Train {dataset.train_count} · Val {dataset.val_count} · Predict{" "}
                      {dataset.predict_count}
                    </div>
                    <div>Path: {paths?.dataset || "default"}</div>
                  </>
                ) : (
                  "Dataset not detected"
                )}
              </div>
            </div>
          </div>
        </section>
        <section className="workspace-grid">
          <div className="panel">
            <div className="panel-header">
              <h2>Input</h2>
              <p>Select a face image from disk or provide a local path.</p>
            </div>
            <div className="input-grid">
              <div>
                <label htmlFor="fileInput">Image file</label>
                <input id="fileInput" type="file" accept="image/*" onChange={handleFileChange} />
              </div>
              <div className="divider">OR</div>
              <div>
                <label htmlFor="pathInput">Local path</label>
                <input
                  id="pathInput"
                  type="text"
                  value={imagePath}
                  onChange={(event) => {
                    setImagePath(event.target.value);
                    if (event.target.value) {
                      setImageBase64("");
                      setPreviewUrl("");
                      setPreviewName(event.target.value.split("/").slice(-1)[0] || "Path image");
                    }
                  }}
                  placeholder="/home/user/image.jpg"
                />
              </div>
            </div>

            <div className="preview">
              <div className="preview-box">
                {previewUrl ? (
                  <img src={previewUrl} alt="Preview" />
                ) : (
                  <div className="preview-empty">Awaiting image</div>
                )}
              </div>
              <p className="preview-name">{previewName}</p>
            </div>

            <div className="panel-actions">
              <div className="model-select">
                <label htmlFor="modelSelect">Model</label>
                <select
                  id="modelSelect"
                  value={selectedModel}
                  onChange={(event) => setSelectedModel(event.target.value)}
                >
                  <option value="">Auto</option>
                  {models.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
              </div>
              <button
                type="button"
                className="primary"
                onClick={runPredict}
                disabled={predictStatus === "loading"}
              >
                {predictStatus === "loading" ? "Running..." : "Run prediction"}
              </button>
              <button
                type="button"
                className="ghost"
                onClick={runCompare}
                disabled={compareStatus === "loading"}
              >
                {compareStatus === "loading" ? "Comparing..." : "Compare all"}
              </button>
            </div>
            <p className="hint">Tip: run Compare after each new model training session.</p>
          </div>

          <div className="panel">
            <div className="panel-header">
              <h2>Prediction</h2>
              <p>Live output from the ASPICV model stack.</p>
            </div>
            <div className="status-line">
              {predictStatus === "idle" && "Waiting for input."}
              {predictStatus === "loading" && "Prediction in progress..."}
              {predictStatus === "error" && `Error: ${predictError}`}
              {predictStatus === "ready" && "Prediction ready."}
            </div>
            <div className="result-card">
              <div>
                <span>Label</span>
                <strong>{predictResult ? predictResult.label : "-"}</strong>
              </div>
              <div>
                <span>Confidence</span>
                <strong>
                  {predictResult ? Number(predictResult.confidence).toFixed(3) : "-"}
                </strong>
              </div>
              <div>
                <span>Model</span>
                <strong>{predictResult ? predictResult.model : "-"}</strong>
              </div>
              <div>
                <span>Latency</span>
                <strong>
                  {predictResult && typeof predictResult.latency_ms === "number"
                    ? `${predictResult.latency_ms} ms`
                    : "-"}
                </strong>
              </div>
              <div className="span-two">
                <span>Scores (normalized)</span>
                <strong>
                  {predictResult ? formatScoreList(predictResult.normalized_scores) : "-"}
                </strong>
              </div>
              {predictResult?.warning && (
                <div className="span-two warning">{predictResult.warning}</div>
              )}
            </div>

            <div className="compare-block">
              <div className="panel-header compact">
                <h3>Compare models</h3>
                <p>Side-by-side labels and confidence scores.</p>
              </div>
              <div className="status-line">
                {compareStatus === "idle" && "No comparison yet."}
                {compareStatus === "loading" && "Running comparison..."}
                {compareStatus === "error" && `Error: ${compareError}`}
                {compareStatus === "ready" && `${compareResults.length} results.`}
              </div>
              <div className="compare-list">
                {compareResults.length === 0 && (
                  <div className="empty-card">Run Compare to see the full model stack.</div>
                )}
                {compareResults.map((item) => (
                  <div key={item.model} className="compare-row">
                    <div>
                      <strong>{item.model}</strong>
                      <span>{item.label || "-"}</span>
                    </div>
                    <div>
                      <span>Conf</span>
                      <strong>
                        {typeof item.confidence === "number"
                          ? item.confidence.toFixed(3)
                          : "-"}
                      </strong>
                    </div>
                    <div className="compare-scores">
                      {item.normalized_scores ? formatScoreList(item.normalized_scores) : "-"}
                    </div>
                    {item.warning && <div className="warning">{item.warning}</div>}
                    {item.error && <div className="warning">{item.error}</div>}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="panel metrics-panel">
          <div className="panel-header">
            <h2>Reports snapshot</h2>
            <p>Latest comparison and validation summaries from reports/.</p>
          </div>
          <div className="reports-grid">
            <div className="report-card">
              <h3>Latest comparison</h3>
              {reports?.comparison ? (
                <pre>{JSON.stringify(reports.comparison, null, 2)}</pre>
              ) : (
                <div className="empty-card">No comparison report found.</div>
              )}
            </div>
            <div className="report-card">
              <h3>Validation (per model)</h3>
              {reports?.validations && Object.keys(reports.validations).length > 0 ? (
                <pre>{JSON.stringify(reports.validations, null, 2)}</pre>
              ) : (
                <div className="empty-card">No validation reports found.</div>
              )}
            </div>
          </div>
        </section>

        <section className="panel log-panel">
          <div className="panel-header">
            <h2>Activity log</h2>
            <p>Recent actions from this session.</p>
          </div>
          <div className="log-list">
            {activityLog.length === 0 && <div className="empty-card">No activity yet.</div>}
            {activityLog.map((entry, index) => (
              <div key={`${entry.type}-${index}`} className="log-row">
                <span className="log-type">{entry.type}</span>
                <span className="log-meta">
                  {entry.label && `Label: ${entry.label}`} {entry.model && `Model: ${entry.model}`} 
                  {entry.modelCount && `Models: ${entry.modelCount}`} {entry.latency && `(${entry.latency} ms)`}
                </span>
                <span className="log-time">{entry.time}</span>
              </div>
            ))}
          </div>
        </section>
      </main>

      <footer>
        ASPICV pipeline deck · HOG + LBP + SVM · Local inference only
      </footer>
    </div>
  );
}
