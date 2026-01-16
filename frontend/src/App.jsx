import { useEffect, useMemo, useRef, useState } from "react";

const formatScoreList = (scores) => {
  if (!scores || typeof scores !== "object") return "-";
  return Object.entries(scores)
    .sort((a, b) => b[1] - a[1])
    .map(([label, value]) => `${label}: ${Number(value).toFixed(3)}`)
    .join(" | ");
};

const formatPercent = (value) => {
  if (typeof value !== "number") return "-";
  return `${(value * 100).toFixed(1)}%`;
};

export default function App() {
  const apiBase = useMemo(
    () => (import.meta.env.VITE_ASPICV_API_BASE || "http://localhost:8000").replace(/\/$/, ""),
    []
  );
  const mirQuickQueries = useMemo(
    () => ["a happy face", "a surprised face", "a neutral face", "a sad face"],
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
  const [mirReady, setMirReady] = useState(null);
  const [mirProgress, setMirProgress] = useState(null);
  const [mirStatusError, setMirStatusError] = useState(null);
  const [mirIndexSize, setMirIndexSize] = useState(null);

  const [imagePath, setImagePath] = useState("");
  const [imageBase64, setImageBase64] = useState("");
  const [previewUrl, setPreviewUrl] = useState("");
  const [previewName, setPreviewName] = useState("No image selected");
  const previewUrlRef = useRef(null);
  const mirAbortRef = useRef(null);
  const mirStoredModelRef = useRef(null);

  const [predictStatus, setPredictStatus] = useState("idle");
  const [predictError, setPredictError] = useState(null);
  const [predictResult, setPredictResult] = useState(null);

  const [compareStatus, setCompareStatus] = useState("idle");
  const [compareError, setCompareError] = useState(null);
  const [compareResults, setCompareResults] = useState([]);

  const [reports, setReports] = useState(null);
  const [activityLog, setActivityLog] = useState([]);
  const [mirQuery, setMirQuery] = useState("");
  const [mirDebouncedQuery, setMirDebouncedQuery] = useState("");
  const [mirResults, setMirResults] = useState([]);
  const [mirStatus, setMirStatus] = useState("idle");
  const [mirError, setMirError] = useState(null);
  const [mirLatencyMs, setMirLatencyMs] = useState(null);
  const [mirExplainStatus, setMirExplainStatus] = useState("idle");
  const [mirExplainError, setMirExplainError] = useState(null);
  const [mirExplanation, setMirExplanation] = useState("");
  const [mirExplainMeta, setMirExplainMeta] = useState(null);
  const [mirOllamaModels, setMirOllamaModels] = useState([]);
  const [mirSelectedModel, setMirSelectedModel] = useState("");
  const [mirOllamaError, setMirOllamaError] = useState(null);

  const resolveImageUrl = (url) => {
    if (!url) return "";
    if (url.startsWith("/")) {
      return `${apiBase}${url}`;
    }
    return url;
  };

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

  const requestMirExplanation = () => {
    if (!mirDebouncedQuery || mirResults.length === 0 || mirExplainStatus === "loading") {
      return;
    }
    if (mirAbortRef.current) {
      mirAbortRef.current.abort();
    }

    const controller = new AbortController();
    mirAbortRef.current = controller;
    setMirExplainStatus("loading");
    setMirExplainError(null);
    setMirExplainMeta(null);

    const payload = {
      query: mirDebouncedQuery,
      model: mirSelectedModel || undefined,
      results: mirResults.slice(0, 6).map((item) => ({
        file_name: item.file_name,
        score: item.score,
        caption: item.caption || null,
        image_url: item.image_url,
      })),
    };

    fetch(`${apiBase}/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Explain failed (${res.status})`);
        }
        return res.json();
      })
      .then((data) => {
        if (data.error) {
          throw new Error(data.error);
        }
        const message = data.explanation || data.answer || data.summary || data.message || "";
        if (!message) {
          throw new Error("Raspunsul de la Ollama nu contine explicatie.");
        }
        setMirExplanation(message);
        setMirExplainStatus("ready");
        setMirExplainMeta({
          model: data.model || data.ollama_model || null,
          duration_ms: data.duration_ms || data.latency_ms || null,
          prompt_tokens: data.prompt_tokens || null,
          completion_tokens: data.completion_tokens || null,
        });
      })
      .catch((err) => {
        if (err.name === "AbortError") return;
        setMirExplainError(err.message);
        setMirExplainStatus("error");
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
        setMirReady(data.mir ? data.mir.ready : null);
        setMirProgress(data.mir ? data.mir.progress : null);
        setMirStatusError(data.mir ? data.mir.error : null);
        setMirIndexSize(data.mir ? data.mir.index_size : null);
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
        if (data.mir_progress) {
          setMirProgress(data.mir_progress);
        }
        if (typeof data.mir_ready === "boolean") {
          setMirReady(data.mir_ready);
        }
        if (data.mir_error) {
          setMirStatusError(data.mir_error);
        }
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
    const timeout = setTimeout(() => {
      setMirDebouncedQuery(mirQuery.trim());
    }, 350);
    return () => clearTimeout(timeout);
  }, [mirQuery]);

  useEffect(() => {
    let active = true;
    const stored = window.localStorage.getItem("mir_ollama_model");
    mirStoredModelRef.current = stored;

    fetch(`${apiBase}/ollama/models`)
      .then((res) => res.json())
      .then((data) => {
        if (!active) return;
        if (data.error) {
          setMirOllamaError(data.error);
          return;
        }
        const models = Array.isArray(data.models) ? data.models : [];
        setMirOllamaModels(models);
        const storedModel =
          mirStoredModelRef.current && models.includes(mirStoredModelRef.current)
            ? mirStoredModelRef.current
            : null;
        const defaultModel =
          data.default && models.includes(data.default) ? data.default : models[0];
        setMirSelectedModel(storedModel || defaultModel || "");
      })
      .catch((err) => {
        if (!active) return;
        setMirOllamaError(err.message);
      });
    return () => {
      active = false;
    };
  }, [apiBase]);

  useEffect(() => {
    if (!mirSelectedModel) return;
    window.localStorage.setItem("mir_ollama_model", mirSelectedModel);
  }, [mirSelectedModel]);

  useEffect(() => {
    if (!mirDebouncedQuery) {
      setMirResults([]);
      setMirStatus("idle");
      setMirError(null);
      setMirExplanation("");
      setMirExplainStatus("idle");
      setMirExplainError(null);
      setMirExplainMeta(null);
      setMirLatencyMs(null);
      return;
    }

    let active = true;
    setMirStatus("loading");
    setMirError(null);
    setMirExplanation("");
    setMirExplainStatus("idle");
    setMirExplainError(null);
    setMirExplainMeta(null);
    setMirLatencyMs(null);

    const params = new URLSearchParams({ query: mirDebouncedQuery, top_k: "6" });
    fetch(`${apiBase}/search?${params.toString()}`)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Search failed (${res.status})`);
        }
        return res.json();
      })
      .then((data) => {
        if (!active) return;
        if (data.error) {
          setMirError(data.error);
          setMirResults([]);
          setMirStatus("error");
          return;
        }
        setMirResults(data.results || []);
        setMirLatencyMs(typeof data.latency_ms === "number" ? data.latency_ms : null);
        setMirStatus("ready");
      })
      .catch((err) => {
        if (!active) return;
        setMirError(err.message);
        setMirResults([]);
        setMirStatus("error");
      });

    return () => {
      active = false;
    };
  }, [apiBase, mirDebouncedQuery]);

  useEffect(() => {
    if (!mirDebouncedQuery) return;
    if (mirAbortRef.current) {
      mirAbortRef.current.abort();
    }
  }, [mirDebouncedQuery]);

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
          <h1>Facial emotion inference with classic models and CNNs.</h1>
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
            <p>Live visibility into models, datasets, MIR index, and RAG availability.</p>
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
            <div className="health-card">
              <div className="health-title">MIR index</div>
              <div className="health-value">
                {mirReady ? "Ready" : mirProgress?.stage === "error" ? "Error" : "Loading"}
              </div>
              <div className="health-meta">
                <div>
                  {mirStatusError
                    ? mirStatusError
                    : mirProgress?.message || "Waiting for MIR initialization"}
                </div>
                <div>Index size: {typeof mirIndexSize === "number" ? mirIndexSize : "-"}</div>
                <div>Last query: {mirLatencyMs !== null ? `${mirLatencyMs} ms` : "-"}</div>
              </div>
            </div>
            <div className="health-card">
              <div className="health-title">RAG (Ollama)</div>
              <div className="health-value">
                {mirOllamaModels.length > 0 ? "Available" : "Offline"}
              </div>
              <div className="health-meta">
                <div>
                  {mirOllamaError
                    ? mirOllamaError
                    : mirOllamaModels.length > 0
                      ? `${mirOllamaModels.length} model(s)`
                      : "Start Ollama to enable explanations"}
                </div>
                <div>Selected: {mirSelectedModel || "-"}</div>
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

        <section className="panel mir-panel">
          <div className="panel-header">
            <h2>CLIP + FAISS retrieval</h2>
            <p>
              Explore the ASPICV dataset with semantic text search and generate
              explanations with the RAG layer.
            </p>
          </div>
          <div className="mir-grid">
            <div className="mir-controls">
              <label htmlFor="mirQuery">Text query</label>
              <input
                id="mirQuery"
                type="text"
                value={mirQuery}
                onChange={(event) => setMirQuery(event.target.value)}
                placeholder="a happy face with bright eyes"
              />
              <div className="mir-quick">
                {mirQuickQueries.map((item) => (
                  <button
                    key={item}
                    type="button"
                    className="ghost mir-chip"
                    onClick={() => setMirQuery(item)}
                  >
                    {item}
                  </button>
                ))}
              </div>

              <div className="mir-status">
                <div className="status-row">
                  <span className="status-label">MIR index</span>
                  <span
                    className={`status-pill ${
                      mirReady === true ? "ready" : mirReady === false ? "offline" : "pending"
                    }`}
                  >
                    {mirReady === null && "checking"}
                    {mirReady === true && "ready"}
                    {mirReady === false && "offline"}
                  </span>
                </div>
                {mirProgress && mirProgress.stage !== "ready" && (
                  <div className="progress-card">
                    <div className="progress-head">
                      <span>{mirProgress.message || "Indexing"}</span>
                      <span>{mirProgress.percent || 0}%</span>
                    </div>
                    <div className="progress-track">
                      <div
                        className="progress-fill"
                        style={{ width: `${mirProgress.percent || 0}%` }}
                      />
                    </div>
                  </div>
                )}
                {mirLatencyMs !== null && (
                  <div className="status-note">Last search: {mirLatencyMs} ms</div>
                )}
                {mirStatusError && <div className="status-error">{mirStatusError}</div>}
              </div>

              <div className="panel-actions">
                <div className="model-select">
                  <label htmlFor="mirModelSelect">RAG model</label>
                  <select
                    id="mirModelSelect"
                    value={mirSelectedModel}
                    onChange={(event) => setMirSelectedModel(event.target.value)}
                  >
                    {mirOllamaModels.length === 0 && <option value="">No model</option>}
                    {mirOllamaModels.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </select>
                </div>
                <button
                  type="button"
                  className="primary"
                  onClick={requestMirExplanation}
                  disabled={mirExplainStatus === "loading" || mirResults.length === 0}
                >
                  {mirExplainStatus === "loading" ? "Explaining..." : "Explain with RAG"}
                </button>
              </div>

              {mirOllamaError && <div className="warning">{mirOllamaError}</div>}
              {mirExplainError && <div className="warning">{mirExplainError}</div>}
              <div className="mir-explain-card">
                {mirExplainStatus === "idle" && (
                  <div className="empty-card">Run a search, then request an explanation.</div>
                )}
                {mirExplainStatus === "ready" && (
                  <pre>{mirExplanation}</pre>
                )}
                {mirExplainMeta?.model && (
                  <div className="mir-meta">
                    Model: {mirExplainMeta.model}
                  </div>
                )}
              </div>
            </div>

            <div className="mir-results">
              <div className="status-line">
                {mirStatus === "idle" && "Type a query to start searching."}
                {mirStatus === "loading" && "Searching the CLIP index..."}
                {mirStatus === "error" && `Error: ${mirError}`}
                {mirStatus === "ready" && `${mirResults.length} results.`}
              </div>
              <div className="mir-results-grid">
                {mirResults.length === 0 && (
                  <div className="empty-card">No results yet.</div>
                )}
                {mirResults.map((item) => (
                  <div
                    key={`${item.file_name}-${item.rank}`}
                    className="mir-result-card"
                  >
                    <div className="mir-thumb">
                      {item.image_url ? (
                        <img
                          src={resolveImageUrl(item.image_url)}
                          alt={item.caption || item.label}
                          loading="lazy"
                        />
                      ) : (
                        <div className="preview-empty">No image</div>
                      )}
                    </div>
                    <div className="mir-result-meta">
                      <strong>{item.label || "unknown"}</strong>
                      <span>{formatPercent(item.score)}</span>
                      <span>{item.file_name}</span>
                    </div>
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
