import type { ModelStatus, PredictionResponse } from "../types";
import { AlertIcon, CheckIcon, InfoIcon } from "./Icons";

export type Phase = "idle" | "loading" | "done" | "error";

interface Props {
  phase: Phase;
  result: PredictionResponse | null;
  error: { code: string; message: string } | null;
  status: ModelStatus | null;
  statusError: string | null;
  hasFile: boolean;
}

export default function ResultPanel({ phase, result, error, status, statusError, hasFile }: Props) {
  return (
    <section className="card card--result" aria-labelledby="result-title" aria-live="polite" aria-busy={phase === "loading"}>
      <div className="card__head">
        <span className="step">3</span>
        <h2 id="result-title">Research output</h2>
      </div>
      {phase === "loading" && <LoadingState />}
      {phase === "error" && error && <ErrorState code={error.code} message={error.message} />}
      {phase === "done" && result && <Result r={result} />}
      {phase === "idle" && <IdleState status={status} statusError={statusError} hasFile={hasFile} />}
    </section>
  );
}

function IdleState({ status, statusError, hasFile }: { status: ModelStatus | null; statusError: string | null; hasFile: boolean }) {
  if (statusError) return <BackendOffline />;
  if (status && !status.loaded) {
    return (
      <div className="state state--warn">
        <InfoIcon width={28} height={28} />
        <h3>No trained model yet</h3>
        <p>
          The interface and API are ready, but no model checkpoint has been trained. Predictions will appear here
          after the supervised baseline (Phase 4) or SSLDEC model (Phase 5) is exported to <code>artifacts/</code>.
        </p>
        <p className="muted">You can still upload an image to test validation.</p>
      </div>
    );
  }
  return (
    <div className="state">
      <div className="state__art" aria-hidden>
        <span />
        <span />
        <span />
      </div>
      <h3>{hasFile ? "Ready to analyze" : "Upload an image to begin"}</h3>
      <p>The model output, its score and how to interpret it will appear here.</p>
    </div>
  );
}

function BackendOffline() {
  return (
    <div className="state state--error">
      <AlertIcon width={28} height={28} />
      <h3>Analysis server is offline</h3>
      <p>Start the backend from the project folder, then refresh this page:</p>
      <pre className="code">uvicorn backend.main:app --port 8000</pre>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="skeleton" aria-label="Analyzing image">
      <div className="skeleton__line skeleton__line--lg" />
      <div className="skeleton__line" />
      <div className="skeleton__bar" />
      <div className="skeleton__bar" />
      <div className="skeleton__line skeleton__line--sm" />
    </div>
  );
}

function ErrorState({ code, message }: { code: string; message: string }) {
  if (code === "BACKEND_UNREACHABLE") return <BackendOffline />;
  const title =
    code === "MODEL_NOT_LOADED"
      ? "Image accepted, but no model is loaded"
      : code === "INVALID_IMAGE" || code === "UNSUPPORTED_TYPE" || code === "EMPTY_FILE" || code === "FILE_TOO_LARGE"
        ? "This image can't be analyzed"
        : "Something went wrong";
  return (
    <div className={`state ${code === "MODEL_NOT_LOADED" ? "state--warn" : "state--error"}`} role="alert">
      {code === "MODEL_NOT_LOADED" ? <InfoIcon width={28} height={28} /> : <AlertIcon width={28} height={28} />}
      <h3>{title}</h3>
      <p>{message}</p>
      <p className="muted mono">{code}</p>
    </div>
  );
}

function Result({ r }: { r: PredictionResponse }) {
  const positive = r.predicted_label !== "no_finding";
  return (
    <div className="result">
      <div className={`verdict ${positive ? "verdict--flag" : "verdict--clear"}`}>
        {positive ? <AlertIcon width={26} height={26} /> : <CheckIcon width={26} height={26} />}
        <div>
          <div className="verdict__eyebrow">Model output</div>
          <div className="verdict__label">{r.predicted_display_name}</div>
        </div>
      </div>

      <div className="scores">
        {r.scores.map((s) => {
          const isPos = s.label !== "no_finding";
          return (
            <div className="score" key={s.label}>
              <div className="score__row">
                <span>{s.display_name}</span>
                <span className="mono">{s.score.toFixed(2)}</span>
              </div>
              <div className="score__track">
                <div
                  className={`score__fill ${isPos ? "score__fill--pos" : ""}`}
                  style={{ width: `${Math.round(s.score * 100)}%` }}
                />
                {isPos && (
                  <div className="score__threshold" style={{ left: `${r.threshold * 100}%` }} title={`Decision threshold ${r.threshold}`} />
                )}
              </div>
            </div>
          );
        })}
        <p className="score-note">
          {r.score_type === "uncalibrated"
            ? "Scores are uncalibrated model outputs between 0 and 1. They are not the probability that a disease is present."
            : "Scores were calibrated on validation data from the training population. They may not transfer to other hospitals or scanners."}{" "}
          The marker shows the decision threshold ({r.threshold.toFixed(2)}) chosen on the validation set.
        </p>
      </div>

      <dl className="facts">
        <div>
          <dt>Model</dt>
          <dd className="mono">{r.model_version}</dd>
        </div>
        <div>
          <dt>Inference</dt>
          <dd className="mono">{r.inference_ms} ms</dd>
        </div>
        <div>
          <dt>Input</dt>
          <dd className="mono">
            {r.image_width}×{r.image_height}
          </dd>
        </div>
      </dl>

      <details className="explain">
        <summary>How to read this result</summary>
        <ul>
          <li>
            "Nodule or mass" is a <strong>radiographic finding</strong> learned from report-mined labels. It is not a
            cancer diagnosis, and most nodules are benign.
          </li>
          <li>"No finding labelled" does not rule out disease. Small or subtle lesions are often missed (false negatives).</li>
          <li>The model can flag normal images (false positives), especially images unlike its training data.</li>
          <li>Only frontal chest X-rays similar to NIH ChestX-ray14 are in scope. Other images give meaningless outputs.</li>
        </ul>
      </details>
    </div>
  );
}
