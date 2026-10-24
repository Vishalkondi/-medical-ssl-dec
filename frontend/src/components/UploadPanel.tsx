import { useRef, useState, type DragEvent } from "react";
import { ACCEPTED_TYPES, formatBytes } from "../validation";
import { AlertIcon, ScanIcon, UploadIcon, XIcon } from "./Icons";

interface Props {
  file: File | null;
  previewUrl: string | null;
  dims: { w: number; h: number } | null;
  fileError: string | null;
  maxMb: number;
  busy: boolean;
  canAnalyze: boolean;
  onFile: (f: File) => void;
  onClear: () => void;
  onDims: (d: { w: number; h: number }) => void;
  onAnalyze: () => void;
}

export default function UploadPanel(p: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) p.onFile(f);
  };

  return (
    <section className="card" aria-labelledby="upload-title">
      <div className="card__head">
        <span className="step">1</span>
        <h2 id="upload-title">Chest X-ray image</h2>
      </div>

      {!p.previewUrl ? (
        <label
          className={`dropzone ${dragging ? "is-dragging" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED_TYPES.join(",")}
            className="visually-hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) p.onFile(f);
              e.target.value = "";
            }}
          />
          <span className="dropzone__icon">
            <UploadIcon width={26} height={26} />
          </span>
          <span className="dropzone__title">Drop an image here or <u>browse</u></span>
          <span className="dropzone__hint">
            Frontal (PA/AP) chest X-ray · PNG or JPEG · up to {p.maxMb} MB
          </span>
        </label>
      ) : (
        <figure className="preview">
          <div className="preview__frame">
            <img
              src={p.previewUrl}
              alt="Uploaded chest X-ray preview"
              onLoad={(e) => p.onDims({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
            />
            {p.busy && <div className="preview__scan" aria-hidden />}
          </div>
          <figcaption className="preview__meta">
            <span>
              {p.dims ? `${p.dims.w} × ${p.dims.h}px` : "…"} · {p.file && formatBytes(p.file.size)}
            </span>
            <span className="preview__actions">
              <button className="btn btn--ghost btn--sm" onClick={() => inputRef.current?.click()} disabled={p.busy}>
                Replace
              </button>
              <button className="btn btn--ghost btn--sm" onClick={p.onClear} disabled={p.busy} aria-label="Remove image">
                <XIcon width={16} height={16} /> Remove
              </button>
            </span>
          </figcaption>
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED_TYPES.join(",")}
            className="visually-hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) p.onFile(f);
              e.target.value = "";
            }}
          />
        </figure>
      )}

      {p.fileError && (
        <p className="inline-error" role="alert">
          <AlertIcon width={16} height={16} /> {p.fileError}
        </p>
      )}

      <div className="card__head card__head--spaced">
        <span className="step">2</span>
        <h2>Research task</h2>
      </div>
      <div className="task-options" role="radiogroup" aria-label="Research task">
        <div className="task-option is-selected" role="radio" aria-checked="true" tabIndex={0}>
          <span className="task-option__title">Classification</span>
          <span className="task-option__desc">No finding vs nodule / mass</span>
        </div>
        <div className="task-option is-disabled" role="radio" aria-checked="false" aria-disabled="true">
          <span className="task-option__title">
            Segmentation <span className="tag">Planned</span>
          </span>
          <span className="task-option__desc">Needs pixel-level masks</span>
        </div>
      </div>

      <button className="btn btn--primary btn--block" onClick={p.onAnalyze} disabled={!p.canAnalyze}>
        {p.busy ? (
          <>
            <span className="spinner" aria-hidden /> Analyzing…
          </>
        ) : (
          <>
            <ScanIcon width={18} height={18} /> Analyze image
          </>
        )}
      </button>
    </section>
  );
}
