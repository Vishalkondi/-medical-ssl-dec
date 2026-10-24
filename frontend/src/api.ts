import type { ApiErrorBody, ModelStatus, PredictionResponse } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public status?: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError(
      "BACKEND_UNREACHABLE",
      "Cannot reach the analysis server. Make sure the backend is running on port 8000.",
    );
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const err = (body as ApiErrorBody | null)?.error;
    if (!err && res.status >= 500) {
      throw new ApiError("BACKEND_UNREACHABLE", "The analysis server is not responding.", res.status);
    }
    throw new ApiError(err?.code ?? "HTTP_ERROR", err?.message ?? `Request failed (${res.status}).`, res.status);
  }
  return body as T;
}

export const getModelStatus = () => request<ModelStatus>("/model/status");

export function predict(file: File): Promise<PredictionResponse> {
  const form = new FormData();
  // Send a neutral filename: original names can contain patient identifiers.
  const ext = file.type === "image/png" ? "png" : "jpg";
  form.append("file", file, `upload.${ext}`);
  return request<PredictionResponse>("/predict", { method: "POST", body: form });
}
