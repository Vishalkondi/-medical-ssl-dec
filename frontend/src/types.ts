// Mirrors backend/schemas.py. Keep both files in sync.

export type ScoreType = "uncalibrated" | "calibrated";

export interface ModelStatus {
  loaded: boolean;
  message: string;
  task: string;
  classes: string[];
  model_version: string | null;
  architecture: string | null;
  score_type: ScoreType | null;
  metrics: Record<string, number>;
  max_upload_mb: number;
}

export interface ClassScore {
  label: string;
  display_name: string;
  score: number;
}

export interface PredictionResponse {
  predicted_label: string;
  predicted_display_name: string;
  scores: ClassScore[];
  score_type: ScoreType;
  threshold: number;
  model_version: string;
  inference_ms: number;
  image_width: number;
  image_height: number;
  disclaimer: string;
}

export interface ApiErrorBody {
  error: { code: string; message: string };
}
