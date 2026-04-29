export type PredictionClass = 'REAL' | 'AI_GENERATED' | 'EDITED';

export interface BlindDetectionResult {
  prediction: PredictionClass;
  confidence: number;
  forgery_type: string;
  heatmap_url: string | null;
  class_probabilities: {
    REAL: number;
    AI_GENERATED: number;
    EDITED: number;
  };
  analysis: {
    cnn_score:     number;
    fft_score:     number;
    gan_score:     number;
    noise_score:   number;
    texture_score: number;
  };
}

export interface ComparativeResult {
  similarity_score: number;
  is_manipulated: boolean;
  diff_map_url: string | null;
  highlighted_url: string | null;
  regions_count: number;
}

export type DetectionStatus = 'idle' | 'uploading' | 'analyzing' | 'done' | 'error';
