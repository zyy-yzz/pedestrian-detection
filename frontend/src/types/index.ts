export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Detection {
  id: number;
  bbox: BBox;
  confidence: number;
  class_name: string;
  track_id?: number;
}

export interface DetectionResponse {
  job_id: string;
  status: string;
  pedestrian_count: number;
  processing_time_ms: number;
  detections: Detection[];
  annotated_image?: string;
}

export interface CompareResult {
  delta_count: number;
  delta_time_ms: number;
  avg_conf_enhanced: number;
  avg_conf_vanilla: number;
}

export interface CompareDetectionResponse {
  job_id: string;
  status: string;
  enhancements_used: string[];
  enhanced: DetectionResponse;
  vanilla: DetectionResponse;
  comparison: CompareResult;
}

export type EnhancementOption =
  | 'clahe'
  | 'denoise'
  | 'histogram_eq'
  | 'sharpen'
  | 'gamma_correct'
  | 'adaptive_threshold';

export const ENHANCEMENT_LABELS: Record<EnhancementOption, string> = {
  clahe: 'CLAHE 对比度增强',
  denoise: '双边滤波去噪',
  histogram_eq: '直方图均衡化',
  sharpen: '锐化增强',
  gamma_correct: 'Gamma 校正',
  adaptive_threshold: '自适应阈值',
};

export const ALL_ENHANCEMENTS: EnhancementOption[] = [
  'clahe', 'denoise', 'histogram_eq', 'sharpen', 'gamma_correct', 'adaptive_threshold',
];

export interface DetectionConfig {
  confThreshold: number;
  iouThreshold: number;
  enhancements: EnhancementOption[];
  compare: boolean;
}

export interface DetectionJob {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  input_type?: 'image' | 'video' | 'stream';
  pedestrian_count?: number;
  processing_time_ms?: number;
  detections?: Detection[];
  annotated_image?: string;
  annotated_image_url?: string;
  created_at?: string;
  completed_at?: string;
  error_message?: string;
}

export interface StreamFrame {
  frame_index: number;
  pedestrian_count: number;
  detections: Array<{ bbox: BBox; confidence: number; class_id: number; track_id?: number }>;
  inference_ms: number;
}

export interface HistoryEntry {
  job_id: string;
  input_type: string;
  original_filename: string | null;
  status: string;
  pedestrian_count: number | null;
  created_at: string | null;
}