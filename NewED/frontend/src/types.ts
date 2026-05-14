export type ReferenceDrawing = {
  id: number;
  filename: string;
  drawing_type: string;
  created_at: string;
};

export type Submission = {
  id: number;
  reference_id: number;
  student_id: string;
  filename: string;
  status: string;
};

export type Job = {
  id: string;
  reference_id: number;
  status: string;
  processed: number;
  total: number;
  error?: string | null;
};

export type Result = {
  id: number;
  submission_id: number;
  student_id: string;
  filename: string;
  score: number;
  percentage: number;
  max_marks: number;
  feedback: string;
  errors: string[];
  metrics: {
    category_scores: Record<string, number>;
    [key: string]: unknown;
  };
  overlay_url?: string | null;
  heatmap_url?: string | null;
  created_at: string;
};
