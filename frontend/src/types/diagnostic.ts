export interface HealthResponse {
  auth_enabled?: boolean
  status: string
  pipeline_mode: string
}

export interface DiagnosticPatient {
  patient_age: number | null
  symptoms: string
  duration: string
}

export interface DiagnosticFeatures {
  area_ratio: number
  irregularity_index: number
  color_variance: number
  asymmetry_index?: number
  border_sharpness?: number
  color_diversity?: number
  max_diameter_ratio?: number
  [key: string]: number | undefined
}

export interface ClassificationResult {
  status: string
  label?: string | null
  candidates?: string[]
  scores?: Record<string, number>
  score_semantics?: string
  reason?: string
  source?: string
  model?: string
  prompt_version?: string
  version?: string
  revision?: string
  elapsed_ms?: number
  request_settings?: { view?: string }
}

export interface ReliabilityResult {
  status: string
  reason?: string
  version?: string
  mean_variance?: number
  p95_variance?: number
  elapsed_ms?: number
  extra_prediction_count?: number
  variance_image_path?: string
  variance_path?: string
  occlusion_image_path?: string
  occlusion?: { status: string; elapsed_ms?: number }
  semantics?: string
}

export interface AIResults {
  classification?: ClassificationResult | null
  reliability?: ReliabilityResult | null
  model_info?: { name?: string; execution?: string; device?: string; version?: string;
    weights_sha256?: string; explanation_status?: string; training_dataset?: string } | null
  risk_level?: string | null
  risk_source?: string | null
  original_path?: string | null
  patient_id?: string | null
  report_status?: string | null
  report_metadata?: { source?: string; model?: string; prompt_version?: string; variant?: string } | null
  overlay_path?: string | null
  probability_path?: string | null
}

export interface AnalyzeResponse extends AIResults {
  id: number
  user_id: number | null
  pipeline_mode: string
  patient: DiagnosticPatient
  features: DiagnosticFeatures
  mask_path: string
  roi_path: string
  inference_time_ms: number
  created_at: string
  status: string
}

export interface AnalyzePayload {
  image: File
  patientAge: number | null
  symptoms: string
  duration: string
}

// ─── SSE 流式分析 ────────────────────────────────────────────────────────────

export interface SSEMetaPayload extends AIResults {
  task_id: number
  pipeline_mode: string
  features: DiagnosticFeatures
  format: string
  llm_enabled: boolean
  gradcam_url?: string
  mask_path?: string
  roi_path?: string
}

// ─── 认证 ────────────────────────────────────────────────────────────────────

export interface AuthUser {
  id: number
  username: string
  email: string | null
  is_active: boolean
  created_at: string
  updated_at?: string
  role?: 'doctor' | 'admin'
}

export interface LoginRequest {
  username: string
  password: string
}

export interface LoginResponse {
  user: AuthUser
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
  refresh_expires_in: number
}

export interface RegisterRequest {
  username: string
  password: string
  email?: string
}

export interface RegisterResponse {
  id: number
  username: string
  email: string | null
  created_at: string
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
}

export interface RefreshResponse {
  access_token: string
  token_type: string
  expires_in: number
}

// ─── 任务历史 ─────────────────────────────────────────────────────────────────

export interface Pagination {
  page: number
  page_size: number
  total_count: number
  total_pages: number
}

export interface TaskSummary extends AIResults {
  id: number
  patient_age: number | null
  patient_symptoms: string
  patient_duration: string
  irregularity_index: number | null
  area_ratio: number | null
  report_version?: number
  status: string
  created_at: string
  updated_at: string
}

export interface TaskDetail extends AIResults {
  id: number
  user_id: number | null
  patient_age: number | null
  patient_symptoms: string
  patient_duration: string
  pipeline_mode: string
  features: DiagnosticFeatures
  mask_path: string | null
  roi_path: string | null
  gradcam_path: string | null
  llm_report: string | null
  edited_report?: string | null
  effective_report?: string | null
  report_version: number
  report_edited_at?: string | null
  status: string
  inference_time_ms: number | null
  request_id: string | null
  created_at: string
}

export interface TasksListResponse {
  data: TaskSummary[]
  pagination: Pagination
}

// ─── 工具函数 ─────────────────────────────────────────────────────────────────

export function riskClass(label?: string | null): string {
  if (label === '高风险') return 'risk-high'
  if (label === '中-高风险') return 'risk-medium-high'
  if (label === '中风险') return 'risk-medium'
  if (label === '低风险') return 'risk-low'
  return 'risk-unknown'
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}
