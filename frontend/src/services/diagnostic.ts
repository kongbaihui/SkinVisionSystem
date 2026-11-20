import { http, getApiBaseUrl } from '@/services/http'
import type { HealthResponse, SSEMetaPayload, ClassificationResult, ReliabilityResult, AIResults } from '@/types/diagnostic'

export async function getHealth() {
  const { data } = await http.get<HealthResponse>('/health')
  return data
}

type EventData = {
  meta: SSEMetaPayload
  classification: ClassificationResult
  reliability: ReliabilityResult
  report_status: { status: string; metadata?: AIResults['report_metadata'] }
  chunk: { text: string }
  error: { module?: string; message: string; task_id?: number }
  done: { task_id: number }
}
export type AnalysisEvent = { [K in keyof EventData]: { type: K; data: EventData[K] } }[keyof EventData]

export async function analyzeStream(form: FormData, token: string | null, signal: AbortSignal, receive: (event: AnalysisEvent) => void) {
  const response = await fetch(getApiBaseUrl() + '/api/v1/diagnostic/analyze_stream', {
    method: 'POST', body: form, signal, headers: token ? { Authorization: 'Bearer ' + token } : {},
  })
  if (!response.ok || !response.body) {
    const issue = await response.json().catch(() => ({}))
    if (response.status === 401) throw new Error('登录已过期，请重新登录。')
    if (response.status === 413) throw new Error('上传图像超过大小限制。')
    if (issue.error === 'invalid image file') throw new Error('图片无法解码，请选择有效的 JPG 或 PNG 图像。')
    throw new Error(typeof issue.detail === 'string' ? issue.detail : '分析请求失败（HTTP ' + response.status + '）')
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let completed = false
  try {
    while (true) {
      const { value, done } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })
      const blocks = buffer.split(/\r?\n\r?\n/)
      buffer = blocks.pop() || ''
      for (const block of blocks) {
        const lines = block.split(/\r?\n/)
        const type = lines.find(line => line.startsWith('event:'))?.slice(6).trim()
        const payload = lines.filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n')
        if (!type || !payload) continue
        receive({ type, data: JSON.parse(payload) } as AnalysisEvent)
        if (type === 'done') completed = true
      }
      if (done) break
    }
    if (!completed) throw new Error('分析连接已中断，请核对已保存的模块状态。')
  } finally { reader.releaseLock() }
}
