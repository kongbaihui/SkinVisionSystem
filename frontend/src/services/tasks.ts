import { http } from '@/services/http'
import type { TasksListResponse, TaskDetail } from '@/types/diagnostic'

export interface TaskListParams {
  page?: number
  page_size?: number
  status?: string
  sort?: string
}

export async function getTasks(params: TaskListParams = {}): Promise<TasksListResponse> {
  const { data } = await http.get<TasksListResponse>('/api/v1/tasks', { params })
  return data
}

const pathFor = (id: number) => '/api/v1/tasks/' + id

export async function getTask(id: number): Promise<TaskDetail> {
  const { data } = await http.get<TaskDetail>(pathFor(id))
  return data
}

export async function deleteTask(id: number): Promise<void> {
  await http.delete(pathFor(id))
}

export async function saveReport(id: number, text: string, version: number): Promise<TaskDetail> {
  const { data } = await http.patch<TaskDetail>(pathFor(id) + '/report', { text, version })
  return data
}

export async function retryReport(id: number): Promise<TaskDetail> {
  const { data } = await http.post<TaskDetail>(pathFor(id) + '/report/retry', null, { timeout: 0 })
  return data
}

export async function exportTask(id: number, format: 'markdown' | 'pdf'): Promise<Blob> {
  const { data } = await http.get<Blob>(pathFor(id) + '/export', {
    params: { format }, responseType: 'blob',
  })
  return data
}

export const stateLabels: Record<string, string> = {
  completed: '已完成', failed: '失败', processing: '处理中', not_evaluated: '未评估',
  not_available: '不可用', not_configured: '未配置', review_required: '待人工复核',
  local_template: '本地模板', interrupted: '已中断', not_requested: '未请求', skipped: '已跳过',
}
export const stateLabel = (value?: string | null) => value ? stateLabels[value] ?? value : '未记录'
