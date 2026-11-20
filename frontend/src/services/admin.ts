import { http } from '@/services/http'
import type { Pagination } from '@/types/diagnostic'

export interface AdminStats {
  total_tasks: number
  completed_tasks: number
  failed_tasks: number
  processing_tasks: number
  total_users: number
  active_users: number
  avg_inference_ms: number
  pipeline_mode: string
}

export interface AdminUser {
  id: number
  username: string
  email: string | null
  role: string
  is_active: boolean
  last_login: string | null
  created_at: string
}

export interface AdminAuditLog {
  id: number
  user_id: number | null
  username: string
  action: string
  ip_address: string | null
  created_at: string
}

export interface AdminTask {
  risk_level?: string | null
  risk_source?: string | null
  report_status?: string | null
  id: number
  user_id: number | null
  username: string
  patient_id: string | null
  patient_age: number | null
  irregularity_index: number | null
  area_ratio: number | null
  status: string
  pipeline_mode: string | null
  inference_time_ms: number | null
  created_at: string | null
}

export interface PagedResponse<T> {
  data: T[]
  pagination: Pagination
}

export async function getAdminStats(): Promise<AdminStats> {
  const { data } = await http.get<AdminStats>('/api/v1/admin/stats')
  return data
}

export async function getAdminUsers(params: {
  page?: number
  page_size?: number
} = {}): Promise<PagedResponse<AdminUser>> {
  const { data } = await http.get<PagedResponse<AdminUser>>('/api/v1/admin/users', { params })
  return data
}

export async function toggleUserStatus(userId: number): Promise<{ id: number; is_active: boolean }> {
  const { data } = await http.patch<{ id: number; is_active: boolean }>(
    `/api/v1/admin/users/${userId}/status`
  )
  return data
}

export async function updateUserRole(userId: number, role: 'doctor' | 'admin'): Promise<{ id: number; role: string }> {
  const { data } = await http.patch<{ id: number; role: string }>(
    `/api/v1/admin/users/${userId}/role`,
    { role }
  )
  return data
}

export async function getAdminLogs(params: {
  page?: number
  page_size?: number
  action?: string
} = {}): Promise<PagedResponse<AdminAuditLog>> {
  const { data } = await http.get<PagedResponse<AdminAuditLog>>('/api/v1/admin/logs', { params })
  return data
}

export async function getAdminTasks(params: {
  page?: number
  page_size?: number
  status?: string
} = {}): Promise<PagedResponse<AdminTask>> {
  const { data } = await http.get<PagedResponse<AdminTask>>('/api/v1/admin/tasks', { params })
  return data
}
