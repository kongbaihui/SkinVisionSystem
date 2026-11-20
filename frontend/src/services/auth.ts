import { http } from '@/services/http'
import type {
  AuthUser,
  LoginRequest,
  LoginResponse,
  RefreshResponse,
  RegisterRequest,
  RegisterResponse,
} from '@/types/diagnostic'

export async function register(body: RegisterRequest): Promise<RegisterResponse> {
  const { data } = await http.post<RegisterResponse>('/api/v1/auth/register', body)
  return data
}

export async function login(body: LoginRequest): Promise<LoginResponse> {
  const { data } = await http.post<LoginResponse>('/api/v1/auth/login', body)
  return data
}

export async function logout(refreshToken?: string): Promise<void> {
  await http.post('/api/v1/auth/logout', { refresh_token: refreshToken ?? null })
}

export async function getMe(): Promise<AuthUser> {
  const { data } = await http.get<AuthUser>('/api/v1/auth/me')
  return data
}

export async function refreshAccessToken(refreshToken: string): Promise<RefreshResponse> {
  const { data } = await http.post<RefreshResponse>('/api/v1/auth/refresh', {
    refresh_token: refreshToken,
  })
  return data
}
