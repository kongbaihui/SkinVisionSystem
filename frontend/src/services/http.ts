import axios from 'axios'
import { useAuthStore } from '@/stores/auth'

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim() || ''

export const http = axios.create({
  baseURL: apiBaseUrl,
  timeout: 90000,
})

// 请求拦截：附加 Bearer Token（在请求发起时访问 Pinia 实例）
http.interceptors.request.use((config) => {
  try {
    const store = useAuthStore()
    if (store.accessToken) {
      config.headers.Authorization = `Bearer ${store.accessToken}`
    }
  } catch {
    // Pinia 尚未初始化，跳过
  }
  return config
})

// 响应拦截：401 时尝试刷新 Token
http.interceptors.response.use(
  (response) => response,
  async (error) => {
    const isAuthEndpoint = (error.config?.url as string)?.includes('/api/v1/auth/')
    if (error.response?.status === 401 && !error.config?._retry && !isAuthEndpoint) {
      error.config._retry = true
      try {
        const store = useAuthStore()
        const newToken = await store.doRefresh()
        if (newToken) {
          error.config.headers.Authorization = `Bearer ${newToken}`
          return http(error.config)
        }
      } catch {
        // refresh 失败，跳转登录
      }
      if (!window.location.pathname.startsWith('/login')) {
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  },
)

export function getApiBaseUrl() {
  return apiBaseUrl
}
