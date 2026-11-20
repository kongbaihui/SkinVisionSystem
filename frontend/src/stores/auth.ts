import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { getHealth } from '@/services/diagnostic'

import { getMe, login as loginApi, logout as logoutApi, refreshAccessToken, register as registerApi } from '@/services/auth'
import type { AuthUser, LoginRequest, RegisterRequest } from '@/types/diagnostic'

const REFRESH_TOKEN_KEY = 'sv_refresh_token'
const USER_KEY = 'sv_user'
const DEMO_REFRESH_TOKEN = 'demo_refresh_sentinel'
const DEMO_ADMIN_REFRESH_TOKEN = 'demo_admin_refresh_sentinel'

export const useAuthStore = defineStore('auth', () => {
  const accessToken = ref<string | null>(null)
  const user = ref<AuthUser | null>(null)

  const isAuthenticated = computed(() => !!accessToken.value)

  /** 从 localStorage 恢复刷新令牌与用户快照（页面刷新后调用） */
  function restoreSession() {
    try {
      const savedUser = localStorage.getItem(USER_KEY)
      if (savedUser) {
        user.value = JSON.parse(savedUser) as AuthUser
      }
    } catch {
      // 忽略
    }
  }

  function _persistUser(u: AuthUser | null) {
    user.value = u
    if (u) {
      localStorage.setItem(USER_KEY, JSON.stringify(u))
    } else {
      localStorage.removeItem(USER_KEY)
    }
  }

  function _persistRefresh(token: string | null) {
    if (token) {
      localStorage.setItem(REFRESH_TOKEN_KEY, token)
    } else {
      localStorage.removeItem(REFRESH_TOKEN_KEY)
    }
  }

  async function login(body: LoginRequest): Promise<void> {
    const demoName = body.username === 'doctor' || body.username === 'admin'
    if (demoName && body.password === '123456' && (await getHealth()).auth_enabled === false) {
      const admin = body.username === 'admin'
      accessToken.value = admin ? 'demo_admin_token' : 'demo_access_token'
      try {
        _persistUser(await getMe())
        _persistRefresh(admin ? DEMO_ADMIN_REFRESH_TOKEN : DEMO_REFRESH_TOKEN)
      } catch (err) {
        _clear()
        throw err
      }
      return
    }
    const res = await loginApi(body)
    accessToken.value = res.access_token
    _persistRefresh(res.refresh_token)
    _persistUser(res.user)
  }

  async function register(body: RegisterRequest): Promise<void> {
    const res = await registerApi(body)
    accessToken.value = res.access_token
    _persistRefresh(res.refresh_token)
    // register response 没有完整 user 对象，构造一个
    _persistUser({
      id: res.id,
      username: res.username,
      email: res.email,
      is_active: true,
      created_at: res.created_at,
    })
  }

  async function logout(): Promise<void> {
    const rt = localStorage.getItem(REFRESH_TOKEN_KEY) ?? undefined
    try {
      await logoutApi(rt)
    } catch {
      // 忽略服务端错误
    }
    accessToken.value = null
    _persistUser(null)
    _persistRefresh(null)
  }

  /** HTTP 拦截器调用 — 用 refresh_token 换新 access_token */
  async function doRefresh(): Promise<string | null> {
    const rt = localStorage.getItem(REFRESH_TOKEN_KEY)
    if (!rt) {
      _clear()
      return null
    }
    // 演示账号：直接从 localStorage 恢复 session，跳过后端刷新
    if (rt === DEMO_REFRESH_TOKEN || rt === DEMO_ADMIN_REFRESH_TOKEN) {
      try {
        if ((await getHealth()).auth_enabled === false) {
          accessToken.value = rt === DEMO_ADMIN_REFRESH_TOKEN ? 'demo_admin_token' : 'demo_access_token'
          return accessToken.value
        }
      } catch {
        // ignore
      }
      _clear()
      return null
    }
    try {
      const res = await refreshAccessToken(rt)
      accessToken.value = res.access_token
      return res.access_token
    } catch {
      _clear()
      return null
    }
  }

  /** 拉取当前用户信息（页面刷新后且有 refresh_token 时调用） */
  async function fetchMe(): Promise<void> {
    try {
      const me = await getMe()
      _persistUser(me)
    } catch {
      // token 失效
    }
  }

  function _clear() {
    accessToken.value = null
    _persistUser(null)
    _persistRefresh(null)
  }

  return {
    accessToken,
    user,
    isAuthenticated,
    restoreSession,
    login,
    register,
    logout,
    doRefresh,
    fetchMe,
  }
})
