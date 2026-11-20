<script setup lang="ts">
import { ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()

const username = ref('')
const password = ref('')
const loading = ref(false)
const errorMsg = ref('')

function validate(): string {
  if (!username.value.trim()) return '请输入用户名'
  if (!password.value) return '请输入密码'
  return ''
}

async function handleSubmit() {
  errorMsg.value = validate()
  if (errorMsg.value) return

  loading.value = true
  try {
    await authStore.login({ username: username.value.trim(), password: password.value })
    const role = authStore.user?.role
    const redirect = role === 'admin' ? '/admin' : ((route.query.redirect as string) || '/')
    await router.push(redirect)
  } catch (err: unknown) {
    const detail = (err as { response?: { data?: { detail?: { message?: string } | string } } })?.response?.data?.detail
    if (typeof detail === 'object' && detail?.message) {
      errorMsg.value = detail.message
    } else if (typeof detail === 'string') {
      errorMsg.value = detail
    } else {
      errorMsg.value = '登录失败，请检查用户名和密码'
    }
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <section class="auth-page">
    <article class="panel auth-card">
      <div class="login-title-area">
        <p class="auth-brand">智肤研析</p>
        <h2>登录账户</h2>
        <p class="subtle login-subtitle">进入皮肤病灶分析工作台</p>
      </div>

      <form class="auth-form" @submit.prevent="handleSubmit">
        <div class="form-field">
          <label class="form-label" for="login-username">用户名</label>
          <input
            id="login-username"
            v-model="username"
            class="form-input"
            type="text"
            placeholder="请输入用户名"
            autocomplete="username"
            :disabled="loading"
          />
        </div>

        <div class="form-field">
          <label class="form-label" for="login-password">密码</label>
          <input
            id="login-password"
            v-model="password"
            class="form-input"
            type="password"
            placeholder="请输入密码"
            autocomplete="current-password"
            :disabled="loading"
          />
        </div>

        <div v-if="errorMsg" class="form-error-box">{{ errorMsg }}</div>

        <button class="btn-primary" type="submit" :disabled="loading">
          {{ loading ? '登录中…' : '登录' }}
        </button>
      </form>

      <p class="auth-link">还没有账号？<RouterLink to="/register">注册账号</RouterLink></p>
      <p class="compliance-notice">
        本系统为皮肤病变智能辅助诊断工具，仅限具备合法执业资质的医护人员使用。辅助结果不替代专业医学诊疗建议。
      </p>
    </article>
  </section>
</template>
