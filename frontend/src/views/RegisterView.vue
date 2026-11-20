<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'

import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const authStore = useAuthStore()

const username = ref('')
const password = ref('')
const confirmPassword = ref('')
const email = ref('')
const loading = ref(false)
const errorMsg = ref('')

function validate(): string {
  if (!username.value.trim()) return '请输入用户名'
  if (!/^[a-zA-Z0-9_]{3,20}$/.test(username.value.trim())) return '用户名为 3-20 位字母、数字或下划线'
  if (!password.value) return '请输入密码'
  if (password.value.length < 8) return '密码至少 8 位'
  if (!/[A-Za-z]/.test(password.value) || !/[0-9]/.test(password.value)) return '密码需同时包含字母和数字'
  if (password.value !== confirmPassword.value) return '两次密码输入不一致'
  if (email.value && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.value)) return '邮箱格式不正确'
  return ''
}

async function handleSubmit() {
  errorMsg.value = validate()
  if (errorMsg.value) return

  loading.value = true
  try {
    await authStore.register({
      username: username.value.trim(),
      password: password.value,
      email: email.value.trim() || undefined,
    })
    await router.push('/')
  } catch (err: unknown) {
    const detail = (err as { response?: { data?: { detail?: { message?: string } | string } } })?.response?.data?.detail
    if (typeof detail === 'object' && detail?.message) {
      errorMsg.value = detail.message
    } else if (typeof detail === 'string') {
      errorMsg.value = detail
    } else {
      errorMsg.value = '注册失败，请重试'
    }
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <section class="auth-page">
    <article class="panel auth-card">
      <div class="login-title-area"><p class="auth-brand">智肤研析</p>
      <h2>注册账号</h2>
      <p class="subtle">创建医护人员账户</p></div>

      <form class="auth-form" @submit.prevent="handleSubmit">
        <div class="form-field">
          <label class="form-label" for="reg-username">用户名 <span class="required">*</span></label>
          <input
            id="reg-username"
            v-model="username"
            class="form-input"
            type="text"
            placeholder="3-20 位字母、数字或下划线"
            autocomplete="username"
            :disabled="loading"
          />
        </div>

        <div class="form-field">
          <label class="form-label" for="reg-password">密码 <span class="required">*</span></label>
          <input
            id="reg-password"
            v-model="password"
            class="form-input"
            type="password"
            placeholder="至少 8 位，包含字母和数字"
            autocomplete="new-password"
            :disabled="loading"
          />
        </div>

        <div class="form-field">
          <label class="form-label" for="reg-confirm">确认密码 <span class="required">*</span></label>
          <input
            id="reg-confirm"
            v-model="confirmPassword"
            class="form-input"
            type="password"
            placeholder="再次输入密码"
            autocomplete="new-password"
            :disabled="loading"
          />
        </div>

        <div class="form-field">
          <label class="form-label" for="reg-email">邮箱（可选）</label>
          <input
            id="reg-email"
            v-model="email"
            class="form-input"
            type="email"
            placeholder="请输入邮箱地址"
            autocomplete="email"
            :disabled="loading"
          />
        </div>

        <div v-if="errorMsg" class="form-error-box">{{ errorMsg }}</div>

        <button class="btn-primary" type="submit" :disabled="loading">
          {{ loading ? '注册中…' : '注册' }}
        </button>
      </form>

      <p class="auth-link">
        已有账号？<RouterLink to="/login">立即登录</RouterLink>
      </p>
    </article>
  </section>
</template>
