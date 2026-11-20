<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAppStore } from '@/stores/app'
import { useAuthStore } from '@/stores/auth'
import AppDialog from './AppDialog.vue'
import UiIcon from './UiIcon.vue'

const route = useRoute()
const router = useRouter()
const app = useAppStore()
const auth = useAuthStore()
const account = ref(false)
const online = computed(() => !!app.pipelineMode && !app.healthError)
async function logout() { account.value = false; await auth.logout(); await router.push('/login') }
onMounted(() => { void app.fetchHealth() })
</script>
<template>
  <header class="site-header"><div class="header-inner">
    <RouterLink to="/" class="brand-link" aria-label="智肤研析首页">
      <img class="brand-mark" src="@/brand.svg" alt=""><span>智肤研析</span>
    </RouterLink>
    <nav class="nav-tabs" aria-label="主导航">
      <template v-if="auth.isAuthenticated"><RouterLink to="/" :class="{ active: route.path === '/' }">工作台</RouterLink><RouterLink to="/history" :class="{ active: route.path === '/history' || route.path.startsWith('/task/') }">历史记录</RouterLink>
        <RouterLink v-if="auth.user?.role === 'admin'" to="/admin" :class="{ active: route.path === '/admin' }">管理后台</RouterLink></template>
      <template v-else><RouterLink to="/login" active-class="active">登录</RouterLink><RouterLink to="/register" active-class="active">注册</RouterLink></template>
    </nav>
    <div class="header-tools"><span class="health-label" :class="{ online }"><i />{{ app.healthLoading ? '检测中' : online ? '在线' : '离线' }}</span>
      <button v-if="auth.isAuthenticated" class="account-trigger" @click="account = true">账户<UiIcon name="chevron" /></button>
    </div>
  </div></header>
  <AppDialog v-if="account" title="账户" @close="account = false">
    <dl class="facts"><dt>用户名</dt><dd>{{ auth.user?.username }}</dd><dt>角色</dt><dd>{{ auth.user?.role === 'admin' ? '管理员' : '医护人员' }}</dd></dl>
    <div class="dialog-actions"><button class="secondary-button" @click="logout">退出登录</button></div>
  </AppDialog>
</template>
