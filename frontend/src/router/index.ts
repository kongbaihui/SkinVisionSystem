import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

import AdminView from '@/views/AdminView.vue'
import HistoryView from '@/views/HistoryView.vue'
import LoginView from '@/views/LoginView.vue'
import RegisterView from '@/views/RegisterView.vue'
import TaskDetailView from '@/views/TaskDetailView.vue'
import WorkbenchView from '@/views/WorkbenchView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'workbench',
      component: WorkbenchView,
      meta: { requiresAuth: true },
    },
    {
      path: '/login',
      name: 'login',
      component: LoginView,
    },
    {
      path: '/register',
      name: 'register',
      component: RegisterView,
    },
    {
      path: '/history',
      name: 'history',
      component: HistoryView,
      meta: { requiresAuth: true },
    },
    {
      path: '/task/:taskId/detail',
      name: 'task-detail',
      component: TaskDetailView,
      meta: { requiresAuth: true },
    },
    {
      path: '/admin',
      name: 'admin',
      component: AdminView,
      meta: { requiresAuth: true, requiresRole: 'admin' },
    },
  ],
})

router.beforeEach(async (to) => {
  if (!to.meta.requiresAuth) return true

  const authStore = useAuthStore()

  // 认证检查
  let authenticated = !!authStore.accessToken
  if (!authenticated) {
    const newToken = await authStore.doRefresh()
    if (newToken) {
      await authStore.fetchMe()
      authenticated = true
    }
  }

  if (!authenticated) {
    return { name: 'login', query: { redirect: to.fullPath } }
  }

  // 角色守卫：admin 页面只允许 admin 账号进入
  if (to.meta.requiresRole === 'admin' && authStore.user?.role !== 'admin') {
    return { name: 'workbench' }
  }

  return true
})

export default router
