<script setup lang="ts">
import { onMounted, ref } from 'vue'


import {
  getAdminStats,
  getAdminUsers,
  getAdminLogs,
  getAdminTasks,
  toggleUserStatus,
  updateUserRole,
  type AdminStats,
  type AdminUser,
  type AdminAuditLog,
  type AdminTask,
} from '@/services/admin'

import type { Pagination } from '@/types/diagnostic'
import { formatDate, riskClass } from '@/types/diagnostic'




// ─── 标签页 ───────────────────────────────────────────────────────────────────
type Tab = 'overview' | 'users' | 'logs' | 'tasks' | 'compliance'
const activeTab = ref<Tab>('overview')

// ─── 概览统计 ─────────────────────────────────────────────────────────────────
const stats = ref<AdminStats | null>(null)
const statsLoading = ref(false)
const statsError = ref('')

async function fetchStats() {
  statsLoading.value = true
  statsError.value = ''
  try {
    stats.value = await getAdminStats()
  } catch (err: unknown) {
    statsError.value = (err as Error).message || '加载失败'
  } finally {
    statsLoading.value = false
  }
}

// ─── 用户管理 ─────────────────────────────────────────────────────────────────
const users = ref<AdminUser[]>([])
const usersPagination = ref<Pagination>({ page: 1, page_size: 20, total_count: 0, total_pages: 1 })
const usersLoading = ref(false)
const usersError = ref('')
const usersPage = ref(1)
const togglingId = ref<number | null>(null)
const updatingRoleId = ref<number | null>(null)

async function fetchUsers(page = 1) {
  usersLoading.value = true
  usersError.value = ''
  try {
    const res = await getAdminUsers({ page, page_size: 20 })
    users.value = res.data
    usersPagination.value = res.pagination
    usersPage.value = page
  } catch (err: unknown) {
    usersError.value = (err as Error).message || '加载失败'
  } finally {
    usersLoading.value = false
  }
}

async function handleToggleStatus(userId: number) {
  togglingId.value = userId
  try {
    const res = await toggleUserStatus(userId)
    const u = users.value.find(x => x.id === userId)
    if (u) u.is_active = res.is_active
  } catch {
    // ignore
  } finally {
    togglingId.value = null
  }
}

async function handleUpdateRole(userId: number, role: 'doctor' | 'admin') {
  updatingRoleId.value = userId
  try {
    const res = await updateUserRole(userId, role)
    const u = users.value.find(x => x.id === userId)
    if (u) u.role = res.role
  } catch {
    // ignore
  } finally {
    updatingRoleId.value = null
  }
}

// ─── 操作日志 ─────────────────────────────────────────────────────────────────
const logs = ref<AdminAuditLog[]>([])
const logsPagination = ref<Pagination>({ page: 1, page_size: 20, total_count: 0, total_pages: 1 })
const logsLoading = ref(false)
const logsError = ref('')
const logsPage = ref(1)
const logsActionFilter = ref('')

async function fetchLogs(page = 1) {
  logsLoading.value = true
  logsError.value = ''
  try {
    const res = await getAdminLogs({
      page,
      page_size: 20,
      action: logsActionFilter.value || undefined,
    })
    logs.value = res.data
    logsPagination.value = res.pagination
    logsPage.value = page
  } catch (err: unknown) {
    logsError.value = (err as Error).message || '加载失败'
  } finally {
    logsLoading.value = false
  }
}

// ─── 全部任务 ─────────────────────────────────────────────────────────────────
const adminTasks = ref<AdminTask[]>([])
const tasksPagination = ref<Pagination>({ page: 1, page_size: 15, total_count: 0, total_pages: 1 })
const tasksLoading = ref(false)
const tasksError = ref('')
const tasksPage = ref(1)
const tasksStatusFilter = ref('')

async function fetchAdminTasks(page = 1) {
  tasksLoading.value = true
  tasksError.value = ''
  try {
    const res = await getAdminTasks({
      page,
      page_size: 15,
      status: tasksStatusFilter.value || undefined,
    })
    adminTasks.value = res.data
    tasksPagination.value = res.pagination
    tasksPage.value = page
  } catch (err: unknown) {
    tasksError.value = (err as Error).message || '加载失败'
  } finally {
    tasksLoading.value = false
  }
}

// ─── 标签切换 ─────────────────────────────────────────────────────────────────
function switchTab(tab: Tab) {
  activeTab.value = tab
  if (tab === 'overview' && !stats.value) fetchStats()
  if (tab === 'users' && users.value.length === 0) fetchUsers()
  if (tab === 'logs' && logs.value.length === 0) fetchLogs()
  if (tab === 'tasks' && adminTasks.value.length === 0) fetchAdminTasks()
}

// ─── 工具函数 ─────────────────────────────────────────────────────────────────
const ACTION_LABELS: Record<string, string> = {
  login: '用户登录',
  logout: '退出登录',
  login_failed: '登录失败',
  register: '用户注册',
}

function actionLabel(a: string): string {
  return ACTION_LABELS[a] ?? a
}

function statusLabel(s: string): string {
  const m: Record<string, string> = { processing: '进行中', completed: '已完成', failed: '失败' }
  return m[s] ?? s
}

// ─── 合规配置（静态） ─────────────────────────────────────────────────────────
const complianceConfig = [
  { label: '强制合规提示',     value: '已开启' },
  { label: '报告免责声明',     value: '已开启' },
  { label: '账户操作日志',     value: '记录登录、注册与退出事件' },
  { label: '角色',             value: '医护人员 / 管理员' },
  { label: 'HTTPS 传输加密',   value: '建议生产环境开启' },
  { label: '账号登录限流',     value: '由部署环境配置失败次数和锁定时长' },
]

onMounted(() => {
  fetchStats()
})
</script>

<template>
  <div class="admin-layout">
    <!-- 页头 -->
    <div class="page-heading"><h1>管理后台</h1></div>

    <!-- 标签页导航 -->
    <div class="panel" style="padding:16px 24px">
      <nav class="admin-tabs" style="margin-bottom:0;border-bottom:none;padding-bottom:0">
        <button class="admin-tab" :class="{ active: activeTab === 'overview' }" @click="switchTab('overview')">概览</button>
        <button class="admin-tab" :class="{ active: activeTab === 'users' }" @click="switchTab('users')">用户管理</button>
        <button class="admin-tab" :class="{ active: activeTab === 'logs' }" @click="switchTab('logs')">操作日志</button>
        <button class="admin-tab" :class="{ active: activeTab === 'tasks' }" @click="switchTab('tasks')">全部任务</button>
        <button class="admin-tab" :class="{ active: activeTab === 'compliance' }" @click="switchTab('compliance')">合规配置</button>
      </nav>
    </div>

    <!-- ══ 概览 ══════════════════════════════════════════════════════════════ -->
    <section v-if="activeTab === 'overview'">
      <div v-if="statsLoading" class="panel" style="text-align:center;padding:32px">
        <span class="spinner" style="display:inline-block" />
        <p class="subtle" style="margin-top:12px">加载统计数据…</p>
      </div>
      <div v-else-if="statsError" class="panel">
        <p class="form-error-box">{{ statsError }}</p>
        <button class="secondary-button" @click="fetchStats">重试</button>
      </div>
      <template v-else-if="stats">
        <!-- 统计卡片 -->
        <div class="stat-grid">
          <div class="stat-card accent">
            <span class="stat-label">总任务数</span>
            <span class="stat-value">{{ stats.total_tasks }}</span>
            <span class="stat-sub">全部分析记录</span>
          </div>
          <div class="stat-card success">
            <span class="stat-label">已完成</span>
            <span class="stat-value">{{ stats.completed_tasks }}</span>
            <span class="stat-sub">成功率 {{ stats.total_tasks ? ((stats.completed_tasks / stats.total_tasks) * 100).toFixed(1) : 0 }}%</span>
          </div>
          <div class="stat-card">
            <span class="stat-label">总用户数</span>
            <span class="stat-value">{{ stats.total_users }}</span>
            <span class="stat-sub">活跃 {{ stats.active_users }} 人</span>
          </div>
          <div class="stat-card">
            <span class="stat-label">平均推理时间</span>
            <span class="stat-value">{{ stats.avg_inference_ms.toFixed(0) }}</span>
            <span class="stat-sub">ms / 次</span>
          </div>
        </div>

        <!-- 系统状态 + 任务分布 -->
        <div class="admin-grid">
          <article class="panel">
            <p class="eyebrow">系统状态</p>
            <h3 style="margin:0 0 16px">推理引擎</h3>
            <div class="mini-info-grid">
              <div class="mini-info-item">
                <span class="mini-info-label">Pipeline 模式</span>
                <span class="mini-info-value">{{ stats.pipeline_mode }}</span>
              </div>
              <div class="mini-info-item">
                <span class="mini-info-label">进行中任务</span>
                <span class="mini-info-value">{{ stats.processing_tasks }}</span>
              </div>
              <div class="mini-info-item">
                <span class="mini-info-label">失败任务</span>
                <span class="mini-info-value" :style="{ color: stats.failed_tasks > 0 ? 'var(--danger)' : 'inherit' }">
                  {{ stats.failed_tasks }}
                </span>
              </div>
            </div>
          </article>

          <article class="panel">
            <p class="eyebrow">任务分布</p>
            <h3 style="margin:0 0 16px">状态概览</h3>
            <div style="display:flex;flex-direction:column;gap:10px">
              <div>
                <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px">
                  <span>已完成</span>
                  <span class="subtle">{{ stats.total_tasks ? ((stats.completed_tasks / stats.total_tasks) * 100).toFixed(1) : 0 }}%</span>
                </div>
                <div class="feature-bar-bg">
                  <div class="feature-bar" :style="{ width: stats.total_tasks ? (stats.completed_tasks / stats.total_tasks * 100) + '%' : '0%' }" />
                </div>
              </div>
              <div>
                <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px">
                  <span>失败</span>
                  <span class="subtle">{{ stats.total_tasks ? ((stats.failed_tasks / stats.total_tasks) * 100).toFixed(1) : 0 }}%</span>
                </div>
                <div class="feature-bar-bg">
                  <div class="feature-bar bar-warn" :style="{ width: stats.total_tasks ? (stats.failed_tasks / stats.total_tasks * 100) + '%' : '0%' }" />
                </div>
              </div>
              <div>
                <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px">
                  <span>进行中</span>
                  <span class="subtle">{{ stats.total_tasks ? ((stats.processing_tasks / stats.total_tasks) * 100).toFixed(1) : 0 }}%</span>
                </div>
                <div class="feature-bar-bg">
                  <div class="feature-bar" style="background:var(--warning)" :style="{ width: stats.total_tasks ? (stats.processing_tasks / stats.total_tasks * 100) + '%' : '0%' }" />
                </div>
              </div>
            </div>
          </article>
        </div>
      </template>
      <div v-else class="panel empty-state"><p>暂无数据</p></div>
    </section>

    <!-- ══ 用户管理 ══════════════════════════════════════════════════════════ -->
    <section v-else-if="activeTab === 'users'" class="panel">
      <div class="admin-toolbar">
        <div class="admin-toolbar-left">
          <p class="eyebrow" style="margin:0">账号管理</p>
          <h3 style="margin:0">医护账号列表</h3>
        </div>
        <button class="secondary-button" :disabled="usersLoading" @click="fetchUsers(usersPage)">
          {{ usersLoading ? '加载中…' : '刷新' }}
        </button>
      </div>
      <p class="subtle" style="margin:0 0 16px">共 {{ usersPagination.total_count }} 个账号</p>

      <div v-if="usersError" class="form-error-box">{{ usersError }}</div>

      <div v-if="!usersLoading && users.length === 0 && !usersError" class="empty-state">
        <p>暂无用户数据</p>
      </div>

      <div v-if="users.length > 0" class="table-wrap">
        <table class="admin-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>用户名</th>
              <th>邮箱</th>
              <th>角色</th>
              <th>状态</th>
              <th>最后登录</th>
              <th>注册时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="user in users" :key="user.id">
              <td class="td-id">{{ user.id }}</td>
              <td><strong>{{ user.username }}</strong></td>
              <td class="subtle" style="font-size:12px">{{ user.email ?? '—' }}</td>
              <td>
                <select
                  class="form-input select-sm"
                  style="max-width:100px"
                  :value="user.role"
                  :disabled="updatingRoleId === user.id"
                  @change="handleUpdateRole(user.id, ($event.target as HTMLSelectElement).value as 'doctor' | 'admin')"
                >
                  <option value="doctor">医护人员</option>
                  <option value="admin">管理员</option>
                </select>
              </td>
              <td>
                <span class="status-badge" :class="user.is_active ? 'status-completed' : 'status-failed'">
                  {{ user.is_active ? '活跃' : '已停用' }}
                </span>
              </td>
              <td class="td-date subtle">{{ user.last_login ? formatDate(user.last_login) : '从未登录' }}</td>
              <td class="td-date subtle">{{ formatDate(user.created_at) }}</td>
              <td>
                <button
                  class="link-btn"
                  :class="{ danger: user.is_active }"
                  :disabled="togglingId === user.id"
                  @click="handleToggleStatus(user.id)"
                >
                  {{ togglingId === user.id ? '…' : (user.is_active ? '停用' : '启用') }}
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- 分页 -->
      <div v-if="usersPagination.total_pages > 1" class="pagination">
        <button class="page-btn" :disabled="usersPage === 1" @click="fetchUsers(usersPage - 1)">上一页</button>
        <template v-for="p in usersPagination.total_pages" :key="p">
          <button
            v-if="Math.abs(p - usersPage) <= 2 || p === 1 || p === usersPagination.total_pages"
            class="page-btn" :class="{ active: p === usersPage }" @click="fetchUsers(p)"
          >{{ p }}</button>
          <span v-else-if="p === usersPage - 3 || p === usersPage + 3" class="page-ellipsis">…</span>
        </template>
        <button class="page-btn" :disabled="usersPage === usersPagination.total_pages" @click="fetchUsers(usersPage + 1)">下一页</button>
        <span class="page-info">共 {{ usersPagination.total_count }} 条</span>
      </div>
    </section>

    <!-- ══ 操作日志 ══════════════════════════════════════════════════════════ -->
    <section v-else-if="activeTab === 'logs'" class="panel">
      <div class="admin-toolbar">
        <div class="admin-toolbar-left">
          <div>
            <p class="eyebrow" style="margin:0">审计溯源</p>
            <h3 style="margin:0">操作日志</h3>
          </div>
          <select
            v-model="logsActionFilter"
            class="form-input select-sm"
            style="max-width:130px"
            @change="fetchLogs(1)"
          >
            <option value="">全部操作</option>
            <option value="login">登录</option>
            <option value="logout">退出</option>
            <option value="login_failed">登录失败</option>
            <option value="register">注册</option>
          </select>
        </div>
        <button class="secondary-button" :disabled="logsLoading" @click="fetchLogs(logsPage)">
          {{ logsLoading ? '加载中…' : '刷新' }}
        </button>
      </div>
      <p class="subtle" style="margin:0 0 16px">共 {{ logsPagination.total_count }} 条记录，满足医疗系统溯源审计要求</p>

      <div v-if="logsError" class="form-error-box">{{ logsError }}</div>

      <div v-if="!logsLoading && logs.length === 0 && !logsError" class="empty-state">
        <p>暂无日志记录</p>
      </div>

      <div v-if="logs.length > 0" class="table-wrap">
        <table class="history-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>时间</th>
              <th>用户</th>
              <th>操作类型</th>
              <th>IP 地址</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="log in logs" :key="log.id">
              <td class="td-id">{{ log.id }}</td>
              <td class="td-date">{{ formatDate(log.created_at) }}</td>
              <td><strong>{{ log.username }}</strong></td>
              <td>
                <span
                  class="status-badge"
                  :class="log.action === 'login_failed' ? 'status-failed' : log.action === 'login' ? 'status-completed' : 'status-processing'"
                >
                  {{ actionLabel(log.action) }}
                </span>
              </td>
              <td class="subtle" style="font-size:12px;font-family:monospace">{{ log.ip_address ?? '—' }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- 分页 -->
      <div v-if="logsPagination.total_pages > 1" class="pagination">
        <button class="page-btn" :disabled="logsPage === 1" @click="fetchLogs(logsPage - 1)">上一页</button>
        <template v-for="p in logsPagination.total_pages" :key="p">
          <button
            v-if="Math.abs(p - logsPage) <= 2 || p === 1 || p === logsPagination.total_pages"
            class="page-btn" :class="{ active: p === logsPage }" @click="fetchLogs(p)"
          >{{ p }}</button>
          <span v-else-if="p === logsPage - 3 || p === logsPage + 3" class="page-ellipsis">…</span>
        </template>
        <button class="page-btn" :disabled="logsPage === logsPagination.total_pages" @click="fetchLogs(logsPage + 1)">下一页</button>
        <span class="page-info">共 {{ logsPagination.total_count }} 条</span>
      </div>
    </section>

    <!-- ══ 全部任务 ══════════════════════════════════════════════════════════ -->
    <section v-else-if="activeTab === 'tasks'" class="panel">
      <div class="admin-toolbar">
        <div class="admin-toolbar-left">
          <div>
            <p class="eyebrow" style="margin:0">任务总览</p>
            <h3 style="margin:0">全部诊断任务</h3>
          </div>
          <select
            v-model="tasksStatusFilter"
            class="form-input select-sm"
            @change="fetchAdminTasks(1)"
          >
            <option value="">全部状态</option>
            <option value="processing">进行中</option>
            <option value="completed">已完成</option>
            <option value="failed">失败</option>
          </select>
        </div>
        <button class="secondary-button" :disabled="tasksLoading" @click="fetchAdminTasks(tasksPage)">
          {{ tasksLoading ? '加载中…' : '刷新' }}
        </button>
      </div>
      <p class="subtle" style="margin:0 0 16px">共 {{ tasksPagination.total_count }} 条记录</p>

      <div v-if="tasksError" class="form-error-box">{{ tasksError }}</div>

      <div v-if="!tasksLoading && adminTasks.length === 0 && !tasksError" class="empty-state">
        <p>暂无任务记录</p>
      </div>

      <div v-if="adminTasks.length > 0" class="table-wrap">
        <table class="history-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>操作用户</th>
              <th>患者编号</th>
              <th>年龄</th>
              <th>不规则指数</th>
              <th>既有规则提示</th>
              <th>模式</th>
              <th>耗时 (ms)</th>
              <th>状态</th>
              <th>时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="task in adminTasks" :key="task.id">
              <td class="td-id">#{{ task.id }}</td>
              <td><span class="subtle" style="font-size:12px">{{ task.username }}</span></td>
              <td><span style="font-family:monospace;font-size:12px">{{ task.patient_id ?? '—' }}</span></td>
              <td>{{ task.patient_age ?? '—' }}</td>
              <td>{{ task.irregularity_index?.toFixed(4) ?? '—' }}</td>
              <td>
                <span class="risk-badge sm" :class="riskClass(task.risk_level)">
                  {{ task.risk_level ?? '未记录' }}
                </span>
              </td>
              <td class="subtle" style="font-size:12px">{{ task.pipeline_mode ?? '—' }}</td>
              <td class="subtle" style="font-size:12px">{{ task.inference_time_ms?.toFixed(0) ?? '—' }}</td>
              <td>
                <span class="status-badge" :class="`status-${task.status}`">
                  {{ statusLabel(task.status) }}
                </span>
              </td>
              <td class="td-date">{{ formatDate(task.created_at) }}</td>
              <td>
                <RouterLink
                  :to="{ name: 'task-detail', params: { taskId: task.id } }"
                  class="link-btn"
                >查看</RouterLink>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- 分页 -->
      <div v-if="tasksPagination.total_pages > 1" class="pagination">
        <button class="page-btn" :disabled="tasksPage === 1" @click="fetchAdminTasks(tasksPage - 1)">上一页</button>
        <template v-for="p in tasksPagination.total_pages" :key="p">
          <button
            v-if="Math.abs(p - tasksPage) <= 2 || p === 1 || p === tasksPagination.total_pages"
            class="page-btn" :class="{ active: p === tasksPage }" @click="fetchAdminTasks(p)"
          >{{ p }}</button>
          <span v-else-if="p === tasksPage - 3 || p === tasksPage + 3" class="page-ellipsis">…</span>
        </template>
        <button class="page-btn" :disabled="tasksPage === tasksPagination.total_pages" @click="fetchAdminTasks(tasksPage + 1)">下一页</button>
        <span class="page-info">共 {{ tasksPagination.total_count }} 条</span>
      </div>
    </section>

    <!-- ══ 合规配置 ══════════════════════════════════════════════════════════ -->
    <section v-else-if="activeTab === 'compliance'" class="panel">
      <p class="eyebrow">合规管理</p>
      <h3>系统合规配置</h3>
      <p class="subtle" style="margin-bottom:20px">账户访问、图像留存及报告复核说明。</p>
      <ul class="compliance-list">
        <li v-for="(item, i) in complianceConfig" :key="i" class="compliance-item">
          <span class="compliance-label">{{ item.label }}</span>
          <span class="compliance-value">{{ item.value }}</span>
        </li>
      </ul>
      <p class="compliance-notice" style="margin-top:20px">
        本系统为皮肤病变智能辅助诊断工具，仅限具备合法执业资质的医护人员使用。辅助结果不替代专业医学诊疗建议。
      </p>
      <div class="admin-grid" style="margin-top:20px">
        <article style="background:rgba(255,255,255,0.5);border:1px solid var(--panel-border);border-radius:14px;padding:16px">
          <p class="eyebrow">数据安全</p>
          <ul class="guide-list">
            <li>任务保留原图和分析产物，删除任务时一起清理</li>
            <li>令牌有效期由部署环境配置</li>
            <li>密码使用 bcrypt 哈希存储</li>
            <li>跨域访问使用部署环境配置的来源</li>
          </ul>
        </article>
        <article style="background:rgba(255,255,255,0.5);border:1px solid var(--panel-border);border-radius:14px;padding:16px">
          <p class="eyebrow">AI 伦理</p>
          <ul class="guide-list">
            <li>所有报告附带免责声明</li>
            <li>AI 结论不替代执业医师诊断</li>
            <li>特征量化结果可溯源验证</li>
            <li>Grad-CAM 提供模型可解释性</li>
          </ul>
        </article>
      </div>
    </section>
  </div>
</template>
