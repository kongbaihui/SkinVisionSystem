<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { deleteTask, getTasks, stateLabel } from '@/services/tasks'
import { formatDate, riskClass, type TaskSummary } from '@/types/diagnostic'
import AppDialog from '@/components/AppDialog.vue'

const tasks = ref<TaskSummary[]>([])
const page = ref(1)
const total = ref(0)
const pages = ref(1)
const filter = ref('')
const loading = ref(false)
const error = ref('')
const selected = ref<number>()
const deleting = ref(false)
const deleteError = ref('')
let serial = 0
async function fetchPage(n = 1) {
  const request = ++serial
  loading.value = true
  error.value = ''
  try {
    const res = await getTasks({ page: n, page_size: 10, status: filter.value || undefined, sort: 'created_at.desc' })
    if (request !== serial) return
    tasks.value = res.data
    page.value = res.pagination.page
    total.value = res.pagination.total_count
    pages.value = res.pagination.total_pages
  } catch { if (request === serial) error.value = '读取历史记录失败，请重试。' }
  finally { if (request === serial) loading.value = false }
}
async function remove() {
  if (!selected.value || deleting.value) return
  deleting.value = true
  try { await deleteTask(selected.value); selected.value = undefined; await fetchPage(tasks.value.length === 1 ? Math.max(page.value - 1, 1) : page.value) }
  catch { deleteError.value = '删除失败，请重试。' }
  finally { deleting.value = false }
}
onMounted(() => fetchPage())
</script>
<template>
  <div class="page-heading"><h1>历史记录</h1><div class="actions">
    <select v-model="filter" aria-label="筛选分割状态" class="form-input" @change="fetchPage(1)"><option value="">全部状态</option><option value="processing">进行中</option><option value="completed">已完成</option><option value="failed">失败</option></select>
    <button class="secondary-button" :disabled="loading" @click="fetchPage(page)">刷新</button></div></div>
  <article class="panel history-panel">
    <p v-if="error" class="form-error-box">{{ error }}</p><p v-if="loading" class="note" aria-live="polite">正在读取记录…</p>
    <div v-if="!loading && !error && !tasks.length" class="empty-state"><h2>暂无分析记录</h2><RouterLink to="/" class="secondary-button">前往工作台</RouterLink></div>
    <div v-if="tasks.length" class="history-table-wrap"><table class="history-table"><thead><tr>
      <th>任务 / 病例</th><th>症状摘要</th><th>既有规则提示</th><th>分割状态</th><th>报告状态</th><th>分析时间</th><th>操作</th>
    </tr></thead><tbody><tr v-for="t in tasks" :key="t.id">
      <td data-label="任务 / 病例"><strong>#{{ t.id }}</strong><small>{{ t.patient_id || '未记录编号' }}{{ t.patient_age == null ? '' : ' · ' + t.patient_age + ' 岁' }}</small></td>
      <td data-label="症状摘要"><span class="symptom-snippet">{{ t.patient_symptoms || '未提供' }}</span></td>
      <td data-label="既有规则提示"><span class="risk-badge" :class="riskClass(t.risk_level)" :title="t.risk_source || ''">{{ t.risk_level || '未记录' }}</span></td>
      <td data-label="分割状态">{{ stateLabel(t.status) }}</td><td data-label="报告状态">{{ stateLabel(t.report_status) }}<small v-if="t.report_version">修订 v{{ t.report_version }}</small></td>
      <td data-label="分析时间">{{ formatDate(t.created_at) }}</td>
      <td class="row-actions"><RouterLink class="text-link" :to="{name: 'task-detail', params: {taskId: t.id}}">查看</RouterLink><button class="link-btn danger" @click="selected = t.id; deleteError = ''">删除</button></td>
    </tr></tbody></table></div>
    <div v-if="total" class="pagination"><span>共 {{ total }} 条</span><button class="secondary-button" :disabled="loading || page <= 1" @click="fetchPage(page - 1)">上一页</button><span>{{ page }} / {{ pages }}</span><button class="secondary-button" :disabled="loading || page >= pages" @click="fetchPage(page + 1)">下一页</button></div>
  </article>
  <AppDialog v-if="selected" title="删除任务" @close="!deleting && (selected = undefined)">
    <p>删除任务 #{{ selected }} 后，关联图像及报告也将删除，此操作不可恢复。</p><p v-if="deleteError" class="form-error-box">{{ deleteError }}</p>
    <div class="dialog-actions"><button class="secondary-button" @click="selected = undefined">取消</button><button class="btn-danger" :disabled="deleting" @click="remove">确认删除</button></div>
  </AppDialog>
</template>
