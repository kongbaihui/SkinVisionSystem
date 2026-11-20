<script setup lang="ts">
import { ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { isAxiosError } from 'axios'
import { deleteTask, getTask } from '@/services/tasks'
import type { TaskDetail } from '@/types/diagnostic'
import AnalysisResults from '@/components/AnalysisResults.vue'
import AppDialog from '@/components/AppDialog.vue'
import UiIcon from '@/components/UiIcon.vue'

const route = useRoute()
const router = useRouter()
const task = ref<TaskDetail>()
const loading = ref(false)
const pageError = ref('')
const showDelete = ref(false)
const deleting = ref(false)
const deleteError = ref('')
watch(() => route.params.taskId, async (id, _, onCleanup) => {
  let valid = true
  onCleanup(() => { valid = false })
  loading.value = true
  pageError.value = ''
  task.value = undefined
  try { const data = await getTask(Number(id)); if (valid) task.value = data }
  catch (err) {
    if (valid) pageError.value = isAxiosError(err) && err.response?.status === 403 ? '您无权访问此任务。' : '无法打开此任务，请返回历史记录核对。'
  } finally { if (valid) loading.value = false }
}, { immediate: true })

async function remove() {
  if (!task.value || deleting.value) return
  deleteError.value = ''
  deleting.value = true
  try { await deleteTask(task.value.id); task.value = undefined; showDelete.value = false; await router.push('/history') }
  catch { deleteError.value = '删除失败，请重试。' }
  finally { deleting.value = false }
}
</script>
<template>
  <div class="page-heading"><h1>分析结果</h1><RouterLink to="/history" class="return-link"><UiIcon name="back" />返回历史记录</RouterLink></div>
  <p v-if="loading" class="panel">正在读取任务…</p>
  <p v-else-if="pageError" class="form-error-box" role="alert">{{ pageError }}</p>
  <template v-else-if="task"><AnalysisResults :task="task" @updated="task = $event" />
    <div class="detail-end"><button class="link-btn danger" @click="showDelete = true">删除任务</button></div>
  </template>
  <AppDialog v-if="showDelete" title="删除任务" @close="!deleting && (showDelete = false)">
    <p>删除任务 #{{ task?.id }} 后，关联图像及报告也将删除，此操作不可恢复。</p>
    <p v-if="deleteError" class="form-error-box">{{ deleteError }}</p>
    <div class="dialog-actions"><button class="secondary-button" @click="showDelete = false">取消</button><button class="btn-danger" :disabled="deleting" @click="remove">确认删除</button></div>
  </AppDialog>
</template>
