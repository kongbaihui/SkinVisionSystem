<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { onBeforeRouteLeave, onBeforeRouteUpdate } from 'vue-router'
import { isAxiosError } from 'axios'
import { exportTask, getTask, retryReport, saveReport, stateLabel } from '@/services/tasks'
import { renderReport } from '@/services/report'
import { formatDate, type TaskDetail } from '@/types/diagnostic'
import AppDialog from './AppDialog.vue'
import UiIcon from './UiIcon.vue'

const props = defineProps<{ task: TaskDetail }>()
const emit = defineEmits<{ updated: [task: TaskDetail] }>()
const mode = ref<'edit' | 'preview'>('edit')
const dialog = ref<'export' | 'conflict' | null>(null)
const draft = ref('')
const baseline = ref('')
const version = ref(0)
const conflict = ref(false)
const saveFailed = ref(false)
const latest = ref<TaskDetail>()
const error = ref('')
const busy = ref<'save' | 'retry' | 'export' | 'load' | null>(null)
const dirty = computed(() => draft.value !== baseline.value)
const canRetry = computed(() => props.task.status === 'completed' && ['failed', 'interrupted'].includes(props.task.report_status || ''))
const status = computed(() => busy.value === 'save' ? '保存中…' : conflict.value ? '版本冲突' : saveFailed.value ? '保存失败' : dirty.value ? '未保存' : props.task.edited_report != null ? '已保存' : '生成草稿')
function sync(task: TaskDetail) {
  draft.value = baseline.value = task.effective_report ?? task.edited_report ?? task.llm_report ?? ''
  version.value = task.report_version ?? 0
  conflict.value = false
  saveFailed.value = false
  error.value = ''
}
watch(() => props.task.id, () => sync(props.task), { immediate: true })
function update(task: TaskDetail) { sync(task); emit('updated', task) }
async function save() {
  if (busy.value || !dirty.value || conflict.value) return
  error.value = ''
  busy.value = 'save'
  try { update(await saveReport(props.task.id, draft.value, version.value)) }
  catch (err) {
    saveFailed.value = true
    conflict.value = isAxiosError(err) && err.response?.status === 409
    error.value = conflict.value ? '此报告的版本已变更。当前输入已保留，请比对最新版本。' : '修改未能保存。当前输入已保留，可再次保存。'
  } finally { busy.value = null }
}
async function loadLatest() {
  error.value = ''
  busy.value = 'load'
  try { latest.value = await getTask(props.task.id); dialog.value = 'conflict' }
  catch { error.value = '读取最新报告失败，请重试。' }
  finally { busy.value = null }
}
function resolveConflict(keepDraft: boolean) {
  if (!latest.value) return
  const current = draft.value
  update(latest.value)
  if (keepDraft) draft.value = current
  dialog.value = null
}
async function retry() {
  error.value = ''
  busy.value = 'retry'
  try { update(await retryReport(props.task.id)) }
  catch (err) {
    error.value = isAxiosError(err) && err.response?.status === 409 ? '当前状态无法重试，请重新打开任务核对报告状态。' : '重试请求失败，请重新打开任务确认结果。'
  } finally { busy.value = null }
}
async function download(format: 'pdf' | 'markdown') {
  busy.value = 'export'
  error.value = ''
  try {
    const blob = await exportTask(props.task.id, format)
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = '智肤研析_任务' + props.task.id + '.' + (format === 'pdf' ? 'pdf' : 'md')
    link.click()
    window.setTimeout(() => URL.revokeObjectURL(url), 30000)
    dialog.value = null
  } catch { error.value = '导出失败，请重试。' }
  finally { busy.value = null }
}
function guard() { return !dirty.value || window.confirm('报告有未保存的修改，确定离开？') }
function onUnload(event: BeforeUnloadEvent) { if (dirty.value) event.preventDefault() }
onBeforeRouteLeave(guard)
onBeforeRouteUpdate((to, from) => to.fullPath === from.fullPath || guard())
onMounted(() => window.addEventListener('beforeunload', onUnload))
onBeforeUnmount(() => window.removeEventListener('beforeunload', onUnload))
</script>
<template>
  <article class="panel report-panel">
    <div class="section-heading"><h2>报告草稿</h2>
      <div class="segments compact" role="group" aria-label="报告视图"><button :aria-pressed="mode === 'edit'" @click="mode = 'edit'">编辑</button><button :aria-pressed="mode === 'preview'" @click="mode = 'preview'">预览</button></div>
    </div>
    <div class="draft-meta"><span class="status-badge">{{ stateLabel(task.report_status) }}</span><span>{{ task.edited_report != null ? '医生修订稿' : '生成原稿' }} · v{{ version }}</span></div>
    <div v-if="canRetry" class="retry-line"><span>报告生成{{ stateLabel(task.report_status) }}，已保留现有分析结果。</span><button class="secondary-button" :disabled="!!busy || dirty" @click="retry">{{ busy === 'retry' ? '重试中…' : '重试报告' }}</button></div>
    <template v-if="mode === 'edit'">
      <label for="report-body" class="form-label">报告正文</label>
      <textarea id="report-body" v-model="draft" class="form-input report-editor" maxlength="200000" :disabled="!!busy" placeholder="填写医生复核意见或编辑报告草稿" @input="saveFailed = false; !conflict && (error = '')" />
    </template>
    <template v-else><div v-if="draft" class="report-content preview-pane" v-html="renderReport(draft)" /><p v-else class="preview-pane note">暂无报告正文。</p></template>
    <p v-if="error && !dialog" class="form-error-box" role="alert">{{ error }} <button v-if="conflict" class="link-btn" :disabled="!!busy" @click="loadLatest">比对最新版本</button></p>
    <div class="editor-footer"><span class="save-state" aria-live="polite"><span :key="status"><UiIcon v-if="status === '已保存'" name="check" />{{ status }}</span></span>
      <div class="actions"><button class="secondary-button" :disabled="!!busy" @click="dialog = 'export'"><UiIcon name="download" />导出</button>
        <button class="btn-primary" :disabled="!!busy || !dirty || !draft.trim() || conflict" @click="save">保存修改</button></div>
    </div>
    <details class="module-details"><summary>报告来源与原稿</summary>
      <dl class="facts"><dt>来源</dt><dd>{{ task.report_metadata?.source || '未记录' }}</dd><dt>模型</dt><dd>{{ task.report_metadata?.model || '未记录' }}</dd>
        <template v-if="task.report_edited_at"><dt>上次修订</dt><dd>{{ formatDate(task.report_edited_at) }}</dd></template></dl>
      <div v-if="task.llm_report" class="report-content original-draft" v-html="renderReport(task.llm_report)" />
    </details>
  </article>
  <AppDialog v-if="dialog === 'export'" title="导出报告" @close="dialog = null">
    <p>{{ task.edited_report != null ? '导出此任务已保存的医生修订稿。' : '导出此任务已保存的生成原稿。' }}</p><p v-if="dirty" class="note">当前编辑内容尚未保存。</p>
    <p v-if="error" class="form-error-box">{{ error }}</p>
    <div class="dialog-actions"><button class="secondary-button" :disabled="!!busy" @click="download('markdown')">导出 Markdown</button><button class="btn-primary" :disabled="!!busy" @click="download('pdf')">导出 PDF</button></div>
  </AppDialog>
  <AppDialog v-if="dialog === 'conflict' && latest" title="版本冲突" @close="dialog = null">
    <p>已保存版本 v{{ latest.report_version }}</p><pre class="conflict-text">{{ latest.effective_report || latest.llm_report }}</pre>
    <p>采用最新版本会替换本页输入。保留本页修改后，可继续编辑并手动保存。</p>
    <div class="dialog-actions"><button class="secondary-button" @click="resolveConflict(false)">采用最新版本</button><button class="btn-primary" @click="resolveConflict(true)">保留本页修改</button></div>
  </AppDialog>
</template>
