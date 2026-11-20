<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import AnalysisResults from '@/components/AnalysisResults.vue'
import AppDialog from '@/components/AppDialog.vue'
import UiIcon from '@/components/UiIcon.vue'
import { analyzeStream, type AnalysisEvent } from '@/services/diagnostic'
import { getTask, stateLabel } from '@/services/tasks'
import { getApiBaseUrl } from '@/services/http'
import { useAuthStore } from '@/stores/auth'
import type { TaskDetail } from '@/types/diagnostic'

const auth = useAuthStore()
const router = useRouter()
const route = useRoute()
const file = ref<File>()
const input = ref<HTMLInputElement>()
const preview = ref('')
const help = ref(false)
const dragDepth = ref(0)
const caseNumber = () => 'PT-' + new Date().toISOString().slice(0, 10).replace(/-/g, '') + '-' + String(Math.floor(100 + Math.random() * 900))
const patientId = ref(caseNumber())
const age = ref<number | ''>('')
const duration = ref('')
const symptoms = ref('')
const classificationView = ref('full')
const reliability = ref(true)
const occlusion = ref(false)
const formError = ref('')
const warning = ref('')
const running = ref(false)
const reading = ref(false)
const loadError = ref('')
const task = ref<TaskDetail>()
const savedId = ref<number>()
const progress = ref<Record<string, string>>({})
const streamText = ref('')
const overlay = ref('')
const progressLabels: Record<string, string> = { segmentation: '病灶分割', classification: '七类分类', reliability: '分割可靠性', report: '报告生成' }
const resultId = computed(() => Number(route.query.task) || undefined)
let controller: AbortController | undefined
let loadSerial = 0

function choose(next?: File) {
  if (!next) return
  formError.value = ''
  if (!['image/jpeg', 'image/jpg', 'image/png'].includes(next.type)) { formError.value = '请选择 JPG 或 PNG 图像。'; return }
  if (next.size > 50 * 1024 * 1024) { formError.value = '图像大小不能超过 50 MB。'; return }
  clearPreview()
  file.value = next
  preview.value = URL.createObjectURL(next)
}
function clearPreview() {
  if (preview.value) URL.revokeObjectURL(preview.value)
  preview.value = ''
  file.value = undefined
}
function drop(event: DragEvent) { dragDepth.value = 0; choose(event.dataTransfer?.files[0]) }
function receive(event: AnalysisEvent) {
  switch (event.type) {
    case 'meta':
      savedId.value = event.data.task_id
      progress.value.segmentation = '已完成'
      overlay.value = event.data.overlay_path ? getApiBaseUrl() + event.data.overlay_path : ''
      break
    case 'classification': case 'reliability': progress.value[event.type] = stateLabel(event.data.status); break
    case 'report_status': progress.value.report = stateLabel(event.data.status); break
    case 'chunk': streamText.value += event.data.text; break
    case 'error':
      warning.value = event.data.message
      savedId.value = event.data.task_id ?? savedId.value
      if (event.data.module) progress.value[event.data.module] = '失败'
      break
    case 'done': savedId.value = event.data.task_id; break
  }
}
async function submit() {
  if (running.value) return
  if (!file.value) { formError.value = '请选择病灶图像。'; return }
  if (age.value !== '' && (!Number.isInteger(age.value) || age.value < 1 || age.value > 150)) { formError.value = '年龄需为 1–150 的整数，也可留空。'; return }
  formError.value = warning.value = streamText.value = overlay.value = ''
  savedId.value = undefined
  progress.value = {}
  running.value = true
  const request = new AbortController()
  controller = request
  const form = new FormData()
  form.append('image', file.value)
  if (age.value !== '') form.append('patient_age', String(age.value))
  form.append('symptoms', symptoms.value.trim() || '未提供')
  form.append('duration', duration.value.trim() || '未提供')
  form.append('patient_id', patientId.value)
  form.append('classification_view', classificationView.value)
  form.append('include_reliability', String(reliability.value))
  form.append('include_occlusion', String(reliability.value && occlusion.value))
  try { await analyzeStream(form, auth.accessToken, request.signal, receive) }
  catch (err) { if (!request.signal.aborted) warning.value = err instanceof Error ? err.message : '分析请求失败。' }
  finally {
    if (!request.signal.aborted) {
      if (savedId.value) await router.replace({ name: 'workbench', query: { task: savedId.value } })
      else formError.value = warning.value || '未收到分析任务，请重试。'
      running.value = false
      window.scrollTo(0, 0)
    }
  }
}
async function loadResult(id: number) {
  const serial = ++loadSerial
  reading.value = true
  loadError.value = ''
  try { const data = await getTask(id); if (serial === loadSerial) task.value = data }
  catch { if (serial === loadSerial) loadError.value = '无法读取任务，请重试或从历史记录打开。' }
  finally { if (serial === loadSerial) reading.value = false }
}
watch(resultId, id => {
  task.value = undefined
  if (id) {
    if (id !== savedId.value) clearPreview()
    void loadResult(id)
  } else {
    ++loadSerial
    reading.value = false
    if (savedId.value) {
      clearPreview()
      age.value = ''; duration.value = symptoms.value = ''; patientId.value = caseNumber()
    }
    savedId.value = undefined
    warning.value = formError.value = ''
  }
}, { immediate: true })
onBeforeUnmount(() => { controller?.abort(); ++loadSerial; clearPreview() })
</script>
<template>
  <template v-if="resultId">
    <div class="page-heading"><h1>分析结果</h1><RouterLink to="/" class="return-link"><UiIcon name="back" />返回工作台</RouterLink></div>
    <p v-if="warning" role="status" class="form-error-box">{{ warning }}</p>
    <p v-if="reading" class="panel">正在读取分析结果…</p>
    <p v-else-if="loadError" class="form-error-box">{{ loadError }} <button class="link-btn" @click="loadResult(resultId)">重试</button></p>
    <AnalysisResults v-else-if="task" :task="task" :preview="preview" @updated="task = $event" />
  </template>
  <template v-else-if="running">
    <div class="page-heading"><h1>正在分析</h1><span class="quiet">{{ patientId }}</span></div>
    <div class="workspace-grid"><article class="panel"><h2>分析图像</h2><div class="progress-image"><img :src="overlay || preview" :alt="overlay ? '已保存的分割叠加图' : '上传原图'"></div></article>
      <article class="panel"><h2>处理进度</h2><ol class="progress-list" aria-live="polite">
        <li v-for="(title, key) in progressLabels" :key="key"><span>{{ title }}</span><strong>{{ progress[key] || '等待结果' }}</strong></li>
      </ol><p v-if="warning" class="form-error-box">{{ warning }}</p><div v-if="streamText" class="stream-preview"><h3>正在接收的报告</h3><pre>{{ streamText }}</pre></div></article>
    </div>
  </template>
  <template v-else>
    <div class="page-heading"><h1>分析工作台</h1><button class="secondary-button" @click="help = true"><UiIcon name="help" />使用帮助</button></div>
    <form class="workspace-grid" @submit.prevent="submit">
      <article class="panel upload-panel"><div class="section-heading"><h2>病灶图像</h2></div>
        <div class="dropzone" :class="{ dragging: dragDepth > 0, loaded: !!preview }" @dragenter.prevent="dragDepth++" @dragleave.prevent="dragDepth = Math.max(0, dragDepth - 1)" @dragover.prevent @drop.prevent="drop">
          <img v-if="preview" :src="preview" class="upload-preview" alt="上传图像预览"><div v-else class="upload-hint"><span class="image-mark"><UiIcon name="image" /></span><h3>拖入病灶图像</h3><p class="note">JPG 或 PNG，最大 50 MB</p></div>
          <button class="secondary-button choose-button" type="button" @click="input?.click()"><UiIcon name="upload" />{{ preview ? '更换图像' : '选择图像' }}</button>
          <input ref="input" type="file" accept="image/jpeg,image/png" hidden @change="choose(($event.target as HTMLInputElement).files?.[0]); ($event.target as HTMLInputElement).value = ''" />
        </div><p v-if="file" class="file-caption">{{ file.name }}</p>
      </article>
      <article class="panel workflow-panel">
        <h2>病例信息</h2>
        <div class="form-grid"><label class="form-field"><span>患者编号</span><input class="form-input" :value="patientId" readonly></label>
          <label class="form-field"><span>患者年龄</span><input v-model.number="age" type="number" min="1" max="150" class="form-input" placeholder="选填，岁"></label>
          <label class="form-field full"><span>病程时间</span><input v-model="duration" class="form-input" placeholder="选填，如 3 个月"></label>
          <label class="form-field full"><span>主诉症状</span><textarea v-model="symptoms" class="form-input" rows="2" maxlength="500" placeholder="选填，可描述颜色、形状或近期变化"></textarea></label></div>
        <h2 class="ruled-heading">分析选项</h2>
        <label class="form-field"><span>分类输入</span><select v-model="classificationView" class="form-input"><option value="full">完整图像</option><option value="roi">分割得到的病灶区域</option></select></label>
        <label class="check-row"><input v-model="reliability" type="checkbox"><span>评估分割稳定性 <small>额外 4 次推理</small></span></label>
        <label class="check-row"><input v-model="occlusion" type="checkbox" :disabled="!reliability"><span>增加遮挡分析 <small>再增加 4 次推理</small></span></label>
        <p class="note">启用远程分类或报告时，将发送分析所需的图像和病例资料，患者编号及账户信息除外。</p>
        <p v-if="formError" role="alert" class="form-error-box">{{ formError }}</p>
        <button class="btn-primary start-button" type="submit">开始分析<UiIcon name="arrow" /></button>
      </article>
    </form>
  </template>
  <AppDialog v-if="help" title="使用帮助" @close="help = false">
    <h3>准备图像</h3><p>选择对焦清晰、照明均匀的皮肤病灶照片，尽量让病灶占据主体。七分类模型用于皮肤镜图像。</p>
    <h3>填写与复核</h3><p>可补充年龄、病程和主诉。开始分析后可查看各模块进度，在结果页切换图层、核对候选并保存报告修订。</p>
    <p class="note">系统供具备执业资质的医护人员辅助分析。</p><div class="dialog-actions"><button class="btn-primary" @click="help = false">知道了</button></div>
  </AppDialog>
</template>
