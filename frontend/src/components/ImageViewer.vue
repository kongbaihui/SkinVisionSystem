<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { http, getApiBaseUrl } from '@/services/http'
import type { TaskDetail } from '@/types/diagnostic'

const props = defineProps<{ task: TaskDetail; preview?: string }>()
const layers = [{ id: 'original', label: '原图' }, { id: 'overlay', label: '分割叠加' }, { id: 'gradcam', label: '热力图' }] as const
type Layer = typeof layers[number]['id']
const active = ref<Layer>(props.task.overlay_path ? 'overlay' : 'original')
const originalUrl = ref('')
const ratio = ref<number | null>(null)
const loading = ref(false)
const failed = ref<Partial<Record<Layer, boolean>>>({})
const api = getApiBaseUrl()
const urls = computed(() => ({
  original: originalUrl.value,
  overlay: props.task.overlay_path ? api + props.task.overlay_path : '',
  gradcam: props.task.gradcam_path ? api + props.task.gradcam_path : '',
}))
const aligning = computed(() => ratio.value === null && !failed.value.original && !!(props.task.original_path || props.preview))
watch([() => props.task.original_path, () => props.preview], async ([path, preview], _, onCleanup) => {
  let valid = true
  let objectUrl = ''
  const controller = new AbortController()
  onCleanup(() => {
    valid = false
    controller.abort()
    if (objectUrl) URL.revokeObjectURL(objectUrl)
  })
  originalUrl.value = preview || ''
  failed.value = {}
  ratio.value = null
  loading.value = !!path && !preview
  if (!path || preview) return
  try {
    const { data } = await http.get<Blob>(path, { responseType: 'blob', signal: controller.signal })
    if (valid) originalUrl.value = objectUrl = URL.createObjectURL(data)
  } catch {
    if (valid && !preview) failed.value.original = true
  } finally {
    if (valid) loading.value = false
  }
}, { immediate: true })
function measure(event: Event) {
  const img = event.target as HTMLImageElement
  if (img.naturalHeight) ratio.value = img.naturalWidth / img.naturalHeight
}
</script>
<template>
  <article class="panel viewer-panel">
    <h2>图像对照</h2>
    <div class="segments layer-controls" role="group" aria-label="图像图层">
      <button v-for="layer in layers" :key="layer.id" :aria-pressed="active === layer.id" @click="active = layer.id">{{ layer.label }}</button>
    </div>
    <div class="view-canvas">
      <div class="registered-frame" :style="{ '--ratio': ratio ?? 1 }">
        <template v-for="layer in layers" :key="layer.id">
          <img v-if="urls[layer.id] && !failed[layer.id]" v-show="active === layer.id && !aligning"
            :src="urls[layer.id]" :alt="layer.id === 'gradcam' ? 'Grad-CAM 热力图' : layer.label" @load="layer.id === 'original' && measure($event)" @error="failed[layer.id] = true" />
        </template>
      </div>
      <p v-if="aligning" class="canvas-note">正在读取图像尺寸…</p>
      <p v-else-if="failed[active]" class="canvas-note">此图像加载失败，请重新打开任务。</p>
      <p v-else-if="!urls[active]" class="canvas-note">{{ active === 'original' ? (loading ? '正在读取原图…' : '此记录未保存原图') : active === 'gradcam' ? '本次分析没有 Grad-CAM 热力图' : '本次分析没有分割叠加图' }}</p>
    </div>
    <div class="viewer-footer"><span>任务 #{{ task.id }}</span><span>{{ active === 'gradcam' ? 'Grad-CAM · 模型关注区域' : active === 'overlay' ? '分割叠加 · 病灶区域' : '上传原图' }}</span></div>
    <p v-if="active !== 'original' && !task.original_path && !preview" class="note">此记录未保存原图。</p>
  </article>
</template>
