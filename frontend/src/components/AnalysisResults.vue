<script setup lang="ts">
import ImageViewer from './ImageViewer.vue'
import CandidateCard from './CandidateCard.vue'
import ReportEditor from './ReportEditor.vue'
import { getApiBaseUrl } from '@/services/http'
import { stateLabel } from '@/services/tasks'
import { formatDate, riskClass, type TaskDetail } from '@/types/diagnostic'

defineProps<{ task: TaskDetail; preview?: string }>()
const emit = defineEmits<{ updated: [task: TaskDetail] }>()
const api = getApiBaseUrl()
const measurements: Record<string, string> = { area_ratio: '面积占比（%）', irregularity_index: '不规则指数',
  color_variance: '色彩变异度', asymmetry_index: '对称性指数', border_sharpness: '边界锐利度', color_diversity: '多色分布', max_diameter_ratio: '直径比' }
</script>
<template>
  <div class="result-grid">
    <ImageViewer :key="task.id" :task="task" :preview="preview" />
    <div class="review-column">
      <CandidateCard :result="task.classification" />
      <ReportEditor :key="task.id" :task="task" @updated="emit('updated', $event)" />
    </div>
  </div>
  <div class="evidence-grid">
    <details class="panel tint-card evidence"><summary>病例信息<span>{{ task.patient_id || '任务 #' + task.id }}</span></summary>
      <dl class="facts"><dt>年龄</dt><dd>{{ task.patient_age == null ? '未提供' : task.patient_age + ' 岁' }}</dd>
        <dt>病程</dt><dd>{{ task.patient_duration || '未提供' }}</dd><dt>主诉症状</dt><dd>{{ task.patient_symptoms || '未提供' }}</dd>
        <dt>分析时间</dt><dd>{{ formatDate(task.created_at) }}</dd></dl></details>
    <details class="panel tint-card evidence"><summary>分割与量化特征<span>{{ stateLabel(task.status) }}</span></summary>
      <dl class="facts"><dt>来源</dt><dd>{{ task.model_info?.execution === 'model' ? '真实模型' : task.model_info?.execution === 'fallback' ? '传统算法降级' : '未记录' }} · {{ task.model_info?.name || task.pipeline_mode }}</dd>
        <dt>设备</dt><dd>{{ task.model_info?.device || '未记录' }}</dd><dt>推理耗时</dt><dd>{{ task.inference_time_ms == null ? '未记录' : task.inference_time_ms.toFixed(0) + ' ms' }}</dd>
        <dt>模型解释</dt><dd>{{ stateLabel(task.model_info?.explanation_status) }}</dd>
        <template v-for="(name, key) in measurements" :key="key"><template v-if="task.features?.[key] != null"><dt>{{ name }}</dt><dd>{{ task.features[key]?.toFixed(key === 'area_ratio' ? 2 : 4) }}</dd></template></template>
      </dl>
      <p><strong>既有规则提示</strong> <span class="risk-badge" :class="riskClass(task.risk_level)">{{ task.risk_level || '未记录' }}</span></p>
      <p class="note">{{ task.risk_source || '规则来源未记录。' }}</p>
      <a v-if="task.probability_path" class="text-link" :href="api + task.probability_path" download>下载像素预测数组</a></details>
    <details class="panel tint-card evidence"><summary>分割可靠性<span>{{ stateLabel(task.reliability?.status) }}</span></summary>
      <template v-if="task.reliability?.status === 'completed'">
        <dl class="facts"><dt>平均预测方差</dt><dd>{{ task.reliability.mean_variance?.toFixed(6) }}</dd><dt>95% 分位数</dt><dd>{{ task.reliability.p95_variance?.toFixed(6) }}</dd>
          <dt>额外推理</dt><dd>{{ task.reliability.extra_prediction_count }} 次</dd><dt>耗时</dt><dd>{{ ((task.reliability.elapsed_ms || 0) / 1000).toFixed(2) }} 秒</dd></dl>
        <img v-if="task.reliability.variance_image_path" class="module-map" :src="api + task.reliability.variance_image_path" alt="预测方差图" loading="lazy">
        <p class="note">固定方差色阶 0–0.25，反映分割预测的变化。</p>
        <template v-if="task.reliability.occlusion_image_path"><img class="module-map" :src="api + task.reliability.occlusion_image_path" alt="遮挡后的预测变化" loading="lazy"><p class="note">遮挡敏感性 · {{ ((task.reliability.occlusion?.elapsed_ms || 0) / 1000).toFixed(2) }} 秒</p></template>
      </template><p v-else class="note">{{ task.reliability?.reason || '该记录没有可靠性评估。' }}</p>
    </details>
  </div>
  <p class="clinical-note">本系统仅供医护人员辅助分析，最终诊疗判断由执业医师作出。</p>
</template>
