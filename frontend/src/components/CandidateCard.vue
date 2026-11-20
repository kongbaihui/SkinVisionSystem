<script setup lang="ts">
import { computed } from 'vue'
import { stateLabel } from '@/services/tasks'
import type { ClassificationResult } from '@/types/diagnostic'

const props = defineProps<{ result?: ClassificationResult | null }>()
const names: Record<string, string> = {
  NV: '黑色素细胞痣', MEL: '黑色素瘤', BCC: '基底细胞癌', AKIEC: '日光性角化病／鲍温病',
  BKL: '良性角化性病变', DF: '皮肤纤维瘤', VASC: '血管性病变',
}
const candidates = computed(() => props.result?.candidates?.length ? props.result.candidates : props.result?.label ? [props.result.label] : [])
</script>
<template>
  <article class="panel tint-card">
    <div class="section-heading"><h2>类别候选</h2><span class="quiet">七分类 · {{ stateLabel(result?.status) }}</span></div>
    <ul v-if="candidates.length" class="candidate-list">
      <li v-for="code in candidates" :key="code"><span>{{ names[code] ?? code }}</span><small>{{ code }}</small></li>
    </ul>
    <p v-else class="note">{{ result?.reason || '该记录没有分类候选。' }}</p>
    <details class="module-details"><summary>分类依据与模型来源</summary>
      <p v-if="candidates.length">{{ result?.reason || '请结合原图与临床资料复核。' }}</p>
      <dl class="facts"><dt>模型</dt><dd>{{ result?.model || '未记录' }}</dd><dt>来源</dt><dd>{{ result?.source || '未记录' }}</dd>
        <template v-if="result?.version || result?.prompt_version"><dt>版本</dt><dd>{{ result.version || result.prompt_version }}</dd></template>
        <template v-if="result?.revision"><dt>修订号</dt><dd>{{ result.revision }}</dd></template>
        <template v-if="result?.request_settings?.view"><dt>分类输入</dt><dd>{{ result.request_settings.view === 'roi' ? '病灶区域' : '完整图像' }}</dd></template>
      </dl>
      <template v-if="result?.scores"><p class="note">{{ result.score_semantics }}</p><ul class="scores"><li v-for="(score, code) in result.scores" :key="code">{{ names[code] ?? code }} <span>{{ score.toFixed(4) }}</span></li></ul></template>
    </details>
  </article>
</template>
