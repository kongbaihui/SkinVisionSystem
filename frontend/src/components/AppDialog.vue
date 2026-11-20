<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, useId } from 'vue'
import UiIcon from './UiIcon.vue'

defineProps<{ title: string }>()
const emit = defineEmits<{ close: [] }>()
const dialog = ref<HTMLDialogElement>()
const titleId = useId()
let trigger: HTMLElement | null = null

onMounted(() => {
  trigger = document.activeElement instanceof HTMLElement ? document.activeElement : null
  dialog.value?.showModal()
})
onBeforeUnmount(() => {
  dialog.value?.close()
  trigger?.focus()
})
</script>
<template>
  <Teleport to="body">
    <dialog ref="dialog" class="app-dialog" :aria-labelledby="titleId" @cancel.prevent="emit('close')" @click="($event.target === dialog) && emit('close')">
      <div class="dialog-inner">
        <header class="section-heading"><h2 :id="titleId">{{ title }}</h2><button type="button" class="icon-btn" aria-label="关闭" @click="emit('close')"><UiIcon name="close" /></button></header>
        <slot />
      </div>
    </dialog>
  </Teleport>
</template>
