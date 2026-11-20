import { defineStore } from 'pinia'
import { ref } from 'vue'

import { getHealth } from '@/services/diagnostic'

export const useAppStore = defineStore('app', () => {
  const pipelineMode = ref('')
  const healthLoading = ref(false)
  const healthError = ref('')

  async function fetchHealth() {
    healthLoading.value = true
    healthError.value = ''

    try {
      const response = await getHealth()
      pipelineMode.value = response.pipeline_mode
    } catch (error) {
      healthError.value = error instanceof Error ? error.message : 'health check failed'
    } finally {
      healthLoading.value = false
    }
  }

  return {
    pipelineMode,
    healthLoading,
    healthError,
    fetchHealth,
  }
})