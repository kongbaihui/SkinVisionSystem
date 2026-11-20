import { createPinia } from 'pinia'
import { createApp } from 'vue'

import App from './App.vue'
import { useAuthStore } from '@/stores/auth'
import router from './router'
import './styles.css'

const app = createApp(App)
const pinia = createPinia()

app.use(pinia)

// 页面刷新时恢复本地用户快照
useAuthStore(pinia).restoreSession()

app.use(router)
app.mount('#app')
