<script setup lang="ts">
defineProps<{ name: 'image' | 'upload' | 'help' | 'arrow' | 'back' | 'download' | 'close' | 'check' | 'chevron' }>()
const paths = {
  image: 'M4 5h16v14H4z M4 16l5-5 4 4 3-3 4 4 M15.5 8.5h.01',
  upload: 'M12 16V3m-5 5 5-5 5 5 M4 16v5h16v-5',
  help: 'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z M9.5 9a2.5 2.5 0 1 1 4 2c-1 .7-1.5 1.3-1.5 2.5 M12 17h.01',
  arrow: 'M4 12h15m-6-6 6 6-6 6', back: 'M20 12H5m6-6-6 6 6 6',
  download: 'M12 3v12m-5-5 5 5 5-5 M4 16v5h16v-5',
  close: 'M6 6l12 12M18 6 6 18', check: 'M4 12l5 5L20 6', chevron: 'm6 9 6 6 6-6',
}
</script>
<template><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" class="ui-icon" aria-hidden="true"><path :d="paths[name]" /></svg></template>
