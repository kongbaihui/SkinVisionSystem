import { marked } from 'marked'
import DOMPurify from 'dompurify'

export function renderReport(markdown: string): string {
  return DOMPurify.sanitize(marked.parse(markdown) as string, {
    USE_PROFILES: { html: true }, FORBID_TAGS: ['img', 'style', 'form', 'input'],
  })
}
