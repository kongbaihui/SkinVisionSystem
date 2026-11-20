import base64
import json
import os

import httpx

_DASHSCOPE_URL = (
	"https://dashscope.aliyuncs.com/api/v1/services/aigc/"
	"multimodal-generation/generation"
)


class LLMGateway:
	def __init__(self, api_key: str = None, timeout: float = 60.0, model: str = "qwen3-vl-plus", temperature=None, max_tokens=None):
		self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY", "")
		self.timeout = timeout
		self.model = model
		self.parameters = {"incremental_output": True}
		if temperature is not None:
			self.parameters['temperature'] = temperature
		if max_tokens is not None:
			self.parameters['max_tokens'] = max_tokens

	async def stream_markdown(self, prompt: str, context: dict = None, roi_image_bytes: bytes = None):
		context = context or {}
		user_text = _build_user_text(context)
		user_content = []
		if roi_image_bytes is not None:
			b64 = base64.b64encode(roi_image_bytes).decode("utf-8")
			user_content.append({"image": f"data:image/png;base64,{b64}"})
		user_content.append({"text": user_text})
		messages = [
			{"role": "system", "content": [{"text": prompt}]},
			{"role": "user", "content": user_content},
		]
		payload = {
			"model": self.model,
			"input": {"messages": messages},
			"parameters": self.parameters,
		}
		headers = {
			"Authorization": f"Bearer {self.api_key}",
			"Content-Type": "application/json",
			"X-DashScope-SSE": "enable",
		}
		async with httpx.AsyncClient(timeout=self.timeout) as client:
			async with client.stream("POST", _DASHSCOPE_URL, headers=headers, json=payload) as response:
				response.raise_for_status()
				has_text = False
				async for line in response.aiter_lines():
					if not line or not line.startswith("data:"):
						continue
					raw = line[5:].strip()
					if raw == "[DONE]":
						break
					try:
						data = json.loads(raw)
						if data.get('code'):
							raise ValueError('Report provider returned an error')
						message = data["output"]["choices"][0]["message"]
						for item in message.get("content", []):
							text = item.get("text", "")
							if text:
								has_text = True
								yield text
					except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
						raise ValueError('Invalid report provider response') from exc
				if not has_text:
					raise ValueError('Empty report provider response')


def _build_user_text(context: dict) -> str:
	from backend.skinvision.ai.report import build_report_input
	return build_report_input(context)
