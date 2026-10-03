import json
import os
import urllib.request


DEFAULT_ENDPOINT = "https://ws-127o6gshpzqjohcm.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.8-flash"


class Client:
    def __init__(self, endpoint=None, model=None, key_env="DASHSCOPE_API_KEY"):
        self.endpoint = (endpoint or os.environ.get("ALIYUN_ENDPOINT", DEFAULT_ENDPOINT)).rstrip("/")
        self.model = model or os.environ.get("ALIYUN_MODEL", DEFAULT_MODEL)
        self.key_env = key_env

    def call(self, system, user, max_tokens=1200):
        key = os.environ.get(self.key_env)
        if not key:
            raise RuntimeError(f"missing {self.key_env}")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "enable_thinking": False,
        }
        request = urllib.request.Request(
            self.endpoint + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.load(response)
        return body["choices"][0]["message"].get("content", "")


def parse_object(content):
    content = content.strip()
    if "{" in content and "}" in content:
        content = content[content.index("{"):content.rindex("}") + 1]
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("the model response is not an object")
    return value
