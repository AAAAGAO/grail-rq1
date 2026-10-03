from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


TOKYO_ENDPOINT = (
    "https://ws-l8mmsqf3krhsrnad.ap-northeast-1.maas.aliyuncs.com/"
    "compatible-mode/v1"
)
DEFAULT_MODEL = "qwen3.8-flash"


def normalize_endpoint(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    suffix = "/chat/completions"
    return value[:-len(suffix)] if value.endswith(suffix) else value


class CachedChat:
    def __init__(self, args):
        self.args = args
        self.calls = Path(args.output_dir) / "calls"
        self.calls.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(name: str) -> str:
        value = os.getenv(name)
        if value:
            return value
        raise RuntimeError(
            f"Environment variable {name} is not available in this process"
        )

    def _request(
        self,
        endpoint: str,
        model: str,
        key: str,
        system: str,
        user: str,
    ) -> dict:
        payload = {
            "model": model,
            "temperature": 0,
            "max_tokens": 1600,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if model.startswith("qwen3.8-"):
            payload["enable_thinking"] = False
        digest = hashlib.sha256(
            json.dumps(
                {
                    "endpoint": normalize_endpoint(endpoint),
                    "payload": payload,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        path = self.calls / f"{digest}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        request = Request(
            normalize_endpoint(endpoint) + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.time()
        with urlopen(request, timeout=150) as response:
            body = json.load(response)
        choice = body["choices"][0]
        record = {
            "provider_model": body.get("model", model),
            "content": choice["message"].get("content", ""),
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage", {}),
            "seconds": time.time() - started,
        }
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return record

    def call(self, system: str, user: str) -> dict:
        record = self._request(
            self.args.endpoint,
            self.args.model,
            self._key(self.args.api_key_env),
            system,
            user,
        )
        record["provider"] = "aliyun_tokyo"
        return record
