"""Ollama（ローカルLLM）との通信を担当するモジュール．

通常応答・ストリーミング応答・埋め込みベクトル取得（RAG用）をまとめている．
"""
import asyncio
import json
import logging

import requests

logger = logging.getLogger(__name__)


class LocalLLMClient:
    """ローカルLLM (Ollama) クライアント"""

    def __init__(self, base_url="http://localhost:11434", model="llama2", embed_model="nomic-embed-text"):
        """接続先や使用モデルなど、Ollamaとの通信に必要な設定を保持する"""
        self.base_url = base_url
        self.model = model
        self.embed_model = embed_model
        self.timeout = 300

    def _build_messages(self, prompt, system_prompt=None, history=None):
        """system_prompt・過去の会話履歴・今回のユーザー入力を
        Ollamaの/api/chatが要求するmessages形式（role/contentのリスト）に組み立てる"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate_response(self, prompt: str, system_prompt: str = None, history=None) -> str:
        """ローカルLLMで応答を生成する（一括取得）"""
        messages = self._build_messages(prompt, system_prompt, history)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 500},
        }

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout),
            )
        except requests.exceptions.ConnectionError:
            raise Exception("Ollamaサーバーに接続できません．Ollamaが起動しているか確認してください．")
        except requests.exceptions.Timeout:
            raise Exception("LLMの応答がタイムアウトしました．")

        if response.status_code != 200:
            raise Exception(f"LLM API Error: {response.status_code}")

        result = response.json()
        return result["message"]["content"].strip()

    async def generate_response_stream(self, prompt: str, system_prompt: str = None, history=None):
        """ローカルLLMの応答をストリーミングで取得する．チャンクごとにテキストをyieldする．"""
        messages = self._build_messages(prompt, system_prompt, history)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0.1, "num_predict": 500},
        }

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def worker():
            """別スレッドで実行され、Ollamaからのストリーミング応答を1行ずつ読み取って
            queueに詰めていく（requestsは同期APIのため、asyncioの外側で動かす必要がある）"""
            try:
                with requests.post(
                    f"{self.base_url}/api/chat", json=payload, timeout=self.timeout, stream=True
                ) as resp:
                    if resp.status_code != 200:
                        loop.call_soon_threadsafe(queue.put_nowait, ("error", f"LLM API Error: {resp.status_code}"))
                        return
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
                        if data.get("done"):
                            break
            except requests.exceptions.ConnectionError:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", "Ollamaサーバーに接続できません．"))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        loop.run_in_executor(None, worker)

        while True:
            kind, value = await queue.get()
            if kind == "chunk":
                yield value
            elif kind == "error":
                raise Exception(value)
            elif kind == "done":
                break

    async def embed(self, text: str):
        """テキストを埋め込みベクトルに変換する（RAG用）"""
        payload = {"model": self.embed_model, "prompt": text}
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(f"{self.base_url}/api/embeddings", json=payload, timeout=self.timeout),
            )
        except requests.exceptions.ConnectionError:
            raise Exception("Ollamaサーバーに接続できません．")
        if response.status_code != 200:
            raise Exception(f"Embedding API Error: {response.status_code}")
        return response.json().get("embedding")

    async def check_connection(self) -> bool:
        """LLMサーバーの接続確認"""
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None, lambda: requests.get(f"{self.base_url}/api/tags", timeout=5)
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
