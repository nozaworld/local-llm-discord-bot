"""チャンネル単位で直近の会話履歴を保持するモジュール。

/ask, /chat, !chat, メンション応答のいずれからも共通で利用し、
直前のやり取りを踏まえた応答ができるようにする。
"""
from collections import defaultdict, deque


class ConversationStore:
    """チャンネルIDをキーに、直近の会話履歴をチャンネルごと独立して保持するクラス。

    メモリ上にのみ保持し、Bot再起動で消える（永続化はしない）"""

    def __init__(self, max_turns: int = 6):
        """max_turns往復分だけ保持するdequeを、チャンネルIDごとに遅延生成する"""
        self.max_turns = max_turns
        # deque(maxlen=...)によって、上限を超えた古いやり取りは自動的に捨てられる
        self._history = defaultdict(lambda: deque(maxlen=max_turns * 2))

    def get(self, channel_id):
        """{"role": ..., "content": ...} のリストを返す（LLMのmessages形式）"""
        return list(self._history[channel_id])

    def add_exchange(self, channel_id, user_text: str, assistant_text: str):
        """1往復分（ユーザーの発言とBotの応答）を履歴に追加する"""
        history = self._history[channel_id]
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})
