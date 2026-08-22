"""
Bot全体の設定を環境変数（.env）から読み込むモジュール．

対応する環境変数:
    DISCORD_BOT_TOKEN / TOKEN   Discord Botのトークン（必須）
    OLLAMA_BASE_URL             OllamaサーバーのURL（既定: http://localhost:11434）
    OLLAMA_MODEL                会話に使うモデル名（既定: gemma3:12B）
    OLLAMA_EMBED_MODEL          RAG用の埋め込みモデル名（既定: nomic-embed-text）
    DOCS_DIRECTORY              自動読み込み対象のディレクトリ（既定: カレントディレクトリ）
    DOCS_INCLUDE_DIRS           自動読み込みの対象とするディレクトリ名（カンマ区切り、ホワイトリスト）
                                 指定した場合、このリストにあるディレクトリの中だけを走査する
    DOCS_EXCLUDE_DIRS           自動読み込みから除外するディレクトリ名（カンマ区切り）
                                 DOCS_INCLUDE_DIRSが未指定の場合のみ使われるブラックリスト
    MASCOT_NAME                 マスコットの名前
    MASCOT_PERSONA              マスコットの性格・話し方の説明
    MASCOT_ENDING                回答文末に付ける語尾（空なら指定しない）
    CONVERSATION_HISTORY_TURNS  会話履歴として保持するやり取りの往復数
    ASK_COOLDOWN_SECONDS        /ask のクールダウン秒数
    CHAT_COOLDOWN_SECONDS       /chat のクールダウン秒数
    RAG_TOP_K                   RAG検索で取得するチャンク数
    RAG_CHUNK_CHARS             ドキュメントを分割する際の1チャンクあたりの文字数
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: str = "") -> str:
    """環境変数を取得する．未設定または空文字の場合はdefaultを返す
    （os.getenvだけだと空文字とNoneを区別してしまうため、ここで吸収する）"""
    value = os.getenv(name)
    return value if value not in (None, "") else default


DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN") or os.getenv("TOKEN")

OLLAMA_BASE_URL = _get_env("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = _get_env("OLLAMA_MODEL", "gemma3:12B")
OLLAMA_EMBED_MODEL = _get_env("OLLAMA_EMBED_MODEL", "nomic-embed-text")

DOCS_DIRECTORY = _get_env("DOCS_DIRECTORY", ".")

# ホワイトリスト: 指定されていれば、このディレクトリだけを自動読み込みの対象にする
# （公開用の既定値は空にしてあるため、自分の環境に合わせて.envで設定する）
DOCS_INCLUDE_DIRS = {
    d.strip()
    for d in _get_env("DOCS_INCLUDE_DIRS", "").split(",")
    if d.strip()
}

# ブラックリスト: DOCS_INCLUDE_DIRSが未指定のときだけ使われる、走査から除外するディレクトリ
DOCS_EXCLUDE_DIRS = {
    d.strip()
    for d in _get_env(
        "DOCS_EXCLUDE_DIRS",
        "myenv,.venv,venv,__pycache__,.git,node_modules,private,data,_to_delete",
    ).split(",")
    if d.strip()
}

MASCOT_NAME = _get_env("MASCOT_NAME", "ミーティー")
MASCOT_PERSONA = _get_env("MASCOT_PERSONA", "元気で丁寧．会議の合意事項とTODOを必ず要約する癖がある．")
MASCOT_ENDING = _get_env("MASCOT_ENDING", "")

CONVERSATION_HISTORY_TURNS = int(_get_env("CONVERSATION_HISTORY_TURNS", "6"))
ASK_COOLDOWN_SECONDS = int(_get_env("ASK_COOLDOWN_SECONDS", "20"))
CHAT_COOLDOWN_SECONDS = int(_get_env("CHAT_COOLDOWN_SECONDS", "10"))

RAG_TOP_K = int(_get_env("RAG_TOP_K", "4"))
RAG_CHUNK_CHARS = int(_get_env("RAG_CHUNK_CHARS", "800"))
