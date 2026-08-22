"""ドキュメントのチャンク分割・埋め込み・類似検索（簡易RAG）を担当するモジュール．

Ollamaの埋め込みモデルが利用できない場合は自動的に無効化し、
呼び出し側（document_manager / bot）は全文書をそのまま渡す方式にフォールバックできる．
"""
import logging
import math

logger = logging.getLogger(__name__)


def chunk_text(text: str, max_chars: int = 800):
    """段落単位でまとめつつ、max_charsを超えないようにチャンク化する"""
    paragraphs = text.split("\n\n")
    chunks = []
    buf = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) + 1 <= max_chars:
            buf = f"{buf}\n{para}" if buf else para
        else:
            if buf:
                chunks.append(buf)
            if len(para) > max_chars:
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i : i + max_chars])
                buf = ""
            else:
                buf = para
    if buf:
        chunks.append(buf)
    return chunks


def cosine_similarity(a, b) -> float:
    """2つの埋め込みベクトルのコサイン類似度を返す（-1〜1、無効な入力は0.0）"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorIndex:
    """文書チャンクの埋め込みベクトルを保持し、質問に関連する箇所だけを検索する"""

    def __init__(self, llm_client, chunk_chars: int = 800):
        """埋め込み取得に使うllm_clientと、チャンク分割時の1チャンクあたりの文字数を保持する"""
        self.llm_client = llm_client
        self.chunk_chars = chunk_chars
        self.entries = []  # [{filename, chunk, vector}]
        self.embedding_available = True

    def clear(self):
        """保持しているチャンク・ベクトルを全て削除する（/clear_docsと連動）"""
        self.entries.clear()

    async def index_document(self, filename: str, content: str):
        """ドキュメントをチャンクに分割し、それぞれ埋め込みベクトルを取得してentriesへ追加する．
        埋め込み取得に一度でも失敗した場合はembedding_availableをFalseにし、
        以降はRAG検索を使わない（全文渡しへのフォールバック）方式に切り替える"""
        if not self.embedding_available:
            return
        chunks = chunk_text(content, self.chunk_chars)
        for chunk in chunks:
            try:
                vector = await self.llm_client.embed(chunk)
            except Exception as e:
                logger.warning(f"埋め込み取得に失敗したため、RAG検索を無効化し全文渡しにフォールバックします: {e}")
                self.embedding_available = False
                return
            if vector:
                self.entries.append({"filename": filename, "chunk": chunk, "vector": vector})

    def remove_document(self, filename: str):
        """指定したファイル名に属するチャンクだけをentriesから取り除く（/delete_doc用）"""
        self.entries = [e for e in self.entries if e["filename"] != filename]

    async def search(self, query: str, top_k: int = 4):
        """質問文を埋め込みベクトル化し、コサイン類似度が高い順にtop_k件のチャンクを返す．
        類似度0以下（無関係）のチャンクは結果から除外する"""
        if not self.embedding_available or not self.entries:
            return []
        try:
            query_vector = await self.llm_client.embed(query)
        except Exception as e:
            logger.warning(f"質問の埋め込み取得に失敗しました: {e}")
            self.embedding_available = False
            return []

        scored = [(cosine_similarity(query_vector, e["vector"]), e) for e in self.entries]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for score, e in scored[:top_k] if score > 0]
