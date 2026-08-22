"""記憶しているドキュメントの管理を行うモジュール．

- ディレクトリからの自動読み込み
  include_dirs（ホワイトリスト）が指定されていればそこだけを，
  未指定ならdocs_directory全体をexclude_dirs（ブラックリスト）で除外しながら走査する
- アップロードされたドキュメントのディスクへの永続化（再起動対策）
- RAG用VectorIndexへの登録
"""
import logging
import os
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

DEFAULT_SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml", ".rst", ".yml", ".yaml",
}


class DocumentManager:
    """記憶しているドキュメント（アップロード分・自動読み込み分）を管理するクラス．

    メモリ上の辞書（self.documents）を正とし，必要に応じてディスクへの永続化と
    RAG用のVectorIndexへの登録も合わせて行う"""

    def __init__(
        self,
        docs_directory: str = ".",
        exclude_dirs=None,
        include_dirs=None,
        uploaded_docs_subdir: str = "data/uploaded_docs",
        supported_extensions=None,
        vector_index=None,
    ):
        """走査対象ディレクトリや除外/包含リストなど，ドキュメント管理に必要な設定を保持する"""
        self.documents: Dict[str, dict] = {}
        self.docs_directory = docs_directory
        self.exclude_dirs = set(exclude_dirs or set())
        self.include_dirs = set(include_dirs or set())  # 空ならホワイトリスト無効（従来通りの全体走査）
        self.uploaded_docs_dir = os.path.join(docs_directory, uploaded_docs_subdir)
        self.supported_extensions = set(supported_extensions or DEFAULT_SUPPORTED_EXTENSIONS)
        self.vector_index = vector_index  # Noneの場合はRAG検索を使わない

    def _scan_roots(self):
        """実際に走査するディレクトリの一覧を返す（ホワイトリストの有無で切り替え）"""
        # include_dirsが指定されていれば，そのディレクトリだけを走査対象にする
        if self.include_dirs:
            return [os.path.join(self.docs_directory, d) for d in sorted(self.include_dirs)]
        # 未指定ならdocs_directory全体を1つの走査対象とする（exclude_dirsで除外しながら歩く）
        return [self.docs_directory]

    async def load_documents_from_directory(self) -> int:
        """ディレクトリを再帰的に探索してドキュメントを読み込む"""
        loaded_count = 0
        try:
            # ホワイトリストが指定されていればそこだけを，なければ全体を走査対象にする
            for scan_root in self._scan_roots():
                if not os.path.isdir(scan_root):
                    logger.warning(f"読み込み対象のディレクトリが見つかりません: {scan_root}")
                    continue

                for root, dirs, files in os.walk(scan_root):
                    # ブラックリスト（exclude_dirs）は，ホワイトリスト未指定時のみ意味を持つ
                    dirs[:] = [d for d in dirs if d not in self.exclude_dirs and not d.startswith(".")]
                    for filename in files:
                        file_ext = os.path.splitext(filename)[1].lower()
                        if file_ext in self.supported_extensions:
                            file_path = os.path.join(root, filename)
                            try:
                                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                    content = f.read()
                                await self.add_document(
                                    filename=os.path.relpath(file_path, self.docs_directory),
                                    content=content,
                                    author="System (Auto-loaded)",
                                    persist=False,
                                )
                                loaded_count += 1
                            except Exception as e:
                                logger.warning(f"Failed to load {file_path}: {e}")

            logger.info(f"Loaded {loaded_count} documents from directory (recursive)")
            return loaded_count
        except Exception as e:
            logger.error(f"Error loading documents from directory: {e}")
            return 0

    async def add_document(self, filename: str, content: str, author: str, persist: bool = True) -> bool:
        """ドキュメントを追加する．persist=Trueならディスクにも保存し，再起動後も自動で復元される"""
        try:
            # persist=Trueの場合のみディスクに保存し，保存先パスを記録する
            # （個別削除の際，このパスがあるものだけ実ファイルも削除する）
            persisted_path = self._persist_to_disk(filename, content) if persist else None

            # ドキュメント本体をメモリ上の辞書に登録
            self.documents[filename] = {
                "content": content,
                "author": author,
                "timestamp": datetime.now(),
                "size": len(content),
                "persisted_path": persisted_path,
            }

            # RAG検索用のベクトルインデックスにも登録
            if self.vector_index is not None:
                await self.vector_index.index_document(filename, content)

            logger.info(f"Document added: {filename} by {author}")
            return True
        except Exception as e:
            logger.error(f"Error adding document {filename}: {e}")
            return False

    def _persist_to_disk(self, filename: str, content: str) -> str:
        """アップロード等で受け取ったドキュメントをディスクに保存し，保存先パスを返す"""
        # ファイル名にディレクトリ区切り文字が含まれる場合に備えて安全な名前に変換
        safe_name = filename.replace(os.sep, "_").replace("/", "_")

        # 保存先ディレクトリがなければ作成
        os.makedirs(self.uploaded_docs_dir, exist_ok=True)

        # 実際にファイルへ書き出す
        path = os.path.join(self.uploaded_docs_dir, safe_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return path

    async def remove_document(self, filename: str) -> bool:
        """指定した1件のドキュメントだけを記憶から削除する"""
        # 記憶している辞書から該当ドキュメントを取り出す（存在しなければNone）
        doc_data = self.documents.pop(filename, None)
        if doc_data is None:
            return False

        # アップロード等でディスクに保存されていた実ファイルがあれば，それも削除する
        # （起動時に自動読み込みしただけのドキュメントはpersisted_pathを持たないため，
        #   プロジェクト内の元ファイルまでは削除しない）
        persisted_path = doc_data.get("persisted_path")
        if persisted_path and os.path.exists(persisted_path):
            try:
                os.remove(persisted_path)
            except OSError as e:
                logger.warning(f"永続化ファイルの削除に失敗しました: {persisted_path} ({e})")

        # RAG検索用のベクトルインデックスからも該当ドキュメントのチャンクを削除
        if self.vector_index is not None:
            self.vector_index.remove_document(filename)

        logger.info(f"Document removed: {filename}")
        return True

    def get_document_list(self) -> List[Dict]:
        """/list_docsの表示用に，本文を含まないメタ情報だけのリストを返す"""
        return [
            {
                "filename": filename,
                "author": doc_data["author"],
                "timestamp": doc_data["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "size": doc_data["size"],
            }
            for filename, doc_data in self.documents.items()
        ]

    def clear_documents(self):
        """記憶している全ドキュメントをRAGインデックスごと削除する（/clear_docs用）"""
        self.documents.clear()
        if self.vector_index is not None:
            self.vector_index.clear()
        logger.info("All documents cleared")
