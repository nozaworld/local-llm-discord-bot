# ローカルLLM Discord Bot

## 使用技術

<p style="display: inline">
  <img src="https://img.shields.io/badge/-Python-3776AB.svg?logo=python&style=for-the-badge&logoColor=white">
  <img src="https://img.shields.io/badge/-Discord-5865F2.svg?logo=discord&style=for-the-badge&logoColor=white">
  <img src="https://img.shields.io/badge/-Ollama-000000.svg?style=for-the-badge">
</p>

## 概要

ローカルLLM（Ollama）によるドキュメントQ&A・自由会話・マスコットペルソナをまとめたDiscord Botです．

アップロードしたドキュメントをRAG（検索拡張生成）で参照しながら質問に答えるほか，マスコットとしてのキャラクター設定を持たせた自由会話も行えます．

## 主な機能

- `/ask` : アップロード・記憶させたドキュメントの内容に基づいて質問に回答します．ドキュメント量に応じて，RAG検索（関連チャンクのみ渡す方式）と全文渡しに自動で切り替わります．
- `/chat`，`!chat`，メンション : ローカルLLMと自由に会話します．応答はストリーミングで随時更新されます．
- `/upload_doc` : ファイルをアップロードして記憶させます．
- `/list_docs`，`/delete_doc`，`/clear_docs` : 記憶しているドキュメントの一覧表示，1件だけの削除，全削除を行います（削除系は管理者のみ）．
- `/set_model`，`/set_persona` : 使用するLLMモデルやマスコットの名前・性格・語尾を変更します（管理者のみ）．

## 制約

- 現状は1つのDiscordサーバーでの利用を想定しています．複数サーバーに導入した場合，記憶しているドキュメントや会話履歴がサーバーをまたいで共有されます．
- 埋め込みベクトルはメモリ上にのみ保持しており，永続化していません．Bot再起動のたびに全ドキュメントの再インデックスが発生します．
- ドキュメント検索はコサイン類似度による線形探索であり，大量のドキュメントを扱う用途には向いていません．

## 要件

- Python 3.10以上
- [Ollama](https://ollama.ai/)（ローカルLLMサーバー本体）
- 依存ライブラリは`requirements.txt`を参照してください．

```bash
pip install -r requirements.txt
```

## 使い方

1. リポジトリの取得と仮想環境の作成

```bash
git clone <このリポジトリのURL>
cd discord-bot
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

2. Ollamaのインストールとモデルの取得

```bash
ollama pull gemma3:12b
ollama pull nomic-embed-text   # RAG検索を使う場合のみ必要
ollama serve
```

3. 環境変数の設定

```bash
cp .env.example .env
```

`.env`を開き，`DISCORD_BOT_TOKEN`にDiscord Developer Portalで発行したBotトークンを設定します．その他の項目は任意で，未設定の場合は`config.py`の既定値が使われます．

4. Discord Developer Portal側の設定

`/great_mention`コマンドがサーバーの全メンバー一覧を取得するため，Developer PortalでBotの`SERVER MEMBERS INTENT`を有効にしてください．`MESSAGE CONTENT INTENT`も併せて必要です．

5. 起動

```bash
python bot.py
```

## プロジェクト構成

- `bot.py`
  Bot本体．各モジュールを読み込んでコマンドを定義する部分
- `config.py`
  設定．環境変数（.env）の読み込みを行う
- `llm_client.py`
  Ollamaとの通信．通常応答・ストリーミング応答・埋め込み取得を担当する
- `document_manager.py`
  記憶しているドキュメントの管理・永続化を行う
- `rag.py`
  ドキュメントのベクトル検索（簡易RAG）を行う
- `persona.py`
  マスコットのキャラクター設定を管理する
- `conversation.py`
  チャンネル単位の会話履歴を保持する
- `private/`
  過去に作成した単体スクリプト（録音専用Bot，LoRA学習ツール）．リポジトリには含めていません．

## ライセンス

このプロジェクトはMITライセンスのもとで公開されています．詳細は[LICENSE](./LICENSE)を参照してください．

## 補足

- Bot起動時に読み込むドキュメントのディレクトリは，既定でカレントディレクトリ全体です．除外したいフォルダがあれば`.env`の`DOCS_EXCLUDE_DIRS`に追加してください．
- アップロードしたドキュメントは`data/`に保存されます．リポジトリには含めていません．
