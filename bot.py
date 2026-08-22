"""
Discord Bot 本体。

ローカルLLM (Ollama) によるドキュメントQ&A・自由会話・マスコットペルソナを提供する。

機能ごとに以下のモジュールへ分割し、本ファイルから参照する構成にしている。
    config.py             環境変数からの設定読み込み
    llm_client.py         Ollamaとの通信（通常応答・ストリーミング・埋め込み）
    document_manager.py   記憶しているドキュメントの管理・永続化
    rag.py                ドキュメントのベクトル検索（簡易RAG）
    persona.py            マスコットのキャラクター設定
    conversation.py       チャンネル単位の会話履歴
"""
import logging
import os
import re

import discord
from discord.ext import commands

import config
from conversation import ConversationStore
from document_manager import DocumentManager
from llm_client import LocalLLMClient
from persona import Persona
from rag import VectorIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentBot(discord.Bot):
    """ドキュメント対応Discord Bot（ローカルLLM使用）"""

    def __init__(self, *args, **kwargs):
        """各機能モジュール（LLM通信・RAG・ドキュメント管理・ペルソナ・会話履歴）を
        まとめて初期化し、self経由でコマンド側から参照できるようにする"""
        super().__init__(*args, **kwargs)

        self.llm_client = LocalLLMClient(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL,
            embed_model=config.OLLAMA_EMBED_MODEL,
        )
        self.vector_index = VectorIndex(self.llm_client, chunk_chars=config.RAG_CHUNK_CHARS)
        self.doc_manager = DocumentManager(
            docs_directory=config.DOCS_DIRECTORY,
            exclude_dirs=config.DOCS_EXCLUDE_DIRS,
            include_dirs=config.DOCS_INCLUDE_DIRS,
            vector_index=self.vector_index,
        )
        self.persona = Persona(
            name=config.MASCOT_NAME,
            description=config.MASCOT_PERSONA,
            ending=config.MASCOT_ENDING,
        )
        self.conversations = ConversationStore(max_turns=config.CONVERSATION_HISTORY_TURNS)

    async def read_attachment_content(self, attachment: discord.Attachment) -> str:
        """添付ファイルの内容を読み取り"""
        if attachment.size > 8 * 1024 * 1024:
            raise ValueError("ファイルサイズが大きすぎます (8MB制限)")

        # 対応拡張子はdoc_manager側の定義をそのまま使う
        # （ここで別に持つと、ディレクトリ自動読み込み側とずれる可能性があるため）
        file_ext = os.path.splitext(attachment.filename)[1].lower()
        if file_ext not in self.doc_manager.supported_extensions:
            raise ValueError(f"サポートされていないファイル形式: {file_ext}")

        file_bytes = await attachment.read()
        try:
            return file_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            raise ValueError(f"ファイル読み取りエラー: {str(e)}")


# Botの初期化
intents = discord.Intents.default()
intents.message_content = True
intents.members = True  # /great_mention でのメンバー一覧取得に必要（Developer Portal側でも有効化が必要）

bot = DocumentBot(intents=intents)


@bot.event
async def on_ready():
    """Discordへのログイン完了時に1度だけ呼ばれる。
    ドキュメントの自動読み込みとOllamaへの接続確認をここで行う"""
    print(f"ログインしました: {bot.user}")

    print("ディレクトリからドキュメントを読み込み中...")
    loaded_count = await bot.doc_manager.load_documents_from_directory()
    print(f"{loaded_count} 個のドキュメントを読み込みました")

    print("ローカルLLMサーバーの接続を確認中...")
    if await bot.llm_client.check_connection():
        print("ローカルLLMサーバーに接続しました")
        print(f"使用モデル: {bot.llm_client.model}")
    else:
        print("ローカルLLMサーバーに接続できません")
        print("Ollamaをインストールして起動してください:")
        print("   https://ollama.ai/")
        print("   ollama pull llama2")
        print("   ollama serve")


@bot.event
async def on_application_command_error(ctx, error):
    """全スラッシュコマンド共通のエラーハンドラ。
    権限不足・クールダウン中は専用メッセージを返し、それ以外は
    ログに記録したうえで汎用のエラーメッセージを返す"""
    if isinstance(error, commands.MissingPermissions):
        await ctx.respond("このコマンドは管理者のみ実行できます。", ephemeral=True)
        return
    if isinstance(error, commands.CommandOnCooldown):
        await ctx.respond(f"少し間隔を空けてから実行してください（あと{error.retry_after:.0f}秒）。", ephemeral=True)
        return
    logger.error(f"Command error: {error}")
    try:
        # deferやrespond済みかどうかでfollowup/respondを使い分ける
        # （二重に応答しようとするとdiscord.HTTPExceptionになるため）
        if ctx.response.is_done():
            await ctx.followup.send("コマンドの実行中にエラーが発生しました。")
        else:
            await ctx.respond("コマンドの実行中にエラーが発生しました。")
    except discord.HTTPException:
        pass


def _build_numbered_context(items):
    """[(ファイル名, 本文), ...] を受け取り、
    "[1] 出典: ファイル名\n本文" の形式で連結した文脈文字列と、
    番号からファイル名を引ける対応表（citation_map）を作る。

    RAG検索はチャンク（文書の一部）単位でヒットを返すため、同じファイルから
    複数チャンクがヒットすることがある。チャンクごとに番号を振ると、
    実際には1つの資料しか無いのに[1][2]のように複数の出典があるかのように
    LLMや利用者に見えてしまうため、ファイル名ごとに番号をまとめる"""
    # まずファイル名ごとにチャンクをまとめる（登場順は保持する）
    grouped = {}
    for filename, text in items:
        grouped.setdefault(filename, []).append(text)

    parts = []
    citation_map = {}
    for i, (filename, texts) in enumerate(grouped.items(), 1):
        combined = "\n\n".join(texts)
        parts.append(f"[{i}] 出典: {filename}\n{combined}")
        citation_map[i] = filename
    return "\n\n".join(parts), citation_map


async def build_answer_context(question: str):
    """RAGが使える場合は関連チャンクのみ、使えない場合は全文書を使って番号付き文脈を作る"""
    if bot.vector_index.embedding_available and bot.vector_index.entries:
        hits = await bot.vector_index.search(question, top_k=config.RAG_TOP_K)
        if hits:
            items = [(h["filename"], h["chunk"]) for h in hits]
            return _build_numbered_context(items)
    # 埋め込みが使えない、またはヒットがない場合は全文書を渡す（フォールバック）
    items = [(filename, doc["content"]) for filename, doc in bot.doc_manager.documents.items()]
    return _build_numbered_context(items)


def extract_cited_files(reply: str, citation_map: dict) -> list:
    """回答テキストの中に登場する[1]や[2]といった番号を拾い、citation_mapでファイル名に変換する。
    表記ゆれのあるファイル名の文字列一致に頼らないため、番号なら確実に拾える"""
    cited_numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", reply)})
    files = [citation_map[n] for n in cited_numbers if n in citation_map]
    # 同じファイルが複数の番号（チャンク）に分かれて引用された場合の重複を除く
    return list(dict.fromkeys(files))


async def classify_used_documents(question: str, reply: str, citation_map: dict) -> list:
    """回答本文に[1]等の番号引用が1つも無かった場合のフォールバック。

    小規模なローカルLLMは、長い回答本文の途中に[N]を挟む指示までは
    守らないことが多い（今回のように箇条書き中心の回答で顕著）。
    そこで、回答生成とは別に「どの文書番号を使ったか」だけを問う
    短く単純な追加の質問をLLMに投げ、数字だけを抜き出す。
    範囲を絞った単純な質問の方が、複雑な指示より遵守されやすいための対策"""
    if not citation_map:
        return []

    doc_list = "\n".join(f"[{n}] {filename}" for n, filename in sorted(citation_map.items()))
    classify_prompt = (
        f"以下は、番号付きの文書を参照して行われた質問と回答のやり取りです。\n\n"
        f"文書一覧:\n{doc_list}\n\n"
        f"質問: {question}\n\n"
        f"回答: {reply}\n\n"
        "この回答の内容を作るために実際に使われた文書の番号だけを、"
        "半角カンマ区切りの数字のみで答えてください（例: 1,3）。"
        "使われた文書が分からない場合は none とだけ答えてください。"
        "数字と例以外の文字は一切書かないでください。"
    )
    try:
        classify_reply = await bot.llm_client.generate_response(classify_prompt)
    except Exception as e:
        # 判定用の呼び出し自体が失敗しても、メインの回答には影響させない
        logger.warning(f"引用文書の判定に失敗しました: {e}")
        return []

    cited_numbers = sorted({int(n) for n in re.findall(r"\d+", classify_reply)})
    files = [citation_map[n] for n in cited_numbers if n in citation_map]
    return list(dict.fromkeys(files))


def format_references(used_files, limit: int = 15) -> str:
    """参照ファイル一覧をDiscordの1メッセージ2000文字制限に収まる長さに整形する"""
    # ドキュメントが1つもなければ「なし」とだけ返す
    if not used_files:
        return "参照ファイル: なし"

    # 表示件数をlimitで打ち切り、残りは件数だけ添える
    # （記憶しているドキュメントが多いと、ファイル名の一覧だけで数千文字になり得るため）
    shown = used_files[:limit]
    text = "参照ファイル: " + ", ".join(shown)
    remaining = len(used_files) - len(shown)
    if remaining > 0:
        text += f" ほか{remaining}件"
    return text


async def handle_chat(channel_id, prompt: str, respond):
    """/chat, !chat, メンション応答で共通利用するチャット処理。
    常にストリーミングで応答する（呼び出し元はすべてストリーミング利用のため、
    以前あった非ストリーミング分岐は使われておらず削除した）"""
    system_prompt = bot.persona.system_prompt(extra_rules="質問に対して簡潔かつ正確に日本語で答えてください。")
    history = bot.conversations.get(channel_id)

    try:
        buffer = ""
        last_edit_len = 0
        sent_message = None
        async for piece in bot.llm_client.generate_response_stream(prompt, system_prompt, history=history):
            buffer += piece
            if sent_message is None:
                sent_message = await respond(f"{buffer}")
                last_edit_len = len(buffer)
            elif len(buffer) - last_edit_len >= 40:
                try:
                    await sent_message.edit(content=f"{buffer}")
                    last_edit_len = len(buffer)
                except discord.HTTPException:
                    pass

        if sent_message is None:
            await respond("（応答がありませんでした）")
        elif last_edit_len != len(buffer):
            try:
                await sent_message.edit(content=f"{buffer}")
            except discord.HTTPException:
                pass
        reply = buffer

        bot.conversations.add_exchange(channel_id, prompt, reply)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        await respond(f"エラー: {str(e)}")


@bot.slash_command(name="llm_status", description="ローカルLLMの状態を確認します")
async def llm_status(ctx):
    """Ollamaサーバーへの接続確認を行い、結果と使用中のモデル名を返す"""
    await ctx.defer()
    if await bot.llm_client.check_connection():
        await ctx.followup.send(f"ローカルLLM接続OK\nモデル: {bot.llm_client.model}")
    else:
        await ctx.followup.send(
            "ローカルLLMに接続できません\n\n"
            "**セットアップ手順:**\n"
            "1. Ollama をインストール: https://ollama.ai/\n"
            "2. モデルをダウンロード: `ollama pull llama2`\n"
            "3. サーバー起動: `ollama serve`"
        )


@bot.slash_command(name="set_model", description="使用するLLMモデルを変更します（管理者のみ）")
@commands.has_permissions(administrator=True)
async def set_model(ctx, model_name: str):
    """会話・回答生成に使うOllamaのモデル名を実行時に切り替える（Bot再起動不要）"""
    old_model = bot.llm_client.model
    bot.llm_client.model = model_name
    await ctx.respond(f"LLMモデルを {old_model} → {model_name} に変更しました")


@bot.slash_command(name="set_persona", description="マスコットの名前・性格・語尾を変更します（管理者のみ）")
@commands.has_permissions(administrator=True)
async def set_persona(ctx, name: str = None, description: str = None, ending: str = None):
    """マスコットの名前・性格説明・語尾を個別に上書きする。
    引数を省略した項目は現在の値をそのまま維持する"""
    if name:
        bot.persona.name = name
    if description:
        bot.persona.description = description
    if ending is not None:
        bot.persona.ending = ending
    await ctx.respond(
        "ペルソナを更新しました\n"
        f"名前: {bot.persona.name}\n"
        f"性格: {bot.persona.description}\n"
        f"語尾: {bot.persona.ending or '（指定なし）'}"
    )


@bot.slash_command(name="upload_doc", description="ドキュメントファイルをアップロードして記憶させます")
async def upload_doc(ctx, file: discord.Attachment):
    """添付ファイルの内容を読み取り、doc_managerへ登録する（/askの参照対象になる）"""
    await ctx.defer()
    try:
        content = await bot.read_attachment_content(file)
        success = await bot.doc_manager.add_document(
            filename=file.filename, content=content, author=str(ctx.author)
        )
        if success:
            await ctx.followup.send(
                f"ドキュメント '{file.filename}' を記憶しました！\n"
                f"サイズ: {len(content)} 文字\n"
                f"`/ask` コマンドで質問できます。"
            )
        else:
            await ctx.followup.send("ドキュメントの保存に失敗しました。")
    except ValueError as e:
        await ctx.followup.send(f"エラー: {str(e)}")
    except Exception as e:
        logger.error(f"Upload error: {e}")
        await ctx.followup.send("予期しないエラーが発生しました。")


@bot.slash_command(name="list_docs", description="記憶しているドキュメント一覧を表示します")
async def list_docs(ctx):
    """記憶しているドキュメントの一覧をEmbedで表示する（作成者・時刻・サイズ付き）"""
    doc_list = bot.doc_manager.get_document_list()
    if not doc_list:
        await ctx.respond("記憶しているドキュメントはありません。")
        return

    embed = discord.Embed(title="記憶しているドキュメント", color=0x00FF00)
    # Discordの仕様上、Embedのフィールドは最大25件までしか追加できない
    for doc in doc_list[:25]:
        embed.add_field(
            name=doc["filename"],
            value=f"作成者: {doc['author']}\n時刻: {doc['timestamp']}\nサイズ: {doc['size']} 文字",
            inline=False,
        )
    await ctx.respond(embed=embed)


@bot.slash_command(name="clear_docs", description="記憶しているドキュメントをすべて削除します（管理者のみ）")
@commands.has_permissions(administrator=True)
async def clear_docs(ctx):
    """記憶しているドキュメントをRAGインデックスごと全件削除する（元に戻せないため管理者限定）"""
    bot.doc_manager.clear_documents()
    await ctx.respond("すべてのドキュメントを削除しました。")


async def document_filename_autocomplete(ctx: discord.AutocompleteContext):
    """/delete_doc 入力中に、現在記憶しているドキュメント名を候補として返す"""
    # 入力途中の文字列を取得（未入力なら空文字扱い）
    typed = (ctx.value or "").lower()

    # 記憶しているドキュメント名のうち、入力文字列を含むものだけに絞り込む
    candidates = [name for name in bot.doc_manager.documents.keys() if typed in name.lower()]

    # Discordの仕様上、候補は最大25件までしか表示できない
    return candidates[:25]


@bot.slash_command(name="delete_doc", description="指定した1件のドキュメントだけを記憶から削除します（管理者のみ）")
@commands.has_permissions(administrator=True)
async def delete_doc(
    ctx,
    filename: discord.Option(str, "削除するドキュメント名", autocomplete=document_filename_autocomplete),
):
    """指定した1件のドキュメントだけをメモリ・ディスク・RAGインデックスから削除する"""
    # 指定された名前のドキュメントが記憶されているか確認
    if filename not in bot.doc_manager.documents:
        await ctx.respond(f"'{filename}' という名前のドキュメントは記憶されていません。`/list_docs` で名前を確認してください。")
        return

    # ドキュメント本体・永続化ファイル・RAGインデックスをまとめて削除
    removed = await bot.doc_manager.remove_document(filename)

    # 結果に応じてメッセージを出し分ける
    if removed:
        await ctx.respond(f"'{filename}' を記憶から削除しました。")
    else:
        await ctx.respond(f"'{filename}' の削除に失敗しました。")


@bot.slash_command(name="ask", description="ドキュメントの内容に基づいて質問に答えます（ローカルLLM使用）")
@commands.cooldown(1, config.ASK_COOLDOWN_SECONDS, commands.BucketType.user)
async def ask(ctx, question: str):
    """記憶しているドキュメントの内容だけを根拠にLLMへ質問し、
    回答と実際に参照されたファイル名を分けて返す"""
    await ctx.defer()

    if not bot.doc_manager.documents:
        await ctx.followup.send("記憶しているドキュメントがありません。まずドキュメントをアップロードしてください。")
        return

    try:
        content, citation_map = await build_answer_context(question)
        system_prompt = bot.persona.system_prompt(
            extra_rules=(
                "あなたはドキュメントに基づいて質問に答えるアシスタントです。"
                "提供された文書の情報のみをもとに回答し、情報が記載されていなければ"
                "「この情報はドキュメントに記載されていません」と答えてください。"
                "各文書には[1]や[2]のような番号が付いています。"
                "回答の中で、その文書の情報を実際に使った箇所には、"
                "必ず対応する番号を[1]のように角括弧で示してください。"
                "複数の文書を参照した場合は[1][2]のように併記してください。"
            )
        )
        history = bot.conversations.get(ctx.channel.id)
        user_prompt = f"Documents:\n{content}\n\nQuestion: {question}"

        reply = await bot.llm_client.generate_response(user_prompt, system_prompt, history=history)
        bot.conversations.add_exchange(ctx.channel.id, question, reply)

        # Discordの1メッセージ2000文字制限があるため、質問・回答・参照ファイルは
        # それぞれ独立したメッセージとして送る（1つにまとめると容易に超過するため）
        SAFE_CHUNK_LEN = 1800

        # 質問自体が長い場合に備えて表示用に切り詰める
        question_display = question if len(question) <= 300 else question[:300] + "…"
        await ctx.followup.send(f"質問: {question_display}")

        # 回答本文をSAFE_CHUNK_LENごとに分割し、それぞれ別メッセージで送る
        if len(reply) > SAFE_CHUNK_LEN:
            chunks = [reply[i : i + SAFE_CHUNK_LEN] for i in range(0, len(reply), SAFE_CHUNK_LEN)]
            for i, chunk in enumerate(chunks, 1):
                await ctx.followup.send(f"回答 ({i}/{len(chunks)}):\n{chunk}")
        else:
            await ctx.followup.send(f"回答:\n{reply}")

        # 回答本文中の[1]や[2]といった番号を拾い、citation_mapでファイル名に変換する
        actually_referenced = extract_cited_files(reply, citation_map)

        # 回答本文に番号引用が1つも無かった場合、追加の短い質問で使用文書を判定する
        if not actually_referenced and citation_map:
            actually_referenced = await classify_used_documents(question, reply, citation_map)

        if actually_referenced:
            await ctx.followup.send(format_references(actually_referenced))
        elif citation_map:
            # 文脈には渡したが、回答本文に番号での引用が見当たらなかった場合
            await ctx.followup.send("参照ファイル: 回答内から特定できませんでした")
        else:
            await ctx.followup.send("参照ファイル: なし")
    except Exception as e:
        logger.error(f"Ask command error: {e}")
        await ctx.followup.send(f"エラー: {str(e)}")


@bot.slash_command(name="chat", description="ローカルLLMと自由に会話します")
@commands.cooldown(1, config.CHAT_COOLDOWN_SECONDS, commands.BucketType.user)
async def chat(ctx, message: str):
    """ドキュメントに縛られない自由会話。ドキュメントを持たない/askとは別コマンドとして分けている"""
    await ctx.defer()
    await handle_chat(ctx.channel.id, message, respond=ctx.followup.send)


@bot.slash_command(name="hello", description="こんにちはと挨拶します")
async def hello(ctx):
    """動作確認用の簡単な挨拶コマンド"""
    await ctx.respond(f"こんにちは、{ctx.author.name} さん！")


@bot.slash_command(name="great_mention", description="複合メンションを行います")
async def great_mention(
    ctx,
    exception: bool = False,
    role1: discord.Role = None,
    role2: discord.Role = None,
    role3: discord.Role = None,
    message: str = "",
):
    """役職（ロール）の組み合わせを指定して対象メンバーへ一括メンションする。

    exception=Falseの場合: role1〜role3を全て持つメンバーのANDでメンションする
    exception=Trueの場合 : role1を持つメンバーからrole2を持つメンバーを除外する
                           （role1未指定ならサーバー全員が母集団になる）"""
    if exception:
        # role1を持つメンバーを母集団にする（未指定ならサーバー全員）
        if role1 is None:
            base_users = ctx.guild.members
        else:
            base_users = [m for m in ctx.guild.members if role1 in m.roles]

        # role2を持つメンバーを除外する「AかつBではない」形の指定
        if role2 is not None:
            exclude_users = [m for m in ctx.guild.members if role2 in m.roles]
            users = [m for m in base_users if m not in exclude_users]
        else:
            users = base_users
    else:
        # 指定された役職を全て持つメンバーのAND条件で絞り込む
        roles = [r for r in [role1, role2, role3] if r is not None]
        if not roles:
            await ctx.respond("メンションしたい役職を1つ以上指定してください。")
            return
        users = [m for m in ctx.guild.members if all(r in m.roles for r in roles)]

    if not users:
        await ctx.respond("メンション対象のユーザーが見つかりませんでした。")
        return

    await ctx.respond(f"メンション対象: {', '.join(m.mention for m in users)} {message}")


@bot.event
async def on_message(message):
    """スラッシュコマンド以外の入力経路（添付ファイル・!chat・メンション）をまとめて処理する。
    ドキュメントの提出とチャットのどちらも、この1つのイベントハンドラで受け付ける"""
    if message.author.bot:
        return

    # ファイルが添付されていれば、そのままドキュメントとして記憶させる
    if message.attachments:
        for attachment in message.attachments:
            try:
                content = await bot.read_attachment_content(attachment)
                success = await bot.doc_manager.add_document(
                    filename=attachment.filename, content=content, author=str(message.author)
                )
                if success:
                    await message.channel.send(
                        f"ドキュメント '{attachment.filename}' を記憶しました！\n"
                        f"サイズ: {len(content)} 文字\n"
                        f"`/ask` コマンドで質問できます。"
                    )
            except ValueError as e:
                await message.channel.send(f"{attachment.filename}: {str(e)}")
            except Exception as e:
                logger.error(f"File processing error: {e}")

    # プレフィックス形式（!chat）でのチャット
    if message.content.startswith("!chat"):
        prompt = message.content[len("!chat") :].strip()
        if not prompt:
            await message.channel.send("メッセージを入力してください。例: `!chat こんにちは`")
        else:
            await handle_chat(message.channel.id, prompt, respond=message.channel.send)

    # Botへのメンションでのチャット（メンション部分は本文から取り除いてから渡す）
    elif bot.user in message.mentions:
        prompt = message.content.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()
        if not prompt:
            await message.channel.send("メッセージを入力してください。例: `@Bot こんにちは`")
        else:
            await handle_chat(message.channel.id, prompt, respond=message.channel.send)

    # このBotはprefixコマンド（@bot.command）を1つも定義しておらず、
    # スラッシュコマンドと上記のメンション/!chat処理だけで完結しているため、
    # process_commands()の呼び出しは不要（discord.Botではそもそも使えない）


# Bot起動
if __name__ == "__main__":
    if not config.DISCORD_TOKEN:
        print("DISCORD_BOT_TOKEN もしくは TOKEN が設定されていません。")
        print("例: export DISCORD_BOT_TOKEN=your_bot_token")
    else:
        print("ローカルLLM Discord Bot を起動中...")
        print("必要な環境:")
        print("   1. Ollama インストール済み")
        print("   2. ollama serve で起動済み")
        print("   3. モデルダウンロード済み (例: ollama pull llama2)")
        print("   4. RAGを使う場合は埋め込みモデルも取得: ollama pull nomic-embed-text")
        bot.run(config.DISCORD_TOKEN)
