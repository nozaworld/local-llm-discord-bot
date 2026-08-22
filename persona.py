"""マスコットのキャラクター設定（ペルソナ）を管理するモジュール．

LoRA2.pyのchatコマンドが持つ mascot_name / persona / ending の考え方を
Discord Bot側でもコマンドから変更できる形にしたもの．
"""
from dataclasses import dataclass


@dataclass
class Persona:
    """マスコットの名前・性格・語尾を保持し、LLMへ渡すsystem_promptを組み立てるクラス．

    /set_personaで実行時に書き換えられる想定のため、フィールドはミュータブルなまま持つ"""

    name: str
    description: str
    ending: str = ""

    def system_prompt(self, extra_rules: str = "") -> str:
        """現在のペルソナ設定から、LLMへ渡すsystem_promptを組み立てる．
        extra_rulesには呼び出し元（/ask, /chat等）ごとの追加指示を渡す"""
        lines = [
            f"あなたは『{self.name}』という名前のアシスタントです．",
            f"性格・話し方: {self.description}",
        ]
        if self.ending:
            lines.append(f"回答の各文末には必ず『{self.ending}』を付けてください．")
        if extra_rules:
            lines.append(extra_rules)
        return "\n".join(lines)
