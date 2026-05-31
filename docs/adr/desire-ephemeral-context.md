# Plan: Desire ターンをエフェメラル mini-context で実行する

**Status:** Draft（未実装）  
**前提:** desire-action-executor.md の実装済み

---

## 動機

現状、SOCIAL_INITIATION / ABSENT_CARE の在席パスは `agent.run("", inner_voice=nudge)` を呼ぶ。
`agent.run()` は `self.messages`（メイン会話履歴）を共有するため、desire ターンの
user メッセージ（現在 `"…"`）が会話履歴に残り続ける。

会話履歴を完全にクリーンに保ちたい場合、desire ターンを独立した使い捨て履歴で実行する。

---

## 設計

```
drive fires (SOCIAL_INITIATION / ABSENT_CARE-在席)
  └─ DriveActionExecutor._social_initiation()
       ├─ mini_messages = agent.messages[-4:] + [user = inner_voice_block]
       │    直近2〜3往復だけ context に含める（self.messages は変更しない）
       ├─ agent.run_ephemeral(mini_messages, tools=[say, memory_read, memory_write])
       │    └─ _stream_with_retry を直接呼ぶ（self.messages に触れない）
       └─ response → TTS / memory 保存 → 終了
```

次にカガヤが返事をすると、それが通常の user ターンとして self.messages に入る。

---

## 変更ファイル

| ファイル | 変更内容 |
|---------|---------|
| `agent.py` | `run_ephemeral(messages, tools, system)` メソッド追加 |
| `drive_executor.py` | `_social_initiation` を `agent.run()` → `agent.run_ephemeral()` に切り替え |
| `agent.py` | `user_input_with_ctx = "…"` の desire ターン分岐を除去 |
| `tests/` | `test_drive_executor.py` に ephemeral パスのテスト追加 |

---

## `run_ephemeral` の要件

- `self.messages` を読まない・書かない
- `_stream_with_retry` を直接呼ぶ（ReAct ループ自体は維持してよい）
- system prompt は `_system_prompt(inner_voice=nudge, ...)` をそのまま使う
- tools は `say`, `memory_read`, `memory_write` 程度に限定
  - カメラ・移動は DriveActionExecutor の SILENT_ACTION が直接担う
- 戻り値は生成テキスト（呼び出し元が TTS / memory 保存を行う）

---

## トレードオフ

| | 現状（`"…"` プレースホルダー） | 本プラン |
|---|---|---|
| 履歴の汚染 | `"…"` が残る | なし |
| 実装コスト | 小 | 中（`run_ephemeral` 新設） |
| 会話文脈 | フル履歴 | 直近4件のみ |
| モデルの混乱リスク | `"…"` が user ターンに見える | inner_voice が user ターンに見える（mini-context 内） |

---

## 保留判断

履歴に `"…"` が残ることで実際に問題（文脈破壊・モデルの誤動作）が起きていないなら、
現状のままでよい。本プランは「起きたときに着手する」対応候補として残す。
