# DriveActionExecutor 復活計画

**対象ブランチ**: `chore/merge-upstream-20260614`（または develop へのマージ後）  
**作成日**: 2026-06-14  
**背景**: upstream マージ時に `desire_tick_prompt() + agent.run("")` に戻ったため、
SILENT_ACTION 系の drive でも毎回 LLM ターンが発生するスパム問題が再発しうる。

---

## なぜ必要か

`AUTO_DESIRE` をオンにすると explore / look_around などが約 37 秒で再発火し、
「内的衝動に従って行動」というプロンプトが延々 LLM に投げられる。
DriveActionExecutor は drive の effect type で発火方式を振り分け、これを防いでいた。

詳細: `docs/adr/desire-action-executor.md`

---

## 現在の状態（マージ後）

### 残っているファイル（削除されていない）

- `src/familiar_agent/drive_executor.py` — DriveActionExecutor 本体
- `src/familiar_agent/desires.py` — DriveEffect enum / DriveSpec.effect_type あり
- `tests/test_drive_executor.py` — 37 件のテスト
- `docs/adr/desire-action-executor.md` — ADR

### マージで失われた配線

`src/familiar_agent/main.py` の desire 発火ループが upstream 方式に戻った:

```python
# 現在 (upstream 方式 — 全 drive が LLM ターンを生成)
tick = desire_tick_prompt(desires, pending_items)
if tick:
    desire_name, prompt, _pending = tick
    await agent.run("", on_action=on_action, on_text=on_text,
                    desires=desires, inner_voice=prompt, ...)
    desires.satisfy(desire_name)
```

```python
# 復活させたい (DriveActionExecutor 方式)
dominant = desires.get_dominant()
if dominant and not pending_items:
    desire_name, _ = dominant
    result = await executor.dispatch(
        desire_name,
        last_interaction_time=last_interaction_time,
        on_action=on_action, on_text=on_text, interrupt_queue=input_queue,
    )
    if result.fired:
        desires.curiosity_target = None
        last_interaction_time = time.time()
```

---

## 実装手順

### Step 1: main.py に DriveActionExecutor を再接続

`src/familiar_agent/main.py` を編集:

**1a. import を戻す**（現在削除されている）:
```python
from .drive_executor import DriveActionExecutor
```

**1b. `repl()` 関数内の executor 初期化を戻す**（現在削除されている）:
```python
loop = asyncio.get_event_loop()
executor = DriveActionExecutor(agent, desires)   # ← これを追加
```

**1c. desire 発火ブロックを置換**:

```python
# 削除する (upstream 方式):
tick = desire_tick_prompt(desires, pending_items)
if tick:
    desire_name, prompt, _pending = tick
    try:
        murmur = _t(f"desire_{desire_name}")
    except KeyError:
        murmur = _t("desire_default")
    print(f"\n{murmur}\n")
    await agent.run(
        "",
        on_action=on_action,
        on_text=on_text,
        desires=desires,
        inner_voice=prompt,
        interrupt_queue=input_queue,
    )
    desires.satisfy(desire_name)
    desires.curiosity_target = None
elif pending_items:
```

```python
# 置き換え後 (DriveActionExecutor 方式):
dominant = desires.get_dominant()
if dominant and not pending_items:
    desire_name, _ = dominant
    try:
        murmur = _t(f"desire_{desire_name}")
    except KeyError:
        murmur = _t("desire_default")
    print(f"\n{murmur}\n")
    result = await executor.dispatch(
        desire_name,
        last_interaction_time=last_interaction_time,
        on_action=on_action,
        on_text=on_text,
        interrupt_queue=input_queue,
    )
    if result.fired:
        desires.curiosity_target = None
        last_interaction_time = time.time()
elif pending_items:
```

**1d. `desire_tick_prompt` の import を削除**（使わなくなる）:

`_ui_helpers` の import から `desire_tick_prompt,` を除去。

### Step 2: Telegram bot の確認

`src/familiar_agent/telegram_bot.py` が desire を発火する場合、同様に executor を使っているか確認。
使っていなければ `agent.run("", inner_voice=prompt)` のままで OK（Telegram は常に在席扱い）。

### Step 3: テスト確認

```bash
uv run pytest tests/test_drive_executor.py -q
uv run pytest tests/test_main_repl_reminder.py -q   # commitment reminder との干渉確認
uv run pytest -q
```

### Step 4: 動作確認ポイント

- `AUTO_DESIRE=1` で起動し explore / look_around が LLM ターンを**生成しない**ことを確認
- `share_memory` が TTS でひとりごとを**生成する**ことを確認
- commitment reminder（upstream の新機能）と desire 発火が干渉しないことを確認

---

## 注意点

- `DriveActionExecutor` は `agent.execute_desire_silent_action()` を内部で呼ぶ。
  このメソッドは `agent.py` に現在も残っている（確認済み）。
- upstream の `desire_tick_prompt` は `_ui_helpers.py` に残しておいてよい
  （commitment reminder 側は独立して動くため）。
- `_BRIEF_REPLY_MAX_TOKENS_THINKING` など今回のマージで追加した定数は触らなくてよい。
