# ADR: Desire System — Action Executor Refactor

**Status:** Implemented  
**Branch:** develop

---

## 問題

`AUTO_DESIRE` をオンにしたまま放置すると、閾値を超えた欲求が
`agent.run("", inner_voice=prompt)` という空のユーザーターンとして発火し続け、
エージェントが「内的衝動に従って行動」というプロンプトを延々と受け取る。

根本的な欠陥は **「欲求が外から命令されるプロンプトとして実体化している」** こと。
`DECAY_ON_SATISFY = 0.5` のため explore なら約 37 秒で再発火し、スパムになる。

---

## 設計原則

欲求の種類によって発火時の **効果（Effect）** を変える。

```
desire fires → DriveActionExecutor.dispatch()
    SILENT_ACTION     → ツールを直接実行（LLM ターン不要）
    EXPRESSIVE_SOLO   → ひとりごと生成（memory/narrative 保存、会話履歴には残さない）
    SOCIAL_INITIATION → コンパニオン在席時のみ LLM ターン生成
    ABSENT_CARE       → 不在時: 感情を記録 → 完全 satisfy → 長い cooldown
    GATE              → 他 drive を抑制、自身はターンを生成しない
```

---

## Drive 再分類

| Drive | Effect | 不在時の動作 | 在席時の動作 |
|-------|--------|-------------|-------------|
| `look_around` | SILENT_ACTION | カメラ直呼び出し → memory 保存 | 同左 |
| `explore` | SILENT_ACTION | カメラ/mobility/MCP → memory 保存 | 同左 |
| `consolidate` | SILENT_ACTION | MemoryJobWorker を直接起動 | 同左 |
| `reflect` | SILENT_ACTION | utility backend 一文生成 → self_narrative | 同左 |
| `curiosity` | SILENT_ACTION | ツール呼び出し(MCP 含む) + memory 保存 | 同左 |
| `share_memory` | EXPRESSIVE_SOLO | 記憶想起 → 独語生成 → narrative/TTS | 記憶想起 → 共有 |
| `play` | EXPRESSIVE_SOLO | 一人遊び(MCP/ツール)、utterance 可 | 一緒に遊ぶ |
| `attachment` | SOCIAL_INITIATION | accumulate（発火しない） | 繋がりを言葉にする |
| `greet_companion` | SOCIAL_INITIATION | accumulate | 挨拶 |
| `repair` | SOCIAL_INITIATION | accumulate | 修復を試みる |
| `worry_companion` | ABSENT_CARE | 心配を memory/narrative に書く → satisfy(0) → cooldown 4h | 心配を表現 |
| `care` | ABSENT_CARE | 気持ちを memory/narrative に書く → satisfy(0) → cooldown 2h | ケアを申し出る |
| `rest` | GATE | 他の active drive を抑制 | 同左 |
| `self_protect` | GATE | social drive を抑制 | 同左 |

**在席判定:** `time.time() - last_interaction_time > ABSENCE_THRESHOLD`  
`last_interaction_time` は main.py にすでにあり、ユーザーメッセージ受信時のみ更新される。  
デフォルト閾値: 1800 秒（30 分）。SOCIAL_INITIATION 以外は閾値の影響を受けない。

---

## 各 Effect の実装詳細

### SILENT_ACTION

```python
async def execute_desire_silent_action(self, desire_name: str) -> None:
    match desire_name:
        case "look_around" | "explore":
            # カメラを直接呼ぶ。LLM ターンを生成しない
            result = await self._camera.capture()
            await self._memory.save_async(result, kind="observation", ...)
            # curiosity_target 更新も可
        case "consolidate":
            self._memory_worker.enqueue_consolidation()
        case "reflect":
            text = await self._utility_backend.complete(reflect_prompt, max_tokens=80)
            self._self_narrative.write(text, trigger="reflect_drive")
        case "curiosity":
            # MCP を含む任意のツールを呼べる
            ...
```

### EXPRESSIVE_SOLO（share_memory、play）

```python
# 会話履歴には追加しない
text = await self._utility_backend.complete(solo_prompt, max_tokens=120)
await self._memory.save_async(text, kind="feeling", ...)
self._self_narrative.write(text, trigger=desire_name)
if self._tts:
    await self._tts.call("say", {"text": text})
desires.satisfy(desire_name)
```

後から確認したい場合は memory recall または self_narrative から読む。

### ABSENT_CARE（worry_companion、care）

```python
# 不在時: 感情を記録してから完全 satisfy
text = await self._utility_backend.complete(absent_care_prompt, max_tokens=80)
await self._memory.save_async(text, kind="feeling", emotion="worried", ...)
self._self_narrative.write(text, trigger=f"{desire_name}_unresolved")
desires.satisfy(desire_name, to=0.0)          # 完全リセット（×0.5 ではない）
desires._last_fired[desire_name] = time.time()  # cooldown 開始
# DriveSpec.min_interval_seconds を worry=14400(4h)、care=7200(2h) に設定

# 在席時: 既存の LLM ターン（inner_voice 注入）をそのまま使う
```

### SOCIAL_INITIATION

```python
if not self._companion_present():
    return DriveActionResult(fired=False, reason="companion_absent")
    # desire level は decay せず accumulate し続ける
    # 在席が検知されたタイミングで発火する
await self._agent.run("", inner_voice=nudge_text, desires=self._desires, ...)
desires.satisfy(desire_name)
```

### GATE（rest、self_protect）

`desires.update_context()` の `schedule_multiplier` / `social_permission` を下げるだけ。
ターンを生成しない。既存ロジックをそのまま使える。

---

## 実装フェーズ

| Phase | 対象ファイル | 作業 |
|-------|-------------|------|
| 0 | `desires.py` | `DriveEffect` enum 追加、`DriveSpec` に `effect_type` フィールド追加、全 drive に分類付与。振る舞い変化なし |
| 1 | `drive_executor.py`（新設） | `DriveActionExecutor`、`DriveActionResult` クラス。SILENT_ACTION と GATE から実装 |
| 2 | `agent.py` | `execute_desire_silent_action()` メソッド追加、`companion_present()` プロパティ追加 |
| 3 | `drive_executor.py` | EXPRESSIVE_SOLO 実装（utility backend 呼び出し、history 非汚染） |
| 4 | `drive_executor.py` | ABSENT_CARE 実装（不在判定 + 感情記録 + 完全 satisfy + cooldown） |
| 5 | `drive_executor.py` | SOCIAL_INITIATION の在席ゲート実装 |
| 6 | `main.py` | `desire_tick_prompt()` を `DriveActionExecutor.dispatch()` に差し替え |
| 7 | `tests/` | `test_drive_executor.py` 新設、`test_desires.py` に DriveEffect 分類テスト追加 |

---

## 変更ファイルサマリー

| ファイル | 変更の性質 |
|---------|-----------|
| `src/familiar_agent/desires.py` | `DriveEffect` enum 追加、`DriveSpec` に `effect_type` フィールド追加 |
| `src/familiar_agent/drive_executor.py` | **新設** |
| `src/familiar_agent/agent.py` | `execute_desire_silent_action()`、`companion_present()` 追加 |
| `src/familiar_agent/main.py` | desire 発火ロジック差し替え（`desire_tick_prompt` 削除） |
| `tests/test_drive_executor.py` | **新設** |
| `tests/test_desires.py` | DriveEffect 分類テスト追加 |

---

## 変わること / 変わらないこと

**変わること:**
- look_around / explore はカメラを叩くが会話には何も現れない
- rest / self_protect は静かに他 drive を抑制するだけ
- social desires は誰かがいる時だけ実際に語りかける
- worry / care は不在時に「書いて手放す」サイクルで回る（蓄積しない）
- 「(内的衝動に従って行動)」の無限ループが消える

**変わらないこと:**
- drive level 計算、circadian modulation、DECAY_ON_SATISFY（ABSENT_CARE 除く）
- GlobalWorkspace への coalition 参加
- `[inner-voice]` の system prompt 注入インフラ（SOCIAL_INITIATION 在席時に存続）
- 既存のメモリ DB スキーマ（migration 不要）
- `dominant_as_prompt()` は SOCIAL_INITIATION 在席時のみ使用

---

## 議論済みの判断

- `share_memory` は **一人で発火してよい**（ひとりごとは自然な内的活動）
- ひとりごとは会話履歴に **残さない**（user/assistant 交互制約の workaround がそのまま現行の inner_voice 方式の再実装になるため）。memory と self_narrative に保存する
- 在席判定は **時間ベースのみ**（last_interaction_time、カメラ検知はオプション）
- `worry_companion` / `care` の satisfy は **完全リセット**（×0.5 ではなく 0.0）、その後長い cooldown
