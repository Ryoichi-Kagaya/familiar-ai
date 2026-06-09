# チャット中ユーザー切り替え — 実装計画書

作成日: 2026-06-09  
対象ブランチ: `develop`

---

## 1. 概要と目的

familiar-ai は家族で共有して使われている。現在は「話しかけている人間」が 1 人固定だが、  
家族の誰かが `/switch <name>` と入力することで、その人のコンテキスト（関係性履歴・記憶・ToM 推論）に切り替えられるようにする。

**AI 側（コンパニオン）は変わらない。変わるのは「今話しかけている人間」の情報。**

---

## 2. 現状の構造（変更対象の把握）

現在、「ユーザー（人間）」は暗黙の 1 人として扱われており、  
その人に関するデータは以下に集約されている：

| コンポーネント | ファイル | ユーザー依存の要素 |
|---|---|---|
| ユーザー名 | `AgentConfig.companion_name` | 環境変数 `COMPANION_NAME`（命名が紛らわしいが、ここが「相手の人間の名前」） |
| 関係性履歴 | `observations.db` `relationship_state` | `state_key='default'` 固定 |
| ToM 推論ターゲット | `agent.py:528` `ToMTool(default_person=config.companion_name)` | 起動時固定 |
| AI の自己記述ナラティブ | `~/.familiar_ai/self_narrative.jsonl` | 1 本固定（誰との関係かが混在） |
| 精神状態ログ | `~/.familiar_ai/mental_state.jsonl` | 1 本固定 |

**AI 自身のデータ（変えない）:**

| コンポーネント | ファイル | 備考 |
|---|---|---|
| AI の人格テキスト | `ME.md` / `~/.familiar_ai/ME.md` | 家族全員に同一 |
| AI の自己状態 | `~/.familiar_ai/self_state.json` | 体の状態・疲労度など |
| AI の欲求 | `~/.familiar_ai/desires.json` | 全ユーザー共通 |
| AI の記憶 DB | `observations.db` の観察・エピソード等 | ユーザー横断で保持 |

---

## 3. 設計方針

1. **ユーザーはディレクトリで表現する**  
   `~/.familiar_ai/users/<id>/` — 追加・削除がファイル操作で完結

2. **識別子はスラッグ（例: `kagaya`, `mom`, `taro`）、表示名は別管理**

3. **関係性 DB は `state_key` をユーザー ID に変える**  
   既存の `state_key='default'` は `ACTIVE_USER` 環境変数 or `'default'` として引き継ぐ

4. **AI の体・欲求・記憶は共有のまま**  
   ユーザーが変わっても AI の疲れや欲求は継続する（実際の体と同じ）

5. **切り替えは `EmbodiedAgent.switch_user()` に閉じる**

---

## 4. ストレージ設計

```
~/.familiar_ai/
  users/                         ← 新規
    default/                     ← 既存データの移行先
      self_narrative.jsonl       ← この人との関係に関する AI の自己記述
      mental_state.jsonl         ← この人と話しているときの精神状態ログ
    kagaya/
      self_narrative.jsonl
      mental_state.jsonl
    mom/
      self_narrative.jsonl
      mental_state.jsonl
  observations.db                ← relationship_state.state_key がユーザー ID に
  self_state.json                ← AI の体（共有・変更なし）
  desires.json                   ← AI の欲求（共有・変更なし）
  ME.md                          ← AI の人格（共有・変更なし）
  heartbeat_state.json           ← 共有・変更なし
```

### 4.1 UserProfile（新規データクラス）

`src/familiar_agent/user_profile.py` として新規作成：

```python
@dataclass
class UserProfile:
    id: str        # スラッグ、例 "kagaya"
    name: str      # 表示名、例 "カガヤ"
    dir: Path      # ~/.familiar_ai/users/kagaya/

    @property
    def mental_state_path(self) -> Path:
        return self.dir / "mental_state.jsonl"

    @property
    def self_narrative_path(self) -> Path:
        return self.dir / "self_narrative.jsonl"
```

### 4.2 UserRegistry（新規クラス）

```python
class UserRegistry:
    def list_users(self) -> list[UserProfile]: ...
    def get(self, user_id: str) -> UserProfile: ...      # 存在しなければ自動作成
    def create(self, user_id: str, name: str) -> UserProfile: ...
    def active_id(self) -> str: ...      # users/active.txt から読む
    def set_active(self, user_id: str): ...
```

`~/.familiar_ai/users/active.txt` に現在アクティブなユーザー ID を保存（起動時復元用）。

---

## 5. エージェント層の変更

### 5.1 EmbodiedAgent の変更

```python
class EmbodiedAgent:
    def __init__(self, config: AgentConfig):
        self._user_registry = UserRegistry()
        self._current_user = self._user_registry.get(config.companion_name or "default")
        # RelationshipTracker にユーザー ID を渡す
        self._relationship = RelationshipTracker(user_id=self._current_user.id)
        # SelfNarrative / MentalStateBus はユーザー固有パスで初期化
        ...

    async def switch_user(self, user_id: str) -> str:
        """チャット中ユーザー切り替え。戻り値は切り替え後の表示名。"""
        # 1. 現在のユーザー状態を保存
        await self._relationship.save()
        self._self_narrative.flush()
        self._mental_state_bus.flush()
        # 2. 新しいユーザーをロード
        user = self._user_registry.get(user_id)
        self._current_user = user
        self.config.companion_name = user.name
        # 3. 関係性トラッカーを切り替え
        self._relationship = RelationshipTracker(user_id=user_id)
        # 4. SelfNarrative をこのユーザー向けに切り替え
        self._self_narrative = SelfNarrative(path=user.self_narrative_path)
        # 5. MentalStateBus のログ先を切り替え
        self._mental_state_bus.set_log_path(user.mental_state_path)
        # 6. ToMTool の推論ターゲットを切り替え
        self._tom_tool.default_person = user.name
        # 7. registry に active を保存
        self._user_registry.set_active(user_id)
        return user.name
```

### 5.2 RelationshipTracker の変更

`familiar_neighbor/mind/relationship.py` の `RelationshipTracker.__init__` に  
`user_id: str = "default"` を追加し、SQL の `state_key = 'default'` を  
`state_key = user_id` に置換する（2 箇所: SELECT / INSERT）。

### 5.3 SelfNarrative / MentalStateBus の変更

コンストラクタで受け取るパスを実行時に変更できるよう `set_log_path(path: Path)` を追加。

---

## 6. UI 層の変更

### 6.1 GUI（メイン対応）

`FamiliarWindow` のヘッダーバー（`_build_ui` の header_layout）に  
**ユーザー切り替えコンボボックス**を追加する。既存の settings_btn の左隣に配置。

```
[ ✦ familiar-ai ]  [👤 kagaya ▾]  [ 設定 ]  [ ↻ STT ]  [ 🎙 Mic ]
```

#### 変更点

**`FamiliarWindow.__init__`**:
```python
self._user_registry = UserRegistry()
self._current_user = self._user_registry.get_active()
self._companion_display_name = self._current_user.name  # 既存フィールドを維持
```

**`_build_ui`** に追加:
```python
self._user_combo = QComboBox()
self._user_combo.setFixedHeight(_px(30))
self._user_combo.setMinimumWidth(_px(120))
self._user_combo.setStyleSheet(...)  # 既存ボタンと統一
self._refresh_user_combo()
self._user_combo.currentIndexChanged.connect(self._on_user_changed)
header_layout.insertWidget(1, self._user_combo)  # title の右隣
```

**新規メソッド**:
```python
def _refresh_user_combo(self) -> None:
    """登録済みユーザーをコンボに反映する。"""
    self._user_combo.blockSignals(True)
    self._user_combo.clear()
    for user in self._user_registry.list_users():
        self._user_combo.addItem(f"👤 {user.name}", userData=user.id)
    # 現在アクティブなユーザーを選択状態にする
    active_id = self._current_user.id
    idx = next(
        (i for i in range(self._user_combo.count())
         if self._user_combo.itemData(i) == active_id), 0
    )
    self._user_combo.setCurrentIndex(idx)
    self._user_combo.blockSignals(False)

def _on_user_changed(self, index: int) -> None:
    """コンボ選択変更 → ユーザー切り替えをエージェントに依頼。"""
    user_id = self._user_combo.itemData(index)
    if user_id and user_id != self._current_user.id:
        self._create_task(self._switch_user_async(user_id))

async def _switch_user_async(self, user_id: str) -> None:
    if self._agent is None:
        return
    name = await self._agent.switch_user(user_id)
    self._current_user = self._user_registry.get(user_id)
    self._companion_display_name = name
    # チャットログのラベルも更新
    self._chat_log.set_companion_label(name)
    # システムメッセージをチャットに表示
    self._chat_log.append_line(f"── {name} に切り替わりました ──")
```

**`ChatLog.set_companion_label()`** を追加（既存クラスに 3 行追加）:
```python
def set_companion_label(self, label: str) -> None:
    self._companion_label = label.strip() or "You"
```

#### エージェント未初期化中の扱い

起動直後（`_agent_ready = False`）はコンボを無効化し、初期化完了後に有効化する。

### 6.2 TUI（サブ対応）

`tui.py` の `_SLASH_COMMANDS` に追加：

```python
("/switch", "👤  Switch user (e.g. /switch kagaya)"),
```

`handle_slash_command()` に分岐を追加：

```python
if text.startswith("/switch"):
    user_id = text[len("/switch"):].strip()
    if not user_id:
        users = self.agent._user_registry.list_users()
        # ユーザー一覧を表示
    else:
        name = await self.agent.switch_user(user_id)
        # システムメッセージ表示: "── カガヤさんに切り替えました ──"
```

オートコンプリートで `/switch ` の後に登録済みユーザー一覧を補完候補として出す。

### 6.3 不明ユーザー ID の扱い

- `UserRegistry.get()` は存在しない ID なら**自動作成**する（名前は ID と同値として仮登録）
- GUI のコンボは登録済みユーザーのみ表示するので未登録 ID の問題は GUI では発生しない
- TUI で未登録 ID を入力した場合のみ自動作成が発動

---

## 7. データベースマイグレーション

### migration/YYYYMMDDHHMMSS_user_state_key.sql

```sql
-- relationship_state の state_key は既に TEXT PRIMARY KEY なので
-- スキーマ変更不要。'default' レコードがそのまま default ユーザーとして機能する。
-- このマイグレーションはドキュメント用。実行内容なし。
SELECT 1; -- no-op
```

### 起動時ファイルシステムマイグレーション（`bootstrap.py` に追加）

```python
def _migrate_legacy_user_files():
    """既存の mental_state.jsonl / self_narrative.jsonl を users/default/ へ移動。"""
    default_dir = FAI_DIR / "users" / "default"
    default_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("mental_state.jsonl", "self_narrative.jsonl"):
        src = FAI_DIR / fname
        dst = default_dir / fname
        if src.exists() and not dst.exists():
            src.rename(dst)
    # active.txt が無ければ 'default' で初期化
    active_txt = FAI_DIR / "users" / "active.txt"
    if not active_txt.exists():
        active_txt.write_text("default")
```

---

## 8. セットアップフローの変更

`setup.py` の `SetupConfig.companion_name` は「デフォルトユーザーの名前」として継続利用。  
セットアップ完了時に `~/.familiar_ai/users/default/` ディレクトリを作成する。

---

## 9. 実装フェーズ

### Phase 1 — ストレージ基盤

1. `user_profile.py` に `UserProfile` + `UserRegistry` を実装
2. 起動時マイグレーション（`bootstrap.py`）
3. `RelationshipTracker` に `user_id` パラメータを追加
4. テスト: `tests/test_user_registry.py`

### Phase 2 — エージェント切り替えロジック

1. `EmbodiedAgent.switch_user()` を実装
2. `SelfNarrative` / `MentalStateBus` に `set_log_path()` を追加
3. テスト: `tests/test_user_switch.py`

### Phase 3 — GUI 統合（メイン）

1. `FamiliarWindow` ヘッダーにユーザーコンボボックスを追加
2. `_switch_user_async()` / `_refresh_user_combo()` を実装
3. `ChatLog.set_companion_label()` を追加
4. エージェント未初期化中はコンボ無効化
5. 切り替え時のシステムメッセージ表示

### Phase 4 — TUI 統合

1. `/switch` コマンドを TUI に追加
2. ユーザー一覧のオートコンプリートを実装
3. 切り替え時のシステムメッセージ表示

---

## 10. 未決定事項

| 項目 | v1 の判断 | 将来の選択肢 |
|---|---|---|
| 会話履歴の扱い | 継続（前のユーザーとの会話も見える） | オプションで `/switch taro --fresh` にてリセット |
| 不明 ID 時の動作 | 自動作成 | 確認プロンプト |
| desires の共有/分離 | 共有 | ユーザーごとに分離（ただし「AI の欲求」なので共有が自然） |
| ユーザー一覧表示コマンド | `/switch`（引数なし）で一覧 | 専用 `/users` コマンド |
| GUI 対応 | v1 は TUI のみ | v2 でドロップダウン追加 |
