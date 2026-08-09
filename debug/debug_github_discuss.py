"""
github-discuss-mcp 接続デバッグスクリプト

以下を順番にチェックします:
  1. 環境変数 / .env の読み込み確認
  2. 認証方式の判定
  3. Python API 直接呼び出し (MCP を介さず get_discussions)
  4. MCP サブプロセス経由の呼び出し (実際の familiar-ai と同じ経路)

使い方:
  cd C:/Users/Blue-/familiar-ai
  uv run python debug/debug_github_discuss.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# ── 1. .env を読み込む ────────────────────────────────────────────────────────

DOTENV_PATH = Path("C:/Users/Blue-/github-discuss/.env")
GITHUB_DISCUSS_SRC = Path("C:/Users/Blue-/github-discuss")

print("=" * 60)
print("STEP 1: .env の読み込み")
print("=" * 60)

if not DOTENV_PATH.exists():
    print(f"[ERROR] .env が見つかりません: {DOTENV_PATH}")
    sys.exit(1)

# python-dotenv が使える場合は使う、なければ手動パース
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=DOTENV_PATH)
    print(f"[OK] dotenv で読み込み: {DOTENV_PATH}")
except ImportError:
    print("[WARN] python-dotenv が未インストール。手動パースします。")
    with open(DOTENV_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())
    print(f"[OK] 手動パース完了: {DOTENV_PATH}")

# ── 2. 環境変数の確認 ────────────────────────────────────────────────────────

print()
print("=" * 60)
print("STEP 2: 環境変数の確認")
print("=" * 60)

ENV_VARS = [
    "GITHUB_TOKEN",
    "GITHUB_APP_ID",
    "GITHUB_APP_PRIVATE_KEY",
    "GITHUB_APP_INSTALLATION_ID",
    "GITHUB_DISCUSS_REPO",
    "GITHUB_DISCUSS_OWNER",
]

for var in ENV_VARS:
    val = os.getenv(var, "")
    if val:
        # トークン類はマスク
        if "TOKEN" in var or "KEY" in var:
            display = val[:12] + "..." if len(val) > 12 else "***"
        else:
            display = val
        print(f"  [OK] {var} = {display}")
    else:
        print(f"  [--] {var} (未設定)")

# GITHUB_DISCUSS_REPO が owner/repo 形式かチェック
repo_env = os.getenv("GITHUB_DISCUSS_REPO", "")
if "/" in repo_env:
    owner_from_repo, _, repo_from_repo = repo_env.partition("/")
    print("\n  [INFO] GITHUB_DISCUSS_REPO を owner/repo 形式で解釈:")
    print(f"         owner={owner_from_repo}, repo={repo_from_repo}")
    if not os.getenv("GITHUB_DISCUSS_OWNER"):
        os.environ["GITHUB_DISCUSS_OWNER"] = owner_from_repo
        # REPO だけを repo 名に上書きして API クラスが使えるようにする
        os.environ["_DEBUG_REPO_ONLY"] = repo_from_repo

# GitHub App Private Key ファイルの存在確認
pem_path = os.getenv("GITHUB_APP_PRIVATE_KEY", "")
if pem_path:
    pem_file = Path(pem_path.replace("~", str(Path.home())))
    if pem_file.exists():
        print(f"\n  [OK] Private Key ファイルあり: {pem_file}")
        auth_mode = "GitHub App"
    else:
        print(f"\n  [WARN] Private Key ファイルなし: {pem_file}")
        print("         → Personal Access Token 認証にフォールバックします")
        auth_mode = "Personal Access Token"
else:
    auth_mode = "Personal Access Token"

print(f"\n  認証方式: {auth_mode}")

# ── 3. Python API 直接呼び出し ───────────────────────────────────────────────

print()
print("=" * 60)
print("STEP 3: Python API 直接呼び出し (MCP を介さない)")
print("=" * 60)

# github-discuss パッケージを sys.path に追加
src_path = GITHUB_DISCUSS_SRC / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from github_discuss.github_api import GitHubDiscussionsAPI
    from github_discuss.utils import DEFAULT_OWNER, DEFAULT_REPO

    # owner / repo を解決
    if "/" in repo_env:
        owner_part, _, repo_part = repo_env.partition("/")
    else:
        owner_part = os.getenv("GITHUB_DISCUSS_OWNER", DEFAULT_OWNER)
        repo_part = repo_env or DEFAULT_REPO

    print(f"  対象リポジトリ: {owner_part}/{repo_part}")

    async def test_get_discussions():
        api = GitHubDiscussionsAPI()
        print("  API インスタンス作成: OK")
        discussions = await api.get_discussions(owner_part, repo_part)
        return discussions

    discussions = asyncio.run(test_get_discussions())
    print(f"  [OK] get_discussions 成功: {len(discussions)} 件取得")
    if discussions:
        print("\n  最初の1件:")
        d = discussions[0]
        print(f"    タイトル : {d.get('title', '?')}")
        print(f"    カテゴリ : {d.get('category', {}).get('name', '?')}")
        print(f"    URL      : {d.get('url', '?')}")
        print(f"    作成日   : {d.get('createdAt', '?')}")
    else:
        print("  [INFO] ディスカッションは 0 件でした")

except ImportError as e:
    print(f"  [ERROR] インポート失敗: {e}")
    print(f"  sys.path に追加した: {src_path}")
    print("  → uv sync で依存関係をインストールしてください")
except Exception as e:
    print(f"  [ERROR] API 呼び出し失敗: {type(e).__name__}: {e}")

# ── 4. MCP サブプロセス経由の呼び出し ────────────────────────────────────────

print()
print("=" * 60)
print("STEP 4: MCP サブプロセス経由の呼び出し")
print("=" * 60)

MCP_CMD = ["uv", "run", "--project", str(GITHUB_DISCUSS_SRC), "github-discuss-mcp"]
print(f"  コマンド: {' '.join(MCP_CMD)}")
print(f"  cwd:     {GITHUB_DISCUSS_SRC}")
print()

# ── StdioServerParameters の cwd サポート確認 ────────────────────────────────
print("  [確認] StdioServerParameters の cwd サポート:")
try:
    from mcp import StdioServerParameters
    import inspect

    sig = inspect.signature(StdioServerParameters)
    if "cwd" in sig.parameters:
        print("    [OK] cwd パラメータあり → familiar-ai は cwd を渡せます")
    else:
        print("    [WARN] cwd パラメータなし → MCP ライブラリのバージョンが古い可能性")
except ImportError:
    print("    [ERROR] mcp パッケージが未インストール")
print()

# メッセージを1つずつ送り、レスポンスを待ってから次を送る。
# subprocess.run で stdin を一括送信すると、GitHub API 呼び出し中に
# EOF になりサーバーが終了してしまうため、asyncio で双方向通信する。
MCP_SEQUENCE = [
    (
        1,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "debug-client", "version": "0.1"},
            },
        },
    ),
    (
        None,  # 通知（レスポンスなし）
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
    ),
    (
        2,
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ),
    (
        3,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            # 引数なし（familiar-ai が使う方法）
            # .env の GITHUB_DISCUSS_REPO が "owner/repo" 合体形式の場合、
            # main.py の call_tool は _parse_repo_info() を使わず
            # GITHUB_DISCUSS_OWNER (未設定→utenadev) を直読みするバグがある
            "params": {"name": "get_discussions", "arguments": {}},
        },
    ),
    (
        4,
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            # 引数で owner/repo を明示（ワークアラウンド）
            "params": {
                "name": "get_discussions",
                "arguments": {"owner": owner_part, "repo": repo_part},
            },
        },
    ),
]


def _print_response(obj: dict) -> bool:
    """レスポンスを表示。エラーがあれば False を返す。"""
    msg_id = obj.get("id")
    if "error" in obj:
        print(f"    [ERROR] id={msg_id}: {obj['error']}")
        return False
    result = obj.get("result", {})
    if msg_id == 1:
        print(f"    [initialize] OK - server: {result.get('serverInfo', {})}")
    elif msg_id == 2:
        tools = result.get("tools", [])
        print(f"    [tools/list] {len(tools)} ツール: {[t['name'] for t in tools]}")
    elif msg_id == 3:
        content = result.get("content", [])
        text = content[0].get("text", "") if content else ""
        preview = text[:400] + ("..." if len(text) > 400 else "")
        label = "引数なし (familiar-ai と同じ呼び出し方)"
        if "エラー" in text or "ERROR" in text or "📭" in text:
            print(f"    [NG] get_discussions {label}:\n      {preview}")
            print("    ↑ .env の GITHUB_DISCUSS_OWNER が未設定のため utenadev が使われている可能性")
        else:
            print(f"    [OK] get_discussions {label}:\n      {preview}")
    elif msg_id == 4:
        content = result.get("content", [])
        text = content[0].get("text", "") if content else ""
        preview = text[:400] + ("..." if len(text) > 400 else "")
        label = f"引数あり (owner={owner_part}, repo={repo_part})"
        if "エラー" in text or "ERROR" in text or "📭" in text:
            print(f"    [NG] get_discussions {label}:\n      {preview}")
        else:
            print(f"    [OK] get_discussions {label}:\n      {preview}")
    return True


async def run_mcp_session():
    proc = await asyncio.create_subprocess_exec(
        *MCP_CMD,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(GITHUB_DISCUSS_SRC),
    )

    stderr_lines: list[str] = []

    async def read_stderr():
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            stderr_lines.append(line.decode(errors="replace").rstrip())

    stderr_task = asyncio.ensure_future(read_stderr())

    print("\n  stdout (JSON-RPC レスポンス):")
    try:
        for expected_id, msg in MCP_SEQUENCE:
            payload = json.dumps(msg) + "\n"
            proc.stdin.write(payload.encode())
            await proc.stdin.drain()

            if expected_id is None:
                # 通知はレスポンスなし
                continue

            # id が一致するレスポンスが来るまで読む（30 秒）
            deadline = asyncio.get_event_loop().time() + 30
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    print(f"    [ERROR] id={expected_id} のレスポンスがタイムアウト")
                    break
                try:
                    line_bytes = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
                except asyncio.TimeoutError:
                    print(f"    [ERROR] id={expected_id} のレスポンスがタイムアウト")
                    break
                if not line_bytes:
                    print(f"    [ERROR] id={expected_id} を待つ前に stdout が閉じました")
                    break
                line = line_bytes.decode(errors="replace").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("id") == expected_id:
                        _print_response(obj)
                        break
                    # id が違うレスポンス（通知など）は読み飛ばす
                except json.JSONDecodeError:
                    print(f"    (非JSON): {line[:120]}")

    finally:
        # セッション終了
        proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
        # stderr タスクが完了するまで待つ
        try:
            await asyncio.wait_for(stderr_task, timeout=3)
        except asyncio.TimeoutError:
            stderr_task.cancel()
        print(f"\n  終了コード: {proc.returncode}")

    if stderr_lines:
        print("\n  stderr:\n    " + "\n    ".join(stderr_lines))
    else:
        print("  (stderr なし)")


try:
    asyncio.run(run_mcp_session())
except FileNotFoundError:
    print("  [ERROR] 'uv' コマンドが見つかりません。PATH を確認してください。")
except Exception as e:
    print(f"  [ERROR] {type(e).__name__}: {e}")

# ── 5. よくある問題のチェック ────────────────────────────────────────────────

print()
print("=" * 60)
print("STEP 5: よくある問題チェック")
print("=" * 60)

checks = []

# .familiar-ai.json の --project パスが ~ 表記かどうか
fai_json = Path("C:/Users/Blue-/familiar-ai/.familiar-ai.json")
if fai_json.exists():
    with open(fai_json, encoding="utf-8") as f:
        cfg = json.load(f)
    gd_cfg = cfg.get("mcpServers", {}).get("github-discuss", {})
    args = gd_cfg.get("args", [])
    if "~/github-discuss" in args:
        checks.append(
            (
                "WARN",
                ".familiar-ai.json の --project に '~/github-discuss' を使用しています。\n"
                "       Windows では ~ が展開されないことがあります。\n"
                "       → 絶対パス 'C:/Users/Blue-/github-discuss' に変更を検討してください。",
            )
        )
    cwd_val = gd_cfg.get("cwd", "")
    if cwd_val:
        cwd_path = Path(cwd_val)
        if cwd_path.exists():
            checks.append(("OK", f".familiar-ai.json の cwd が設定されています: {cwd_val}"))
        else:
            checks.append(
                (
                    "ERROR",
                    f".familiar-ai.json の cwd が存在しません: {cwd_val}\n"
                    f"       → パスを確認してください",
                )
            )
    else:
        checks.append(
            (
                "WARN",
                ".familiar-ai.json の github-discuss に 'cwd' が設定されていません。\n"
                "       uv はプロジェクトルートを検出できず、github-discuss-mcp が起動しない可能性があります。\n"
                "       → 'cwd': 'C:/Users/Blue-/github-discuss' を追加してください。",
            )
        )

    if "env" not in gd_cfg:
        checks.append(
            (
                "INFO",
                ".familiar-ai.json の github-discuss に 'env' セクションがありません。\n"
                "       .env の自動検出に依存しています。",
            )
        )
else:
    checks.append(("WARN", f".familiar-ai.json が見つかりません: {fai_json}"))

# Private Key ファイルの確認（再掲）
if pem_path and not Path(pem_path).exists():
    checks.append(
        (
            "ERROR",
            f"GITHUB_APP_PRIVATE_KEY のファイルが存在しません:\n"
            f"       {pem_path}\n"
            f"       → Personal Access Token 認証 (GITHUB_TOKEN) に切り替えるか、パスを修正してください。",
        )
    )

if not checks:
    print("  特に問題は検出されませんでした。")
else:
    for level, msg in checks:
        print(f"  [{level}] {msg}")

print()
print("デバッグ完了。")
