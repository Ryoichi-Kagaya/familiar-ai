"""Art critique long-term memory — store and recall artwork critiques with 7-motivation scoring.

Scoring framework based on Tetsuo Osaki's "What is Contemporary Art?" (7 motivations).
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .memory import ObservationMemory

logger = logging.getLogger(__name__)

DB_PATH = str(Path.home() / ".familiar_ai" / "observations.db")

MOTIVATION_LABELS = {
    1: "新しい視覚・感覚の追求",
    2: "メディウムと知覚の探究",
    3: "制度への言及と異議",
    4: "アクチュアリティと政治",
    5: "思想・哲学・科学・世界認識",
    6: "私と世界・記憶・歴史・共同体",
    7: "エロス・タナトス・聖性",
}

# Known aliases for artist name normalization (katakana/hiragana → normalized English)
_ARTIST_ALIASES: dict[str, str] = {
    "マティス": "Henri Matisse",
    "ピカソ": "Pablo Picasso",
    "ゴッホ": "Vincent van Gogh",
    "モネ": "Claude Monet",
    "ダリ": "Salvador Dali",
    "バスキア": "Jean-Michel Basquiat",
    "ウォーホル": "Andy Warhol",
    "カンディンスキー": "Wassily Kandinsky",
    "クリムト": "Gustav Klimt",
    "フリーダ": "Frida Kahlo",
    "村上隆": "Takashi Murakami",
    "草間彌生": "Yayoi Kusama",
    "奈良美智": "Yoshitomo Nara",
}


def _now_iso() -> str:
    return datetime.now().isoformat()


def _normalize_artist_name(name: str, db: sqlite3.Connection) -> str:
    """Normalize artist name: check aliases, then existing DB records, then return as-is."""
    if name in _ARTIST_ALIASES:
        return _ARTIST_ALIASES[name]

    # Check if any existing normalized name partially matches
    row = db.execute(
        "SELECT artist_name_normalized FROM artworks "
        "WHERE artist_name = ? OR artist_name_normalized = ? LIMIT 1",
        (name, name),
    ).fetchone()
    if row and row[0]:
        return str(row[0])

    # Fuzzy: check if input matches any known alias value
    for alias, normalized in _ARTIST_ALIASES.items():
        if name.lower() in normalized.lower() or normalized.lower() in name.lower():
            return normalized

    return name


class ArtCritiqueStore:
    """SQLite-backed storage for artwork critiques and motivation scores."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        self._db_path = db_path
        self._db: sqlite3.Connection | None = None
        self._db_lock = threading.Lock()

    def _ensure_connected(self) -> sqlite3.Connection:
        if self._db is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self._db_path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.execute("PRAGMA synchronous = NORMAL")
            self._db.execute("PRAGMA foreign_keys = ON")
        return self._db

    def store_art_critique(
        self,
        artwork_info: dict[str, Any],
        critique_text: str,
        scores: list[dict[str, Any]],
        personal_resonance: int,
        observation_memory: "ObservationMemory | None" = None,
    ) -> str:
        """Store or update a critique for a given artwork.

        artwork_info keys: title, artist_name, year?, medium?, dimensions?,
                           collection?, description?, image_ref?
        scores: list of dicts with motivation_index, motivation_degree, achievement, notes?
        Returns critique_id.
        """
        now = _now_iso()
        title = str(artwork_info.get("title", "")).strip()
        artist_name = str(artwork_info.get("artist_name", "")).strip()

        with self._db_lock:
            db = self._ensure_connected()

            artist_name_normalized = _normalize_artist_name(artist_name, db)

            # Find or create artwork
            artwork_row = db.execute(
                "SELECT id FROM artworks WHERE title = ? AND artist_name_normalized = ?",
                (title, artist_name_normalized),
            ).fetchone()

            if artwork_row:
                artwork_id = str(artwork_row["id"])
            else:
                artwork_id = str(uuid.uuid4())
                db.execute(
                    """
                    INSERT INTO artworks
                    (id, title, artist_name, artist_name_normalized, year, medium,
                     dimensions, collection, description, image_ref, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artwork_id,
                        title,
                        artist_name,
                        artist_name_normalized,
                        artwork_info.get("year"),
                        artwork_info.get("medium"),
                        artwork_info.get("dimensions"),
                        artwork_info.get("collection"),
                        artwork_info.get("description"),
                        artwork_info.get("image_ref"),
                        now,
                    ),
                )

            # UPSERT critique (one row per artwork)
            existing_critique = db.execute(
                "SELECT id FROM art_critiques WHERE artwork_id = ?",
                (artwork_id,),
            ).fetchone()

            if existing_critique:
                critique_id = str(existing_critique["id"])
                db.execute(
                    """
                    UPDATE art_critiques
                    SET critique_text = ?, personal_resonance = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (critique_text, personal_resonance, now, critique_id),
                )
            else:
                critique_id = str(uuid.uuid4())
                db.execute(
                    """
                    INSERT INTO art_critiques
                    (id, artwork_id, critique_text, personal_resonance, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (critique_id, artwork_id, critique_text, personal_resonance, now, now),
                )

            # INSERT OR REPLACE motivation scores
            for score in scores:
                idx = int(score["motivation_index"])
                degree = int(score["motivation_degree"])
                achievement = int(score["achievement"])
                notes = score.get("notes")
                db.execute(
                    """
                    INSERT OR REPLACE INTO art_motivation_scores
                    (critique_id, motivation_index, motivation_degree, achievement, notes)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (critique_id, idx, degree, achievement, notes),
                )

            db.commit()

        # Also save to observations for semantic search
        if observation_memory and critique_text:
            obs_content = f"[美術批評] {artist_name}「{title}」— {critique_text[:400]}"
            observation_memory.save(
                obs_content,
                kind="observation",
                emotion="moved",
                dedupe_key=f"art_critique:{critique_id}",
            )

        return critique_id

    def recall_art_critiques(
        self,
        query: str,
        artist: str | None = None,
        observation_memory: "ObservationMemory | None" = None,
    ) -> list[dict[str, Any]]:
        """Recall critiques by artist name or semantic query."""
        with self._db_lock:
            db = self._ensure_connected()

            if artist:
                normalized = _normalize_artist_name(artist, db)
                rows = db.execute(
                    """
                    SELECT a.id as artwork_id, a.title, a.artist_name, a.artist_name_normalized,
                           a.year, a.medium, a.collection,
                           c.id as critique_id, c.critique_text, c.personal_resonance,
                           c.updated_at
                    FROM artworks a
                    JOIN art_critiques c ON c.artwork_id = a.id
                    WHERE a.artist_name_normalized LIKE ?
                    ORDER BY c.updated_at DESC
                    """,
                    (f"%{normalized}%",),
                ).fetchall()
            else:
                rows = db.execute(
                    """
                    SELECT a.id as artwork_id, a.title, a.artist_name, a.artist_name_normalized,
                           a.year, a.medium, a.collection,
                           c.id as critique_id, c.critique_text, c.personal_resonance,
                           c.updated_at
                    FROM artworks a
                    JOIN art_critiques c ON c.artwork_id = a.id
                    WHERE a.title LIKE ? OR a.artist_name LIKE ? OR a.artist_name_normalized LIKE ?
                    ORDER BY c.updated_at DESC
                    LIMIT 10
                    """,
                    (f"%{query}%", f"%{query}%", f"%{query}%"),
                ).fetchall()

            results = []
            for row in rows:
                critique_id = str(row["critique_id"])
                score_rows = db.execute(
                    """
                    SELECT motivation_index, motivation_degree, achievement, notes
                    FROM art_motivation_scores
                    WHERE critique_id = ?
                    ORDER BY motivation_index
                    """,
                    (critique_id,),
                ).fetchall()
                scores = [
                    {
                        "motivation_index": int(s["motivation_index"]),
                        "label": MOTIVATION_LABELS.get(int(s["motivation_index"]), "?"),
                        "motivation_degree": int(s["motivation_degree"]),
                        "achievement": int(s["achievement"]),
                        "notes": s["notes"],
                    }
                    for s in score_rows
                ]
                results.append(
                    {
                        "artwork_id": str(row["artwork_id"]),
                        "title": row["title"],
                        "artist_name": row["artist_name"],
                        "artist_name_normalized": row["artist_name_normalized"],
                        "year": row["year"],
                        "medium": row["medium"],
                        "collection": row["collection"],
                        "critique_id": critique_id,
                        "critique_text": row["critique_text"],
                        "personal_resonance": row["personal_resonance"],
                        "updated_at": row["updated_at"],
                        "scores": scores,
                    }
                )

        # If no DB hits and we have semantic memory, try that too
        if not results and observation_memory:
            mem_results = observation_memory.recall(f"美術批評 {query}", n=5)
            for m in mem_results:
                summary = str(m.get("summary", ""))
                if summary.startswith("[美術批評]"):
                    results.append(
                        {
                            "source": "semantic_memory",
                            "critique_text": summary,
                            "date": m.get("date"),
                            "confidence": m.get("confidence"),
                        }
                    )

        return results

    def get_artist_profile(self, artist_name: str) -> dict[str, Any]:
        """Return aggregated motivation profile for an artist."""
        with self._db_lock:
            db = self._ensure_connected()
            normalized = _normalize_artist_name(artist_name, db)

            artwork_count_row = db.execute(
                "SELECT COUNT(*) as cnt FROM artworks WHERE artist_name_normalized LIKE ?",
                (f"%{normalized}%",),
            ).fetchone()
            artwork_count = int(artwork_count_row["cnt"]) if artwork_count_row else 0

            critique_count_row = db.execute(
                """
                SELECT COUNT(*) as cnt FROM art_critiques c
                JOIN artworks a ON c.artwork_id = a.id
                WHERE a.artist_name_normalized LIKE ?
                """,
                (f"%{normalized}%",),
            ).fetchone()
            critique_count = int(critique_count_row["cnt"]) if critique_count_row else 0

            score_rows = db.execute(
                """
                SELECT s.motivation_index,
                       AVG(s.motivation_degree) as avg_degree,
                       AVG(s.achievement) as avg_achievement,
                       COUNT(*) as cnt
                FROM art_motivation_scores s
                JOIN art_critiques c ON s.critique_id = c.id
                JOIN artworks a ON c.artwork_id = a.id
                WHERE a.artist_name_normalized LIKE ?
                GROUP BY s.motivation_index
                ORDER BY s.motivation_index
                """,
                (f"%{normalized}%",),
            ).fetchall()

            motivations = [
                {
                    "motivation_index": int(r["motivation_index"]),
                    "label": MOTIVATION_LABELS.get(int(r["motivation_index"]), "?"),
                    "avg_motivation_degree": round(float(r["avg_degree"]), 2),
                    "avg_achievement": round(float(r["avg_achievement"]), 2),
                    "sample_count": int(r["cnt"]),
                }
                for r in score_rows
            ]

        return {
            "artist_name": artist_name,
            "artist_name_normalized": normalized,
            "artwork_count": artwork_count,
            "critique_count": critique_count,
            "motivation_profile": motivations,
        }

    def compare_artworks(self, title_a: str, title_b: str) -> dict[str, Any]:
        """Compare 7-motivation scores between two artworks."""
        with self._db_lock:
            db = self._ensure_connected()

            def _fetch_scores(title: str) -> tuple[str | None, str | None, list[dict[str, Any]]]:
                row = db.execute(
                    """
                    SELECT a.artist_name, c.id as critique_id
                    FROM artworks a JOIN art_critiques c ON c.artwork_id = a.id
                    WHERE a.title LIKE ?
                    LIMIT 1
                    """,
                    (f"%{title}%",),
                ).fetchone()
                if not row:
                    return None, None, []
                artist = str(row["artist_name"])
                critique_id = str(row["critique_id"])
                score_rows = db.execute(
                    """
                    SELECT motivation_index, motivation_degree, achievement, notes
                    FROM art_motivation_scores WHERE critique_id = ?
                    ORDER BY motivation_index
                    """,
                    (critique_id,),
                ).fetchall()
                scores = [
                    {
                        "motivation_index": int(s["motivation_index"]),
                        "label": MOTIVATION_LABELS.get(int(s["motivation_index"]), "?"),
                        "motivation_degree": int(s["motivation_degree"]),
                        "achievement": int(s["achievement"]),
                    }
                    for s in score_rows
                ]
                return artist, critique_id, scores

            artist_a, _, scores_a = _fetch_scores(title_a)
            artist_b, _, scores_b = _fetch_scores(title_b)

        # Build comparison dict keyed by motivation_index
        scores_a_map = {s["motivation_index"]: s for s in scores_a}
        scores_b_map = {s["motivation_index"]: s for s in scores_b}

        comparison = []
        for idx in range(1, 8):
            sa = scores_a_map.get(idx)
            sb = scores_b_map.get(idx)
            diff_degree = (sa["motivation_degree"] - sb["motivation_degree"]) if sa and sb else None
            diff_achievement = (sa["achievement"] - sb["achievement"]) if sa and sb else None
            comparison.append(
                {
                    "motivation_index": idx,
                    "label": MOTIVATION_LABELS[idx],
                    f"{title_a[:20]}_degree": sa["motivation_degree"] if sa else None,
                    f"{title_a[:20]}_achievement": sa["achievement"] if sa else None,
                    f"{title_b[:20]}_degree": sb["motivation_degree"] if sb else None,
                    f"{title_b[:20]}_achievement": sb["achievement"] if sb else None,
                    "diff_degree": diff_degree,
                    "diff_achievement": diff_achievement,
                }
            )

        # Find motivations with largest difference
        diffs_with_data = [c for c in comparison if c["diff_achievement"] is not None]
        diffs_with_data.sort(key=lambda x: abs(x["diff_achievement"]), reverse=True)  # type: ignore[arg-type]
        largest_diff = diffs_with_data[:3] if diffs_with_data else []

        return {
            "artwork_a": {"title": title_a, "artist": artist_a},
            "artwork_b": {"title": title_b, "artist": artist_b},
            "comparison": comparison,
            "largest_differences": largest_diff,
        }


class ArtCritiqueTool:
    """Agent-callable art critique tools."""

    def __init__(
        self,
        store: ArtCritiqueStore,
        observation_memory: "ObservationMemory | None" = None,
    ) -> None:
        self._store = store
        self._memory = observation_memory

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "store_art_critique",
                "description": (
                    "作品の批評をデータベースに保存する。"
                    "画像を見せて「批評して」と指示されたとき、または自分で批評したくなったときに使用する。"
                    "小崎哲哉の7動機フレームワークで各動機の度合いと達成度を1–10で評価して保存する。"
                    "同一作品の批評は上書き更新される。"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "artwork_info": {
                            "type": "object",
                            "description": "作品情報",
                            "properties": {
                                "title": {"type": "string", "description": "作品タイトル"},
                                "artist_name": {"type": "string", "description": "アーティスト名"},
                                "year": {
                                    "type": "string",
                                    "description": "制作年（例: '1888', 'c.1920'）",
                                },
                                "medium": {"type": "string", "description": "素材・技法"},
                                "dimensions": {"type": "string", "description": "寸法"},
                                "collection": {"type": "string", "description": "所蔵先"},
                                "description": {
                                    "type": "string",
                                    "description": "作品の簡単な説明",
                                },
                                "image_ref": {"type": "string", "description": "画像パスまたはURL"},
                            },
                            "required": ["title", "artist_name"],
                        },
                        "critique_text": {
                            "type": "string",
                            "description": "批評本文（自由記述）",
                        },
                        "scores": {
                            "type": "array",
                            "description": "7動機スコア（最大7項目）",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "motivation_index": {
                                        "type": "integer",
                                        "description": "動機番号 1–7",
                                        "minimum": 1,
                                        "maximum": 7,
                                    },
                                    "motivation_degree": {
                                        "type": "integer",
                                        "description": "動機の度合い 1–10（作品にどれほど内在しているか）",
                                        "minimum": 1,
                                        "maximum": 10,
                                    },
                                    "achievement": {
                                        "type": "integer",
                                        "description": "達成度 1–10（その動機をどれほど成功裏に実現しているか）",
                                        "minimum": 1,
                                        "maximum": 10,
                                    },
                                    "notes": {
                                        "type": "string",
                                        "description": "この動機についての補足メモ",
                                    },
                                },
                                "required": [
                                    "motivation_index",
                                    "motivation_degree",
                                    "achievement",
                                ],
                            },
                        },
                        "personal_resonance": {
                            "type": "integer",
                            "description": "個人的共鳴度 1–10",
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                    "required": ["artwork_info", "critique_text", "scores", "personal_resonance"],
                },
            },
            {
                "name": "recall_art_critiques",
                "description": (
                    "過去に批評した作品を検索・想起する。"
                    "アーティスト名または作品タイトルで絞り込める。"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "検索キーワード（作品名・アーティスト名など）",
                        },
                        "artist": {
                            "type": "string",
                            "description": "アーティスト名で絞り込む場合に指定（省略可）",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_artist_profile",
                "description": "アーティストの7動機傾向プロフィールを集計して返す。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "artist_name": {
                            "type": "string",
                            "description": "アーティスト名",
                        },
                    },
                    "required": ["artist_name"],
                },
            },
            {
                "name": "compare_artworks",
                "description": "2作品の7動機スコアを並べて比較する。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title_a": {
                            "type": "string",
                            "description": "作品Aのタイトル（部分一致可）",
                        },
                        "title_b": {
                            "type": "string",
                            "description": "作品Bのタイトル（部分一致可）",
                        },
                    },
                    "required": ["title_a", "title_b"],
                },
            },
        ]

    async def call(self, tool_name: str, tool_input: dict[str, Any]) -> tuple[str, list[str]]:
        if tool_name == "store_art_critique":
            critique_id = await asyncio.to_thread(
                self._store.store_art_critique,
                tool_input["artwork_info"],
                tool_input.get("critique_text", ""),
                tool_input.get("scores", []),
                int(tool_input.get("personal_resonance", 5)),
                self._memory,
            )
            info = tool_input["artwork_info"]
            return (
                f"批評を保存しました。critique_id={critique_id}\n"
                f"作品: {info.get('artist_name')}「{info.get('title')}」"
            ), []

        if tool_name == "recall_art_critiques":
            results = await asyncio.to_thread(
                self._store.recall_art_critiques,
                tool_input.get("query", ""),
                tool_input.get("artist"),
                self._memory,
            )
            if not results:
                return "該当する批評記録が見つかりませんでした。", []
            lines = []
            for r in results:
                if r.get("source") == "semantic_memory":
                    lines.append(f"[意味記憶] {r.get('critique_text', '')[:120]}")
                    continue
                resonance = r.get("personal_resonance")
                lines.append(
                    f"【{r.get('artist_name')}「{r.get('title')}」"
                    f"{' (' + str(r.get('year')) + ')' if r.get('year') else ''}】"
                    f" 共鳴度:{resonance}/10"
                )
                lines.append(f"  {str(r.get('critique_text', ''))[:200]}")
                for s in r.get("scores", []):
                    lines.append(
                        f"  [{s['motivation_index']}]{s['label']}: "
                        f"度合={s['motivation_degree']} 達成={s['achievement']}"
                        + (f" — {s['notes']}" if s.get("notes") else "")
                    )
            return "\n".join(lines), []

        if tool_name == "get_artist_profile":
            profile = await asyncio.to_thread(
                self._store.get_artist_profile,
                tool_input["artist_name"],
            )
            if profile["artwork_count"] == 0:
                return f"「{tool_input['artist_name']}」の批評記録はまだありません。", []
            lines = [
                f"【{profile['artist_name_normalized']}】"
                f" 作品数:{profile['artwork_count']} 批評数:{profile['critique_count']}"
            ]
            for m in profile["motivation_profile"]:
                lines.append(
                    f"  [{m['motivation_index']}]{m['label']}: "
                    f"平均度合={m['avg_motivation_degree']} "
                    f"平均達成={m['avg_achievement']}"
                )
            return "\n".join(lines), []

        if tool_name == "compare_artworks":
            result = await asyncio.to_thread(
                self._store.compare_artworks,
                tool_input["title_a"],
                tool_input["title_b"],
            )
            a = result["artwork_a"]
            b = result["artwork_b"]
            if not a["artist"] or not b["artist"]:
                missing = tool_input["title_a"] if not a["artist"] else tool_input["title_b"]
                return f"「{missing}」の批評記録が見つかりません。", []

            lines = [f"比較: 「{a['title']}」({a['artist']}) vs 「{b['title']}」({b['artist']})"]
            for c in result["comparison"]:
                keys = [k for k in c if k.endswith("_degree") or k.endswith("_achievement")]
                vals = " | ".join(f"{k}={c[k]}" for k in keys if c[k] is not None)
                if vals:
                    lines.append(f"  [{c['motivation_index']}]{c['label']}: {vals}")

            if result["largest_differences"]:
                lines.append("差が大きい動機:")
                for d in result["largest_differences"]:
                    lines.append(
                        f"  [{d['motivation_index']}]{d['label']} "
                        f"達成差={d.get('diff_achievement')}"
                    )
            return "\n".join(lines), []

        return f"Unknown art critique tool: {tool_name}", []
