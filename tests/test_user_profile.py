"""Tests for UserProfile and UserRegistry."""

from __future__ import annotations

from familiar_agent.user_profile import UserRegistry, _default_display_name, _slugify


def test_slugify_basic():
    assert _slugify("Kagaya") == "kagaya"


def test_slugify_spaces():
    slug = _slugify("hello world")
    assert " " not in slug
    assert len(slug) > 0


def test_registry_get_creates_entry(tmp_path):
    reg = UserRegistry(users_dir=tmp_path)
    user = reg.get("alice")
    assert user.id == "alice"
    assert user.name == _default_display_name()
    assert (tmp_path / "alice").is_dir()


def test_registry_get_persists_to_json(tmp_path):
    reg = UserRegistry(users_dir=tmp_path)
    reg.get("alice")
    entries = reg._read()
    assert any(e["id"] == "alice" for e in entries)


def test_registry_create_sets_display_name(tmp_path):
    reg = UserRegistry(users_dir=tmp_path)
    user = reg.create("bob", "ボブ")
    assert user.name == "ボブ"


def test_registry_create_round_trips_aliases(tmp_path):
    reg = UserRegistry(users_dir=tmp_path)
    user = reg.create("sister", "ほのるる", aliases=["姉", "お姉ちゃん", "姉"])

    assert user.aliases == ("姉", "お姉ちゃん")
    assert reg.get("sister").aliases == ("姉", "お姉ちゃん")


def test_registry_scopes_relationship_aliases_to_the_speaker(tmp_path):
    reg = UserRegistry(users_dir=tmp_path)
    user = reg.create(
        "sister",
        "ほのるる",
        aliases_by_user={"default": ["姉", "お姉ちゃん", "姉"]},
    )

    assert "姉" in user.references_from("default")
    assert "姉" not in user.references_from("mom")


def test_registry_create_updates_existing(tmp_path):
    reg = UserRegistry(users_dir=tmp_path)
    reg.create("carol", "キャロル")
    reg.create("carol", "キャロル改")
    assert reg.get("carol").name == "キャロル改"
    assert sum(1 for e in reg._read() if e["id"] == "carol") == 1


def test_registry_name_update_preserves_aliases_when_omitted(tmp_path):
    reg = UserRegistry(users_dir=tmp_path)
    reg.create("sister", "ほのるる", aliases=["姉"])
    reg.create("sister", "ほのるる改")

    assert reg.get("sister").aliases == ("姉",)


def test_registry_get_reads_from_json(tmp_path):
    reg = UserRegistry(users_dir=tmp_path)
    reg.create("carol", "キャロル")
    user = reg.get("carol")
    assert user.name == "キャロル"


def test_registry_list_users(tmp_path):
    reg = UserRegistry(users_dir=tmp_path)
    reg.create("alice", "アリス")
    reg.create("bob", "ボブ")
    ids = [u.id for u in reg.list_users()]
    assert "alice" in ids
    assert "bob" in ids


def test_registry_get_active_returns_default(tmp_path):
    reg = UserRegistry(users_dir=tmp_path)
    user = reg.get_active()
    assert user.id == "default"


def test_user_profile_paths(tmp_path):
    reg = UserRegistry(users_dir=tmp_path)
    user = reg.get("eve")
    assert user.mental_state_path == tmp_path / "eve" / "mental_state.jsonl"
    assert user.self_narrative_path == tmp_path / "eve" / "self_narrative.jsonl"


def test_registry_invalid_id_slugified(tmp_path):
    reg = UserRegistry(users_dir=tmp_path)
    user = reg.get("Hello World!")
    assert " " not in user.id
    assert "!" not in user.id
