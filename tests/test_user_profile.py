"""Tests for UserProfile and UserRegistry."""

from __future__ import annotations

from familiar_agent.user_profile import UserRegistry, _slugify


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
    assert user.name == "ユーザー"
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


def test_registry_create_updates_existing(tmp_path):
    reg = UserRegistry(users_dir=tmp_path)
    reg.create("carol", "キャロル")
    reg.create("carol", "キャロル改")
    assert reg.get("carol").name == "キャロル改"
    assert sum(1 for e in reg._read() if e["id"] == "carol") == 1


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


def test_registry_active(tmp_path):
    reg = UserRegistry(users_dir=tmp_path)
    assert reg.active_id() == "default"
    reg.set_active("kagaya")
    assert reg.active_id() == "kagaya"


def test_registry_get_active(tmp_path):
    reg = UserRegistry(users_dir=tmp_path)
    reg.create("dave", "デイブ")
    reg.set_active("dave")
    user = reg.get_active()
    assert user.id == "dave"
    assert user.name == "デイブ"


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
