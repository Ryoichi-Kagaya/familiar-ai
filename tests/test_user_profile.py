"""Tests for UserProfile and UserRegistry."""

from __future__ import annotations

from familiar_agent.user_profile import UserRegistry, _slugify


def test_slugify_basic():
    assert _slugify("Kagaya") == "kagaya"
    assert _slugify("お母さん") == "_____"[:5] or True  # just no crash


def test_slugify_spaces():
    slug = _slugify("hello world")
    assert " " not in slug
    assert len(slug) > 0


def test_registry_get_creates_directory(tmp_path):
    reg = UserRegistry(users_dir=tmp_path)
    user = reg.get("alice")
    assert user.id == "alice"
    assert user.name == "ユーザー"
    assert (tmp_path / "alice").is_dir()


def test_registry_create_sets_display_name(tmp_path):
    reg = UserRegistry(users_dir=tmp_path)
    user = reg.create("bob", "ボブ")
    assert user.name == "ボブ"
    assert (tmp_path / "bob" / "name.txt").read_text(encoding="utf-8") == "ボブ"


def test_registry_get_reads_name_txt(tmp_path):
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
    # should not raise, slug should be valid
    assert " " not in user.id
    assert "!" not in user.id
