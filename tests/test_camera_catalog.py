"""Tests for the persistent multi-camera catalog and legacy migration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from familiar_agent.camera_catalog import (
    CameraCatalog,
    CameraProfile,
    load_camera_catalog,
    migrate_legacy_camera_catalog,
    save_camera_catalog,
)


def test_catalog_round_trip_and_url_encoding(tmp_path) -> None:
    path = tmp_path / "cameras.json"
    profile = CameraProfile.from_dict(
        {
            "id": "travel",
            "label": "外出用",
            "host": "172.20.10.10",
            "username": "camera user",
            "password": "local/pass",
            "go2rtc_stream": "tapo_cam_remote",
            "tapo_password": "cloud pass",
        }
    )
    catalog = CameraCatalog(default_camera="travel", cameras=(profile,))

    save_camera_catalog(catalog, path)

    loaded = load_camera_catalog(path)
    assert loaded == catalog
    assert path.stat().st_mode & 0o777 == 0o600
    assert profile.go2rtc_sources() == [
        "rtsp://camera%20user:local%2Fpass@172.20.10.10/stream1",
        "tapo://cloud%20pass@172.20.10.10",
    ]


def test_full_rtsp_url_is_preserved_as_go2rtc_source() -> None:
    profile = CameraProfile.from_dict(
        {
            "id": "atom",
            "host": "rtsp://embedded:secret@192.0.2.30/live0",
            "tapo_password": "cloud secret",
        }
    )

    assert profile.go2rtc_sources() == [
        "rtsp://embedded:secret@192.0.2.30/live0",
        "tapo://cloud%20secret@192.0.2.30",
    ]


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            {
                "cameras": [
                    {"id": "same", "host": "192.0.2.1"},
                    {"id": "same", "host": "192.0.2.2"},
                ]
            },
            "ids must be unique",
        ),
        (
            {
                "default_camera": "missing",
                "cameras": [{"id": "main", "host": "192.0.2.1"}],
            },
            "is not defined",
        ),
        (
            {"cameras": [{"id": "Not Valid", "host": "192.0.2.1"}]},
            "Invalid camera id",
        ),
    ],
)
def test_catalog_rejects_invalid_entries(payload, message) -> None:
    with pytest.raises(ValueError, match=message):
        CameraCatalog.from_dict(payload)


def test_migration_combines_primary_and_profile_files(monkeypatch, tmp_path) -> None:
    profile_path = tmp_path / ".env.camera-remote"
    profile_path.write_text(
        "\n".join(
            [
                "CAMERA_LABEL=外出用",
                "CAMERA_HOST=172.20.10.10",
                "CAMERA_USERNAME=remote",
                "CAMERA_PASSWORD=secret",
                "CAMERA_ONVIF_PORT=2020",
                "CAMERA_TAPO_PASSWORD=cloud-secret",
                "GO2RTC_STREAM=tapo_cam_remote",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "state" / "cameras.json"
    primary = SimpleNamespace(
        host="192.0.2.10",
        username="main",
        password="main-secret",
        port=2020,
        preview=False,
        ptz_host_override="",
        ptz_username_override="",
        ptz_password_override="",
        ptz_port_override=None,
    )
    monkeypatch.setenv("GO2RTC_STREAM", "tapo_cam")

    catalog = migrate_legacy_camera_catalog(primary, search_dir=tmp_path, path=output_path)

    assert catalog is not None
    assert catalog.default_camera == "main"
    assert [camera.id for camera in catalog.cameras] == ["main", "remote"]
    assert catalog.cameras[0].connection == "warm"
    assert catalog.cameras[1].connection == "on_demand"
    assert catalog.cameras[1].label == "外出用"
    assert catalog.cameras[1].go2rtc_stream == "tapo_cam_remote"
    assert json.loads(output_path.read_text(encoding="utf-8"))["version"] == 1


def test_existing_catalog_is_not_rewritten(tmp_path) -> None:
    path = tmp_path / "cameras.json"
    original = CameraCatalog(
        default_camera="saved",
        cameras=(CameraProfile.from_dict({"id": "saved", "host": "192.0.2.40"}),),
    )
    save_camera_catalog(original, path)
    before = path.read_bytes()

    loaded = migrate_legacy_camera_catalog(
        SimpleNamespace(host="192.0.2.50"),
        search_dir=tmp_path,
        path=path,
    )

    assert loaded == original
    assert path.read_bytes() == before
