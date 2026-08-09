"""Tests for the persistent multi-camera catalog."""

from __future__ import annotations

import json

import pytest

from familiar_agent.camera_catalog import CameraProfile, load_camera_catalog


def test_load_catalog(tmp_path) -> None:
    path = tmp_path / "cameras.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "default_camera": "travel",
                "cameras": [
                    {
                        "id": "travel",
                        "label": "外出用",
                        "host": "172.20.10.10",
                        "username": "camera user",
                        "password": "local/pass",
                        "connection": "on_demand",
                        # Older catalogs may contain speaker settings. They are
                        # harmless but not part of image-camera selection.
                        "go2rtc_stream": "tapo_cam_remote",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_camera_catalog(path)

    assert loaded is not None
    assert loaded.default_camera == "travel"
    assert loaded.cameras == (
        CameraProfile(
            id="travel",
            label="外出用",
            host="172.20.10.10",
            username="camera user",
            password="local/pass",
            connection="on_demand",
        ),
    )


def test_missing_catalog_returns_none(tmp_path) -> None:
    assert load_camera_catalog(tmp_path / "missing.json") is None


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
        (
            {"cameras": [{"id": "main", "host": "192.0.2.1", "preview": "false"}]},
            "preview must be a boolean",
        ),
    ],
)
def test_catalog_rejects_invalid_entries(tmp_path, payload, message) -> None:
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_camera_catalog(path)
