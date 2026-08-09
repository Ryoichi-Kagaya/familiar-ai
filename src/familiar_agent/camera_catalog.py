"""Validated persistent multi-camera catalog."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CAMERA_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CONNECTION_MODES = {"warm", "on_demand"}


def default_camera_catalog_path() -> Path:
    override = os.environ.get("FAMILIAR_CAMERAS_CONFIG", "").strip()
    return (
        Path(override).expanduser() if override else Path.home() / ".familiar_ai" / "cameras.json"
    )


@dataclass(frozen=True, slots=True)
class CameraProfile:
    id: str
    label: str
    host: str
    username: str = "admin"
    password: str = ""
    onvif_port: int = 2020
    preview: bool = False
    ptz_host: str = ""
    ptz_username: str = ""
    ptz_password: str = ""
    ptz_port: int | None = None
    connection: str = "on_demand"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CameraProfile:
        camera_id = str(raw.get("id", "")).strip()
        if not _CAMERA_ID_RE.fullmatch(camera_id):
            raise ValueError(
                f"Invalid camera id {camera_id!r}; use lowercase letters, digits, and underscores"
            )
        host = str(raw.get("host", "")).strip()
        if not host:
            raise ValueError(f"Camera {camera_id!r} has no host")
        connection = str(raw.get("connection", "on_demand")).strip().lower()
        if connection not in _CONNECTION_MODES:
            raise ValueError(f"Camera {camera_id!r} connection must be 'warm' or 'on_demand'")
        preview = raw.get("preview", False)
        if not isinstance(preview, bool):
            raise ValueError(f"Camera {camera_id!r} preview must be a boolean")
        onvif_port = int(raw.get("onvif_port") or 2020)
        ptz_port_raw = raw.get("ptz_port")
        ptz_port = int(str(ptz_port_raw)) if ptz_port_raw not in (None, "") else None
        return cls(
            id=camera_id,
            label=str(raw.get("label", camera_id)).strip() or camera_id,
            host=host,
            username=str(raw.get("username", "admin")),
            password=str(raw.get("password", "")),
            onvif_port=onvif_port,
            preview=preview,
            ptz_host=str(raw.get("ptz_host", "")),
            ptz_username=str(raw.get("ptz_username", "")),
            ptz_password=str(raw.get("ptz_password", "")),
            ptz_port=ptz_port,
            connection=connection,
        )


@dataclass(frozen=True, slots=True)
class CameraCatalog:
    default_camera: str
    cameras: tuple[CameraProfile, ...]
    version: int = 1

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CameraCatalog:
        version = int(raw.get("version", 1))
        if version != 1:
            raise ValueError(f"Unsupported cameras.json version: {version}")
        camera_items = raw.get("cameras")
        if not isinstance(camera_items, list) or not camera_items:
            raise ValueError("cameras.json must contain a non-empty 'cameras' list")
        cameras = tuple(
            CameraProfile.from_dict(item) for item in camera_items if isinstance(item, dict)
        )
        if len(cameras) != len(camera_items):
            raise ValueError("Every cameras.json camera entry must be an object")
        ids = [camera.id for camera in cameras]
        if len(ids) != len(set(ids)):
            raise ValueError("Camera ids must be unique")
        default_camera = str(raw.get("default_camera", "")).strip() or ids[0]
        if default_camera not in ids:
            raise ValueError(f"Default camera {default_camera!r} is not defined")
        return cls(default_camera=default_camera, cameras=cameras, version=version)

    def get(self, camera_id: str) -> CameraProfile | None:
        return next((camera for camera in self.cameras if camera.id == camera_id), None)


def load_camera_catalog(path: Path | None = None) -> CameraCatalog | None:
    resolved = (path or default_camera_catalog_path()).expanduser()
    if not resolved.is_file():
        return None
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read camera catalog {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("cameras.json root must be an object")
    return CameraCatalog.from_dict(raw)


def apply_default_camera(config: Any, catalog: CameraCatalog) -> None:
    """Apply the catalog's default endpoint to legacy single-camera config fields."""
    from .config import CameraConfig

    profile = catalog.get(catalog.default_camera)
    if profile is None:
        return
    config.camera = CameraConfig(
        host=profile.host,
        username=profile.username,
        password=profile.password,
        port=profile.onvif_port,
        preview=profile.preview,
        ptz_host_override=profile.ptz_host,
        ptz_username_override=profile.ptz_username,
        ptz_password_override=profile.ptz_password,
        ptz_port_override=profile.ptz_port,
    )
