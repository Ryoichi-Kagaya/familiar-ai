"""Validated multi-camera catalog with migration from legacy dotenv profiles."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from dotenv import dotenv_values

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
    go2rtc_stream: str = ""
    tapo_password: str = ""
    tapo_hash: str = ""
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
        onvif_port = int(raw.get("onvif_port") or 2020)
        ptz_port_raw = raw.get("ptz_port")
        ptz_port = int(str(ptz_port_raw)) if ptz_port_raw not in (None, "") else None
        stream = str(raw.get("go2rtc_stream", "")).strip() or f"tapo_cam_{camera_id}"
        return cls(
            id=camera_id,
            label=str(raw.get("label", camera_id)).strip() or camera_id,
            host=host,
            username=str(raw.get("username", "admin")),
            password=str(raw.get("password", "")),
            onvif_port=onvif_port,
            preview=bool(raw.get("preview", False)),
            ptz_host=str(raw.get("ptz_host", "")),
            ptz_username=str(raw.get("ptz_username", "")),
            ptz_password=str(raw.get("ptz_password", "")),
            ptz_port=ptz_port,
            go2rtc_stream=stream,
            tapo_password=str(raw.get("tapo_password", "")),
            tapo_hash=str(raw.get("tapo_hash", "")),
            connection=connection,
        )

    def go2rtc_sources(self) -> list[str]:
        sources: list[str] = []
        parsed = urlparse(self.host) if "://" in self.host else None
        tapo_host = parsed.hostname if parsed is not None else self.host
        if parsed is not None and parsed.scheme.lower() == "rtsp":
            sources.append(self.host)
        elif self.username and self.password:
            user = quote(self.username, safe="")
            password = quote(self.password, safe="")
            sources.append(f"rtsp://{user}:{password}@{self.host}/stream1")
        if self.tapo_password and tapo_host:
            sources.append(f"tapo://{quote(self.tapo_password, safe='')}@{tapo_host}")
        if self.tapo_hash and tapo_host:
            sources.append(f"tapo://admin:{quote(self.tapo_hash, safe='')}@{tapo_host}")
        return sources


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
        streams = [camera.go2rtc_stream for camera in cameras]
        if len(streams) != len(set(streams)):
            raise ValueError("Camera go2rtc_stream values must be unique")
        default_camera = str(raw.get("default_camera", "")).strip() or ids[0]
        if default_camera not in ids:
            raise ValueError(f"Default camera {default_camera!r} is not defined")
        return cls(default_camera=default_camera, cameras=cameras, version=version)

    def get(self, camera_id: str) -> CameraProfile | None:
        return next((camera for camera in self.cameras if camera.id == camera_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "default_camera": self.default_camera,
            "cameras": [asdict(camera) for camera in self.cameras],
        }


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


def save_camera_catalog(catalog: CameraCatalog, path: Path | None = None) -> Path:
    resolved = (path or default_camera_catalog_path()).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(catalog.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        resolved.chmod(0o600)
    except OSError:
        pass
    return resolved


def _dotenv_camera_profile(path: Path) -> CameraProfile | None:
    values = dotenv_values(path)
    host = str(values.get("CAMERA_HOST") or values.get("TAPO_CAMERA_HOST") or "").strip()
    if not host:
        return None
    suffix = path.name.removeprefix(".env.camera-")
    fallback_id = re.sub(r"[^a-z0-9_]+", "_", suffix.lower()).strip("_") or "camera"
    camera_id = str(values.get("CAMERA_ID") or fallback_id).strip()
    raw: dict[str, Any] = {
        "id": camera_id,
        "label": values.get("CAMERA_LABEL") or camera_id,
        "host": host,
        "username": values.get("CAMERA_USERNAME") or values.get("TAPO_USERNAME") or "admin",
        "password": values.get("CAMERA_PASSWORD") or values.get("TAPO_PASSWORD") or "",
        "onvif_port": values.get("CAMERA_ONVIF_PORT") or values.get("TAPO_ONVIF_PORT") or 2020,
        "preview": str(values.get("CAMERA_PREVIEW") or "false").lower() == "true",
        "ptz_host": values.get("CAMERA_PTZ_HOST") or "",
        "ptz_username": values.get("CAMERA_PTZ_USERNAME") or "",
        "ptz_password": values.get("CAMERA_PTZ_PASSWORD") or "",
        "ptz_port": values.get("CAMERA_PTZ_PORT") or None,
        "go2rtc_stream": values.get("GO2RTC_STREAM") or f"tapo_cam_{camera_id}",
        "tapo_password": values.get("CAMERA_TAPO_PASSWORD") or "",
        "tapo_hash": values.get("CAMERA_TAPO_HASH") or "",
        "connection": values.get("CAMERA_CONNECTION") or "on_demand",
    }
    return CameraProfile.from_dict(raw)


def migrate_legacy_camera_catalog(
    primary: Any,
    *,
    search_dir: Path,
    path: Path | None = None,
) -> CameraCatalog | None:
    """Create cameras.json once from the primary env and .env.camera-* files."""
    resolved = path or default_camera_catalog_path()
    existing = load_camera_catalog(resolved)
    if existing is not None:
        return existing

    cameras: list[CameraProfile] = []
    primary_host = str(getattr(primary, "host", "") or "").strip()
    if primary_host:
        cameras.append(
            CameraProfile.from_dict(
                {
                    "id": "main",
                    "label": "普段のカメラ",
                    "host": primary_host,
                    "username": getattr(primary, "username", "admin"),
                    "password": getattr(primary, "password", ""),
                    "onvif_port": getattr(primary, "port", 2020),
                    "preview": getattr(primary, "preview", False),
                    "ptz_host": getattr(primary, "ptz_host_override", ""),
                    "ptz_username": getattr(primary, "ptz_username_override", ""),
                    "ptz_password": getattr(primary, "ptz_password_override", ""),
                    "ptz_port": getattr(primary, "ptz_port_override", None),
                    "go2rtc_stream": os.environ.get("GO2RTC_STREAM", "tapo_cam"),
                    "tapo_password": os.environ.get("CAMERA_TAPO_PASSWORD", ""),
                    "tapo_hash": os.environ.get("CAMERA_TAPO_HASH", ""),
                    "connection": "warm",
                }
            )
        )

    for profile_path in sorted(search_dir.glob(".env.camera-*")):
        profile = _dotenv_camera_profile(profile_path)
        if profile is not None and all(existing.id != profile.id for existing in cameras):
            cameras.append(profile)

    if not cameras:
        return None
    default_id = "main" if any(camera.id == "main" for camera in cameras) else cameras[0].id
    catalog = CameraCatalog(default_camera=default_id, cameras=tuple(cameras))
    save_camera_catalog(catalog, resolved)
    return catalog


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
    config.tts.go2rtc_stream = profile.go2rtc_stream
