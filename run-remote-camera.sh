#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

profile="${FAMILIAR_CAMERA_PROFILE:-.env.camera-remote}"
if [[ ! -f "$profile" ]]; then
    echo "Camera profile not found: $profile" >&2
    echo "Copy config/camera-remote.env.example to .env.camera-remote first." >&2
    exit 2
fi

exec uv run familiar --camera-profile "$profile" "$@"
