"""PySide6 GUI for familiar-ai — Monokai dark theme.

Provides a native desktop window with:
- Scrollable conversation log with styled HTML-like bubbles (ChatLog)
- Live streaming text display with blinking cursor (StreamLabel)
- Camera image viewer (CameraView)
- Animated desire level bars (DesirePanel)
- Pill-shaped text input with circular send button
- Settings dialog (⚙) for .env configuration

Launch via:
    uv run familiar --gui

or via run-gui.bat / run-gui.sh
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import html as _html
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote

import qasync
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QSize, Qt, QTimer
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from familiar_runtime.models import ImageAttachment, UserTurn, coerce_user_turn

from ._i18n import _t
from ._ui_helpers import (
    DESIRE_COOLDOWN,
    IDLE_CHECK_INTERVAL,
    commitment_reminder_prompt,
    format_action,
    format_tool_result,
    should_fire_commitment_reminder,
    should_fire_idle_desire,
    should_run_sleep_consolidation,
)
from .bootstrap import resolve_env_path
from .drive_executor import DriveActionExecutor
from .user_profile import UserRegistry
from .diagnostics import (
    build_gui_diagnostics,
    format_gui_diagnostics,
    test_backend_connection,
    test_camera_connection_from_config,
    test_realtime_stt_connection_from_config,
)
from .realtime_stt_session import create_realtime_stt_controller
from .image_input import (
    ImageInputError,
    load_image_attachment,
    validate_image_count,
)
from .settings_schema import (
    SECTION_LABELS,
    SetupConfig,
    SettingField,
    iter_setting_fields,
    sections_for_mode,
    setup_config_from_agent_config,
    validate_setup_config,
)
from .setup import save_setup_config

if TYPE_CHECKING:
    from familiar_agent.agent import EmbodiedAgent
    from familiar_agent.config import AgentConfig
    from familiar_agent.desires import DesireSystem
    from familiar_agent.realtime_stt_session import RealtimeSttController

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

_BG_BASE = "#272822"
_BG_SURFACE = "#2D2E27"
_BG_CARD = "#2D2E27"
_BG_ELEVATED = "#3E3D32"
_BG_HOVER = "#49483E"
_ACCENT = "#A6E22E"
_ACCENT_DEEP = "#F92672"
_ACCENT_DIM = "#75715E"
_TEXT_PRIMARY = "#F8F8F2"
_TEXT_SECONDARY = "#A59F85"
_BORDER = "#49483E"
_BUBBLE_USER_BG = "#3E3D32"
_BUBBLE_AGENT_BG = "#2D2E27"
_BUBBLE_TOOL_BG = "#1E1F1C"
_BUBBLE_ERROR_BG = "#3E1F24"
_UI_FONT_STACK = (
    "'Noto Sans CJK JP', 'Yu Gothic UI', 'Hiragino Sans', 'Meiryo', 'Segoe UI', sans-serif"
)
_MONO_FONT_STACK = "'Cascadia Mono', 'Consolas', 'Courier New', monospace"
_FONT_SCALE = 1.5

# Semantic accent colors not in the Monokai base palette
_ACCENT_MIC = "#7C3AED"

# Translucent surface tokens — rgba() because hex cannot encode alpha
_DISABLED_BG = "rgba(127,115,148,0.12)"
_DISABLED_BORDER = "rgba(127,115,148,0.20)"
_PROGRESS_TRACK = "rgba(127,115,148,0.16)"
_STOP_BG = "rgba(255,107,115,0.20)"
_STOP_BORDER = "rgba(255,107,115,0.52)"
_STOP_TEXT = "#e34f5d"
_STOP_HOVER_BG = "rgba(255,107,115,0.30)"
_STOP_HOVER_TEXT = "#c13f4d"
_SEND_HOVER_STOP0 = "#FF5F96"
_SEND_HOVER_STOP1 = "#FF9FBC"

_DESIRE_COLORS: dict[str, str] = {
    "look_around": "#57b8ff",
    "look_outside": "#4f8cff",
    "miss_companion": "#ff7ea3",
    "browse_curiosity": "#58c5b7",
    "explore": "#5ea1ff",
    "greet_companion": "#ffb35f",
    "worry_companion": "#ff6b73",
}

# Flush streamed text at most this often (ms)
_STREAM_FLUSH_INTERVAL_MS = 50
_GUI_LOOP_LAG_CHECK_SEC = 1.0
_GUI_LOOP_LAG_WARN_SEC = 0.35
_GUI_QUEUE_WARN_SIZE = 20
_GUI_LOOK_PREVIEW_FPS = 8
_GUI_LOOK_PREVIEW_MIN_SEC = 0.8
_GUI_LOOK_PREVIEW_MAX_SEC = 2.0
_GUI_LOOK_PREVIEW_GRACE_SEC = 0.3
_GUI_LOOK_PREVIEW_READ_TIMEOUT_SEC = 0.35
_SUBPROCESS_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_APP_ICON_ENV = "FAMILIAR_APP_ICON"


def _runtime_base_dir() -> Path:
    """Return runtime base dir (frozen exe dir for PyInstaller builds, else cwd)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def resolve_app_icon_path() -> Path | None:
    """Find app icon: env override → exe dir → assets/ fallback."""
    base_dir = _runtime_base_dir()
    env_icon = (os.environ.get(_APP_ICON_ENV, "") or "").strip()
    candidates: list[Path] = []
    if env_icon:
        p = Path(env_icon)
        candidates.append(p if p.is_absolute() else base_dir / p)
    candidates.extend(
        [
            base_dir / "app.ico",
            Path(__file__).resolve().parents[2] / "assets" / "app.ico",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _subprocess_exec_kwargs() -> dict[str, Any]:
    """Return platform-specific kwargs to hide console windows on Windows."""
    if _SUBPROCESS_NO_WINDOW:
        return {"creationflags": _SUBPROCESS_NO_WINDOW}
    return {}


def _px(size: int) -> int:
    """Scale font-size in px for large readable UI."""
    return max(1, int(round(size * _FONT_SCALE)))


# Resolve .env path once for settings / setup flows.
_ENV_PATH: Path = resolve_env_path()


# ---------------------------------------------------------------------------
# Global stylesheet
# ---------------------------------------------------------------------------


def _apply_global_style(app: QApplication) -> None:
    """Apply Monokai dark theme to the whole application."""
    app.setStyleSheet(
        f"""
        QWidget {{
            color: {_TEXT_PRIMARY};
            font-family: {_UI_FONT_STACK};
            font-size: {_px(13)}px;
            background: {_BG_BASE};
        }}

        QMainWindow {{
            background: {_BG_BASE};
        }}

        QDialog {{
            background: {_BG_SURFACE};
            border: 1px solid {_BORDER};
            border-radius: {_px(12)}px;
        }}

        QLabel {{ color: {_TEXT_PRIMARY}; }}

        /* Scrollbar — Monokai style */
        QScrollBar:vertical {{
            background: transparent; width: 8px;
        }}
        QScrollBar::handle:vertical {{
            background: {_ACCENT_DIM};
            border-radius: 4px;
            min-height: 16px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {_ACCENT};
        }}
        QScrollBar::add-line:vertical, 
        QScrollBar::sub-line:vertical {{
            height: 0;
        }}

        /* Inputs */
        QLineEdit {{
            background: {_BG_SURFACE};
            color: {_TEXT_PRIMARY};
            border: 1px solid {_BORDER};
            border-radius: {_px(6)}px;
            padding: {_px(5)}px {_px(10)}px;
            selection-background-color: {_ACCENT};
        }}
        QLineEdit:focus {{
            border-color: {_ACCENT};
            background: {_BG_ELEVATED};
        }}

        /* Buttons */
        QPushButton {{
            background: {_BG_ELEVATED};
            color: {_TEXT_PRIMARY};
            border: 1px solid {_BORDER};
            border-radius: {_px(6)}px;
            padding: {_px(5)}px {_px(12)}px;
        }}
        QPushButton:hover {{
            background: {_BG_HOVER};
            border-color: {_ACCENT};
        }}
        QPushButton:pressed {{
            background: {_ACCENT_DEEP};
            color: {_BG_BASE};
        }}
        QPushButton:disabled {{
            background: {_BG_SURFACE};
            color: {_TEXT_SECONDARY};
            border-color: {_BORDER};
        }}

        /* ComboBox */
        QComboBox {{
            background: {_BG_SURFACE};
            color: {_TEXT_PRIMARY};
            border: 1px solid {_BORDER};
            border-radius: {_px(6)}px;
            padding: {_px(5)}px {_px(10)}px;
        }}
        QComboBox::drop-down {{
            border: none;
        }}
        QComboBox QAbstractItemView {{
            background: {_BG_CARD};
            color: {_TEXT_PRIMARY};
            selection-background-color: {_ACCENT};
            border: 1px solid {_BORDER};
        }}

        /* Tabs */
        QTabWidget::pane {{
            border: 1px solid {_BORDER};
            background: {_BG_SURFACE};
            border-radius: {_px(10)}px;
        }}

        QTabBar::tab {{
            background: {_BG_ELEVATED};
            color: {_TEXT_SECONDARY};
            padding: {_px(5)}px {_px(12)}px;
            border-radius: {_px(7)}px;
            margin-right: {_px(4)}px;
            border: 1px solid {_BORDER};
        }}

        QTabBar::tab:selected {{
            background: {_ACCENT};
            color: {_BG_BASE};
            border-color: {_ACCENT};
        }}

        QTabBar::tab:hover:!selected {{
            background: {_BG_HOVER};
            color: {_TEXT_PRIMARY};
        }}
        """
    )


# ---------------------------------------------------------------------------
# ChatLog — bubble-based conversation log
# ---------------------------------------------------------------------------


class ChatLog(QScrollArea):
    """Scrollable chat log that renders messages as styled bubbles.

    Public API (backward-compatible with ActionLog):
        append_line(text: str) -> None
        append_action(name: str, tool_input: dict) -> None
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        agent_label: str = "Agent",
        companion_label: str = "You",
    ) -> None:
        super().__init__(parent)
        self._agent_label = (agent_label or "Agent").strip() or "Agent"
        self._companion_label = (companion_label or "You").strip() or "You"
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet(f"QScrollArea {{ background: {_BG_BASE}; border: none; }}")

        self._container = QWidget()
        self._container.setStyleSheet(f"background: {_BG_BASE}; border: none;")
        self._vbox = QVBoxLayout(self._container)
        self._vbox.setContentsMargins(10, 10, 10, 10)
        self._vbox.setSpacing(4)
        # Top stretch pushes messages to the bottom of the viewport (chat-app style).
        # As messages accumulate and overflow the viewport the stretch shrinks to zero
        # and the scrollbar takes over.
        self._vbox.addStretch(1)

        self.setWidget(self._container)

        # Auto-scroll: follow new content unless the user has scrolled up.
        self._auto_scroll = True
        self.verticalScrollBar().valueChanged.connect(self._on_scroll_value_changed)
        self.verticalScrollBar().rangeChanged.connect(self._on_scroll_range_changed)

    def _on_scroll_value_changed(self, value: int) -> None:
        sb = self.verticalScrollBar()
        self._auto_scroll = value >= sb.maximum() - 20

    def _on_scroll_range_changed(self, _min: int, maximum: int) -> None:
        if self._auto_scroll:
            self.verticalScrollBar().setValue(maximum)

    def _scroll_to_bottom(self) -> None:
        self._auto_scroll = True
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    @staticmethod
    def _extract_prefixed_text(text: str, label: str) -> str | None:
        marker = f"[{label}]"
        if text.startswith(marker):
            return text[len(marker) :].strip()
        return None

    def append_line(self, text: str) -> None:
        """Add a styled bubble. Prefix determines bubble style."""
        text = text.strip()
        if not text:
            return

        user_text = self._extract_prefixed_text(text, self._companion_label)
        if user_text is None and self._companion_label != "You":
            user_text = self._extract_prefixed_text(text, "You")
        if user_text is not None:
            self._add_bubble(
                user_text,
                prefix=self._companion_label,
                prefix_color=_TEXT_SECONDARY,
                bg=_BUBBLE_USER_BG,
                ml=60,
                mr=4,
            )
            return

        agent_text = self._extract_prefixed_text(text, self._agent_label)
        if agent_text is None and self._agent_label != "Agent":
            agent_text = self._extract_prefixed_text(text, "Agent")
        if agent_text is not None:
            self._add_bubble(
                agent_text,
                prefix=self._agent_label,
                prefix_color=_ACCENT,
                bg=_BUBBLE_AGENT_BG,
                ml=4,
                mr=60,
                accent_left=True,
            )
            return

        if text.startswith("[error]"):
            self._add_bubble(
                f"⚠ {text[7:].strip()}",
                bg=_BUBBLE_ERROR_BG,
                text_color=_ACCENT_DEEP,
                ml=20,
                mr=20,
                small=True,
            )
        else:
            self._add_bubble(
                text,
                bg=_BUBBLE_TOOL_BG,
                text_color=_TEXT_SECONDARY,
                ml=20,
                mr=20,
                small=True,
                monospace=True,
            )

    def set_companion_label(self, label: str) -> None:
        self._companion_label = label.strip() or "You"

    def append_action(self, name: str, tool_input: dict) -> None:
        """Format a tool call and append it as a bubble."""
        self.append_line(format_action(name, tool_input))

    def _add_bubble(
        self,
        text: str,
        bg: str,
        ml: int = 0,
        mr: int = 0,
        prefix: str = "",
        prefix_color: str = _TEXT_SECONDARY,
        text_color: str = _TEXT_PRIMARY,
        small: bool = False,
        monospace: bool = False,
        accent_left: bool = False,
    ) -> None:
        escaped = _html.escape(text).replace("\n", "<br>")
        fs = f"{_px(11) if small else _px(13)}px"
        ff = _MONO_FONT_STACK if monospace else _UI_FONT_STACK

        if prefix:
            inner_html = (
                f'<span style="color:{prefix_color};font-size:{_px(10)}px;font-weight:600;'
                f'letter-spacing:0.04em;text-transform:uppercase;">'
                f"{_html.escape(prefix)}</span><br>"
                f'<span style="color:{text_color};font-size:{fs};font-family:{ff};">'
                f"{escaped}</span>"
            )
        else:
            inner_html = (
                f'<span style="color:{text_color};font-size:{fs};font-family:{ff};">'
                f"{escaped}</span>"
            )

        label = QLabel(inner_html)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if accent_left:
            label.setStyleSheet(
                f"background: {bg}; border-radius: {_px(10)}px; padding: {_px(6)}px {_px(12)}px;"
                f" border: 1px solid {_BORDER};"
                f" border-left: 3px solid {_ACCENT};"
            )
        else:
            label.setStyleSheet(
                f"background: {bg}; border-radius: {_px(10)}px; padding: {_px(6)}px {_px(12)}px;"
                f" border: 1px solid {_BORDER};"
            )
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        row = QWidget()
        row.setStyleSheet("background: transparent;")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(ml, 2, mr, 2)
        rl.setSpacing(0)
        rl.addWidget(label)

        self._vbox.addWidget(row)
        self._scroll_to_bottom()


# ---------------------------------------------------------------------------
# StreamLabel — live LLM token display with blinking cursor
# ---------------------------------------------------------------------------


class StreamLabel(QWidget):
    """Displays streaming LLM text tokens in real time.

    Public API:
        append_chunk(chunk: str) -> None
        commit_and_clear() -> str
    """

    _BLINK_MS = 500

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._chunks: list[str] = []
        self._text = ""
        self._status_text = ""
        self._cursor_on = False

        self._label = QLabel("")
        self._label.setWordWrap(True)
        self._label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._label.setStyleSheet(
            f"background: {_BG_SURFACE}; color: {_TEXT_PRIMARY};"
            f" padding: {_px(6)}px {_px(12)}px; border-radius: {_px(10)}px;"
            f" border: 1px solid {_BORDER}; border-left: 3px solid {_ACCENT};"
            f" font-family: {_UI_FONT_STACK}; font-size: {_px(13)}px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(_STREAM_FLUSH_INTERVAL_MS)
        self._flush_timer.timeout.connect(self._flush)
        self._flush_timer.start()

        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(self._BLINK_MS)
        self._blink_timer.timeout.connect(self._blink)

    def append_chunk(self, chunk: str) -> None:
        self._chunks.append(chunk)
        self._status_text = ""
        if not self._blink_timer.isActive():
            self._blink_timer.start()

    def set_status(self, text: str) -> None:
        """Display non-streaming status text (e.g., thinking elapsed time)."""
        self._status_text = text
        if not self._text and not self._chunks:
            self._label.setText(self._status_text)

    def clear_status(self) -> None:
        """Clear the status text if no streaming text is currently shown."""
        self._status_text = ""
        if not self._text and not self._chunks:
            self._label.setText("")

    def has_content(self) -> bool:
        """Whether streamed assistant text is currently visible/pending."""
        return bool(self._text or self._chunks)

    def _flush(self) -> None:
        if self._chunks:
            self._text += "".join(self._chunks)
            self._chunks.clear()
            self._update_label()

    def _blink(self) -> None:
        self._cursor_on = not self._cursor_on
        self._update_label()

    def _update_label(self) -> None:
        if self._text:
            cursor = "▊" if self._cursor_on else " "
            self._label.setText(self._text + cursor)
        elif self._status_text:
            self._label.setText(self._status_text)
        else:
            self._label.setText("")

    def commit_and_clear(self) -> str:
        """Flush pending chunks, return full accumulated text, then clear."""
        self._flush()
        text = self._text
        self._text = ""
        self._chunks.clear()
        self._status_text = ""
        self._cursor_on = False
        self._blink_timer.stop()
        self._label.setText("")
        return text


# ---------------------------------------------------------------------------
# CameraView
# ---------------------------------------------------------------------------


class CameraView(QLabel):
    """Displays the latest camera image (base64-encoded JPEG/PNG)."""

    _PLACEHOLDER_SIZE = QSize(640, 360)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(self._PLACEHOLDER_SIZE)
        self.setStyleSheet(
            f"background: {_BG_CARD}; border-radius: {_px(18)}px;"
            f" border: 1px solid {_BORDER}; color: {_TEXT_SECONDARY};"
        )
        self.setText("No camera image yet")
        self.setWordWrap(True)

    def update_image(self, b64_data: str) -> None:
        """Decode a base64 image and display it."""
        try:
            raw = base64.b64decode(b64_data)
            qimage = QImage.fromData(raw)
            if qimage.isNull():
                logger.warning("CameraView: could not decode image data")
                return
            pixmap = QPixmap.fromImage(qimage)
            scaled = pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
        except Exception:
            logger.exception("CameraView.update_image failed")


# ---------------------------------------------------------------------------
# DesireBar — single desire with animated progress bar
# ---------------------------------------------------------------------------


class DesireBar(QWidget):
    """A labeled progress bar with smooth animation for one desire."""

    def __init__(self, name: str, color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 2, 0, 2)
        vbox.setSpacing(2)

        # Header: desire name + percentage
        header = QHBoxLayout()
        try:
            display_name = _t(f"desire_label_{name}")
        except KeyError:
            display_name = name.replace("_", " ").title()
        name_lbl = QLabel(display_name)
        name_lbl.setStyleSheet(
            f"color: {color}; font-size: {_px(10)}px; font-weight: 600; background: transparent;"
            f" letter-spacing: 0.03em;"
        )
        header.addWidget(name_lbl)
        header.addStretch()
        self._pct_label = QLabel("0%")
        self._pct_label.setStyleSheet(
            f"color: {_TEXT_SECONDARY}; font-size: {_px(10)}px; background: transparent;"
        )
        header.addWidget(self._pct_label)
        vbox.addLayout(header)

        # Gradient progress bar
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(_px(4))
        self._bar.setStyleSheet(
            f"QProgressBar {{ background: {_PROGRESS_TRACK}; border-radius: 3px; border: none; }}"
            f"QProgressBar::chunk {{"
            f" background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f" stop:0 rgba(0,0,0,0), stop:0.3 {color}55, stop:1 {color});"
            f" border-radius: 3px;"
            f"}}"
        )
        vbox.addWidget(self._bar)

        # Smooth animation
        self._anim = QPropertyAnimation(self._bar, b"value")
        self._anim.setDuration(500)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def set_level(self, level: float) -> None:
        target = int(level * 100)
        self._anim.stop()
        self._anim.setStartValue(self._bar.value())
        self._anim.setEndValue(target)
        self._anim.start()
        self._pct_label.setText(f"{target}%")


# ---------------------------------------------------------------------------
# DesirePanel
# ---------------------------------------------------------------------------


class DesirePanel(QWidget):
    """Shows desire levels as animated bars, refreshed every 2 seconds."""

    def __init__(self, desires: "DesireSystem", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._desires = desires
        self._bars: dict[str, DesireBar] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)

        title = QLabel(_t("desire_panel_title"))
        title.setStyleSheet(
            f"color: {_TEXT_SECONDARY}; font-size: {_px(10)}px; font-weight: 600;"
            f" background: transparent; letter-spacing: 0.1em;"
        )
        layout.addWidget(title)

        for name, color in _DESIRE_COLORS.items():
            bar = DesireBar(name, color)
            self._bars[name] = bar
            layout.addWidget(bar)

        layout.addStretch()

        timer = QTimer(self)
        timer.timeout.connect(self._refresh)
        timer.start(2000)
        self._refresh()

    def _refresh(self) -> None:
        for name, bar in self._bars.items():
            try:
                level = self._desires.level(name)
                bar.set_level(level)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# SettingsDialog
# ---------------------------------------------------------------------------


class SettingsDialog(QDialog):
    """Settings dialog with 4 tabs: Agent, Voice, Camera, Advanced."""

    def __init__(
        self,
        config: "AgentConfig",
        env_path: Path,
        *,
        setup_mode: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._env_path = env_path
        self._setup_mode = setup_mode
        self._setup = setup_config_from_agent_config(config)
        self._field_widgets: dict[str, QWidget] = {}
        self.setWindowTitle("Set up familiar-ai" if setup_mode else _t("settings_window_title"))
        self.setMinimumWidth(760)
        self.setModal(True)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(16, 16, 16, 12)
        vbox.setSpacing(12)

        if setup_mode:
            intro = QLabel(
                "Enter the minimum settings to get familiar-ai running. "
                "You can fill the rest in later from Settings."
            )
            intro.setWordWrap(True)
            intro.setStyleSheet(
                f"color: {_TEXT_SECONDARY}; font-size: {_px(12)}px; background: transparent;"
            )
            vbox.addWidget(intro)

        tabs = QTabWidget()
        vbox.addWidget(tabs)

        for section in sections_for_mode(setup_mode=setup_mode):
            tab = QWidget()
            tab.setStyleSheet("background: transparent;")
            form = QFormLayout(tab)
            self._style_form(form)
            for field in iter_setting_fields(section=section, setup_mode=setup_mode):
                widget = self._create_field_widget(field)
                self._field_widgets[field.attr] = widget
                form.addRow(self._form_label(field.label), widget)
            tabs.addTab(tab, _t(SECTION_LABELS[section]))

        # Buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._save)
        btn_box.rejected.connect(self.reject)
        vbox.addWidget(btn_box)

    def _save(self) -> None:
        config = self._build_setup_config()
        errors = validate_setup_config(config, setup_mode=self._setup_mode)
        if errors:
            QMessageBox.warning(self, "Setup incomplete", "\n".join(errors))
            return
        try:
            save_setup_config(
                config,
                path=self._env_path,
                preserve_empty_sensitive=True,
            )
        except Exception as exc:
            QMessageBox.warning(self, _t("settings_save_failed_title"), str(exc))
            return

        QMessageBox.information(
            self,
            _t("settings_saved_title"),
            (
                "Configuration saved. familiar-ai will use it on the next launch."
                if self._setup_mode
                else _t("settings_saved_message")
            ),
        )
        self.accept()

    @staticmethod
    def _style_form(form: QFormLayout) -> None:
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(11)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

    def _form_label(self, key_or_text: str) -> QLabel:
        try:
            text = _t(key_or_text)
        except KeyError:
            text = key_or_text
        label = QLabel(text)
        label.setMinimumWidth(140)
        label.setStyleSheet(
            f"color: {_TEXT_PRIMARY}; font-size: {_px(13)}px; font-weight: 600;"
            f"padding-right: 6px; background: transparent;"
        )
        return label

    def _create_field_widget(self, field: SettingField) -> QWidget:
        value = getattr(self._setup, field.attr)
        if field.widget == "combo":
            combo = QComboBox()
            combo.addItems(list(field.options))
            _set_combo(combo, str(value))
            return combo
        if field.widget == "bool":
            combo = QComboBox()
            combo.addItems(["false", "true"])
            _set_combo(combo, "true" if bool(value) else "false")
            return combo
        line = QLineEdit(str(value))
        if field.widget == "password":
            line.setEchoMode(QLineEdit.EchoMode.Password)
        if field.placeholder_key:
            line.setPlaceholderText(_t(field.placeholder_key))
        return line

    def _read_field_value(self, field: SettingField) -> str | bool:
        widget = self._field_widgets[field.attr]
        if isinstance(widget, QComboBox):
            current = widget.currentText()
            if field.widget == "bool":
                return current == "true"
            return current
        if isinstance(widget, QLineEdit):
            return widget.text()
        raise TypeError(f"Unsupported widget for {field.attr}")

    def _build_setup_config(self) -> SetupConfig:
        values: dict[str, str | bool] = {}
        for field in iter_setting_fields(setup_mode=self._setup_mode):
            values[field.attr] = self._read_field_value(field)
        for field in iter_setting_fields(setup_mode=False):
            values.setdefault(field.attr, cast(str | bool, getattr(self._setup, field.attr)))
        return SetupConfig(**cast(dict[str, Any], values))


def _set_combo(combo: QComboBox, value: str) -> None:
    """Select a combo box item by value, ignoring if not found."""
    idx = combo.findText(value)
    if idx >= 0:
        combo.setCurrentIndex(idx)


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------


class FamiliarWindow(QMainWindow):
    """Main application window."""

    def __init__(
        self, config: "AgentConfig", desires: "DesireSystem", *, shared_camera: "Any | None" = None
    ) -> None:
        super().__init__()
        self._config = config
        self._shared_camera = shared_camera
        self._agent: EmbodiedAgent | None = None
        self._desires = desires
        self._drive_executor: DriveActionExecutor | None = None
        self._agent_display_name = (config.agent_name or "Agent").strip() or "Agent"
        _init_registry = UserRegistry()
        active_user = _init_registry.get_active()
        self._companion_display_name = active_user.name
        self._current_user_id = active_user.id
        self._input_queue: asyncio.Queue[str | UserTurn | None] = asyncio.Queue()
        self._pending_images: list[ImageAttachment] = []
        self._agent_running = False
        self._agent_ready = False
        self._agent_init_failed = False
        self._startup_status = "Starting familiar-ai..."
        self._last_error = ""
        self._closing = False
        self._shutdown_requested = False
        self._shutdown_done = False
        self._shutdown_task: asyncio.Task[None] | None = None
        self._cancel_requested = False
        self._agent_task: asyncio.Task[str] | None = None
        self._queue_task: asyncio.Task[None] | None = None
        self._init_task: asyncio.Task[None] | None = None
        self._startup_status_task: asyncio.Task[None] | None = None
        self._look_preview_task: asyncio.Task[None] | None = None
        self._look_preview_until: float = 0.0
        self._look_preview_disabled = False
        self._realtime_stt: RealtimeSttController | None = create_realtime_stt_controller()
        self._realtime_stt_task: asyncio.Task[None] | None = None
        self._batch_stt_recording = False
        self._batch_stt_stop: asyncio.Event = asyncio.Event()
        self._batch_stt_task: asyncio.Task[None] | None = None
        self._last_lag_tick = time.perf_counter()
        self._lag_timer = QTimer(self)
        self._lag_timer.setInterval(int(_GUI_LOOP_LAG_CHECK_SEC * 1000))
        self._lag_timer.timeout.connect(self._report_event_loop_lag)
        self._lag_timer.start()
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(350)
        self._status_timer.timeout.connect(self._refresh_status_card)
        self._status_timer.start()
        self._log_path = self._open_log_file()

        self.setWindowTitle("familiar-ai")
        self.resize(1280, 860)
        self.setStyleSheet(f"background: {_BG_BASE};")
        self._build_ui()
        self._set_input_enabled(False)
        self._stream.set_status(self._startup_status)
        self._refresh_status_card()

        self._queue_task = self._create_task(self._process_queue())
        self._init_task = self._create_task(self._initialize_agent())

    def _open_log_file(self) -> Path:
        log_dir = Path.home() / ".cache" / "familiar-ai"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "chat.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n{'─' * 60}\n[{datetime.now():%Y-%m-%d %H:%M:%S}] セッション開始\n")
        return log_path

    def _append_log(self, line: str) -> None:
        plain = re.sub(r"\[/?[^\[\]]*\]", "", line)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(plain + "\n")

    def _create_task(self, coro) -> asyncio.Task[Any]:
        """Create an asyncio task from GUI sync callbacks safely.

        `asyncio.create_task()` requires a running loop and fails during
        early window construction on some platforms (notably Windows).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        return loop.create_task(coro)

    @staticmethod
    def _build_rtsp_url(host: str, username: str, password: str) -> str | None:
        """Build RTSP URL from camera config, preserving explicit URI hosts."""
        host = (host or "").strip()
        if not host:
            return None
        if "://" in host:
            return host
        user = quote((username or "").strip(), safe="")
        pw = quote((password or "").strip(), safe="")
        auth = ""
        if user and pw:
            auth = f"{user}:{pw}@"
        elif user:
            auth = f"{user}@"
        return f"rtsp://{auth}{host}:554/stream1"

    def _camera_rtsp_url(self) -> str | None:
        config = getattr(self, "_config", None) or getattr(
            getattr(self, "_agent", None), "config", None
        )
        if config is None:
            return None
        cam = config.camera
        return self._build_rtsp_url(cam.host, cam.username, cam.password)

    @staticmethod
    def _look_preview_seconds_for_degrees(degrees: int | None) -> float:
        """Map look degrees to preview duration (short look -> shorter preview)."""
        raw = 30 if degrees is None else degrees
        clamped = max(1, min(90, int(raw)))
        sec = clamped / 45.0
        return max(_GUI_LOOK_PREVIEW_MIN_SEC, min(_GUI_LOOK_PREVIEW_MAX_SEC, sec))

    @staticmethod
    def _extract_jpeg_frames(buffer: bytearray, max_frames: int = 2) -> list[bytes]:
        """Extract complete JPEG frames from a byte buffer and consume them."""
        frames: list[bytes] = []
        while len(frames) < max_frames:
            start = buffer.find(b"\xff\xd8")
            if start < 0:
                if len(buffer) > 1_000_000:
                    buffer.clear()
                break
            if start > 0:
                del buffer[:start]
            end = buffer.find(b"\xff\xd9", 2)
            if end < 0:
                break
            frame = bytes(buffer[: end + 2])
            del buffer[: end + 2]
            frames.append(frame)
        return frames

    def _request_look_preview(self, degrees: int | None = None) -> None:
        """Start/extend a short live preview window for look() actions."""
        if self._look_preview_disabled:
            return
        stream_url = self._camera_rtsp_url()
        if not stream_url:
            return
        duration = self._look_preview_seconds_for_degrees(degrees) + _GUI_LOOK_PREVIEW_GRACE_SEC
        self._look_preview_until = max(self._look_preview_until, time.perf_counter() + duration)
        if self._look_preview_task and not self._look_preview_task.done():
            return
        self._look_preview_task = self._create_task(self._run_look_preview(stream_url))

    async def _run_look_preview(self, stream_url: str) -> None:
        """Render low-FPS RTSP frames while look() is in progress.

        Uses cv2.VideoCapture (bundled in the opencv-python wheel) instead of
        shelling out to ffmpeg — no system ffmpeg required.
        """
        try:
            import cv2
        except ImportError:
            logger.warning("Live look preview disabled: opencv-python not installed")
            self._look_preview_disabled = True
            return

        loop = asyncio.get_event_loop()
        frame_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=2)
        min_interval = 1.0 / max(1, _GUI_LOOK_PREVIEW_FPS)

        def _capture() -> None:
            cap = cv2.VideoCapture(stream_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                loop.call_soon_threadsafe(frame_queue.put_nowait, None)
                return
            try:
                while not self._closing and time.perf_counter() < self._look_preview_until:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (640, 360))
                    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    try:
                        loop.call_soon_threadsafe(frame_queue.put_nowait, jpg.tobytes())
                    except Exception:
                        pass
                    time.sleep(min_interval)
            finally:
                cap.release()
                loop.call_soon_threadsafe(frame_queue.put_nowait, None)

        capture_future = loop.run_in_executor(None, _capture)
        last_emit = 0.0

        try:
            while not self._closing and time.perf_counter() < self._look_preview_until:
                try:
                    jpg_bytes = await asyncio.wait_for(
                        frame_queue.get(), timeout=_GUI_LOOK_PREVIEW_READ_TIMEOUT_SEC
                    )
                except asyncio.TimeoutError:
                    continue
                if jpg_bytes is None:
                    break
                now = time.perf_counter()
                if now - last_emit < min_interval:
                    continue
                self._camera.update_image(base64.b64encode(jpg_bytes).decode("ascii"))
                last_emit = now
        finally:
            with contextlib.suppress(Exception):
                await capture_future
            self._look_preview_task = None

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        central.setStyleSheet(f"background: {_BG_BASE};")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Left panel ──────────────────────────────────────────
        left = QWidget()
        left.setStyleSheet(f"background: {_BG_BASE};")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Header bar
        header = QWidget()
        header.setStyleSheet(
            f"background: {_BG_SURFACE}; border-radius: {_px(16)}px; border: 1px solid {_BORDER};"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 6, 8, 6)
        header_layout.setSpacing(8)

        title_lbl = QLabel("✦ familiar-ai")
        title_lbl.setStyleSheet(
            f"color: {_ACCENT_DEEP}; font-size: {_px(15)}px; font-weight: 700; background: transparent;"
            f" letter-spacing: -0.02em;"
        )
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()

        # User switcher combo
        self._user_combo = QComboBox()
        self._user_combo.setFixedHeight(_px(30))
        self._user_combo.setMinimumWidth(_px(120))
        self._user_combo.setStyleSheet(
            f"QComboBox {{ background: {_BG_ELEVATED}; border-radius: {_px(8)}px;"
            f" border: 1px solid {_BORDER};"
            f" padding: 0 {_px(8)}px; font-size: {_px(12)}px; color: {_TEXT_SECONDARY}; }}"
            f"QComboBox:hover {{ background: {_BG_HOVER}; color: {_TEXT_PRIMARY}; }}"
            f"QComboBox::drop-down {{ border: none; }}"
            f"QComboBox QAbstractItemView {{ background: {_BG_ELEVATED}; color: {_TEXT_PRIMARY};"
            f" selection-background-color: {_BG_HOVER}; }}"
        )
        self._user_combo.setEnabled(False)  # enabled once agent is ready
        self._refresh_user_combo()
        self._user_combo.currentIndexChanged.connect(self._on_user_combo_changed)
        header_layout.addWidget(self._user_combo)

        settings_btn = QPushButton(_t("settings_button"))
        settings_btn.setToolTip(_t("settings_button_tooltip"))
        settings_btn.setFixedHeight(_px(30))
        settings_btn.setMinimumWidth(_px(110))
        settings_btn.setStyleSheet(
            f"QPushButton {{ background: {_BG_SURFACE}; border-radius: {_px(8)}px;"
            f" border: 1px solid {_BORDER};"
            f" padding: 0 {_px(12)}px; font-size: {_px(12)}px; color: {_TEXT_SECONDARY}; }}"
            f"QPushButton:hover {{ background: {_BG_ELEVATED}; color: {_TEXT_PRIMARY}; }}"
        )
        settings_btn.clicked.connect(self._open_settings)
        header_layout.addWidget(settings_btn)

        self._restart_stt_btn = QPushButton("↻ STT")
        self._restart_stt_btn.setToolTip("Restart realtime STT")
        self._restart_stt_btn.setFixedHeight(_px(30))
        self._restart_stt_btn.setMinimumWidth(_px(90))
        self._restart_stt_btn.setStyleSheet(
            f"QPushButton {{ background: {_BG_ELEVATED}; border-radius: {_px(8)}px;"
            f" border: 1px solid {_BORDER};"
            f" padding: 0 {_px(12)}px; font-size: {_px(12)}px; color: {_TEXT_SECONDARY}; }}"
            f"QPushButton:hover {{ background: {_BG_HOVER}; color: {_TEXT_PRIMARY}; }}"
            f"QPushButton:disabled {{ background: {_DISABLED_BG}; color: {_TEXT_SECONDARY}; }}"
        )
        self._restart_stt_btn.setEnabled(self._realtime_stt is not None)
        self._restart_stt_btn.clicked.connect(self._on_restart_stt_clicked)
        header_layout.addWidget(self._restart_stt_btn)

        self._mic_btn = QPushButton("🎙 Mic")
        self._mic_btn.setToolTip("Start / stop batch voice recording (STT)")
        self._mic_btn.setFixedHeight(_px(30))
        self._mic_btn.setMinimumWidth(_px(90))
        self._mic_btn.setStyleSheet(
            f"QPushButton {{ background: {_BG_ELEVATED}; border-radius: {_px(8)}px;"
            f" border: 1px solid {_BORDER};"
            f" padding: 0 {_px(12)}px; font-size: {_px(12)}px; color: {_TEXT_SECONDARY}; }}"
            f"QPushButton:hover {{ background: {_BG_HOVER}; color: {_TEXT_PRIMARY}; }}"
            f"QPushButton:checked {{ background: {_ACCENT_MIC}; color: white; border-color: {_ACCENT_MIC}; }}"
            f"QPushButton:disabled {{ background: {_DISABLED_BG}; color: {_TEXT_SECONDARY}; }}"
        )
        self._mic_btn.setCheckable(True)
        self._mic_btn.setEnabled(False)  # enabled after agent init if stt is configured
        self._mic_btn.clicked.connect(self._on_mic_btn_clicked)
        header_layout.addWidget(self._mic_btn)
        left_layout.addWidget(header)

        status_card = QWidget()
        status_card.setStyleSheet(
            f"background: {_BG_SURFACE}; border-radius: {_px(16)}px; border: 1px solid {_BORDER};"
        )
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(12, 8, 12, 8)
        status_layout.setSpacing(8)

        self._status_headline = QLabel("Starting up")
        self._status_headline.setStyleSheet(
            f"color: {_TEXT_PRIMARY}; font-size: {_px(13)}px; font-weight: 700; background: transparent;"
        )
        status_layout.addWidget(self._status_headline)

        self._status_detail = QLabel(self._startup_status)
        self._status_detail.setWordWrap(True)
        self._status_detail.setStyleSheet(
            f"color: {_TEXT_SECONDARY}; font-size: {_px(11)}px; background: transparent;"
        )
        status_layout.addWidget(self._status_detail)

        self._status_readiness = QLabel("")
        self._status_readiness.setWordWrap(True)
        self._status_readiness.setStyleSheet(
            f"color: {_TEXT_SECONDARY}; font-size: {_px(10)}px; background: transparent;"
            f"font-family: {_MONO_FONT_STACK};"
        )
        status_layout.addWidget(self._status_readiness)

        # Consciousness profile line (empty unless FAMILIAR_CONSCIOUSNESS_PROFILE).
        self._status_consciousness = QLabel("")
        self._status_consciousness.setWordWrap(True)
        self._status_consciousness.setStyleSheet(
            f"color: {_TEXT_SECONDARY}; font-size: {_px(10)}px; background: transparent;"
            f"font-family: {_MONO_FONT_STACK};"
        )
        status_layout.addWidget(self._status_consciousness)
        left_layout.addWidget(status_card)

        # Chat log
        self._log = ChatLog(
            agent_label=self._agent_display_name,
            companion_label=self._companion_display_name,
        )
        left_layout.addWidget(self._log, stretch=5)

        # Stream label
        self._stream = StreamLabel()
        self._stream.setMinimumHeight(_px(70))
        left_layout.addWidget(self._stream, stretch=1)

        # Pending image attachments
        self._attachment_bar = QWidget()
        self._attachment_layout = QHBoxLayout(self._attachment_bar)
        self._attachment_layout.setContentsMargins(4, 2, 4, 2)
        self._attachment_layout.setSpacing(6)
        self._attachment_bar.hide()
        left_layout.addWidget(self._attachment_bar)

        # Input row — attach button + pill QLineEdit + circular send button
        input_row = QHBoxLayout()
        input_row.setSpacing(8)

        self._attach_btn = QPushButton("＋")
        self._attach_btn.setFixedSize(_px(40), _px(40))
        self._attach_btn.setToolTip("Attach images")
        self._attach_btn.setStyleSheet(
            f"QPushButton {{ background: {_BG_ELEVATED}; border: 1px solid {_BORDER};"
            f" border-radius: {_px(20)}px; color: {_TEXT_PRIMARY}; font-size: {_px(18)}px; }}"
            f"QPushButton:hover {{ background: {_BG_HOVER}; }}"
        )
        self._attach_btn.clicked.connect(self._on_attach_images)
        input_row.addWidget(self._attach_btn)

        self._input = QLineEdit()
        self._input.setPlaceholderText(_t("gui_input_placeholder"))
        self._input.setObjectName("msgInput")
        self._input.setStyleSheet(
            f"QLineEdit#msgInput {{"
            f" border-radius: 999px; padding: {_px(6)}px {_px(14)}px;"
            f" background: {_BG_SURFACE}; border: 1px solid {_BORDER};"
            f" color: {_TEXT_PRIMARY}; font-size: {_px(13)}px;"
            f" font-family: {_UI_FONT_STACK};"
            f"}}"
            f"QLineEdit#msgInput:focus {{"
            f" border-color: {_ACCENT}; background: {_BG_SURFACE};"
            f"}}"
        )
        self._input.returnPressed.connect(self._on_send)
        input_row.addWidget(self._input)

        self._send_btn = QPushButton("⬆")
        self._send_btn.setFixedSize(_px(40), _px(40))
        self._send_btn.setObjectName("sendBtn")
        self._send_btn.setStyleSheet(
            f"QPushButton#sendBtn {{"
            f" background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            f" stop:0 {_ACCENT_DEEP}, stop:1 {_ACCENT});"
            f" border-radius: {_px(20)}px; border: none;"
            f" font-size: {_px(14)}px; color: white;"
            f"}}"
            f"QPushButton#sendBtn:hover {{"
            f" background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            f" stop:0 {_SEND_HOVER_STOP0}, stop:1 {_SEND_HOVER_STOP1});"
            f"}}"
            f"QPushButton#sendBtn:disabled {{"
            f" background: {_DISABLED_BG}; color: {_TEXT_SECONDARY};"
            f"}}"
        )
        self._send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self._send_btn)

        self._stop_btn = QPushButton("■")
        self._stop_btn.setFixedSize(_px(40), _px(40))
        self._stop_btn.setObjectName("stopBtn")
        self._stop_btn.setToolTip(_t("gui_cancel_turn_tooltip"))
        self._stop_btn.setStyleSheet(
            f"QPushButton#stopBtn {{"
            f" background: {_STOP_BG};"
            f" border-radius: {_px(20)}px; border: 1px solid {_STOP_BORDER};"
            f" font-size: {_px(12)}px; color: {_STOP_TEXT};"
            f"}}"
            f"QPushButton#stopBtn:hover {{"
            f" background: {_STOP_HOVER_BG}; color: {_STOP_HOVER_TEXT};"
            f"}}"
            f"QPushButton#stopBtn:disabled {{"
            f" background: {_DISABLED_BG}; color: {_TEXT_SECONDARY};"
            f" border-color: {_DISABLED_BORDER};"
            f"}}"
        )
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_cancel_clicked)
        input_row.addWidget(self._stop_btn)
        left_layout.addLayout(input_row)

        # ── Right panel ─────────────────────────────────────────
        right = QWidget()
        right.setFixedWidth(640)
        right.setStyleSheet(f"background: {_BG_BASE};")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self._camera = CameraView()
        right_layout.addWidget(self._camera, stretch=3)

        desire_card = QWidget()
        desire_card.setStyleSheet(
            f"background: {_BG_ELEVATED}; border-radius: {_px(18)}px; border: 1px solid {_BORDER};"
        )
        desire_card_vbox = QVBoxLayout(desire_card)
        desire_card_vbox.setContentsMargins(0, 0, 0, 0)
        self._desire_panel = DesirePanel(self._desires)
        self._desire_panel.setStyleSheet("background: transparent;")
        desire_card_vbox.addWidget(self._desire_panel)
        right_layout.addWidget(desire_card, stretch=2)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setStyleSheet("QSplitter::handle { background: transparent; width: 2px; }")
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    def _set_input_enabled(self, enabled: bool) -> None:
        self._input.setEnabled(enabled)
        self._send_btn.setEnabled(enabled)
        if hasattr(self, "_attach_btn"):
            self._attach_btn.setEnabled(enabled)
        if hasattr(self, "_user_combo"):
            self._user_combo.setEnabled(enabled)

    def _refresh_user_combo(self) -> None:
        self._user_combo.blockSignals(True)
        self._user_combo.clear()
        users = self._agent.list_users() if self._agent else UserRegistry().list_users()
        for user in users:
            self._user_combo.addItem(f"👤 {user.name}", userData=user.id)
        active_id = self._current_user_id
        idx = next(
            (
                i
                for i in range(self._user_combo.count())
                if self._user_combo.itemData(i) == active_id
            ),
            0,
        )
        self._user_combo.setCurrentIndex(idx)
        self._user_combo.blockSignals(False)

    def _on_user_combo_changed(self, index: int) -> None:
        user_id = self._user_combo.itemData(index)
        if user_id and user_id != self._current_user_id:
            self._create_task(self._switch_user_async(user_id))

    async def _switch_user_async(self, user_id: str) -> None:
        if self._agent is None:
            return
        name = await self._agent.switch_user(user_id)
        self._current_user_id = user_id
        self._companion_display_name = name
        self._log.set_companion_label(name)
        self._log.append_line(f"[システム] ── {name} に切り替わりました ──")

    def _set_startup_status(self, text: str) -> None:
        self._startup_status = text
        if not self._agent_running and not self._stream.has_content():
            self._stream.set_status(text)
        self.setWindowTitle(f"familiar-ai  ⏳ {text}")
        self._refresh_status_card()

    def _set_last_error(self, message: str | None) -> None:
        self._last_error = (message or "").strip()
        self._refresh_status_card()

    def _refresh_status_card(self) -> None:
        snapshot = build_gui_diagnostics(self)
        headline = getattr(self, "_status_headline", None)
        detail = getattr(self, "_status_detail", None)
        readiness = getattr(self, "_status_readiness", None)
        if headline is not None:
            headline.setText(snapshot.headline)
        if detail is not None:
            detail.setText(snapshot.detail)
        if readiness is not None:
            readiness.setText(
                f"{snapshot.readiness}\nqueue={snapshot.queue_backlog} | stt_connected={snapshot.realtime_stt_connected}"
            )
        consciousness = getattr(self, "_status_consciousness", None)
        if consciousness is not None:
            consciousness.setText(snapshot.consciousness_line)

    def _copy_diagnostics(self) -> None:
        snapshot = build_gui_diagnostics(self)
        QApplication.clipboard().setText(format_gui_diagnostics(snapshot))
        self._log.append_line("📋 Diagnostics copied")

    async def _run_backend_test(self) -> None:
        config = getattr(self, "_config", None) or getattr(
            getattr(self, "_agent", None), "config", None
        )
        if config is None:
            self._log.append_line("[error] Backend test unavailable before configuration")
            return
        self._log.append_line("🧪 Testing backend connection...")
        ok, message = await test_backend_connection(config)
        if ok:
            self._set_last_error(None)
            self._log.append_line(f"✅ Backend OK: {message}")
        else:
            self._set_last_error(f"Backend test failed: {message}")
            self._log.append_line(f"[error] Backend test failed: {message}")

    async def _run_camera_test(self) -> None:
        config = getattr(self, "_config", None) or getattr(
            getattr(self, "_agent", None), "config", None
        )
        if config is None:
            self._log.append_line("[error] Camera test unavailable before configuration")
            return
        self._log.append_line("🧪 Testing camera connection...")
        ok, message = await test_camera_connection_from_config(config)
        if ok:
            self._set_last_error(None)
            self._log.append_line("✅ Camera OK")
        else:
            self._set_last_error(f"Camera test failed: {message}")
            self._log.append_line(f"[error] Camera test failed: {message}")

    async def _run_stt_test(self) -> None:
        config = getattr(self, "_config", None) or getattr(
            getattr(self, "_agent", None), "config", None
        )
        if config is None:
            self._log.append_line("[error] Realtime STT test unavailable before configuration")
            return
        self._log.append_line("🧪 Testing realtime STT connection...")
        ok, message = await test_realtime_stt_connection_from_config(config)
        if ok:
            self._set_last_error(None)
            self._log.append_line(f"✅ Realtime STT OK: {message}")
        else:
            self._set_last_error(f"Realtime STT test failed: {message}")
            self._log.append_line(f"[error] Realtime STT test failed: {message}")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _open_settings(self) -> None:
        config = getattr(self, "_config", None) or getattr(
            getattr(self, "_agent", None), "config", None
        )
        if config is None:
            return
        dlg = SettingsDialog(config, _ENV_PATH, parent=self)
        dlg.exec()

    def _on_restart_stt_clicked(self) -> None:
        self._create_task(self._restart_realtime_stt(reason="manual"))

    def _on_send(self) -> None:
        if getattr(self, "_agent", None) is None and not getattr(self, "_agent_ready", True):
            self._stream.set_status(getattr(self, "_startup_status", "Initializing familiar-ai..."))
            return
        text = self._input.text().strip()
        pending_images = tuple(getattr(self, "_pending_images", ()))
        if not text and not pending_images:
            return
        if not text:
            text = "この画像を見て。"
        self._input.clear()
        user_input = coerce_user_turn(
            UserTurn(text=text, images=pending_images) if pending_images else text
        )
        if pending_images:
            self._pending_images.clear()
            self._refresh_attachment_bar()
        self._stream.clear_status()
        image_note = f" 🖼×{len(pending_images)}" if pending_images else ""
        self._log.append_line(f"[{self._companion_display_name}] {text}{image_note}")
        self._append_log(f"{self._companion_display_name} ▶ {text}{image_note}")
        self._input_queue.put_nowait(user_input)
        qsize = self._input_queue.qsize()
        if qsize >= _GUI_QUEUE_WARN_SIZE:
            logger.warning("GUI input queue backlog: %d", qsize)
        else:
            logger.debug("GUI input queued (size=%d)", qsize)

    def _on_attach_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Attach images",
            "",
            "Images (*.jpg *.jpeg *.png *.webp)",
        )
        if not paths:
            return
        try:
            validate_image_count(len(self._pending_images) + len(paths))
            attachments = [load_image_attachment(path) for path in paths]
        except ImageInputError as exc:
            QMessageBox.warning(self, "Image attachment", str(exc))
            return
        self._pending_images.extend(attachments)
        self._refresh_attachment_bar()

    def _remove_pending_image(self, index: int) -> None:
        if 0 <= index < len(self._pending_images):
            self._pending_images.pop(index)
            self._refresh_attachment_bar()

    def _refresh_attachment_bar(self) -> None:
        layout = getattr(self, "_attachment_layout", None)
        bar = getattr(self, "_attachment_bar", None)
        if layout is None or bar is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for index, attachment in enumerate(self._pending_images):
            chip = QWidget()
            chip.setStyleSheet(
                f"background: {_BG_ELEVATED}; border: 1px solid {_BORDER};"
                f" border-radius: {_px(8)}px;"
            )
            chip_layout = QHBoxLayout(chip)
            chip_layout.setContentsMargins(4, 4, 4, 4)
            preview = QLabel()
            pixmap = QPixmap.fromImage(QImage.fromData(attachment.data))
            preview.setPixmap(
                pixmap.scaled(
                    _px(48),
                    _px(48),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            chip_layout.addWidget(preview)
            filename = QLabel(attachment.filename or f"image-{index + 1}")
            filename.setStyleSheet(f"color: {_TEXT_PRIMARY}; border: none;")
            chip_layout.addWidget(filename)
            remove = QPushButton("×")
            remove.setFixedSize(_px(24), _px(24))
            remove.setStyleSheet("border: none; background: transparent;")
            remove.clicked.connect(
                lambda _checked=False, image_index=index: self._remove_pending_image(image_index)
            )
            chip_layout.addWidget(remove)
            layout.addWidget(chip)
        layout.addStretch(1)
        bar.setVisible(bool(self._pending_images))

    def _on_realtime_stt_partial(self, text: str) -> None:
        """Display partial STT transcript while idle."""
        if self._closing:
            return
        partial = text.strip()
        if not partial:
            return
        if self._agent_running or self._stream.has_content():
            return
        self._stream.set_status(f"🎤 {partial}")

    def _on_realtime_stt_committed(self, text: str) -> None:
        """Display committed STT transcript as a user message bubble."""
        if self._closing:
            return
        spoken = text.strip()
        if not spoken:
            return
        self._stream.clear_status()
        self._log.append_line(f"[{self._companion_display_name}] {spoken}")
        self._append_log(f"{self._companion_display_name} 🎤 {spoken}")

    def _on_realtime_stt_restart(self, reason: str) -> None:
        if self._closing:
            return
        label = "watchdog" if reason == "watchdog_loop" else reason
        self._log.append_line(f"🎤 Realtime STT restarting ({label})")

    async def _start_realtime_stt(self) -> None:
        """Initialize realtime STT and feed transcripts into the GUI input queue."""
        assert self._realtime_stt is not None
        try:
            loop = asyncio.get_event_loop()
            self._realtime_stt.on_partial = self._on_realtime_stt_partial
            self._realtime_stt.on_committed = self._on_realtime_stt_committed
            self._realtime_stt.on_restart = self._on_realtime_stt_restart
            await self._realtime_stt.start(loop, self._input_queue)
            self._set_last_error(None)
            self._log.append_line("🎤 Realtime STT ON (ElevenLabs)")
        except Exception as exc:
            logger.warning("Realtime STT init failed: %s", exc)
            self._set_last_error(f"Realtime STT init failed: {exc}")
            self._log.append_line(f"[error] Realtime STT init failed: {exc}")
            self._realtime_stt = None
            restart_btn = getattr(self, "_restart_stt_btn", None)
            if restart_btn is not None:
                restart_btn.setEnabled(False)

    async def _restart_realtime_stt(self, reason: str = "manual") -> None:
        controller = self._realtime_stt
        if controller is None:
            self._log.append_line("[error] Realtime STT is not configured")
            return
        try:
            restarted = await controller.restart(reason=reason)
        except Exception as exc:
            logger.warning("Realtime STT restart failed: %s", exc)
            self._set_last_error(f"Realtime STT restart failed: {exc}")
            self._log.append_line(f"[error] Realtime STT restart failed: {exc}")
            return
        if restarted:
            self._set_last_error(None)
            self._log.append_line("🎤 Realtime STT restarted")
        else:
            self._log.append_line("[error] Realtime STT restart unavailable before startup")

    def _on_mic_btn_clicked(self) -> None:
        if not self._batch_stt_recording:
            self._batch_stt_recording = True
            self._batch_stt_stop.clear()
            self._mic_btn.setText("⏹ Stop")
            self._batch_stt_task = self._create_task(self._do_batch_stt())
        else:
            self._batch_stt_stop.set()

    async def _do_batch_stt(self) -> None:
        agent = getattr(self, "_agent", None)
        if agent is None or agent.stt is None:
            self._log.append_line("[error] Batch STT not configured")
            self._batch_stt_recording = False
            self._mic_btn.setText("🎙 Mic")
            self._mic_btn.setChecked(False)
            return
        try:
            self._stream.set_status("🎙 Recording… (click ⏹ Stop to finish)")
            record_task = asyncio.create_task(agent.stt.record_and_transcribe(self._batch_stt_stop))
            await self._batch_stt_stop.wait()
            self._stream.set_status("🔄 Transcribing…")
            text = await record_task
            if text:
                self._log.append_line(f"{self._companion_display_name} 🎙 {text}")
                self._append_log(f"{self._companion_display_name} 🎙 {text}")
                self._input_queue.put_nowait(coerce_user_turn(text))
        except Exception as exc:
            logger.warning("Batch STT error: %s", exc)
            self._log.append_line(f"[error] STT: {exc}")
        finally:
            self._batch_stt_recording = False
            self._mic_btn.setText("🎙 Mic")
            self._mic_btn.setChecked(False)
            self._stream.clear_status()

    def _on_cancel_clicked(self) -> None:
        self._cancel_turn(reason="user")

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Escape:
            self._cancel_turn(reason="user")
            event.accept()
            return
        super().keyPressEvent(event)

    def _cancel_turn(self, reason: str = "user") -> None:
        """Cancel current turn if running.

        reason: "user" (explicit cancel) or "shutdown" (window/app exit).
        """
        if not self._agent_running:
            return
        self._cancel_requested = True
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
        if reason == "user":
            self._log.append_line("[interrupted]")
        logger.info("GUI turn cancel requested (%s), queue=%d", reason, self._input_queue.qsize())

    def _set_turn_ui_state(self, running: bool) -> None:
        self._stop_btn.setEnabled(running)

    def _thinking_status_text(self, elapsed_sec: int) -> str:
        """Status line shown while waiting for the first response chunk."""
        return _t(
            "thinking_status",
            name=self._agent_display_name,
            seconds=str(elapsed_sec),
        )

    @staticmethod
    def _startup_status_text(elapsed_sec: int) -> str:
        """Status line shown while first-turn startup work is in progress."""
        return f"{_t('initializing')}... ({elapsed_sec}s)"

    # ------------------------------------------------------------------
    # Agent loop
    # ------------------------------------------------------------------

    async def _process_queue(self) -> None:
        """Dequeue user messages and run the agent; fire desires when idle."""
        from .wake import WakeListener, wait_input_or_wake

        window_config = getattr(self, "_config", None)
        wake_listener = WakeListener(
            getattr(window_config, "daemon_socket", "") or None,
            enabled=getattr(window_config, "daemon", False) is True,
        )
        last_interaction = time.time()
        while True:
            try:
                kind, text = await wait_input_or_wake(
                    self._input_queue, wake_listener, IDLE_CHECK_INTERVAL
                )
                if kind != "input":
                    # A familiard wake takes the same path as the poll timeout:
                    # it is only an early poll — every idle gate re-checks below.
                    raise asyncio.TimeoutError
            except asyncio.TimeoutError:
                now = time.time()
                if self._closing:
                    continue
                if not getattr(self, "_agent_ready", True):
                    continue
                agent_obj = getattr(self, "_agent", None)
                agent_config = getattr(agent_obj, "config", None)
                heartbeat = getattr(agent_obj, "_heartbeat", None)
                quiet_now = heartbeat.routine_state().quiet_hours if heartbeat else False
                # Nightly consolidation: a background job, never a turn — it
                # does not participate in idle precedence.
                try:
                    if (
                        agent_obj is not None
                        and bool(getattr(agent_config, "sleep_consolidation", False))
                        and should_run_sleep_consolidation(
                            enabled=True,
                            agent_running=self._agent_running,
                            has_pending_input=not self._input_queue.empty(),
                            quiet_hours=quiet_now,
                            now_dt=datetime.now(),
                            last_night_key=agent_obj.last_consolidation_night_key(),
                            # Must match the job's configured end hour.
                            quiet_end_hour=agent_obj.consolidation_quiet_end_hour(),
                        )
                    ):
                        agent_obj.start_sleep_consolidation()
                except Exception:
                    logger.exception("sleep consolidation gate failed; skipping")
                # Proactive commitment reminders fire independently of auto_desire.
                store = getattr(agent_obj, "_commitment_store", None)
                if (
                    store is not None
                    and agent_config is not None
                    and getattr(agent_config, "proactive_reminders", True)
                ):
                    try:
                        quiet = quiet_now
                        reminders = should_fire_commitment_reminder(
                            agent_running=self._agent_running,
                            has_pending_input=not self._input_queue.empty(),
                            last_interaction=last_interaction,
                            now=now,
                            store=store,
                            quiet_hours=quiet,
                            routine_store=getattr(agent_obj, "_routine_store", None),
                        )
                        if reminders and self._input_queue.empty() and not self._agent_running:
                            # Record the fire BEFORE the turn: the cadence advances
                            # regardless of the turn's outcome, and a mid-turn
                            # snooze reset survives intact.
                            store.mark_reminded([c.id for c in reminders], at=time.time())
                        else:
                            reminders = []
                    except Exception:
                        logger.exception("reminder gate failed; skipping this tick")
                        reminders = []
                    if reminders:
                        await self._run_agent("", inner_voice=commitment_reminder_prompt(reminders))
                        last_interaction = time.time()
                        continue
                # Skip desire-driven turns when auto_desire is disabled (default OFF)
                if agent_config is not None and not getattr(agent_config, "auto_desire", False):
                    continue
                self._desires.set_schedule_multiplier(0.0 if quiet_now else 1.0)
                if not should_fire_idle_desire(
                    agent_running=self._agent_running,
                    has_pending_input=not self._input_queue.empty(),
                    quiet_hours=quiet_now,
                    last_interaction=last_interaction,
                    now=now,
                    cooldown=DESIRE_COOLDOWN,
                ):
                    continue

                dominant = self._desires.get_dominant()
                if dominant:
                    # If user input arrived meanwhile, prioritize that over autonomous desire.
                    if not self._input_queue.empty():
                        continue
                    desire_name, _ = dominant
                    executor = self._drive_executor
                    if executor is None:
                        logger.warning("desire tick skipped: drive executor is not ready")
                        continue

                    async def run_social_turn(inner_voice: str) -> None:
                        await self._run_agent("", inner_voice=inner_voice)

                    result = await executor.dispatch(
                        desire_name,
                        last_interaction_time=last_interaction,
                        run_social_turn=run_social_turn,
                        interrupt_queue=self._input_queue,
                    )
                    if result.fired:
                        self._desires.curiosity_target = None
                        last_interaction = time.time()
                continue

            if text is None:
                break
            while (
                not getattr(self, "_agent_ready", True)
                and getattr(self, "_agent", None) is None
                and not self._closing
                and not getattr(self, "_agent_init_failed", False)
            ):
                await asyncio.sleep(0.05)
            if not getattr(self, "_agent_ready", True) and getattr(self, "_agent", None) is None:
                break
            last_interaction = time.time()
            logger.debug(
                "GUI dequeued input (remaining queue=%d, running=%s)",
                self._input_queue.qsize(),
                self._agent_running,
            )
            await self._run_agent(text)

    async def _run_agent(self, user_input: str | UserTurn, inner_voice: str = "") -> None:
        if self._agent is None:
            self._stream.set_status(self._startup_status)
            return
        user_turn = coerce_user_turn(user_input)
        turn_started = time.perf_counter()
        self._agent_running = True
        self._cancel_requested = False
        self._set_turn_ui_state(True)
        self._refresh_status_card()
        logger.info(
            "GUI turn start (user_input=%d chars, inner_voice=%s, queue=%d)",
            len(user_turn.text),
            bool(inner_voice),
            self._input_queue.qsize(),
        )

        phase = (
            "startup"
            if (getattr(self._agent, "_turn_count", 0) == 0 or not self._agent.is_embedding_ready)
            else "thinking"
        )
        phase_started = time.perf_counter()
        thinking_timer = QTimer(self)
        thinking_timer.setInterval(200)

        def _update_thinking_status() -> None:
            nonlocal phase_started
            if self._stream.has_content():
                return
            elapsed = int(time.perf_counter() - phase_started)
            if phase == "startup":
                self._stream.set_status(self._startup_status_text(elapsed))
            else:
                self._stream.set_status(self._thinking_status_text(elapsed))

        thinking_timer.timeout.connect(_update_thinking_status)
        _update_thinking_status()
        thinking_timer.start()

        def on_phase(new_phase: str) -> None:
            nonlocal phase, phase_started
            if new_phase not in {"startup", "thinking"}:
                return
            if phase == new_phase:
                return
            phase = new_phase
            phase_started = time.perf_counter()
            if not self._stream.has_content():
                _update_thinking_status()
            self._refresh_status_card()

        say_fired = False

        def on_text(chunk: str) -> None:
            if say_fired:
                # Discard post-say text: LLMs often re-emit say() content as plain
                # text after the tool call (with audio tags intact). Suppressing it
                # prevents the raw tagged text from appearing below the clean version.
                return
            self._stream.clear_status()
            self._stream.append_chunk(chunk)

        def on_action(name: str, tool_input: dict) -> None:
            nonlocal say_fired
            if name == "say":
                # Discard pre-say text, then commit each say() immediately so
                # multiple calls all appear in order rather than overwriting.
                say_fired = True
                self._stream.commit_and_clear()
                raw = str(tool_input.get("text", ""))
                clean = re.sub(r"\[.*?\]", "", raw).strip()
                if clean:
                    self._log.append_line(f"[{self._agent_display_name}] {clean}")
                    self._append_log(f"{self._agent_display_name} 🔊 {clean}")
            else:
                self._log.append_action(name, tool_input)
            if name == "look":
                self._request_look_preview(tool_input.get("degrees"))

        def on_tool_result(name: str, tool_input: dict, result: str) -> None:
            formatted = format_tool_result(name, tool_input, result)
            if formatted:
                self._log.append_line(formatted)

        def on_image(b64: str) -> None:
            self._camera.update_image(b64)

        try:
            self._agent_task = self._create_task(
                self._agent.run(
                    user_input,
                    on_action=on_action,
                    on_text=on_text,
                    on_image=on_image,
                    on_phase=on_phase,
                    on_tool_result=on_tool_result,
                    desires=self._desires,
                    inner_voice=inner_voice,
                    interrupt_queue=self._input_queue,
                )
            )
            final_text = await self._agent_task
            committed = self._stream.commit_and_clear()
            # When say() was called, the spoken content is already in the log
            # (clean, tags stripped). Suppress final_text to avoid the LLM's
            # post-say text echo (same content with raw audio tags) appearing again.
            if not say_fired:
                display = committed.strip() or final_text.strip()
                if display and display != "(no response)":
                    self._log.append_line(f"[{self._agent_display_name}] {display}")
                    self._append_log(f"{self._agent_display_name} ▶ {display}")
        except asyncio.CancelledError:
            self._stream.commit_and_clear()
            if not self._cancel_requested:
                self._log.append_line("[interrupted]")
                self._append_log("[interrupted]")
        except Exception as exc:
            logger.exception("Agent run error")
            self._set_last_error(str(exc))
            self._log.append_line(f"[error] {exc}")
            self._append_log(f"[error] {exc}")
        finally:
            thinking_timer.stop()
            self._stream.clear_status()
            self._agent_task = None
            self._agent_running = False
            self._set_turn_ui_state(False)
            self._refresh_status_card()
            logger.info(
                "GUI turn end (duration=%.2fs, cancelled=%s, queue=%d)",
                time.perf_counter() - turn_started,
                self._cancel_requested,
                self._input_queue.qsize(),
            )
            self._cancel_requested = False

    async def _show_init_status(self) -> None:
        """Update status until the agent and embedding model are ready."""
        start = time.time()
        while not self._closing:
            elapsed = int(time.time() - start)
            agent = getattr(self, "_agent", None)
            if getattr(self, "_agent_init_failed", False):
                return
            if agent is None:
                self._set_startup_status(f"{_t('initializing')} agent... ({elapsed}s)")
            elif not agent.is_embedding_ready:
                self._set_startup_status(f"{_t('initializing')} memory... ({elapsed}s)")
            else:
                break
            await asyncio.sleep(0.5)
        if self._closing or getattr(self, "_agent", None) is None:
            return
        elapsed = int(time.time() - start)
        self._log.append_line(f"✅ {_t('initializing_done')} ({elapsed}s)")
        self._stream.clear_status()
        self.setWindowTitle("familiar-ai")

    async def _initialize_agent(self) -> None:
        """Build EmbodiedAgent after the window is already visible."""
        self._set_startup_status(f"{_t('initializing')} agent...")
        self._startup_status_task = self._create_task(self._show_init_status())
        try:
            config = self._config

            def _build_agent():
                from .agent import EmbodiedAgent  # noqa: PLC0415

                return EmbodiedAgent(
                    config, shared_camera=self._shared_camera, camera_gui_priority=True
                )

            agent = await asyncio.to_thread(_build_agent)
            agent.bind_desires(self._desires)
            # Start the MCP handshake now so tools are ready by the first turn (#188).
            start_mcp_early = getattr(agent, "start_mcp_early", None)
            if callable(start_mcp_early):
                start_mcp_early()
            self._agent = agent
            self._drive_executor = DriveActionExecutor(agent, self._desires)
            if not agent.is_embedding_ready:
                self._set_startup_status(f"{_t('initializing')} memory...")
            self._agent_ready = True
            self._set_last_error(None)
            self._set_input_enabled(True)
            if agent.stt is not None:
                mic_btn = getattr(self, "_mic_btn", None)
                if mic_btn is not None:
                    mic_btn.setEnabled(True)
            if self._realtime_stt and self._realtime_stt_task is None:
                self._realtime_stt_task = self._create_task(self._start_realtime_stt())
        except Exception as exc:
            self._agent_init_failed = True
            logger.exception("Agent initialization failed")
            self._set_startup_status("Initialization failed")
            self._set_last_error(f"Agent initialization failed: {exc}")
            self._log.append_line(f"[error] Agent initialization failed: {exc}")
            self._set_input_enabled(False)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._shutdown_done:
            event.accept()
            return
        if self._shutdown_requested:
            event.ignore()
            return
        self._shutdown_requested = True
        self._closing = True
        self.setEnabled(False)
        self.setWindowTitle("familiar-ai  ⏳ shutting down...")
        self._ensure_shutdown_task()
        event.ignore()

    def _ensure_shutdown_task(self) -> asyncio.Task[None]:
        if self._shutdown_task and not self._shutdown_task.done():
            return self._shutdown_task
        self._shutdown_task = self._create_task(self._shutdown())
        self._shutdown_task.add_done_callback(lambda _task: self._finalize_close())
        return self._shutdown_task

    def _finalize_close(self) -> None:
        if not self._shutdown_done:
            return
        self.close()

    def _report_event_loop_lag(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_lag_tick
        self._last_lag_tick = now
        lag = elapsed - _GUI_LOOP_LAG_CHECK_SEC
        if lag > _GUI_LOOP_LAG_WARN_SEC:
            logger.warning(
                "GUI event-loop lag detected: %.3fs (queue=%d running=%s)",
                lag,
                self._input_queue.qsize(),
                self._agent_running,
            )

    async def _shutdown(self) -> None:
        """Best-effort async cleanup on window close."""
        self._lag_timer.stop()
        self._status_timer.stop()
        self._cancel_turn(reason="shutdown")
        if self._realtime_stt_task and not self._realtime_stt_task.done():
            self._realtime_stt_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._realtime_stt_task
        self._realtime_stt_task = None
        if self._realtime_stt:
            with contextlib.suppress(asyncio.TimeoutError, Exception):
                await asyncio.wait_for(self._realtime_stt.stop(), timeout=2.0)
            self._realtime_stt = None
        if self._look_preview_task and not self._look_preview_task.done():
            self._look_preview_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._look_preview_task
        self._input_queue.put_nowait(None)
        if self._queue_task and not self._queue_task.done():
            self._queue_task.cancel()
        if self._init_task and not self._init_task.done():
            self._init_task.cancel()
        startup_status_task = getattr(self, "_startup_status_task", None)
        if startup_status_task and not startup_status_task.done():
            startup_status_task.cancel()
        try:
            if self._agent_task and not self._agent_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(self._agent_task), timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    pass
            if self._agent is not None:
                await asyncio.wait_for(self._agent.close(), timeout=3.0)
        except (asyncio.TimeoutError, Exception):
            pass
        self._shutdown_done = True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_gui(
    config: "AgentConfig",
    desires: "DesireSystem",
    background_coros: "list[Any] | None" = None,
    *,
    shared_camera: "Any | None" = None,
) -> None:
    """Launch the PySide6 GUI with qasync event loop.

    background_coros: optional list of awaitables to schedule as asyncio tasks
    on the qasync event loop (e.g. the Telegram bot coroutine).  They are
    cancelled automatically when the Qt window closes.
    """
    import signal

    existing = QApplication.instance()
    qt_app = existing if isinstance(existing, QApplication) else QApplication(sys.argv)
    _apply_global_style(qt_app)
    icon_path = resolve_app_icon_path()
    if icon_path:
        icon = QIcon(str(icon_path))
        if not icon.isNull():
            qt_app.setWindowIcon(icon)

    loop = qasync.QEventLoop(qt_app)
    asyncio.set_event_loop(loop)

    window = FamiliarWindow(config, desires, shared_camera=shared_camera)
    if icon_path:
        icon = QIcon(str(icon_path))
        if not icon.isNull():
            window.setWindowIcon(icon)
    window.show()

    # Schedule background coroutines (e.g. Telegram bot) on the qasync loop
    bg_tasks: list[asyncio.Task] = []
    if background_coros:
        for coro in background_coros:
            bg_tasks.append(loop.create_task(coro))

    def _on_quit() -> None:
        window._ensure_shutdown_task()
        for t in bg_tasks:
            if not t.done():
                t.cancel()

    qt_app.aboutToQuit.connect(_on_quit)

    # Qt's event loop does not yield to CPython's signal-checking mechanism on its own.
    # A periodic no-op timer wakes up the Python interpreter so that SIGINT (Ctrl+C)
    # is processed promptly instead of being silently ignored.
    from PySide6.QtCore import QTimer

    _signal_timer = QTimer()
    _signal_timer.start(200)
    _signal_timer.timeout.connect(lambda: None)
    signal.signal(signal.SIGINT, lambda *_: qt_app.quit())

    with loop:
        loop.run_forever()

    # Non-daemon threads (ThreadPoolExecutor from run_in_executor, httpx connection
    # pools, etc.) keep the process alive after the Qt event loop exits.
    # Force-exit so the CMD window closes cleanly — same pattern as the REPL.
    os._exit(0)


def run_setup_wizard(config: "AgentConfig", env_path: Path | None = None) -> bool:
    """Launch the setup dialog without constructing the full main window."""
    existing = QApplication.instance()
    qt_app = existing if isinstance(existing, QApplication) else QApplication(sys.argv)
    _apply_global_style(qt_app)
    icon_path = resolve_app_icon_path()
    if icon_path:
        icon = QIcon(str(icon_path))
        if not icon.isNull():
            qt_app.setWindowIcon(icon)

    dialog = SettingsDialog(config, env_path or _ENV_PATH, setup_mode=True)
    if icon_path:
        icon = QIcon(str(icon_path))
        if not icon.isNull():
            dialog.setWindowIcon(icon)
    return dialog.exec() == int(QDialog.DialogCode.Accepted)
