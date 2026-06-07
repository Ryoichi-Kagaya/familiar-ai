"""PTY-over-WebSocket serve mode — exposes the familiar TUI in a browser via xterm.js."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import pty
import shutil
import signal
import struct
import sys
import termios
import tty as _tty

from aiohttp import WSMsgType, web

log = logging.getLogger(__name__)

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>familiar</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/xterm@5.3.0/css/xterm.css">
<script src="https://cdn.jsdelivr.net/npm/xterm@5.3.0/lib/xterm.js"></script>
<script src="https://cdn.jsdelivr.net/npm/xterm-addon-fit@0.8.0/lib/xterm-addon-fit.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { height: 100%; background: #000; overflow: hidden; }
#t { position: absolute; inset: 0; padding: 4px; }
</style>
</head>
<body>
<div id="t"></div>
<script>
const term = new Terminal({ cursorBlink: true, allowProposedApi: true });
const fit  = new FitAddon.FitAddon();
term.loadAddon(fit);
term.open(document.getElementById('t'));
fit.fit();

const ws = new WebSocket('ws://' + location.host + '/ws');
ws.binaryType = 'arraybuffer';

ws.onopen    = sendResize;
ws.onmessage = e => term.write(
  e.data instanceof ArrayBuffer ? new Uint8Array(e.data) : e.data
);
ws.onclose = () => term.write('\\r\\n[connection closed — reload to reconnect]\\r\\n');

term.onData(d => ws.readyState === 1 && ws.send(new TextEncoder().encode(d)));

function sendResize() {
  ws.readyState === 1 &&
    ws.send(JSON.stringify({ type: 'resize', cols: term.cols, rows: term.rows }));
}

const ro = new ResizeObserver(() => { fit.fit(); sendResize(); });
ro.observe(document.getElementById('t'));
</script>
</body>
</html>"""


def _set_size(fd: int, cols: int, rows: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def _child_argv() -> list[str]:
    """Build argv for the TUI child process, stripping server-only flags."""
    args = list(sys.argv)
    while "--serve" in args:
        args.remove("--serve")
    i = 0
    while i < len(args):
        if args[i] in ("--port", "--host"):
            args.pop(i)
            if i < len(args):
                args.pop(i)
        else:
            i += 1
    return args



async def _run_dual(host: str, port: int, argv: list[str]) -> None:
    """Run TUI in the local terminal AND mirror it to browsers via WebSocket."""
    master_fd, slave_fd = pty.openpty()
    cols, rows = shutil.get_terminal_size((220, 50))
    _set_size(master_fd, cols, rows)

    env = os.environ.copy()
    env.update({"TERM": "xterm-256color", "COLORTERM": "truecolor", "FAMILIAR_IN_SERVE": "1"})

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
        close_fds=True,
    )
    os.close(slave_fd)

    clients: set[web.WebSocketResponse] = set()
    loop = asyncio.get_running_loop()
    done = asyncio.Event()

    def _on_sigwinch() -> None:
        c, r = shutil.get_terminal_size((cols, rows))
        with contextlib.suppress(OSError):
            _set_size(master_fd, c, r)

    with contextlib.suppress(OSError, ValueError):
        loop.add_signal_handler(signal.SIGWINCH, _on_sigwinch)

    async def _pty_to_all() -> None:
        while True:
            try:
                chunk = await loop.run_in_executor(None, os.read, master_fd, 4096)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                for ws in list(clients):
                    with contextlib.suppress(Exception):
                        await ws.send_bytes(chunk)
            except OSError:
                break
        done.set()

    stdin_fd = sys.stdin.fileno()
    stdin_queue: asyncio.Queue[bytes] = asyncio.Queue()

    def _stdin_readable() -> None:
        with contextlib.suppress(OSError):
            data = os.read(stdin_fd, 256)
            if data:
                stdin_queue.put_nowait(data)

    loop.add_reader(stdin_fd, _stdin_readable)

    async def _stdin_to_pty() -> None:
        while not done.is_set():
            try:
                chunk = await asyncio.wait_for(stdin_queue.get(), timeout=0.1)
                os.write(master_fd, chunk)
            except asyncio.TimeoutError:
                continue
            except OSError:
                break

    async def _ws_handler(req: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(req)
        clients.add(ws)
        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    with contextlib.suppress(OSError):
                        os.write(master_fd, msg.data)
                elif msg.type == WSMsgType.TEXT:
                    with contextlib.suppress(Exception):
                        d = json.loads(msg.data)
                        if d.get("type") == "resize":
                            # Local terminal controls PTY size in dual mode
                            pass
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            clients.discard(ws)
        return ws

    async def _index(req: web.Request) -> web.Response:
        return web.Response(text=_HTML, content_type="text/html")

    app = web.Application()
    app.router.add_get("/", _index)
    app.router.add_get("/ws", _ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()

    # Print URL before raw mode so the message is briefly visible
    import socket as _socket
    if host in ("0.0.0.0", ""):
        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM) as _s:
                _s.connect(("8.8.8.8", 80))
                _lan_ip = _s.getsockname()[0]
        except OSError:
            _lan_ip = "localhost"
        sys.stderr.write(
            f"\r\n[familiar-ai] Web mirror → http://localhost:{port}"
            f"  (LAN: http://{_lan_ip}:{port})\r\n\r\n"
        )
    else:
        sys.stderr.write(f"\r\n[familiar-ai] Web mirror → http://{host}:{port}\r\n\r\n")
    sys.stderr.flush()

    stdin_attrs = termios.tcgetattr(stdin_fd)
    _tty.setraw(stdin_fd)
    try:
        await asyncio.gather(_pty_to_all(), _stdin_to_pty(), return_exceptions=True)
    finally:
        loop.remove_reader(stdin_fd)
        with contextlib.suppress(Exception):
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, stdin_attrs)
        with contextlib.suppress(OSError, ValueError):
            loop.remove_signal_handler(signal.SIGWINCH)
        await runner.cleanup()
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.terminate()
        with contextlib.suppress(OSError):
            os.close(master_fd)


def serve_dual(host: str = "localhost", port: int = 8080) -> None:
    """Run TUI locally and simultaneously mirror it to a browser at http://localhost:PORT."""
    argv = _child_argv()
    try:
        asyncio.run(_run_dual(host, port, argv))
    except KeyboardInterrupt:
        pass
