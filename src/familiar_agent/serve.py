"""PTY-over-WebSocket serve mode — exposes the familiar TUI in a browser via xterm.js."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import pty
import signal
import struct
import sys
import termios

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


def _child_argv() -> list[str]:
    """Build the argv for the TUI child process, stripping --serve / --port flags."""
    args = list(sys.argv)
    while "--serve" in args:
        args.remove("--serve")
    i = 0
    while i < len(args):
        if args[i] == "--port":
            args.pop(i)
            if i < len(args):
                args.pop(i)
        else:
            i += 1
    return args


async def _ws_session(request: web.Request, argv: list[str]) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    master_fd, slave_fd = pty.openpty()

    def _set_size(cols: int, rows: int) -> None:
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    _set_size(220, 50)

    env = os.environ.copy()
    env.update({"TERM": "xterm-256color", "COLORTERM": "truecolor"})

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
        close_fds=True,
    )
    os.close(slave_fd)

    loop = asyncio.get_running_loop()

    async def _pump() -> None:
        while True:
            try:
                chunk = await loop.run_in_executor(None, os.read, master_fd, 4096)
                if not chunk:
                    break
                await ws.send_bytes(chunk)
            except OSError:
                break
        with contextlib.suppress(Exception):
            await ws.close()

    pump = asyncio.create_task(_pump())

    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            with contextlib.suppress(Exception):
                d = json.loads(msg.data)
                if d.get("type") == "resize":
                    _set_size(int(d["cols"]), int(d["rows"]))
        elif msg.type == WSMsgType.BINARY:
            with contextlib.suppress(OSError):
                os.write(master_fd, msg.data)
        elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
            break

    pump.cancel()
    with contextlib.suppress(ProcessLookupError, OSError):
        proc.send_signal(signal.SIGTERM)
    with contextlib.suppress(OSError):
        os.close(master_fd)
    await asyncio.gather(pump, return_exceptions=True)
    return ws


async def _run(host: str, port: int, argv: list[str]) -> None:
    async def index(req: web.Request) -> web.Response:
        return web.Response(text=_HTML, content_type="text/html")

    async def ws_handler(req: web.Request) -> web.WebSocketResponse:
        return await _ws_session(req, argv)

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, host, port).start()

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


def serve(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Serve the familiar TUI over HTTP/WebSocket with an xterm.js frontend."""
    argv = _child_argv()
    try:
        asyncio.run(_run(host, port, argv))
    except KeyboardInterrupt:
        pass
