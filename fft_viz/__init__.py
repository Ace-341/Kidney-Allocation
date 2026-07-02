"""
FFT visualization component.

Uses declare_component with url= (not path=) so Streamlit sets the iframe src
directly — no server registry lookup, no ScriptRunContext timing dependency.

A tiny background HTTP server serves the static frontend files on a random port.
The server starts once (thread-safe) and lives for the Python process lifetime.
"""

from __future__ import annotations

import http.server
import os
import socket
import socketserver
import threading

import streamlit.components.v1 as components

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

# ── Background static server ──────────────────────────────────────────────────

_server_port: int | None = None
_server_lock  = threading.Lock()


def _start_server() -> int:
    global _server_port
    with _server_lock:
        if _server_port is not None:
            return _server_port

        # Grab a free OS port before binding.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        class _Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=_FRONTEND_DIR, **kw)

            def log_message(self, *args):  # silence access logs
                pass

        srv = socketserver.TCPServer(("127.0.0.1", port), _Handler)
        srv.allow_reuse_address = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        _server_port = port
        return port


# ── Component wrapper ─────────────────────────────────────────────────────────

_component_func = None


def fft_viz(
    tree: dict,
    editing: bool = False,
    params: list[str] | None = None,
    key: str | None = None,
) -> dict | None:
    """
    Render the FFT decision tree.

    Returns the edited tree dict when the user clicks Apply, otherwise None.
    """
    global _component_func
    if _component_func is None:
        port = _start_server()
        # url= tells Streamlit to point the iframe directly at our local server.
        # This bypasses the component registry entirely, so there is no
        # ScriptRunContext requirement and no path= serving issue.
        _component_func = components.declare_component(
            "fft_tree_viz",
            url=f"http://127.0.0.1:{port}/",
        )

    return _component_func(
        tree=tree,
        editing=editing,
        params=params or [],
        key=key,
        default=None,
    )
