"""
FFT visualization component.

Uses declare_component with path= so Streamlit serves the static frontend
files directly through its own server — works in both local and deployed
environments. The url= / local-HTTP-server approach only works when the
browser and server are on the same machine (local dev).
"""

from __future__ import annotations

import os

import streamlit.components.v1 as components

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

_component_func = components.declare_component(
    "fft_tree_viz",
    path=_FRONTEND_DIR,
)


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
    return _component_func(
        tree=tree,
        editing=editing,
        params=params or [],
        key=key,
        default=None,
    )
