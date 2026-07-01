"""
Interactive Fast-and-Frugal Tree visualisation (SVG renderer).
SURA 2026 · IIT Delhi

Renders a learned/edited FFT as a clean, themed SVG flow diagram. Given a test
pair's difference values it highlights the decision path and the predicted exit,
so editing a threshold (which re-renders this SVG) immediately shows the new
prediction. The accompanying editor widgets and persistence live in app.py.

The renderer is intentionally pure (string in → string out) so it is trivial to
test and reuse outside Streamlit.
"""

import html


def pretty_feature(feature):
    """'urgency_score_diff' -> 'Urgency Score (Δ A−B)'."""
    base = feature[:-5] if feature.endswith("_diff") else feature
    return base.replace("_", " ").title() + " (Δ A−B)"


def _eval_path(tree, diffs):
    """Return ('exit', node_index, class) or ('default', -1, class)."""
    for i, n in enumerate(tree["nodes"]):
        x = float(diffs.get(n["feature"], 0.0))
        cond = (x >= n["threshold"]) if n["op"] == ">=" else (x <= n["threshold"])
        if cond:
            return "exit", i, n["exit_class"]
    return "default", -1, tree["default_class"]


def fft_svg(tree, palette, test_diffs=None, width=760):
    """
    Build an SVG string for the given tree dict.

    tree       : FastFrugalTree.to_dict()
    palette    : dict with bg, card, border, text, dim, muted, accent, a, b
    test_diffs : optional {feature: difference_value} for the live test pair.
                 When present, the matching path and exit are highlighted and a
                 prediction badge is shown.
    """
    p = palette
    nodes = tree["nodes"]
    n = len(nodes)

    # geometry
    pad_top = 70 if test_diffs is not None else 30
    row_h = 122
    node_x, node_w, node_h = 36, 372, 66
    leaf_x, leaf_w, leaf_h = 540, 150, 46
    height = pad_top + n * row_h + 150

    kind, exit_i, pred_class = (None, -2, None)
    if test_diffs is not None:
        kind, exit_i, pred_class = _eval_path(tree, test_diffs)

    def leaf_fill(cls):
        return p["a"] if cls == 1 else p["b"]

    def leaf_text(cls):
        return "Prefer A" if cls == 1 else "Prefer B"

    s = []
    s.append(
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'font-family="-apple-system,Segoe UI,Roboto,sans-serif">'
    )
    s.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="none"/>')

    # arrow marker
    s.append(
        f'<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="6" refY="3" '
        f'orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="{p["muted"]}"/></marker>'
        f'<marker id="ahx" markerWidth="9" markerHeight="9" refX="6" refY="3" '
        f'orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="{p["accent"]}"/></marker></defs>'
    )

    # prediction badge
    if test_diffs is not None:
        pf = leaf_fill(pred_class)
        s.append(
            f'<rect x="{node_x}" y="18" rx="14" width="300" height="34" '
            f'fill="{pf}" opacity="0.16" stroke="{pf}"/>'
            f'<text x="{node_x + 16}" y="40" font-size="15" font-weight="700" '
            f'fill="{pf}">Prediction:&#160;{leaf_text(pred_class)}</text>'
        )

    for i, node in enumerate(nodes):
        y = pad_top + i * row_h
        cy = y + node_h / 2
        on_path = (test_diffs is None) or (kind == "default") or (i <= exit_i)
        is_exit = (kind == "exit" and i == exit_i)
        op_active = (i < exit_i) and kind == "exit"  # condition was false, fell through
        dim = (test_diffs is not None) and (not on_path)
        node_op = 0.32 if dim else 1.0

        box_stroke = p["accent"] if is_exit else p["border"]
        box_sw = 2.5 if is_exit else 1.2

        # decision box
        s.append(
            f'<g opacity="{node_op}">'
            f'<rect x="{node_x}" y="{y}" rx="11" width="{node_w}" height="{node_h}" '
            f'fill="{p["card"]}" stroke="{box_stroke}" stroke-width="{box_sw}"/>'
            f'<text x="{node_x + 16}" y="{y + 22}" font-size="11.5" '
            f'fill="{p["muted"]}" font-weight="600">STEP {i + 1}</text>'
            f'<text x="{node_x + 16}" y="{y + 44}" font-size="15.5" '
            f'fill="{p["text"]}" font-weight="600">'
            f'{html.escape(pretty_feature(node["feature"]))}</text>'
            f'<text x="{node_x + 16}" y="{y + 60}" font-size="13" '
            f'fill="{p["dim"]}" font-family="monospace">'
            f'{html.escape(node["op"])} {round(node["threshold"], 2)}</text>'
            f'</g>'
        )

        # YES branch -> exit leaf (to the right)
        yes_hl = is_exit
        yes_col = p["accent"] if yes_hl else p["muted"]
        marker = "ahx" if yes_hl else "ah"
        s.append(
            f'<line x1="{node_x + node_w}" y1="{cy}" x2="{leaf_x - 6}" y2="{cy}" '
            f'stroke="{yes_col}" stroke-width="{2.4 if yes_hl else 1.3}" '
            f'opacity="{node_op}" marker-end="url(#{marker})"/>'
            f'<text x="{node_x + node_w + 12}" y="{cy - 8}" font-size="11" '
            f'fill="{yes_col}" opacity="{node_op}" font-weight="600">YES</text>'
        )
        lf = leaf_fill(node["exit_class"])
        leaf_op = node_op if (test_diffs is None or is_exit or not on_path) else 0.5
        if is_exit:
            leaf_op = 1.0
        s.append(
            f'<g opacity="{leaf_op}">'
            f'<rect x="{leaf_x}" y="{cy - leaf_h / 2}" rx="10" width="{leaf_w}" '
            f'height="{leaf_h}" fill="{lf}" opacity="0.16"/>'
            f'<rect x="{leaf_x}" y="{cy - leaf_h / 2}" rx="10" width="{leaf_w}" '
            f'height="{leaf_h}" fill="none" stroke="{lf}" '
            f'stroke-width="{2.4 if is_exit else 1.4}"/>'
            f'<text x="{leaf_x + leaf_w / 2}" y="{cy + 5}" font-size="14" '
            f'text-anchor="middle" fill="{lf}" font-weight="700">'
            f'{leaf_text(node["exit_class"])}</text>'
            f'</g>'
        )

        # NO branch -> down to next node (or default leaf)
        no_y2 = y + row_h
        no_hl = (kind == "exit" and i < exit_i) or (kind == "default")
        no_col = p["accent"] if no_hl else p["muted"]
        no_marker = "ahx" if no_hl else "ah"
        no_op = 1.0 if no_hl else node_op
        s.append(
            f'<line x1="{node_x + 24}" y1="{y + node_h}" x2="{node_x + 24}" '
            f'y2="{no_y2 - 4}" stroke="{no_col}" '
            f'stroke-width="{2.2 if no_hl else 1.3}" opacity="{no_op}" '
            f'marker-end="url(#{no_marker})"/>'
            f'<text x="{node_x + 32}" y="{y + node_h + 24}" font-size="11" '
            f'fill="{no_col}" opacity="{no_op}" font-weight="600">NO</text>'
        )

    # default leaf
    y = pad_top + n * row_h
    dcls = tree["default_class"]
    df = leaf_fill(dcls)
    d_hl = (kind == "default")
    d_op = 1.0 if d_hl else (0.4 if test_diffs is not None else 1.0)
    s.append(
        f'<g opacity="{d_op}">'
        f'<rect x="{node_x + 24 - leaf_w / 2 + 14}" y="{y + 4}" rx="10" '
        f'width="{leaf_w}" height="{leaf_h}" fill="{df}" opacity="0.16"/>'
        f'<rect x="{node_x + 24 - leaf_w / 2 + 14}" y="{y + 4}" rx="10" '
        f'width="{leaf_w}" height="{leaf_h}" fill="none" stroke="{df}" '
        f'stroke-width="{2.4 if d_hl else 1.4}"/>'
        f'<text x="{node_x + 24 + 14}" y="{y + 4 + leaf_h / 2 + 5}" font-size="13" '
        f'text-anchor="middle" fill="{df}" font-weight="700">'
        f'Default · {leaf_text(dcls)}</text>'
        f'</g>'
    )

    s.append("</svg>")
    return "".join(s)
