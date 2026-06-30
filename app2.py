"""
Preference Elicitation Portal
SURA 2026 · IIT Delhi

Run:  streamlit run app.py
Install: pip install streamlit pandas numpy scikit-learn matplotlib imodels

CSV file must be in the same folder as app.py.
Responses are saved to responses/<username>_responses.csv
"""



import streamlit as st
import pandas as pd
import numpy as np
import json, os, warnings
from datetime import datetime

# Load .env file for local development (GROQ_API_KEY etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — env vars set by Railway or system

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# GROQ LLM HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _groq_explain(prompt, fallback_text):
    """
    Call Groq API for a natural language explanation.
    Falls back to fallback_text if API is unavailable or key is missing.
    Responses are cached in st.session_state to avoid repeated API calls.
    """
    cache_key = f"groq_{hash(prompt)}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return fallback_text

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        result = response.choices[0].message.content.strip()
        st.session_state[cache_key] = result
        return result
    except Exception:
        return fallback_text


def _build_patient_context(d, params):
    """Build plain-English patient description for LLM prompt."""
    lines = []
    for p in params:
        av = d.get(f"A_{p}", "?")
        bv = d.get(f"B_{p}", "?")
        lines.append(f"  - {p.replace('_',' ').title()}: A={av}, B={bv}")
    return "\n".join(lines)


def explain_prediction_llm(d, params, pred, rules_df, fallback_text):
    """
    Generate a 2-3 sentence LLM explanation of why the model
    predicted pred for decision d.
    """
    pred_label = "A" if pred == 1 else "B"

    # Find top fired rule for context
    top_rule = ""
    if rules_df is not None and len(rules_df) > 0:
        top_rule = rules_df.iloc[0]["rule"]

    patient_ctx = _build_patient_context(d, params)
    # Build plain-English feature summary for the prompt
    feature_lines = []
    for p in params:
        av = float(d.get(f"A_{p}", 0))
        bv = float(d.get(f"B_{p}", 0))
        pn = p.replace("_"," ").title()
        diff_pct = (av - bv) / ((av + bv) / 2 + 0.01) * 100
        if abs(diff_pct) > 5:
            who = "A" if diff_pct > 0 else "B"
            feature_lines.append(
                f"  - {pn}: A={av:.0f}, B={bv:.0f} "
                f"({'A is higher' if diff_pct>0 else 'B is higher'} by {abs(diff_pct):.0f}%)"
            )
        else:
            feature_lines.append(f"  - {pn}: A={av:.0f}, B={bv:.0f} (nearly equal)")

    # Add composite context if available
    if "age" in params and "health_score" in params:
        age_idx    = list(params).index("age")
        health_idx = list(params).index("health_score")
        rem_a = max(1, 85 - float(d.get(f"A_age", 0)))
        rem_b = max(1, 85 - float(d.get(f"B_age", 0)))
        tb_a  = float(d.get(f"A_health_score", 0)) * rem_a
        tb_b  = float(d.get(f"B_health_score", 0)) * rem_b
        feature_lines.append(
            f"  - Expected treatment benefit: A={tb_a:.0f}, B={tb_b:.0f} "
            f"(health × remaining life years)"
        )
    if "urgency_score" in params and "years_waiting" in params:
        vi_a = float(d.get("A_urgency_score", 0)) * float(d.get("A_years_waiting", 0))
        vi_b = float(d.get("B_urgency_score", 0)) * float(d.get("B_years_waiting", 0))
        feature_lines.append(
            f"  - Vulnerability index: A={vi_a:.0f}, B={vi_b:.0f} "
            f"(urgency × years waiting)"
        )

    feature_ctx = "\n".join(feature_lines)

    prompt = f"""You are explaining an organ allocation decision to a non-technical person such as a doctor or ethicist.

Two patients need an organ transplant. Only one can receive it.

Patient comparison:
{feature_ctx}

The preference model predicted: Option {pred_label} should be preferred.
Key pattern: {top_rule if top_rule else "no single dominant rule — based on overall patient profile"}

In exactly 2-3 sentences, explain in plain English why Option {pred_label} was preferred.
Mention specific numbers from the comparison above. Do not use words like "rule", "coefficient", "feature", or "model".
Write as if a knowledgeable colleague is briefly explaining their reasoning."""

    return _groq_explain(prompt, fallback_text)


def explain_model_learned_llm(rules_df, params, stats, fallback_text):
    """
    Generate a natural language summary of what the model learned
    from the participant's decisions.
    """
    if rules_df is None or len(rules_df) == 0:
        return fallback_text

    rules_summary = []
    for i, (_, row) in enumerate(rules_df.iterrows()):
        pref  = "A" if row["coef"] > 0 else "B"
        rules_summary.append(
            f"{i+1}. When {row['rule']} → prefer {pref} "
            f"(applies to {row['support']:.0%} of cases)"
        )
    rules_text = "\n".join(rules_summary)

    prompt = f"""You are summarising what a person's organ allocation decisions reveal about their values.

They made {stats.get('n_decisions', '?')} decisions comparing pairs of transplant patients.
The analysis identified these patterns (expressed in plain language):
{rules_text}

The features used include: life years difference (%), urgency difference (%),
health difference (%), waiting time difference (%), vulnerability index (urgency × waiting),
expected treatment benefit (health × remaining life), and social responsibility (dependents × remaining life).

In 2-3 sentences, summarise what this person seems to value most in organ allocation decisions.
Be specific — mention which factors they weight most. Do not use technical jargon.
Write as if presenting findings to a medical ethics committee."""

    return _groq_explain(prompt, fallback_text)


def explain_inconsistency_llm(d_i, d_j, cf, params, pred_i, pred_j, fallback_i, fallback_j):
    """
    Generate LLM explanation for why two similar scenarios got different predictions.
    """
    ctx_i = _build_patient_context(d_i, params)
    ctx_j = _build_patient_context(d_j, params)
    pred_i_lbl = "A" if pred_i == 1 else "B"
    pred_j_lbl = "A" if pred_j == 1 else "B"

    prompt_i = f"""You are explaining an organ allocation decision to a non-technical person.

Scenario {cf['scenario_i']} — patient values:
{ctx_i}
The model predicted: Option {pred_i_lbl}

In 2-3 sentences, explain in plain English why the model preferred Option {pred_i_lbl}.
Focus on the most important factor. No technical jargon. No mention of "rules" or "coefficients"."""

    prompt_j = f"""You are explaining an organ allocation decision to a non-technical person.

Scenario {cf['scenario_j']} — patient values:
{ctx_j}
The model predicted: Option {pred_j_lbl}

In 2-3 sentences, explain in plain English why the model preferred Option {pred_j_lbl}.
Focus on the most important factor. No technical jargon. No mention of "rules" or "coefficients"."""

    exp_i = _groq_explain(prompt_i, fallback_i)
    exp_j = _groq_explain(prompt_j, fallback_j)
    return exp_i, exp_j



# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Preference Elicitation", page_icon="🫘", layout="wide"
)

# CSV file must sit in the same folder as app.py
APP_DIR       = os.path.dirname(os.path.abspath(__file__))
RESPONSES_DIR = os.path.join(APP_DIR, "responses")
USERS_FILE    = os.path.join(APP_DIR, "users.json")
SEED          = 42

# ══════════════════════════════════════════════════════════════════════════════
# THEME SYSTEM
# All colours come from get_theme(). Never hardcode colours elsewhere.
# ══════════════════════════════════════════════════════════════════════════════

THEMES = {
    "light": {
        "BG":        "#ffffff",
        "BG2":       "#f8fafc",
        "BG3":       "#f1f5f9",
        "BORDER":    "#e2e8f0",
        "TEXT":      "#0f172a",
        "TEXT_DIM":  "#64748b",
        "TEXT_MUTED":"#94a3b8",
        "CARD_BG":   "#ffffff",
        "CARD_BORDER":"#e2e8f0",
        "INPUT_BG":  "#f8fafc",
        "A_WIN":     "#16a34a",
        "B_WIN":     "#dc2626",
        "COL_A":     "#b91c1c",
        "COL_B":     "#1d4ed8",
        "ACCENT":    "#2563eb",
        "SUCCESS":   "#dcfce7",
        "SUCCESS_BORDER":"#86efac",
        "WARNING":   "#fef9c3",
        "WARNING_BORDER":"#fde047",
        "DANGER":    "#fee2e2",
        "DANGER_BORDER":"#fca5a5",
        "INFO":      "#eff6ff",
        "INFO_BORDER":"#93c5fd",
    },
    "dark": {
        "BG":        "#0d1117",
        "BG2":       "#161b22",
        "BG3":       "#21262d",
        "BORDER":    "#30363d",
        "TEXT":      "#e6edf3",
        "TEXT_DIM":  "#8b949e",
        "TEXT_MUTED":"#6e7681",
        "CARD_BG":   "#161b22",
        "CARD_BORDER":"#30363d",
        "INPUT_BG":  "#21262d",
        "A_WIN":     "#3fb950",
        "B_WIN":     "#f85149",
        "COL_A":     "#f87171",
        "COL_B":     "#60a5fa",
        "ACCENT":    "#58a6ff",
        "SUCCESS":   "#0d2a1a",
        "SUCCESS_BORDER":"#238636",
        "WARNING":   "#2d2100",
        "WARNING_BORDER":"#9e6a03",
        "DANGER":    "#2d0f0f",
        "DANGER_BORDER":"#da3633",
        "INFO":      "#0d1f3c",
        "INFO_BORDER":"#1f6feb",
    }
}


def get_theme():
    """Return the current theme dict based on session state."""
    mode = st.session_state.get("theme_mode", "light")
    return THEMES[mode]


def inject_theme_css():
    """Inject CSS that overrides Streamlit defaults for the active theme."""
    T = get_theme()
    is_dark = st.session_state.get("theme_mode", "light") == "dark"

    # Streamlit config overrides via CSS custom properties
    css = f"""
<style>
/* ── Root variables ────────────────────────────────── */
:root {{
    --bg:           {T['BG']};
    --bg2:          {T['BG2']};
    --bg3:          {T['BG3']};
    --border:       {T['BORDER']};
    --text:         {T['TEXT']};
    --text-dim:     {T['TEXT_DIM']};
    --accent:       {T['ACCENT']};
    --col-a:        {T['COL_A']};
    --col-b:        {T['COL_B']};
}}

/* ── App background ───────────────────────────────── */
.stApp, .main, [data-testid="stAppViewContainer"] {{
    background-color: {T['BG']} !important;
    color: {T['TEXT']} !important;
}}

/* ── Top header / toolbar ─────────────────────────── */
[data-testid="stHeader"],
header[data-testid="stHeader"],
.stApp > header {{
    background-color: {T['BG2']} !important;
    border-bottom: 1px solid {T['BORDER']} !important;
}}
[data-testid="stToolbar"],
[data-testid="stToolbarActions"] {{
    background-color: {T['BG2']} !important;
}}
[data-testid="stToolbar"] button,
[data-testid="stToolbarActions"] button,
[data-testid="stToolbarActions"] span,
[data-testid="stToolbarActions"] svg {{
    color: {T['TEXT']} !important;
    fill: {T['TEXT']} !important;
}}

[data-testid="stDecoration"] {{
    background-color: {T['ACCENT']} !important;
}}

/* ── Sidebar ──────────────────────────────────────── */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
    background-color: {T['BG2']} !important;
    border-right: 1px solid {T['BORDER']} !important;
}}
[data-testid="stSidebar"] * {{
    color: {T['TEXT']} !important;
}}

/* ── Main content area ────────────────────────────── */
[data-testid="stMainBlockContainer"] {{
    background-color: {T['BG']} !important;
}}

/* ── Text ─────────────────────────────────────────── */
p, span, div, h1, h2, h3, h4, h5, h6, label, li, td, th, b, strong, small {{
    color: {T['TEXT']} !important;
}}
.stMarkdown, .stText {{
    color: {T['TEXT']} !important;
}}

/* ── Tabs ─────────────────────────────────────────── */
.stTabs [data-testid="stTab"] {{
    background-color: {T['BG2']} !important;
    color: {T['TEXT_DIM']} !important;
    border-bottom: 2px solid {T['BORDER']} !important;
}}
.stTabs [aria-selected="true"] {{
    background-color: {T['BG3']} !important;
    color: {T['TEXT']} !important;
    border-bottom: 2px solid {T['ACCENT']} !important;
}}
[data-testid="stTabsContent"] {{
    background-color: {T['BG']} !important;
}}

/* ── Buttons ──────────────────────────────────────── */
.stButton button[kind="primary"] {{
    background-color: {T['ACCENT']} !important;
    color: white !important;
    border: none !important;
}}
.stButton button[kind="secondary"] {{
    background-color: {T['BG3']} !important;
    color: {T['TEXT']} !important;
    border: 1px solid {T['BORDER']} !important;
}}

/* ── Inputs ───────────────────────────────────────── */
input, textarea, select,
[data-testid="textInput"] input,
[data-testid="stTextInput"] input {{
    background-color: {T['INPUT_BG']} !important;
    color: {T['TEXT']} !important;
    border: 1px solid {T['BORDER']} !important;
}}

/* ── Number inputs ────────────────────────────────── */
[data-testid="stNumberInput"] input {{
    background-color: {T['INPUT_BG']} !important;
    color: {T['TEXT']} !important;
    border: 1px solid {T['BORDER']} !important;
}}

/* ── Selectbox / dropdown ─────────────────────────── */
[data-testid="stSelectbox"] > div > div {{
    background-color: {T['INPUT_BG']} !important;
    color: {T['TEXT']} !important;
    border: 1px solid {T['BORDER']} !important;
}}

/* ── Slider ───────────────────────────────────────── */
[data-testid="stSlider"] {{
    color: {T['TEXT']} !important;
}}

/* ── Dataframe / table ────────────────────────────── */
[data-testid="stDataFrame"], .stDataFrame {{
    background-color: {T['BG2']} !important;
}}
[data-testid="stDataFrame"] th {{
    background-color: {T['BG3']} !important;
    color: {T['TEXT']} !important;
}}
[data-testid="stDataFrame"] td {{
    background-color: {T['BG2']} !important;
    color: {T['TEXT']} !important;
}}

/* ── Metric ───────────────────────────────────────── */
[data-testid="stMetric"] {{
    background-color: {T['BG2']} !important;
    border: 1px solid {T['BORDER']} !important;
    border-radius: 8px !important;
    padding: 12px !important;
}}
[data-testid="stMetricValue"] {{
    color: {T['TEXT']} !important;
}}
[data-testid="stMetricLabel"] {{
    color: {T['TEXT_DIM']} !important;
}}

/* ── Expander ─────────────────────────────────────── */
[data-testid="stExpander"] {{
    background-color: {T['BG2']} !important;
    border: 1px solid {T['BORDER']} !important;
    border-radius: 8px !important;
}}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p,
.streamlit-expanderHeader,
.streamlit-expanderHeader p,
.streamlit-expanderHeader span {{
    background-color: {T['BG2']} !important;
    color: {T['TEXT']} !important;
}}
[data-testid="stExpander"] details,
[data-testid="stExpanderDetails"],
.streamlit-expanderContent {{
    background-color: {T['BG2']} !important;
    color: {T['TEXT']} !important;
}}

/* ── Progress bar ─────────────────────────────────── */
[data-testid="stProgressBar"] > div {{
    background-color: {T['BG3']} !important;
}}
[data-testid="stProgressBar"] > div > div {{
    background-color: {T['ACCENT']} !important;
}}

/* ── Alerts / info boxes ─────────────────────────── */
[data-testid="stAlert"] {{
    background-color: {T['BG2']} !important;
    border-left-color: {T['ACCENT']} !important;
    color: {T['TEXT']} !important;
}}

/* ── Divider ─────────────────────────────────────── */
hr {{
    border-color: {T['BORDER']} !important;
}}

/* ── Code blocks ─────────────────────────────────── */
code, pre {{
    background-color: {T['BG3']} !important;
    color: {T['TEXT']} !important;
    border: 1px solid {T['BORDER']} !important;
}}

/* ── Caption / small text ────────────────────────── */
.stCaption, small {{
    color: {T['TEXT_DIM']} !important;
}}

/* ── Columns ─────────────────────────────────────── */
[data-testid="stHorizontalBlock"] {{
    background-color: transparent !important;
}}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)


# ── Named colour shortcuts — resolved at call time from active theme ──────────
# These are plain variables set before each render block that needs them.
# Use _setup_colours() at the top of any section that uses these names.

def _setup_colours():
    """Call at the start of any render section to get current theme colours."""
    T = get_theme()
    return (
        T["BG"], T["BG2"], T["BG3"], T["BORDER"],
        T["TEXT"], T["TEXT_DIM"], T["TEXT_MUTED"],
        T["A_WIN"], T["B_WIN"], T["COL_A"], T["COL_B"], T["ACCENT"]
    )

# ── Session defaults ──────────────────────────────────────────────────────────
DEFAULTS = {
    "logged_in":   False,
    "username":    "",
    "scenarios":   [],
    "params":      [],
    "sc_index":    0,
    "decisions":   [],
    "page":        "login",
    "pred_result": None,
    "theme_mode":  "light",   # always start in light mode
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Inject theme CSS immediately after session state is ready
inject_theme_css()

# ── CSV auto-loader ───────────────────────────────────────────────────────────
def find_and_load_csv():
    path = os.path.join(APP_DIR, "organ_allocation_scenarios.csv")
    if not os.path.exists(path):
        raise ValueError(
            f"File not found: {path}\n"
            "Make sure organ_allocation_scenarios.csv is in the same folder as app.py."
        )
    df     = pd.read_csv(path)
    params, scenarios = parse_csv(df)
    return params, scenarios, "organ_allocation_scenarios.csv"


def parse_csv(df):
    """
    Validate and parse a scenarios CSV.
    Columns must be A_<name> and B_<name>.
    Returns (params, scenarios) or raises ValueError.
    """
    a_cols = [c for c in df.columns if str(c).startswith("A_")]
    if not a_cols:
        raise ValueError(
            "No columns starting with 'A_' found.\n"
            "Format: A_param1, A_param2, ..., B_param1, B_param2, ..."
        )
    params  = [c[2:] for c in a_cols]
    missing = [f"B_{p}" for p in params if f"B_{p}" not in df.columns]
    if missing:
        raise ValueError(f"Missing B_ columns: {', '.join(missing)}")
    for col in a_cols + [f"B_{p}" for p in params]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[a_cols].isnull().any().any():
        raise ValueError("Some values are not numeric. Check your CSV.")
    scenarios = []
    for _, row in df.iterrows():
        scenarios.append({
            "A": {p: float(row[f"A_{p}"]) for p in params},
            "B": {p: float(row[f"B_{p}"]) for p in params},
        })
    return params, scenarios


# ── Persistence ───────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO DIFFICULTY
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_difficulty(sc, params):
    """
    Score how hard a scenario is to decide (0=easy, 1=hardest).
    Based on how close A and B are across all parameters,
    normalized by the overall range in the dataset.
    A near-tie on all params = difficulty ~1.0
    One option dominates on everything = difficulty ~0.0
    """
    diffs = []
    for p in params:
        a = sc["A"].get(p, 0)
        b = sc["B"].get(p, 0)
        diffs.append(abs(float(a) - float(b)))
    if not diffs or max(diffs) == 0:
        return 1.0   # all identical = maximally hard
    # normalise by max diff in this scenario
    max_d    = max(diffs)
    avg_norm = sum(d / max_d for d in diffs) / len(diffs)
    # high avg_norm means diffs are consistently large (easy)
    # low avg_norm means most params are close (hard)
    score = 1.0 - avg_norm
    return round(float(score), 3)


def difficulty_label(score):
    if score >= 0.70:
        return "🔴 Hard",   "#ef4444"
    elif score >= 0.40:
        return "🟡 Medium", "#f59e0b"
    else:
        return "🟢 Easy",   "#22c55e"

def load_users():
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_users(u):
    with open(USERS_FILE, "w") as f:
        json.dump(u, f, indent=2)


def save_session():
    u = load_users()
    u[st.session_state.username] = {
        "sc_index":  st.session_state.sc_index,
        "decisions": st.session_state.decisions,
    }
    save_users(u)


def record_decision(choice):
    """Save current scenario decision and advance index."""
    os.makedirs(RESPONSES_DIR, exist_ok=True)

    idx = st.session_state.sc_index
    sc  = st.session_state.scenarios[idx]
    row = {
        "username":  st.session_state.username,
        "scenario":  idx + 1,
        "choice":    choice,
        "timestamp": datetime.now().isoformat(),
    }
    for p in st.session_state.params:
        row[f"A_{p}"] = sc["A"][p]
        row[f"B_{p}"] = sc["B"][p]

    # Append to <username>_responses.csv inside responses/
    user_file = os.path.join(
        RESPONSES_DIR,
        f"{st.session_state.username}_responses.csv"
    )
    new_row_df = pd.DataFrame([row])
    if os.path.exists(user_file):
        existing = pd.read_csv(user_file)
        combined = pd.concat([existing, new_row_df], ignore_index=True)
    else:
        combined = new_row_df
    combined.to_csv(user_file, index=False)

    st.session_state.decisions.append(row)
    st.session_state.sc_index += 1
    save_session()


# ── Auto-load CSV at startup ──────────────────────────────────────────────────
if not st.session_state.scenarios:
    try:
        params, scenarios, csv_name = find_and_load_csv()
        st.session_state.params    = params
        st.session_state.scenarios = scenarios
        st.session_state._csv_name = csv_name
    except ValueError as _csv_err:
        st.error(str(_csv_err))
        st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# SYMMETRIC FEATURE LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# CONSISTENCY CHECKER
# ═══════════════════════════════════════════════════════════════════════════════

def check_consistency(decisions, params, threshold=0.85):
    """
    Detect potentially inconsistent decisions.
    Two decisions are inconsistent if:
      - In decision X: A beats B on most params AND user chose A
      - In decision Y: A beats B on same set of params AND user chose B
    Uses cosine similarity of normalised difference vectors.
    Returns list of (idx_i, idx_j, similarity, choice_i, choice_j) tuples.
    """
    if len(decisions) < 4:
        return []

    vecs   = []
    labels = []
    for d in decisions:
        diff = np.array(
            [float(d.get(f"A_{p}", 0)) - float(d.get(f"B_{p}", 0))
             for p in params]
        )
        norm = np.linalg.norm(diff)
        vecs.append(diff / norm if norm > 0 else diff)
        labels.append(d.get("choice", "?"))

    conflicts = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sim = float(np.dot(vecs[i], vecs[j]))   # cosine similarity
            if sim >= threshold and labels[i] != labels[j]:
                conflicts.append({
                    "scenario_i": decisions[i].get("scenario", i + 1),
                    "scenario_j": decisions[j].get("scenario", j + 1),
                    "similarity": sim,
                    "choice_i":   labels[i],
                    "choice_j":   labels[j],
                    "decision_i": decisions[i],
                    "decision_j": decisions[j],
                })
    return conflicts

# ── Feature name lookup for LLM prompts ──────────────────────────────────────
HIGHER_MEANS = {
    "age":               "older patient age",
    "years_waiting":     "longer-waiting patient wait time",
    "health_score":      "healthier patient health score",
    "dependents":        "patient with more dependents",
    "prior_transplants": "patient with more prior transplants",
    "urgency_score":     "more urgent patient urgency",
}
LOWER_MEANS = {
    "age":               "younger patient age",
    "years_waiting":     "shorter-waiting patient wait time",
    "health_score":      "less healthy patient health score",
    "dependents":        "patient with fewer dependents",
    "prior_transplants": "patient with fewer prior transplants",
    "urgency_score":     "less urgent patient urgency",
}
MAX_AGE = 85   # assumed maximum lifespan for life-years calculation


def build_features(decisions, params):
    """
    Build interpretable feature library from pairwise decisions.

    Features (20 total):
      Difference (%): how much A leads B on each parameter, as a percentage
        - Age uses life years remaining (85-age) so younger = positive
        - All others: positive = A is higher, negative = B is higher
        - Model learns from data whether higher = better for each param

      Similarity (%): how similar A and B are (100% = identical, 0% = very different)
        - Omitted for prior_transplants (mostly 0/1/2, division issues)

      Higher/Lower patient: symmetric — what is the best/worst value between them
        - Model learns which matters for each parameter

      Composite differences: domain-specific combined features
        - Treatment benefit: health × remaining life years
        - Social responsibility: dependents × remaining life years
        - Vulnerability index: urgency × years waiting
    """
    a_vals = np.array(
        [[d.get(f"A_{p}", 0) for p in params] for d in decisions], dtype=float
    )
    b_vals = np.array(
        [[d.get(f"B_{p}", 0) for p in params] for d in decisions], dtype=float
    )

    feats = {}

    # ── Per-parameter features ────────────────────────────────────────────────
    for i, p in enumerate(params):
        a  = a_vals[:, i]
        b  = b_vals[:, i]
        pn = p.replace("_", " ").title()
        mean = (a + b) / 2 + 0.01
        mx   = np.maximum(np.abs(a), np.abs(b)) + 0.01

        # Difference (%) — antisymmetric
        if p == "age":
            # Life years remaining: younger patient gets positive value
            rem_a = np.maximum(1, MAX_AGE - a)
            rem_b = np.maximum(1, MAX_AGE - b)
            rem_mean = (rem_a + rem_b) / 2 + 0.01
            feats["Life years difference (%)"] = (rem_a - rem_b) / rem_mean * 100
        else:
            feats[f"{pn} difference (%)"] = (a - b) / mean * 100

        # Similarity (%) — symmetric — omit for prior_transplants
        if p != "prior_transplants":
            feats[f"{pn} similarity (%)"] = (
                1 - np.abs(a - b) / mx
            ) * 100

        # Higher/lower patient — symmetric
        feats[f"{pn} higher patient"] = np.maximum(a, b)
        feats[f"{pn} lower patient"]  = np.minimum(a, b)

    # ── Composite features ────────────────────────────────────────────────────
    # Only compute if the required parameters exist in this CSV
    p_list = list(params)
    if "age" in p_list and "health_score" in p_list:
        i_age    = p_list.index("age")
        i_health = p_list.index("health_score")
        rem_a = np.maximum(1, MAX_AGE - a_vals[:, i_age])
        rem_b = np.maximum(1, MAX_AGE - b_vals[:, i_age])
        tb_a  = a_vals[:, i_health] * rem_a
        tb_b  = b_vals[:, i_health] * rem_b
        mean_tb = (tb_a + tb_b) / 2 + 0.01
        feats["Expected treatment benefit difference (%)"] = (tb_a - tb_b) / mean_tb * 100

    if "age" in p_list and "dependents" in p_list:
        i_age  = p_list.index("age")
        i_dep  = p_list.index("dependents")
        rem_a  = np.maximum(1, MAX_AGE - a_vals[:, i_age])
        rem_b  = np.maximum(1, MAX_AGE - b_vals[:, i_age])
        sr_a   = a_vals[:, i_dep] * rem_a
        sr_b   = b_vals[:, i_dep] * rem_b
        mean_sr = (sr_a + sr_b) / 2 + 0.01
        feats["Social responsibility difference (%)"] = (sr_a - sr_b) / mean_sr * 100

    if "urgency_score" in p_list and "years_waiting" in p_list:
        i_urg  = p_list.index("urgency_score")
        i_wait = p_list.index("years_waiting")
        vi_a   = a_vals[:, i_urg] * a_vals[:, i_wait]
        vi_b   = b_vals[:, i_urg] * b_vals[:, i_wait]
        mean_vi = (vi_a + vi_b) / 2 + 0.01
        feats["Vulnerability index difference (%)"] = (vi_a - vi_b) / mean_vi * 100

    F = pd.DataFrame(feats)
    return F, list(F.columns)


def augment(F, y):
    """
    Double the dataset by swapping A↔B.
    Antisymmetric features (difference, composite differences) get negated.
    Symmetric features (similarity, higher/lower patient) stay the same.
    """
    F_swap = F.copy()
    for col in F.columns:
        # Antisymmetric: all difference (%) and composite difference features
        if "difference" in col.lower():
            F_swap[col] = -F[col]
    return (
        pd.concat([F, F_swap], ignore_index=True),
        np.concatenate([y, 1 - y])
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RULEFIT TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def train_rulefit(decisions_json, params):
    """
    Train RuleFit on symmetric augmented features.
    Cached so it doesn't retrain on every rerender.
    Returns (rulefit, rules_df, stats, feat_names, error_msg)
    """
    try:
        
        from imodels import RuleFitClassifier
    except (ImportError, AttributeError):
        return None, None, None, None, (
            "RuleFit requires Python 3.9–3.11. "
            "Run the app from your imodels_env conda environment.\n"
            "BTL model and basic tree still work on any Python version."
        )

    decisions = json.loads(decisions_json)
    F, feat_names = build_features(decisions, params)
    y = np.array([1 if d["choice"] == "A" else 0 for d in decisions])
    F_aug, y_aug = augment(F, y)

    MIN_IMPORTANCE = 0.05   # higher = stricter, fewer but better rules
    MIN_SUPPORT    = 0.10   # rule must apply to 10%+ of decisions
    rulefit  = None
    rules_df = None

    for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0]:
        clf = RuleFitClassifier(
            max_rules=30, n_estimators=10, tree_size=4,
            random_state=SEED, alpha=alpha, include_linear=False
        )
        clf.fit(F_aug.values, y_aug, feature_names=feat_names)
        rows = []
        for rule, coef in zip(clf.rules_, clf.coef):
            imp = abs(float(coef)) * float(rule.support)
            rows.append({
                "rule":       rule.rule,
                "support":    float(rule.support),
                "coef":       float(coef),
                "importance": imp,
            })
        df_try = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
        df_try = df_try[
            (df_try["importance"] >= MIN_IMPORTANCE) &
            (df_try["support"]    >= MIN_SUPPORT)
        ]
        df_try = df_try.sort_values("importance", ascending=False)

        # Deduplicate: keep only the most important rule per parameter.
        # Extract ALL parameter names mentioned in a rule, then skip the
        # rule if all of those params have already been seen.
        # This prevents getting both "Age diff > X" and "Age diff <= X".
        import re
        seen_params = set()
        unique_rows = []
        for _, row in df_try.iterrows():
            rule_str = row["rule"]
            # Extract all parameter names in this rule
            # Feature names end before ">", "<=", or " " after the name
            # e.g. "Life years difference (%) > 4.65 and Urgency Score difference (%) > -3"
            # Split on " and " to get individual conditions
            conditions = rule_str.split(" and ")
            rule_params = set()
            for cond in conditions:
                # param name is everything before the operator
                param = re.split(r"[ ]*[><!=]+", cond)[0].strip()
                rule_params.add(param)
            # Only keep this rule if it introduces at least one new parameter
            if not rule_params.issubset(seen_params):
                seen_params.update(rule_params)
                unique_rows.append(row)
        df_try = pd.DataFrame(unique_rows).head(10).reset_index(drop=True)
        if len(df_try) > 0:
            rulefit  = clf
            rules_df = df_try
            break

    if rulefit is None:
        return None, None, None, feat_names, "No rules found. Answer more scenarios."

    preds = rulefit.predict(F_aug.values)
    acc   = balanced_accuracy_score(y_aug, preds)
    F_swap = F.copy()
    for col in F.columns:
        if "difference" in col.lower():
            F_swap[col] = -F[col]
    p_orig = rulefit.predict(F.values)
    p_swap = rulefit.predict(F_swap.values)
    sym    = ((p_orig + p_swap) == 1).mean()

    # Per-decision accuracy
    F_orig, _ = build_features(decisions, params)
    y_orig     = np.array([1 if d["choice"] == "A" else 0 for d in decisions])
    preds_orig = rulefit.predict(F_orig.values)
    per_decision = []
    for i, (d, actual, predicted) in enumerate(zip(decisions, y_orig, preds_orig)):
        per_decision.append({
            "scenario":  d.get("scenario", i + 1),
            "actual":    "A" if actual    == 1 else "B",
            "predicted": "A" if predicted == 1 else "B",
            "match":     bool(actual == predicted),
            "decision":  d,
        })

    # Most important parameter
    top_param = rules_df.iloc[0]["rule"].split(":")[0].strip() if len(rules_df) > 0 else "—"

    stats = {
        "acc":          acc,
        "sym":          sym,
        "n_rules":      len(rules_df),
        "n_decisions":  len(decisions),
        "top_param":    top_param,
        "alpha_used":   alpha,
        "per_decision": per_decision,
    }
    return rulefit, rules_df, stats, feat_names, None


# ═══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _build_feat_row(params, a_dict, b_dict, feat_names):
    """Build a single-row feature DataFrame using the new interpretable feature library."""
    row_d = {"choice": "A"}
    for p in params:
        row_d[f"A_{p}"] = float(a_dict.get(p, 0))
        row_d[f"B_{p}"] = float(b_dict.get(p, 0))
    F_tmp, _ = build_features([row_d], params)
    return F_tmp.reindex(columns=feat_names, fill_value=0)


def _decision_prob(rulefit, feat_names, params, d):
    """Return (pred, prob_a) for a single decision dict."""
    a_d = {p: float(d.get(f"A_{p}", 0)) for p in params}
    b_d = {p: float(d.get(f"B_{p}", 0)) for p in params}
    F   = _build_feat_row(params, a_d, b_d, feat_names)
    if hasattr(rulefit, "predict_proba"):
        prob = rulefit.predict_proba(F.values)[0]
        return int(rulefit.predict(F.values)[0]), float(prob[1])  # prob of class1=A
    return int(rulefit.predict(F.values)[0]), None


def _explain_prediction(rulefit, feat_names, params, d, pred):
    """
    Return a plain-English string explaining WHY the model predicted pred
    for decision d, based on which rules fired and their direction.
    """
    a_d = {p: float(d.get(f"A_{p}", 0)) for p in params}
    b_d = {p: float(d.get(f"B_{p}", 0)) for p in params}
    F   = _build_feat_row(params, a_d, b_d, feat_names)
    fv  = F.values[0]

    fired_for_pred = []
    fired_against  = []
    for rule, coef in zip(rulefit.rules_, rulefit.coef):
        c = float(coef)
        if abs(c) < 1e-6:
            continue
        try:
            es = rule.rule
            for fn, fval in zip(feat_names, fv):
                es = es.replace(fn, str(round(float(fval), 4)))
            if eval(es):
                if (c > 0 and pred == 1) or (c < 0 and pred == 0):
                    fired_for_pred.append((rule.rule, abs(c)))
                else:
                    fired_against.append((rule.rule, abs(c)))
        except Exception:
            pass

    pred_label = "A" if pred == 1 else "B"
    opp_label  = "B" if pred == 1 else "A"

    # Most influential params for this prediction
    param_votes = {}
    for p in params:
        av = float(d.get(f"A_{p}", 0))
        bv = float(d.get(f"B_{p}", 0))
        if av > bv:
            param_votes[p] = "A"
        elif bv > av:
            param_votes[p] = "B"

    if fired_for_pred:
        top = sorted(fired_for_pred, key=lambda x: -x[1])[:2]
        rule_strs = " and ".join(f"`{r}`" for r, _ in top)
        explanation = (
            f"The model chose **Option {pred_label}** because {rule_strs} "
            f"— this pattern in your training decisions consistently predicted {pred_label}."
        )
    else:
        # Fall back to raw parameter advantage
        adv_params = [p for p, v in param_votes.items() if v == pred_label]
        if adv_params:
            p_str = ", ".join(p.replace("_", " ").title() for p in adv_params[:2])
            explanation = (
                f"The model chose **Option {pred_label}** mainly because {pred_label} "
                f"has higher values on: {p_str}. No single rule fired — "
                f"the combined feature weights tipped toward {pred_label}."
            )
        else:
            explanation = (
                f"The model chose **Option {pred_label}** based on the overall "
                f"weighted combination of feature differences. "
                f"No single dominant rule fired for this scenario."
            )

    if fired_against:
        top_ag = sorted(fired_against, key=lambda x: -x[1])[:1]
        explanation += (
            f" However, `{top_ag[0][0]}` pushed toward {opp_label} — "
            f"the model still preferred {pred_label} overall."
        )

    return explanation


def chart_decision_boundary(rulefit, feat_names, params, decisions):
    """
    Decision Boundary Heatmap — where does the model draw the line?
    Uses the 2 most important parameters as axes.
    """
    BG,BG2,BG3,BORDER,TEXT,TEXT_DIM,TEXT_MUTED,A_WIN,B_WIN,COL_A,COL_B,ACCENT = _setup_colours()
    import matplotlib.colors as mcolors
    import matplotlib.patches as mp2

    # Rank params by total rule importance
    param_imp = {p: 0.0 for p in params}
    for rule, coef in zip(rulefit.rules_, rulefit.coef):
        for p in params:
            if p.replace("_", " ").title().lower() in rule.rule.lower():
                param_imp[p] += abs(float(coef)) * float(rule.support)
                break
    top2 = sorted(param_imp.items(), key=lambda x: -x[1])[:2]
    p1, p2 = top2[0][0], top2[1][0]
    pn1    = p1.replace("_", " ").title()
    pn2    = p2.replace("_", " ").title()

    medians = {}
    for p in params:
        vals = [float(d.get(f"A_{p}", 0)) - float(d.get(f"B_{p}", 0))
                for d in decisions]
        medians[p] = float(np.median(vals)) if vals else 0.0

    grid = np.linspace(-6, 6, 28)
    Z    = np.zeros((len(grid), len(grid)))
    for i, v1 in enumerate(grid):
        for j, v2 in enumerate(grid):
            a_d = {p: medians[p] / 2 + 0.001 for p in params}
            b_d = {p: -medians[p] / 2 + 0.001 for p in params}
            a_d[p1] = v1 / 2 + medians.get(p1, 0)
            b_d[p1] = -v1 / 2 + medians.get(p1, 0)
            a_d[p2] = v2 / 2 + medians.get(p2, 0)
            b_d[p2] = -v2 / 2 + medians.get(p2, 0)
            row_tmp = {"choice": "A"}
            for p in params:
                row_tmp[f"A_{p}"] = a_d[p]
                row_tmp[f"B_{p}"] = b_d[p]
            F_pt_df, _ = build_features([row_tmp], params)
            F_pt = F_pt_df.reindex(columns=feat_names, fill_value=0)
            if hasattr(rulefit, "predict_proba"):
                Z[i, j] = rulefit.predict_proba(F_pt.values)[0][1]
            else:
                Z[i, j] = float(rulefit.predict(F_pt.values)[0])

    actual_x, actual_y, actual_c = [], [], []
    for d in decisions:
        actual_x.append(float(d.get(f"A_{p1}", 0)) - float(d.get(f"B_{p1}", 0)))
        actual_y.append(float(d.get(f"A_{p2}", 0)) - float(d.get(f"B_{p2}", 0)))
        actual_c.append(A_WIN if d.get("choice") == "A" else B_WIN)

    a_rgb = mcolors.to_rgb(A_WIN)
    b_rgb = mcolors.to_rgb(B_WIN)
    cmap  = mcolors.LinearSegmentedColormap.from_list("AB", [b_rgb, get_theme()["BG"], a_rgb])

    theme_rcparams()
    fig, ax = plt.subplots(figsize=(8, 6))
    _fbg, _abg, _alt = theme_fig_bg()
    fig.patch.set_facecolor(_fbg); ax.set_facecolor(_abg)
    im = ax.pcolormesh(grid, grid, Z.T, cmap=cmap, vmin=0, vmax=1, alpha=0.85)
    ax.contour(grid, grid, Z.T, levels=[0.5], colors=[get_theme()["TEXT"]], linewidths=[2])
    ax.scatter(actual_x, actual_y, c=actual_c, s=80,
               edgecolors="white", linewidths=1.2, zorder=4)
    ax.axhline(0, color=get_theme()["TEXT_MUTED"], linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color=get_theme()["TEXT_MUTED"], linewidth=0.8, linestyle="--", alpha=0.5)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("P(A preferred)", fontsize=9)
    ax.set_xlabel(f"{pn1}: A − B", fontsize=10, labelpad=8)
    ax.set_ylabel(f"{pn2}: A − B", fontsize=10, labelpad=8)
    ax.set_title("Decision Boundary — where does the model prefer A vs B?",
                 fontsize=12, pad=12, loc="left", fontweight="bold", color=get_theme()["TEXT"])
    ax.legend(handles=[
        mp2.Patch(color=A_WIN, label="You chose A"),
        mp2.Patch(color=B_WIN, label="You chose B"),
    ], fontsize=8, loc="lower right", frameon=True, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(pad=1.2)
    return fig, pn1, pn2


def chart_confidence_strip(rulefit, feat_names, params, decisions):
    """Per-decision confidence strip — one dot per answered scenario."""
    BG,BG2,BG3,BORDER,TEXT,TEXT_DIM,TEXT_MUTED,A_WIN,B_WIN,COL_A,COL_B,ACCENT = _setup_colours()
    import matplotlib.patches as mp3
    conf_vals, colors, labels = [], [], []
    for d in decisions:
        pred, prob_a = _decision_prob(rulefit, feat_names, params, d)
        conf = prob_a if prob_a is not None else (1.0 if pred == 1 else 0.0)
        conf = max(conf, 1.0 - conf)
        actual = 1 if d.get("choice") == "A" else 0
        conf_vals.append(conf)
        colors.append("#22c55e" if conf >= 0.80 else
                      "#f59e0b" if conf >= 0.65 else "#ef4444")
        labels.append("✓" if pred == actual else "✗")

    n = len(conf_vals)
    theme_rcparams()
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), 2.5))
    _fbg2, _abg2, _ = theme_fig_bg(); fig.patch.set_facecolor(_fbg2); ax.set_facecolor(_abg2)
    for i, (cv, col, lbl) in enumerate(zip(conf_vals, colors, labels)):
        ax.scatter(i, 0, s=260, color=col, zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.text(i, 0, lbl, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold", zorder=4)
        ax.text(i, -0.35, str(decisions[i].get("scenario", i+1)),
                ha="center", va="top", fontsize=7, color=get_theme()["TEXT_MUTED"])
    ax.set_xlim(-0.8, max(n - 0.2, 0.2))
    ax.set_ylim(-0.7, 0.6); ax.axis("off")
    ax.set_title("Model confidence per decision  (✓=agreed  ✗=disagreed)",
                 fontsize=11, pad=10, loc="left", fontweight="bold")
    ax.legend(handles=[
        mp3.Patch(color="#22c55e", label="High ≥80%"),
        mp3.Patch(color="#f59e0b", label="Moderate ≥65%"),
        mp3.Patch(color="#ef4444", label="Low <65%"),
    ], fontsize=8, loc="upper right", frameon=True, ncol=3)
    plt.tight_layout(pad=0.8)
    return fig


def chart_parameter_radar(rulefit, params):
    """Parameter importance radar chart."""
    BG,BG2,BG3,BORDER,TEXT,TEXT_DIM,TEXT_MUTED,A_WIN,B_WIN,COL_A,COL_B,ACCENT = _setup_colours()
    import matplotlib.patches as mp4
    param_imp = {p: 0.0 for p in params}
    for rule, coef in zip(rulefit.rules_, rulefit.coef):
        for p in params:
            if p.replace("_", " ").title().lower() in rule.rule.lower():
                param_imp[p] += abs(float(coef)) * float(rule.support)
                break
    total  = sum(param_imp.values()) or 1.0
    imps   = [param_imp[p] / total for p in params]
    labels = [p.replace("_", "\n").title() for p in params]
    N      = len(params)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + \
             [0 / float(N) * 2 * np.pi]
    imps_c = imps + imps[:1]

    theme_rcparams()
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    _rfbg, _rabg, _ = theme_fig_bg()
    fig.patch.set_facecolor(_rfbg); ax.set_facecolor(_rabg)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color=get_theme()["TEXT_MUTED"])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, color=get_theme()["TEXT"], fontweight="600")
    ax.tick_params(pad=12)
    ax.spines["polar"].set_color(theme_border_color())
    ax.grid(color=theme_border_color(), linewidth=0.8)
    ax.fill(angles, imps_c, color=A_WIN, alpha=0.25)
    ax.plot(angles, imps_c, color=A_WIN, linewidth=2.5)
    ax.scatter(angles[:-1], imps, s=80, color=A_WIN, zorder=4,
               edgecolors="white", linewidths=1.5)
    for angle, imp in zip(angles[:-1], imps):
        ax.text(angle, imp + 0.1, f"{imp*100:.0f}%",
                ha="center", va="center", fontsize=8,
                color=get_theme()["TEXT"], fontweight="600")
    ax.set_title("Parameter Importance", fontsize=12, pad=20,
                 fontweight="bold", color=get_theme()["TEXT"])
    plt.tight_layout(pad=1.0)
    return fig

def dark_rcparams():
    """Apply dark chart theme using current dark palette."""
    T = THEMES["dark"]
    plt.rcParams.update({
        "font.family":       "monospace",
        "text.color":        T["TEXT"],
        "axes.labelcolor":   T["TEXT"],
        "xtick.color":       T["TEXT_DIM"],
        "ytick.color":       T["TEXT"],
        "axes.facecolor":    T["BG2"],
        "figure.facecolor":  T["BG"],
        "axes.edgecolor":    T["BORDER"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "grid.color":        T["BORDER"],
        "grid.alpha":        0.4,
        "grid.linewidth":    0.5,
    })


def light_rcparams():
    """Apply light chart theme using current light palette."""
    T = THEMES["light"]
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "text.color":        T["TEXT"],
        "axes.labelcolor":   T["TEXT_DIM"],
        "xtick.color":       T["TEXT_DIM"],
        "ytick.color":       T["TEXT_DIM"],
        "axes.facecolor":    T["BG"],
        "figure.facecolor":  T["BG"],
        "axes.edgecolor":    T["BORDER"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def theme_rcparams():
    """Apply chart theme matching the currently active app theme."""
    mode = st.session_state.get("theme_mode", "light")
    if mode == "dark":
        dark_rcparams()
    else:
        light_rcparams()


def theme_fig_bg():
    """Return (fig_bg, ax_bg, alt_bg) for current theme."""
    T = get_theme()
    return T["BG"], T["BG2"], T["BG3"]


def theme_border_color():
    return get_theme()["BORDER"]


def chart_rulefit_coef(rules_df):
    theme_rcparams()
    BG,BG2,BG3,BORDER,TEXT,TEXT_DIM,TEXT_MUTED,A_WIN,B_WIN,COL_A,COL_B,ACCENT = _setup_colours()
    n   = len(rules_df)
    _fbg, _abg, _alt = theme_fig_bg()
    fig, ax = plt.subplots(figsize=(12, max(4, n * 0.7)), facecolor=_fbg)
    ax.set_facecolor(_abg)
    for i in range(n):
        ax.axhspan(i - 0.45, i + 0.45,
                   color=_alt if i % 2 == 0 else _abg, zorder=0)
    ax.axvline(0, color=theme_border_color(), linewidth=1.5, zorder=2)
    for i, (_, row) in enumerate(rules_df.iterrows()):
        c     = row["coef"]
        color = A_WIN if c > 0 else B_WIN
        ax.barh(i, c, height=0.55, color=color, alpha=0.85, zorder=3, linewidth=0)
        ha   = "left"  if c > 0 else "right"
        xpos = c + (0.012 if c > 0 else -0.012)
        ax.text(xpos, i, row["rule"], va="center", ha=ha,
                fontsize=8, color=get_theme()["TEXT"], zorder=5)
        ax.text(c, i - 0.28, f"{c:+.3f}", va="top", ha="center",
                fontsize=6.5, color=color, alpha=0.9, zorder=5)
    ax.set_yticks([])
    cab = float(rules_df["coef"].abs().max())
    ax.set_xlim(-cab * 1.6, cab * 1.6)
    ax.set_xlabel("← Favours A         Coefficient         Favours B →",
                  fontsize=9, color=get_theme()["TEXT_DIM"], labelpad=8)
    ax.set_title("RuleFit · Rule Coefficients",
                 fontsize=13, color=get_theme()["TEXT"], pad=14, fontweight="bold", loc="left")
    ax.legend(handles=[
        mpatches.Patch(color=A_WIN, label="Predicts A preferred"),
        mpatches.Patch(color=B_WIN, label="Predicts B preferred"),
    ], fontsize=8, facecolor=get_theme()["BG2"], edgecolor=get_theme()["BORDER"],
       loc="lower right")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(theme_border_color())
    plt.tight_layout(pad=1.2)
    return fig


def chart_rulefit_lollipop(rules_df):
    theme_rcparams()
    BG,BG2,BG3,BORDER,TEXT,TEXT_DIM,TEXT_MUTED,A_WIN,B_WIN,COL_A,COL_B,ACCENT = _setup_colours()
    n       = len(rules_df)
    max_imp = float(rules_df["importance"].max())
    _fbg, _abg, _alt = theme_fig_bg()
    fig, ax = plt.subplots(figsize=(12, max(4, n * 0.7)), facecolor=_fbg)
    ax.set_facecolor(_abg)
    for i in range(n):
        ax.axhspan(i - 0.45, i + 0.45,
                   color=_alt if i % 2 == 0 else _abg, zorder=0)
    for i, (_, row) in enumerate(rules_df.iterrows()):
        imp   = row["importance"]
        color = A_WIN if row["coef"] > 0 else B_WIN
        ax.plot([0, imp], [i, i], color=color, alpha=0.4, linewidth=2, zorder=2)
        ax.scatter(imp, i, s=160, color=color, zorder=4,
                   edgecolors="white", linewidths=0.8)
        ax.text(max_imp * 1.03, i, f"support {row['support']:.0%}",
                va="center", ha="left", fontsize=7, color=get_theme()["TEXT_DIM"])
        ax.text(-max_imp * 0.02, i, row["rule"],
                va="center", ha="right", fontsize=8, color=get_theme()["TEXT"])
    ax.set_yticks([])
    ax.set_xlim(-max_imp * 0.85, max_imp * 1.35)
    ax.axvline(0, color=theme_border_color(), linewidth=1)
    ax.set_xlabel("Importance  ( |coef| × support )",
                  fontsize=9, color=get_theme()["TEXT_DIM"], labelpad=8)
    ax.set_title("RuleFit · Rule Importance Ranking",
                 fontsize=13, color=get_theme()["TEXT"], pad=14, fontweight="bold", loc="left")
    ax.legend(handles=[
        mpatches.Patch(color=A_WIN, label="Predicts A preferred"),
        mpatches.Patch(color=B_WIN, label="Predicts B preferred"),
    ], fontsize=8, facecolor=get_theme()["BG2"], edgecolor=get_theme()["BORDER"],
       loc="lower right")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(theme_border_color())
    plt.tight_layout(pad=1.2)
    return fig


def chart_rulefit_bubble(rules_df):
    theme_rcparams()
    BG,BG2,BG3,BORDER,TEXT,TEXT_DIM,TEXT_MUTED,A_WIN,B_WIN,COL_A,COL_B,ACCENT = _setup_colours()
    top = rules_df.head(10).copy().reset_index(drop=True)
    _bbg, _babg, _ = theme_fig_bg()
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(_bbg); ax.set_facecolor(_babg)
    ax.grid(True, color=theme_border_color(), linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(theme_border_color())
    med_x    = top["support"].median()
    med_y    = top["coef"].abs().median()
    ylim_max = float(top["coef"].abs().max()) * 1.4
    xlim_max = float(top["support"].max()) * 1.3
    ax.axvline(med_x, color=theme_border_color(), linewidth=1, linestyle="--", zorder=1)
    ax.axhline(med_y, color=theme_border_color(), linewidth=1, linestyle="--", zorder=1)
    for xp, yp, label in [
        (0.01,        ylim_max * 0.93, "rare + strong"),
        (med_x + 0.01, ylim_max * 0.93, "common + strong ★"),
        (0.01,        ylim_max * 0.05, "rare + weak"),
        (med_x + 0.01, ylim_max * 0.05, "common + weak"),
    ]:
        ax.text(xp, yp, label, fontsize=8, color=get_theme()["TEXT_MUTED"],
                style="italic", va="bottom")
    for i, (_, row) in enumerate(top.iterrows()):
        color = A_WIN if row["coef"] > 0 else B_WIN
        ax.scatter(row["support"], abs(row["coef"]),
                   s=row["importance"] * 1000 + 10, color=color, alpha=0.75,
                   edgecolors="white", linewidths=1.5, zorder=3)
        ax.text(row["support"], abs(row["coef"]), str(i + 1),
                ha="center", va="center", fontsize=8,
                fontweight="bold", color="white", zorder=4)
    ax.set_xlabel("Coverage — fraction of decisions this rule applies to",
                  fontsize=9, color=get_theme()["TEXT_DIM"], labelpad=8)
    ax.set_ylabel("Strength — |coefficient|",
                  fontsize=9, color=get_theme()["TEXT_DIM"], labelpad=8)
    ax.set_title(f"Top {len(top)} rules: coverage vs strength",
                 fontsize=12, color=get_theme()["TEXT"], pad=12,
                 loc="left", fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlim(left=0, right=xlim_max)
    ax.set_ylim(bottom=0, top=ylim_max)
    sh = [
        plt.scatter([], [], s=0.05*5000+120, color=get_theme()["TEXT_MUTED"],
                    alpha=0.6, edgecolors="white", label="low importance"),
        plt.scatter([], [], s=0.20*5000+120, color=get_theme()["TEXT_MUTED"],
                    alpha=0.6, edgecolors="white", label="high importance"),
        plt.scatter([], [], s=120, color=A_WIN, alpha=0.75,
                    edgecolors="white", label="A preferred"),
        plt.scatter([], [], s=120, color=B_WIN, alpha=0.75,
                    edgecolors="white", label="B preferred"),
    ]
    ax.legend(handles=sh, title="size = importance", fontsize=8, title_fontsize=8,
              frameon=True, framealpha=0.9, edgecolor=theme_border_color(), loc="upper right")
    plt.tight_layout(pad=1.2)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Theme toggle ──────────────────────────────────────────────────────────
    _mode = st.session_state.get("theme_mode", "light")
    _icon = "🌙" if _mode == "light" else "☀️"
    _label = f"{_icon} {'Dark' if _mode == 'light' else 'Light'} mode"
    if st.button(_label, key="theme_toggle", use_container_width=False):
        st.session_state["theme_mode"] = "dark" if _mode == "light" else "light"
        st.rerun()

    st.markdown("## 🫘 Preference Portal")
    st.caption("SURA 2026 · IIT Delhi")


    st.divider()

    # Login
    st.markdown("**Login**")
    if not st.session_state.logged_in:
        uname_input = st.text_input(
            "Username", placeholder="e.g. participant_01",
            label_visibility="collapsed", key="uname_input"
        )
        if st.button("Enter →", type="primary",
                     use_container_width=True, key="login_btn"):
            raw = uname_input.strip()
            if len(raw) < 2:
                st.error("Username must be at least 2 characters.")
            else:
                users = load_users()
                st.session_state.username  = raw
                st.session_state.logged_in = True
                if raw in users:
                    ud = users[raw]
                    st.session_state.sc_index  = ud.get("sc_index",  0)
                    st.session_state.decisions = ud.get("decisions", [])
                else:
                    st.session_state.sc_index  = 0
                    st.session_state.decisions = []
                    users[raw] = {"sc_index": 0, "decisions": []}
                    save_users(users)
                st.session_state.page = "scenarios"
                st.rerun()
    else:
        st.markdown(f"**👤 {st.session_state.username}**")
        n_total = len(st.session_state.scenarios)
        st.progress(min(st.session_state.sc_index / max(n_total, 1), 1.0))
        st.caption(
            f"Scenario {min(st.session_state.sc_index + 1, n_total)}/{n_total}"
        )
        st.divider()
        for label, key in [
            ("🧪 Scenarios",       "scenarios"),
            ("🌳 Model & Results", "model"),
        ]:
            if st.button(
                label, use_container_width=True, key=f"nav_{key}",
                type="primary" if st.session_state.page == key else "secondary"
            ):
                st.session_state.page = key
                st.rerun()
        st.divider()
        if st.button("🚪 Log out", use_container_width=True, key="logout_btn"):
            save_session()
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — login gate
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.title("🫘 Preference Elicitation Portal")
    st.info("Enter your username in the sidebar to begin.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "scenarios":
    BG,BG2,BG3,BORDER,TEXT,TEXT_DIM,TEXT_MUTED,A_WIN,B_WIN,COL_A,COL_B,ACCENT = _setup_colours()

    n_total = len(st.session_state.scenarios)

    # All done
    if st.session_state.sc_index >= n_total:
        st.success(f"✅ All {n_total} scenarios complete!")
        n_ans = len([d for d in st.session_state.decisions
                     if d.get("choice") in ("A", "B")])
        st.markdown(f"**{n_ans}** answered · **{n_total - n_ans}** skipped.")
        if st.button("🌳 See Model & Results",
                     type="primary", use_container_width=False):
            st.session_state.page = "model"
            st.rerun()
        st.stop()

    current_idx = st.session_state.sc_index
    sc          = st.session_state.scenarios[current_idx]
    params      = st.session_state.params

    st.markdown(f"### Scenario {current_idx + 1} of {n_total}")
    st.progress(current_idx / n_total)

    # Difficulty badge
    diff_score = scenario_difficulty(sc, params)
    diff_lbl, diff_col = difficulty_label(diff_score)
    st.markdown(
        f"<span style='background:{diff_col}33;color:{diff_col};"
        f"border:1px solid {diff_col};border-radius:20px;"
        f"padding:3px 12px;font-size:12px;font-weight:600'>"
        f"{diff_lbl} &nbsp;·&nbsp; difficulty {diff_score:.2f}</span>",
        unsafe_allow_html=True,
    )
    if diff_score >= 0.70:
        st.caption("These two options are very similar — take your time.")

    st.markdown("## Which option should be preferred?")
    st.divider()

    col_a, col_b = st.columns(2, gap="large")
    for col, label, data, color in [
        (col_a, "A", sc["A"], COL_A),
        (col_b, "B", sc["B"], COL_B),
    ]:
        with col:
            st.markdown(
                f"<div style='border:2px solid {color};border-radius:10px;background:{get_theme()['CARD_BG']};"
                f"padding:1.2rem'>"
                f"<p style='color:{color};font-weight:700;font-size:15px;"
                f"margin:0 0 12px'>OPTION {label}</p>",
                unsafe_allow_html=True,
            )
            for p in params:
                st.markdown(
                    f"<div style='margin-bottom:10px'>"
                    f"<div style='font-size:11px;color:{get_theme()['TEXT_DIM']};text-transform:uppercase;"
                    f"letter-spacing:.05em'>{p.replace('_', ' ')}</div>"
                    f"<div style='font-size:26px;font-weight:600;"
                    f"font-family:monospace'>{data[p]:g}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        if st.button("✅  Option A is better", use_container_width=True,
                     type="primary", key=f"btn_A_{current_idx}"):
            record_decision("A")
            st.rerun()
    with c2:
        if st.button("✅  Option B is better", use_container_width=True,
                     type="primary", key=f"btn_B_{current_idx}"):
            record_decision("B")
            st.rerun()
    with c3:
        if st.button("⏭ Skip", use_container_width=True,
                     key=f"btn_skip_{current_idx}"):
            record_decision("skip")
            st.rerun()

    n_done = len([d for d in st.session_state.decisions
                  if d.get("choice") in ("A", "B")])
    st.caption(f"📝 {n_done} answered so far")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL & RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "model":
    BG,BG2,BG3,BORDER,TEXT,TEXT_DIM,TEXT_MUTED,A_WIN,B_WIN,COL_A,COL_B,ACCENT = _setup_colours()
    st.markdown("## 🌳 Model & Results")

    params    = st.session_state.params
    decisions = [d for d in st.session_state.decisions
                 if d.get("choice") in ("A", "B")]

    if len(decisions) < 6:
        st.warning(
            f"Need at least 6 answered scenarios (you have {len(decisions)})."
        )
        if st.button("← Back to Scenarios"):
            st.session_state.page = "scenarios"
            st.rerun()
        st.stop()

    # Train RuleFit
    decisions_json = json.dumps(decisions)
    with st.spinner("Training model…"):
        rulefit, rules_df, rf_stats, feat_names, rf_error = train_rulefit(
            decisions_json, params
        )

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decisions answered", len(decisions))
    c2.metric("Parameters",         len(params))
    if rf_error is None and rf_stats:
        c3.metric("Model accuracy",  f"{rf_stats['acc']*100:.0f}%")
        c4.metric("Symmetry",        f"{rf_stats['sym']*100:.0f}%",
                  help="% of predictions that correctly flip when A and B are swapped")
    st.divider()

    tab0, tab2, tab3, tab4, tab5 = st.tabs(["📄 Model Card", "📊 Charts", "🎯 Predict New Pair", "🔍 Examples", "🧠 Analysis"])


    # ── Tab 0: Model Card ─────────────────────────────────────────────────────
    with tab0:
        st.markdown("### 📄 Model Documentation")
        if rf_error:
            st.error(rf_error)
        elif rf_stats is None:
            st.warning("No model trained yet.")
        else:
            col_l, col_r = st.columns([2, 1])
            with col_l:
                acc_pct = rf_stats["acc"] * 100
                sym_pct = rf_stats["sym"] * 100
                st.markdown(f"""
<div style='background:{get_theme()["INFO"]};border:1px solid {get_theme()["INFO_BORDER"]};
border-radius:10px;padding:18px 20px;'>
<h4 style='margin:0 0 12px;color:{get_theme()["ACCENT"]}'>Model Summary</h4>
<table style='width:100%;border-collapse:collapse;font-size:14px'>
<tr><td style='padding:5px 0;color:{get_theme()["TEXT_DIM"]};width:45%'>Trained on</td>
    <td style='padding:5px 0;font-weight:600;color:{get_theme()["TEXT"]}'>{rf_stats["n_decisions"]} decisions
    by {st.session_state.username}</td></tr>
<tr><td style='padding:5px 0;color:{get_theme()["TEXT_DIM"]}'>Parameters</td>
    <td style='padding:5px 0;font-weight:600;color:{get_theme()["TEXT"]}'>
    {", ".join(p.replace("_"," ").title() for p in params)}</td></tr>
<tr><td style='padding:5px 0;color:{get_theme()["TEXT_DIM"]}'>Rules learned</td>
    <td style='padding:5px 0;font-weight:600;color:{get_theme()["TEXT"]}'>{rf_stats["n_rules"]}</td></tr>
<tr><td style='padding:5px 0;color:{get_theme()["TEXT_DIM"]}'>Top parameter</td>
    <td style='padding:5px 0;font-weight:600;color:{get_theme()["TEXT"]}'>{rf_stats["top_param"]}</td></tr>
<tr><td style='padding:5px 0;color:{get_theme()["TEXT_DIM"]}'>Regularisation α</td>
    <td style='padding:5px 0;font-weight:600;color:{get_theme()["TEXT"]}'>{rf_stats["alpha_used"]}</td></tr>
</table></div>""", unsafe_allow_html=True)
            with col_r:
                acc_col = "#22c55e" if acc_pct >= 75 else "#f59e0b" if acc_pct >= 60 else "#ef4444"
                st.markdown(f"""
<div style='border:1px solid {get_theme()["CARD_BORDER"]};background:{get_theme()["CARD_BG"]};border-radius:10px;
padding:16px;text-align:center;margin-bottom:8px'>
<div style='font-size:12px;color:{get_theme()["TEXT_DIM"]}'>Model Accuracy</div>
<div style='font-size:38px;font-weight:700;color:{acc_col}'>{acc_pct:.0f}%</div>
<div style='font-size:11px;color:{get_theme()["TEXT_MUTED"]}'>on training decisions</div>
</div>
<div style='border:1px solid {get_theme()["CARD_BORDER"]};background:{get_theme()["CARD_BG"]};border-radius:10px;
padding:16px;text-align:center'>
<div style='font-size:12px;color:{get_theme()["TEXT_DIM"]}'>Symmetry</div>
<div style='font-size:38px;font-weight:700;color:{get_theme()["ACCENT"]}'>{sym_pct:.0f}%</div>
<div style='font-size:11px;color:{get_theme()["TEXT_MUTED"]}'>swap A↔B → prediction flips</div>
</div>""", unsafe_allow_html=True)

            st.divider()
            n = rf_stats["n_decisions"]
            if n < 10:
                st.warning(f"⚠️ Only {n} decisions — model may not be reliable. Aim for 15–20.")
            elif n < 15:
                st.info(f"ℹ️ {n} decisions — more scenarios will improve accuracy.")
            else:
                st.success(f"✅ {n} decisions — sufficient for a reliable model.")

            st.markdown("#### What the model learned")
            if rules_df is not None and len(rules_df) > 0:
                # Build rule-based fallback text
                fallback_lines = []
                for i, (_, row) in enumerate(rules_df.iterrows()):
                    pref = "A" if row["coef"] > 0 else "B"
                    conf = ("strongly" if abs(row["coef"]) > 1.0
                            else "moderately" if abs(row["coef"]) > 0.4
                            else "weakly")
                    fallback_lines.append(
                        f"{i+1}. When `{row['rule']}` → prefer {pref} "
                        f"({conf}, covers {row['support']:.0%} of cases)"
                    )
                fallback_text = "\n\n".join(fallback_lines)

                with st.spinner("Generating summary…"):
                    llm_text = explain_model_learned_llm(
                        rules_df, params, rf_stats, fallback_text
                    )

                if llm_text == fallback_text:
                    # Show rule-based fallback
                    for line in fallback_lines:
                        st.markdown(line)
                else:
                    st.markdown(llm_text)
                    with st.expander("📋 View raw rules"):
                        for line in fallback_lines:
                            st.markdown(line)

            st.divider()
            st.markdown("#### ⚠️ Limitations")
            st.markdown("""
- Reflects **your** preferences only — not a universal standard
- Training accuracy ≠ accuracy on genuinely new unseen cases
- Assumes your preferences are consistent across scenarios
- RuleFit may suppress weak but genuine preferences via regularisation
- With <20 scenarios some parameter interactions may not be captured
""")

    # ── Tab 2: Charts ─────────────────────────────────────────────────────────
    with tab2:
        if rf_error or rules_df is None or len(rules_df) == 0:
            st.warning("RuleFit unavailable — insufficient data or Python version.")
        else:
            # Chart 1: Rule Coefficients
            st.markdown("#### Chart 1 — Rule Coefficients")
            st.caption(
                "Each bar is one learned rule. Length = how strongly it predicts. "
                "Green bars push toward A, red bars toward B."
            )
            fig1 = chart_rulefit_coef(rules_df)
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

            st.divider()

            # Chart 2: Lollipop + Confidence strip
            st.markdown("#### Chart 2 — Rule Importance + Decision Confidence")
            st.caption(
                "**Top:** Rule importance ranking — dot size = how much each rule matters. "
                "**Bottom:** One dot per decision you made — "
                "colour = how confident the model was. ✓ = agreed with you, ✗ = disagreed."
            )
            fig2 = chart_rulefit_lollipop(rules_df)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
            with st.spinner("Computing confidence per decision…"):
                fig_strip = chart_confidence_strip(rulefit, feat_names, params, decisions)
            st.pyplot(fig_strip, use_container_width=True)
            plt.close(fig_strip)

            st.divider()

            # Chart 3: Decision boundary heatmap
            st.markdown("#### Chart 3 — Decision Boundary Heatmap")
            st.caption(
                "Shows what the model predicts across every combination of the two "
                "most important parameters. **Green = A preferred, Red = B preferred.** "
                "The black line is the decision boundary. "
                "Coloured dots = your actual answered decisions."
            )
            with st.spinner("Building decision boundary…"):
                fig3, pn1, pn2 = chart_decision_boundary(
                    rulefit, feat_names, params, decisions
                )
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)
            st.caption(
                f"X-axis: {pn1} difference (A minus B). "
                f"Y-axis: {pn2} difference (A minus B). "
                "All other parameters held at their median difference."
            )

            st.divider()

            # Chart 4: Parameter radar
            st.markdown("#### Chart 4 — Parameter Importance Radar")
            st.caption(
                "How much each parameter drives the model's decisions. "
                "A larger area on one axis = that parameter dominates. "
                "A balanced shape = the model weighs all parameters roughly equally."
            )
            fig4 = chart_parameter_radar(rulefit, params)
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)

    # ── Tab 3: Predict new pair ───────────────────────────────────────────────
    with tab3:
        st.markdown("#### Predict outcome for a new patient pair")
        st.caption(
            "Enter values for two patients — the model predicts which is preferred."
        )

        if rf_error or rulefit is None:
            st.warning("RuleFit unavailable. Run from imodels_env to use predictions.")
            st.stop()

        # ── Inputs ────────────────────────────────────────────────────────────
        col_a2, col_b2 = st.columns(2)
        patient_a_input = {}
        patient_b_input = {}
        with col_a2:
            st.markdown("**Patient A**")
            for p in params:
                patient_a_input[p] = st.number_input(
                    p.replace("_", " ").title(),
                    key=f"pred_a_{p}", value=0.0, step=1.0
                )
        with col_b2:
            st.markdown("**Patient B**")
            for p in params:
                patient_b_input[p] = st.number_input(
                    p.replace("_", " ").title(),
                    key=f"pred_b_{p}", value=0.0, step=1.0
                )

        # ── Helper: build feature row ─────────────────────────────────────────
        def _make_F(pa, pb):
            """Build single-row feature DataFrame using the new feature library."""
            row = {"choice": "A"}
            for p in params:
                row[f"A_{p}"] = float(pa[p])
                row[f"B_{p}"] = float(pb[p])
            F_tmp, _ = build_features([row], params)
            return F_tmp.reindex(columns=feat_names, fill_value=0)


        # ── Predict button — results shown after click ────────────────────────
        predict_clicked = st.button(
            "🎯 Predict", type="primary",
            use_container_width=False, key="predict_btn"
        )

        # Store prediction in session state so results survive rerenders
        # (slider moves, selectbox changes) without requiring re-click
        if predict_clicked:
            F_tmp = _make_F(patient_a_input, patient_b_input)
            p_tmp = rulefit.predict(F_tmp.values)[0]
            pr_tmp = (rulefit.predict_proba(F_tmp.values)[0]
                      if hasattr(rulefit, "predict_proba") else None)
            st.session_state["pred_result"] = {
                "pred":  p_tmp,
                "prob":  pr_tmp.tolist() if pr_tmp is not None else None,
                "pa":    dict(patient_a_input),
                "pb":    dict(patient_b_input),
            }

        # Use stored result — only render if prediction exists
        _res = st.session_state.get("pred_result")
        if _res is None:
            st.info("Enter patient values above and click 🎯 Predict to see results.")
        else:
            F_new   = _make_F(_res["pa"], _res["pb"])
            pred    = _res["pred"]
            prob    = _res["prob"]
            winner  = "A preferred" if pred == 1 else "B preferred"
            w_color = COL_A if pred == 1 else COL_B
            p_a = prob[1] if prob is not None else (1.0 if pred == 1 else 0.0)
            p_b = 1.0 - p_a
            conf_val = max(p_a, p_b)
            if conf_val >= 0.80:
                conf_label, conf_color, conf_icon = "High confidence",           "#22c55e", "🟢"
            elif conf_val >= 0.65:
                conf_label, conf_color, conf_icon = "Moderate confidence",       "#f59e0b", "🟡"
            else:
                conf_label, conf_color, conf_icon = "Low confidence — near tie", "#ef4444", "🔴"

            st.divider()
            st.markdown(
                f"<div style='border:2px solid {w_color};border-radius:10px;"
                f"padding:16px;text-align:center;margin:12px 0'>"
                f"<span style='font-size:22px;font-weight:700;color:{w_color}'>"
                f"{winner}</span><br>"
                f"<span style='font-size:13px;color:{conf_color};font-weight:600'>"
                f"{conf_icon} {conf_label}: {conf_val:.1%}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if conf_val < 0.65:
                st.warning(
                    "⚠️ This prediction is uncertain — A and B are nearly equal "
                    "on your learned preferences."
                )


            # Gauge bar
            theme_rcparams()
            fig_g, ax_g = plt.subplots(figsize=(8, 1.6))
            ax_g.set_xlim(0,1); ax_g.set_ylim(0,1); ax_g.axis("off")
            ax_g.barh(0.5, 1.0, height=0.4, color=get_theme()["BG3"], left=0, zorder=1)
            ax_g.barh(0.5, p_a, height=0.4, color=A_WIN, left=0, zorder=2, alpha=0.9)
            ax_g.barh(0.5, p_b, height=0.4, color=B_WIN, left=p_a, zorder=2, alpha=0.9)
            if p_a > 0.08:
                ax_g.text(p_a/2, 0.5, f"A  {p_a:.0%}", ha="center", va="center",
                          fontsize=11, color="white", fontweight="bold", zorder=3)
            if p_b > 0.08:
                ax_g.text(p_a+p_b/2, 0.5, f"B  {p_b:.0%}", ha="center", va="center",
                          fontsize=11, color="white", fontweight="bold", zorder=3)
            plt.tight_layout(pad=0.3)
            st.pyplot(fig_g, use_container_width=True)
            plt.close(fig_g)

            # ── Parameter breakdown table ──────────────────────────────────────────
            st.markdown("**Parameter breakdown:**")
            rows_bd = []
            for p in params:
                a = float(patient_a_input[p]); b = float(patient_b_input[p])
                rows_bd.append({
                    "Parameter": p.replace("_", " ").title(),
                    "A value":   a, "B value": b,
                    "A − B":     round(a - b, 2),
                    "Advantage": "A" if a > b else ("B" if b > a else "Tie"),
                })
            st.dataframe(pd.DataFrame(rows_bd), use_container_width=True, hide_index=True)

            # ── Rules that fired ──────────────────────────────────────────────────
            if rules_df is not None and len(rules_df) > 0:
                fv_arr = F_new.values[0]
                fired  = []
                for _, row in rules_df.iterrows():
                    try:
                        es = row["rule"]
                        for fn, fval in zip(feat_names, fv_arr):
                            es = es.replace(fn, str(round(float(fval), 4)))
                        if eval(es):
                            fired.append(row)
                    except Exception:
                        pass
                if fired:
                    st.markdown("**Rules that fired for this pair:**")
                    for row in fired:
                        direction = "→ A preferred" if row["coef"] > 0 else "→ B preferred"
                        color = COL_A if row["coef"] > 0 else COL_B
                        st.markdown(
                            f"<div style='border-left:3px solid {color};"
                            f"padding:6px 14px;margin:4px 0;"
                            f"background:{get_theme()['BG2']};border-radius:4px;font-size:13px'>"
                            f"<code>{row['rule']}</code> "
                            f"<span style='color:{color};font-weight:600'>{direction}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No individual rules fired — prediction based on combined model score.")

            st.divider()

            # ── Counterfactual ────────────────────────────────────────────────────
            st.markdown("#### 🔄 What would flip this prediction?")
            st.caption("Minimum single-parameter change to A that reverses the outcome.")
            flips = []
            for p in params:
                orig = float(patient_a_input[p])
                for delta in [1,-1,2,-2,3,-3,5,-5,10,-10]:
                    test_a = {k: float(v) for k, v in patient_a_input.items()}
                    test_a[p] = orig + delta
                    F2 = _make_F(test_a, patient_b_input)
                    if rulefit.predict(F2.values)[0] != pred:
                        ds = f"+{delta}" if delta > 0 else str(delta)
                        fc = COL_A if rulefit.predict(F2.values)[0] == 1 else COL_B
                        np_lbl = "A preferred" if rulefit.predict(F2.values)[0] == 1 else "B preferred"
                        flips.append({"param": p.replace("_"," ").title(),
                                      "ds": ds, "fc": fc, "np_lbl": np_lbl})
                        break
            if flips:
                for fl in flips:
                    fl_fc     = fl["fc"]
                    fl_param  = fl["param"]
                    fl_ds     = fl["ds"]
                    fl_np_lbl = fl["np_lbl"]
                    st.markdown(
                        f"<div style='border-left:3px solid {fl_fc};"
                        f"padding:6px 14px;margin:4px 0;"
                        f"background:{get_theme()['BG2']};border-radius:4px;font-size:13px'>"
                        f"If <b style='color:{get_theme()['TEXT']}'>{fl_param}</b>"
                        f" of A changes by <b style='color:{get_theme()['TEXT']}'>{fl_ds}</b> → "
                        f"<span style='color:{fl_fc};font-weight:600'>{fl_np_lbl}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No change within ±10 flips this prediction — model is confident.")

            st.divider()

            # ── Parameter contributions waterfall ─────────────────────────────────
            st.markdown("#### 📊 Parameter contributions")
            st.caption("How much each parameter pushes toward A (green) or B (red).")
            param_contribs = {p.replace("_"," ").title(): 0.0 for p in params}
            for rule, coef in zip(rulefit.rules_, rulefit.coef):
                c = float(coef)
                if abs(c) < 1e-6:
                    continue
                for p in params:
                    pn = p.replace("_"," ").title()
                    if pn.lower() in rule.rule.lower():
                        try:
                            es = rule.rule
                            for fn, fval in zip(feat_names, F_new.values[0]):
                                es = es.replace(fn, str(round(float(fval), 4)))
                            if eval(es):
                                param_contribs[pn] += c
                        except Exception:
                            pass
                        break
            if all(v == 0.0 for v in param_contribs.values()):
                ca = np.array(rulefit.coef)
                fv = F_new.values[0][:len(ca)]
                for fn, ct in zip(feat_names[:len(ca)], ca * fv):
                    pn = fn.split(":")[0].strip()
                    param_contribs[pn] = param_contribs.get(pn, 0.0) + float(ct)
            pc_s  = sorted(param_contribs.items(), key=lambda x: x[1])
            nm_pc = [x[0] for x in pc_s]; vl_pc = [x[1] for x in pc_s]
            cl_pc = [A_WIN if v >= 0 else B_WIN for v in vl_pc]
            theme_rcparams()
            _wfbg, _wabg, _walt = theme_fig_bg()
            fig_wf, ax_wf = plt.subplots(
                figsize=(9, max(3, len(nm_pc)*0.65)), facecolor=_wfbg)
            ax_wf.set_facecolor(_wabg)
            for i in range(len(nm_pc)):
                ax_wf.axhspan(i-0.4, i+0.4,
                              color=_walt if i%2==0 else _wabg, zorder=0)
            ax_wf.axvline(0, color=theme_border_color(), linewidth=1.5, zorder=2)
            for i, (nm, vl, cl) in enumerate(zip(nm_pc, vl_pc, cl_pc)):
                ax_wf.barh(i, vl, height=0.5, color=cl, alpha=0.85, zorder=3, linewidth=0)
                ha = "left" if vl >= 0 else "right"
                ax_wf.text(vl+(0.008 if vl>=0 else -0.008), i, nm,
                           va="center", ha=ha, fontsize=9, color=get_theme()["TEXT"], zorder=4)
            ax_wf.set_yticks([])
            ax_wf.set_xlabel(
                "← pushes toward B          Contribution          pushes toward A →",
                fontsize=8, color=get_theme()["TEXT_DIM"], labelpad=6)
            ax_wf.spines["left"].set_visible(False)
            ax_wf.spines["bottom"].set_color(theme_border_color())
            plt.tight_layout(pad=1.0)
            st.pyplot(fig_wf, use_container_width=True)
            plt.close(fig_wf)

            st.divider()

            # ── Sensitivity slider ─────────────────────────────────────────────────
            # Lives OUTSIDE the button block so it persists on re-render
            st.markdown("#### 🎚 Sensitivity analysis")
            st.caption(
                "Select a parameter and move the slider to see how changing "
                "Patient A's value affects the prediction in real time."
            )
            sens_param = st.selectbox(
                "Parameter to vary",
                options=params,
                format_func=lambda p: p.replace("_"," ").title(),
                key="sens_param_sel",
            )
            curr_val = float(patient_a_input.get(sens_param, 0))
            sens_val = st.slider(
                f"{sens_param.replace('_',' ').title()} of A",
                min_value=float(curr_val - 10),
                max_value=float(curr_val + 10),
                value=float(curr_val),
                step=1.0,
                key="sens_slider",
            )

            # Vectorised sweep — build all 21 rows at once, single predict_proba call
            sweep_vals = np.arange(curr_val - 10, curr_val + 11, 1.0)
            sweep_rows = []
            for sv in sweep_vals:
                test_a3 = {k: float(v) for k, v in patient_a_input.items()}
                test_a3[sens_param] = sv
                sweep_rows.append(_make_F(test_a3, patient_b_input).values[0])
            F_sweep = np.vstack(sweep_rows)
            if hasattr(rulefit, "predict_proba"):
                sweep_probs = rulefit.predict_proba(F_sweep)[:, 1].tolist()
            else:
                sweep_probs = rulefit.predict(F_sweep).astype(float).tolist()

            ci_sweep = int(np.argmin(np.abs(sweep_vals - sens_val)))
            cp_a     = sweep_probs[ci_sweep]

            theme_rcparams()
            fig_s, ax_s = plt.subplots(figsize=(9, 3.5))
            _sbg, _sabg, _ = theme_fig_bg()
            _sbg, _sabg, _ = theme_fig_bg()
            fig_s.patch.set_facecolor(_sbg); ax_s.set_facecolor(_sabg)
            ax_s.axhline(0.5, color=get_theme()["TEXT_MUTED"], linewidth=1, linestyle="--", alpha=0.7)
            ax_s.axvline(sens_val, color="#f59e0b", linewidth=2, linestyle="--",
                         label=f"Current: {sens_val:.0f}")
            ax_s.fill_between(sweep_vals, sweep_probs, 0.5,
                              where=[p >= 0.5 for p in sweep_probs],
                              alpha=0.15, color=A_WIN, label="A preferred")
            ax_s.fill_between(sweep_vals, sweep_probs, 0.5,
                              where=[p < 0.5 for p in sweep_probs],
                              alpha=0.15, color=B_WIN, label="B preferred")
            ax_s.annotate(
                f"P(A) = {cp_a:.0%}",
                xy=(sens_val, cp_a),
                xytext=(sens_val + 0.8, min(cp_a + 0.08, 0.92)),
                fontsize=10, color="#f59e0b", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=1.5),
            )
            ax_s.set_xlabel(f"{sens_param.replace('_',' ').title()} of A", fontsize=10)
            ax_s.set_ylabel("P(A preferred)", fontsize=10)
            ax_s.set_ylim(0, 1)
            ax_s.set_title(
                f"How does changing {sens_param.replace('_',' ').title()} affect the outcome?",
                fontsize=11, pad=10, color=get_theme()["TEXT"]
            )
            ax_s.legend(fontsize=8, loc="upper left")
            ax_s.spines[["top","right"]].set_visible(False)
            ax_s.grid(axis="y", alpha=0.3)
            plt.tight_layout(pad=1.0)
            st.pyplot(fig_s, use_container_width=True)
            plt.close(fig_s)

    # ── Tab 4: Examples ───────────────────────────────────────────────────────
    with tab4:
        st.markdown("### 🔍 Prediction Examples")
        st.caption(
            "Comparing what the model predicts vs what you actually chose."
        )
        if rf_error or rulefit is None or rf_stats is None:
            st.warning("RuleFit unavailable — see Rules tab.")
        else:
            per_dec    = rf_stats.get("per_decision", [])
            matches    = [d for d in per_dec if d["match"]]
            mismatches = [d for d in per_dec if not d["match"]]

            def render_example(item, is_match):
                d         = item["decision"]
                sc_num    = item["scenario"]
                actual    = item["actual"]
                predicted = item["predicted"]
                pred_int  = 1 if predicted == "A" else 0
                T_ex         = get_theme()
                _ex_text     = T_ex["TEXT"]
                _ex_dim      = T_ex["TEXT_DIM"]
                _ex_bg3      = T_ex["BG3"]
                match_txt = "✅ Agreed"             if is_match else "❌ Disagreed"
                match_col = "#22c55e"               if is_match else "#ef4444"
                bg_col    = T_ex["SUCCESS"]         if is_match else T_ex["DANGER"]
                border    = T_ex["SUCCESS_BORDER"]  if is_match else T_ex["DANGER_BORDER"]

                # Parameter table rows
                rows_html = ""
                for p in params:
                    av = d.get(f"A_{p}", "?"); bv = d.get(f"B_{p}", "?")
                    try:
                        adv = "🔵 A" if float(av) > float(bv) else ("🔴 B" if float(bv) > float(av) else "—")
                    except Exception:
                        adv = "—"
                    rows_html += (
                        f"<tr><td style='padding:3px 8px;font-size:12px;"
                        f"color:{_ex_dim}'>{p.replace('_',' ').title()}</td>"
                        f"<td style='padding:3px 8px;font-size:12px;"
                        f"text-align:center;color:{_ex_text}'>{av}</td>"
                        f"<td style='padding:3px 8px;font-size:12px;"
                        f"text-align:center;color:{_ex_text}'>{bv}</td>"
                        f"<td style='padding:3px 8px;font-size:12px;"
                        f"text-align:center;color:{_ex_text}'>{adv}</td></tr>"
                    )
                ac = COL_A if actual    == "A" else COL_B
                pc = COL_A if predicted == "A" else COL_B

                # Build rule-based fallback first
                fallback_exp = _explain_prediction(
                    rulefit, feat_names, params, d, pred_int
                )
                # Try LLM explanation
                explanation = explain_prediction_llm(
                    d, params, pred_int, rules_df, fallback_exp
                )

                # Confidence
                _, prob_a = _decision_prob(rulefit, feat_names, params, d)
                if prob_a is not None:
                    conf_val = max(prob_a, 1.0 - prob_a)
                    conf_str = f"{conf_val:.0%} confidence"
                    if conf_val < 0.65:
                        conf_str += " 🔴 (uncertain)"
                    elif conf_val < 0.80:
                        conf_str += " 🟡"
                    else:
                        conf_str += " 🟢"
                else:
                    conf_str = ""

                st.markdown(
                    f"<div style='border:1px solid {border};border-radius:10px;"
                    f"padding:14px 16px;margin:8px 0;background:{bg_col}'>"
                    f"<div style='display:flex;justify-content:space-between;"
                    f"margin-bottom:8px'>"
                    f"<b>Scenario {sc_num}</b>"
                    f"<span style='color:{match_col};font-weight:600'>{match_txt}</span>"
                    f"</div>"
                    f"<table style='width:100%;border-collapse:collapse;margin-bottom:8px'>"
                    f"<tr style='background:{_ex_bg3}'>"
                    f"<th style='padding:3px 8px;font-size:11px;text-align:left;color:{_ex_text}'>Param</th>"
                    f"<th style='padding:3px 8px;font-size:11px;color:{_ex_text}'>A</th>"
                    f"<th style='padding:3px 8px;font-size:11px;color:{_ex_text}'>B</th>"
                    f"<th style='padding:3px 8px;font-size:11px;color:{_ex_text}'>Adv</th>"
                    f"</tr>{rows_html}</table>"
                    f"<div style='display:flex;gap:24px;font-size:13px;"
                    f"margin-bottom:8px'>"
                    f"<span style='color:{_ex_text}'>You chose: <b style='color:{ac}'>Option {actual}</b></span>"
                    f"<span style='color:{_ex_text}'>Model: <b style='color:{pc}'>Option {predicted}</b></span>"
                    f"<span style='color:{_ex_dim}'>{conf_str}</span>"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                # Model explanation as expandable note
                with st.expander(f"💡 Why did the model choose Option {predicted}?",
                                 expanded=not is_match):
                    st.markdown(explanation)

            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                st.markdown(f"#### ✅ Agreements ({len(matches)})")
                if not matches:
                    st.info("No agreements yet.")
                else:
                    for item in matches[:3]:
                        render_example(item, True)
                    if len(matches) > 3:
                        st.caption(f"…and {len(matches)-3} more.")
            with col_ex2:
                st.markdown(f"#### ❌ Disagreements ({len(mismatches)})")
                if not mismatches:
                    st.success("Model agrees with all your decisions!")
                else:
                    for item in mismatches[:3]:
                        render_example(item, False)
                    if len(mismatches) > 3:
                        st.caption(f"…and {len(mismatches)-3} more.")

            st.divider()
            total     = len(per_dec)
            match_pct = len(matches) / total * 100 if total > 0 else 0
            st.markdown("#### 💡 Insight")
            if match_pct >= 80:
                st.success(
                    f"The model agrees with **{match_pct:.0f}%** of your decisions — "
                    "it has learned your preferences well."
                )
            elif match_pct >= 60:
                st.info(
                    f"The model agrees with **{match_pct:.0f}%** of your decisions. "
                    "Some preferences may be complex or context-dependent."
                )
            else:
                st.warning(
                    f"The model only agrees with **{match_pct:.0f}%** of your decisions. "
                    "Try answering more scenarios, especially ones similar to the disagreements."
                )

    # ── Tab 5: Analysis (Consistency Checker) ────────────────────────────────
    with tab5:
        st.markdown("### 🧠 Decision Analysis")

        answered_decisions = [d for d in st.session_state.decisions
                              if d.get("choice") in ("A", "B")]

        # ── Consistency checker ───────────────────────────────────────────
        st.markdown("#### 🔄 Consistency Check")
        st.caption(
            "Detects pairs of scenarios where A is similar to B in both, "
            "but you chose opposite options. Potential inconsistencies may "
            "reflect genuinely complex preferences or accidental misclicks."
        )

        if len(answered_decisions) < 4:
            st.info("Answer at least 4 scenarios to run consistency check.")
        else:
            conflicts = check_consistency(answered_decisions, params, threshold=0.85)

            if not conflicts:
                st.success(
                    "✅ No inconsistencies detected — your decisions appear consistent "
                    "across similar scenarios."
                )
            else:
                st.warning(
                    f"⚠️ Found **{len(conflicts)}** potentially inconsistent decision "
                    f"pair{'s' if len(conflicts) > 1 else ''}."
                )
                for cf in conflicts:
                    d_i = cf["decision_i"]
                    d_j = cf["decision_j"]
                    sim = cf["similarity"]
                    ci  = cf["choice_i"]
                    cj  = cf["choice_j"]
                    col_ci = COL_A if ci == "A" else COL_B
                    col_cj = COL_A if cj == "A" else COL_B

                    # Build side-by-side comparison
                    T_cf     = get_theme()
                    _cf_text = T_cf["TEXT"]
                    _cf_dim  = T_cf["TEXT_DIM"]
                    rows_html = ""
                    for p in params:
                        ai = d_i.get(f"A_{p}", "?"); bi = d_i.get(f"B_{p}", "?")
                        aj = d_j.get(f"A_{p}", "?"); bj = d_j.get(f"B_{p}", "?")
                        rows_html += (
                            f"<tr>"
                            f"<td style='padding:3px 8px;font-size:12px;"
                            f"color:{_cf_dim}'>{p.replace('_',' ').title()}</td>"
                            f"<td style='padding:3px 8px;font-size:12px;"
                            f"text-align:center;color:{_cf_text}'>{ai} vs {bi}</td>"
                            f"<td style='padding:3px 8px;font-size:12px;"
                            f"text-align:center;color:{_cf_text}'>{aj} vs {bj}</td>"
                            f"</tr>"
                        )

                    st.markdown(
                        f"<div style='border:1px solid {T_cf['DANGER_BORDER']};border-radius:10px;"
                        f"padding:14px 16px;margin:8px 0;background:{T_cf['DANGER']}'>"
                        f"<div style='font-weight:600;margin-bottom:8px;"
                        f"font-size:14px;color:{T_cf['TEXT']}'>"
                        f"Scenario {cf['scenario_i']} vs Scenario {cf['scenario_j']} "
                        f"<span style='font-size:12px;color:{T_cf['TEXT_DIM']};font-weight:400'>"
                        f"(similarity {sim:.0%})</span></div>"
                        f"<table style='width:100%;border-collapse:collapse;"
                        f"margin-bottom:10px'>"
                        f"<tr style='background:{T_cf['BG3']}'>"
                        f"<th style='padding:3px 8px;font-size:11px;"
                        f"text-align:left;color:{T_cf['TEXT']}'>Parameter</th>"
                        f"<th style='padding:3px 8px;font-size:11px;color:{T_cf['TEXT']}'>"
                        f"Scenario {cf['scenario_i']}</th>"
                        f"<th style='padding:3px 8px;font-size:11px;color:{T_cf['TEXT']}'>"
                        f"Scenario {cf['scenario_j']}</th>"
                        f"</tr>{rows_html}</table>"
                        f"<div style='display:flex;gap:32px;font-size:13px;color:{T_cf['TEXT']}'>"
                        f"<span>Scenario {cf['scenario_i']}: chose "
                        f"<b style='color:{col_ci}'>Option {ci}</b></span>"
                        f"<span>Scenario {cf['scenario_j']}: chose "
                        f"<b style='color:{col_cj}'>Option {cj}</b></span>"
                        f"</div>"
                        f"<div style='font-size:12px;color:{T_cf['TEXT_DIM']};margin-top:8px'>"
                        f"💡 These scenarios are {sim:.0%} similar in profile "
                        f"but you chose differently. This could reflect a "
                        f"nuanced preference the model hasn't captured yet."
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

                    # Model explanation for each scenario in the conflict
                    if rulefit is not None and feat_names:
                        pred_i, _ = _decision_prob(rulefit, feat_names, params, d_i)
                        pred_j, _ = _decision_prob(rulefit, feat_names, params, d_j)
                        # Rule-based fallbacks
                        fallback_i = _explain_prediction(rulefit, feat_names, params, d_i, pred_i)
                        fallback_j = _explain_prediction(rulefit, feat_names, params, d_j, pred_j)
                        # LLM explanations
                        exp_i, exp_j = explain_inconsistency_llm(
                            d_i, d_j, cf, params,
                            pred_i, pred_j,
                            fallback_i, fallback_j
                        )
                        pred_i_lbl = "A" if pred_i == 1 else "B"
                        pred_j_lbl = "A" if pred_j == 1 else "B"

                        col_exp1, col_exp2 = st.columns(2)
                        with col_exp1:
                            with st.expander(
                                f"💡 Model explanation — Scenario {cf['scenario_i']} "
                                f"(predicted Option {pred_i_lbl})",
                                expanded=True
                            ):
                                st.markdown(exp_i)
                        with col_exp2:
                            with st.expander(
                                f"💡 Model explanation — Scenario {cf['scenario_j']} "
                                f"(predicted Option {pred_j_lbl})",
                                expanded=True
                            ):
                                st.markdown(exp_j)

                        # Plain-English reconciliation note
                        if pred_i_lbl == ci and pred_j_lbl == cj:
                            st.info(
                                "The model **agrees with both** your choices — but since "
                                "the scenarios are similar, this suggests the model found "
                                "a subtle difference to distinguish them. "
                                "Check the explanations above to see what."
                            )
                        elif pred_i_lbl == pred_j_lbl:
                            st.warning(
                                f"The model gives the **same answer (Option {pred_i_lbl}) "
                                f"for both** scenarios — it cannot distinguish between them. "
                                "This suggests these scenarios test a preference the model "
                                "hasn't captured yet. Answering more similar scenarios "
                                "will help."
                            )
                        else:
                            st.info(
                                f"The model chose Option {pred_i_lbl} for Scenario "
                                f"{cf['scenario_i']} and Option {pred_j_lbl} for "
                                f"Scenario {cf['scenario_j']} — it does distinguish "
                                "between them, but differently from you."
                            )

                st.markdown(
                    "**Note:** Inconsistencies are not necessarily errors. "
                    "They may reflect context-sensitive reasoning that simple rules "
                    "cannot capture. However, resolving clear misclicks improves the model."
                )
