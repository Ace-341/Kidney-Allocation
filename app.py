

"""
Preference Elicitation Portal
SURA 2026 · IIT Delhi

Run:  streamlit run app.py
Install: pip install streamlit pandas numpy scikit-learn matplotlib

CSV file must be in the same folder as app.py.
Responses are saved to responses/<username>_responses.csv

The only predictive model is a Fast-and-Frugal Tree (see fft_model.py); its
interactive visualisation lives in fft_component.py.
"""



import streamlit as st
import streamlit.components.v1 as components
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Fast-and-Frugal Tree model + interactive visualisation ────────────────────
from fft_model import (
    build_difference_features,
    train_fft,
    FastFrugalTree,
    feature_row,
    decision_prob,
    explain_prediction as fft_explain_prediction,
)
from fft_component import fft_svg, pretty_feature

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# GROQ LLM HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _groq_explain(prompt, fallback_text):
    cache_key = f"groq_{hash(prompt)}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return fallback_text

    try:
        from groq import Groq
        import time                                          # ADD THIS

        client = Groq(api_key=api_key)

        start = time.time()                                  # ADD THIS
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        latency = (time.time() - start) * 1000               # ADD THIS
        print(f"Groq latency: {latency:.0f}ms")              # ADD THIS

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
    # Build plain-English feature summary for the prompt — raw values + A−B difference
    feature_lines = []
    for p in params:
        av = float(d.get(f"A_{p}", 0))
        bv = float(d.get(f"B_{p}", 0))
        pn = p.replace("_"," ").title()
        diff = av - bv
        if abs(diff) < 1e-9:
            feature_lines.append(f"  - {pn}: A={av:.0f}, B={bv:.0f} (equal)")
        else:
            who = "A" if diff > 0 else "B"
            feature_lines.append(
                f"  - {pn}: A={av:.0f}, B={bv:.0f} "
                f"({who} is higher by {abs(diff):.0f})"
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

The features used are ONLY the differences between the two patients on each factor,
i.e. (A's value − B's value) for: age, years waiting, health score, dependents,
prior transplants, and urgency score. A positive difference means A scores higher.

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


def explain_rule_llm(rule_str, coef, support, params, fallback_text):
    """
    Generate a plain-English explanation of a single Fast-and-Frugal Tree cue
    for a medical-ethics / non-technical audience.

    Covers:
    - What patient profile triggers the cue
    - Why it might represent a coherent allocation preference
    - Fallback to a structured text version if the API is unavailable
    """
    pref     = "A" if coef > 0 else "B"
    opp      = "B" if coef > 0 else "A"
    strength = (
        "very strongly" if abs(coef) > 0.8 else
        "strongly"      if abs(coef) > 0.5 else
        "moderately"    if abs(coef) > 0.25 else
        "weakly"
    )

    # Feature-name glossary injected into the prompt so the LLM can translate
    # the raw feature strings without hallucinating their meaning.
    glossary = """Feature name glossary (use this to interpret the cue):
- Every feature is "<factor>_diff" = Patient A's value MINUS Patient B's value on that factor.
- A positive "<factor>_diff" means Patient A scores higher on that factor; negative means Patient B does.
- "age_diff > 0" means A is OLDER; "age_diff < 0" means A is YOUNGER.
- Factors: age, years_waiting, health_score, dependents, prior_transplants, urgency_score."""

    prompt = f"""You are explaining a single decision pattern from a medical resource-allocation model \
to a doctor or medical ethicist. The model was trained on human pairwise judgements about who \
should receive an organ transplant.

{glossary}

Pattern to explain:
  Condition : {rule_str}
  Conclusion: when this condition is true, the model {strength} prefers Patient {pref} over Patient {opp}.
  Frequency : applies to {support:.0%} of the training decisions.

Write exactly 2 sentences:
  Sentence 1 — What kind of patient pair triggers this pattern (describe the clinical/ethical situation in concrete terms — ages, urgency, waiting time, etc.).
  Sentence 2 — Why this preference might reflect a coherent allocation principle (e.g. fairness, medical utility, vulnerability, social responsibility).

Rules:
- No technical jargon: do not say "coefficient", "feature", "model", "rule", "weight", "training", "positive value", or "negative value".
- Mention specific clinical factors by name (e.g. "younger patient", "longer wait", "higher urgency").
- Write in the third person, as a knowledgeable colleague explaining their reasoning."""

    return _groq_explain(prompt, fallback_text)



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

/* ── Primary buttons — larger, more comfortable ──── */
.stButton > button[kind="primary"] {{
    font-size: 17px !important;
    font-weight: 600 !important;
    padding: 14px 24px !important;
    min-height: 56px !important;
    border-radius: 8px !important;
}}
.stButton > button[kind="secondary"] {{
    font-size: 15px !important;
    padding: 10px 18px !important;
    min-height: 44px !important;
    border-radius: 8px !important;
}}

/* ── Progress bar — slightly thicker ─────────────── */
[data-testid="stProgressBar"] > div {{
    height: 6px !important;
    border-radius: 3px !important;
}}

/* ── Base font size boost ─────────────────────────── */
.stMarkdown p, .stMarkdown li {{
    font-size: 15px !important;
    line-height: 1.65 !important;
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

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA CONFIGURATION (EXMOS-style)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_param_data(decisions, params):
    """
    Compute min, max, and full value list for each parameter
    across all answered decisions (both Patient A and B values combined).
    Used to set slider bounds in the configuration panel.
    """
    data = {}
    for p in params:
        a_vals = [float(d.get(f"A_{p}", 0)) for d in decisions]
        b_vals = [float(d.get(f"B_{p}", 0)) for d in decisions]
        all_vals = a_vals + b_vals
        if not all_vals:
            data[p] = {"min": 0.0, "max": 1.0, "values": []}
            continue
        data[p] = {
            "min":    float(min(all_vals)),
            "max":    float(max(all_vals)),
            "values": all_vals,
        }
    return data


def filter_decisions_by_config(decisions, params, config):
    """
    Return only decisions where BOTH Patient A and Patient B values
    fall within the configured [lo, hi] range for every enabled parameter.
    Parameters whose checkbox is unchecked bypass the range check entirely.
    """
    filtered = []
    for d in decisions:
        keep = True
        for p in params:
            if not config.get(f"{p}_enabled", True):
                continue          # parameter filter is disabled — skip
            lo = config.get(f"{p}_lo", -1e18)
            hi = config.get(f"{p}_hi",  1e18)
            av = float(d.get(f"A_{p}", 0))
            bv = float(d.get(f"B_{p}", 0))
            if not (lo <= av <= hi) or not (lo <= bv <= hi):
                keep = False
                break
        if keep:
            filtered.append(d)
    return filtered


def chart_param_histogram(values, lo, hi):
    """
    Small matplotlib histogram for a single parameter in the
    Configure Training Data panel.

    - Bars for the full distribution; in-range bars coloured with the
      theme accent, out-of-range bars dimmed.
    - Shaded span over the selected [lo, hi] window.
    - Dashed vertical lines at the lo / hi boundaries when they do not
      coincide with the data edges.
    - Compact x-axis tick labels so the reader can see the data range.
    Returns a matplotlib Figure, or None if values are constant / empty.
    """
    if not values:
        return None
    mn, mx = float(min(values)), float(max(values))
    if mx == mn:
        return None

    T       = get_theme()
    accent  = T["ACCENT"]
    dim     = T["TEXT_DIM"]
    text    = T["TEXT"]
    border  = T["BORDER"]
    bg      = T["CARD_BG"]

    theme_rcparams()
    n_bins = min(14, max(6, len(set(round(v, 1) for v in values))))
    counts, edges = np.histogram(values, bins=n_bins, range=(mn, mx))
    bar_w = (edges[1] - edges[0]) * 0.80   # 80 % width → thin gap between bars

    fig, ax = plt.subplots(figsize=(3.4, 1.55))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    for i in range(len(counts)):
        left  = edges[i]
        right = edges[i + 1]
        mid   = (left + right) / 2
        c     = counts[i]
        # A bar is "in range" if any part of its bucket overlaps [lo, hi]
        in_range = right >= lo and left <= hi
        ax.bar(
            mid, c, width=bar_w,
            color=accent if in_range else dim,
            alpha=0.85   if in_range else 0.28,
            linewidth=0, zorder=2,
        )

    # Shaded selection window
    shade_lo = max(lo, mn)
    shade_hi = min(hi, mx)
    if shade_lo < shade_hi:
        ax.axvspan(shade_lo, shade_hi,
                   color=accent, alpha=0.10, zorder=1)

    # Boundary dashed lines (only if not at data edge)
    eps = (mx - mn) * 0.01
    if lo > mn + eps:
        ax.axvline(lo, color=accent, linewidth=1.4,
                   linestyle="--", alpha=0.75, zorder=3)
    if hi < mx - eps:
        ax.axvline(hi, color=accent, linewidth=1.4,
                   linestyle="--", alpha=0.75, zorder=3)

    # x-axis: show only min, lo, hi, max as ticks
    tick_vals = sorted({mn, lo, hi, mx})
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(
        [f"{v:.0f}" for v in tick_vals],
        fontsize=6.5, color=dim,
    )
    ax.tick_params(axis="x", length=2, pad=2)

    ax.set_xlim(mn - (mx - mn) * 0.04,
                mx + (mx - mn) * 0.04)
    ax.set_ylim(0, max(counts) * 1.35 if max(counts) > 0 else 1)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(border)
    ax.spines["bottom"].set_linewidth(0.8)

    plt.tight_layout(pad=0.25)
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


# ═══════════════════════════════════════════════════════════════════════════════
# FFT HELPERS  —  override persistence + theme palette for the SVG component
# ═══════════════════════════════════════════════════════════════════════════════

def load_fft_override(username):
    """Return the user's committed (edited) tree dict, or None."""
    try:
        return load_users().get(username, {}).get("fft_override")
    except Exception:
        return None


def save_fft_override(username, tree_dict):
    """Persist an edited tree as this user's active model."""
    u = load_users()
    u.setdefault(username, {})
    u[username]["fft_override"] = tree_dict
    save_users(u)


def clear_fft_override(username):
    """Drop the edited tree so the learned tree is used again."""
    u = load_users()
    if username in u and "fft_override" in u[username]:
        del u[username]["fft_override"]
        save_users(u)


def fft_palette():
    """Colour palette for the SVG FFT renderer, pulled from the active theme."""
    T = get_theme()
    return {
        "bg":     T["BG"],
        "card":   T["CARD_BG"],
        "border": T["CARD_BORDER"],
        "text":   T["TEXT"],
        "dim":    T["TEXT_DIM"],
        "muted":  T["TEXT_MUTED"],
        "accent": T["ACCENT"],
        "a":      T["COL_A"],
        "b":      T["COL_B"],
    }


def diffs_from_pair(params, a_dict, b_dict):
    """{feature_diff: A-B} for the SVG test-pair highlight."""
    return {f"{p}_diff": float(a_dict.get(p, 0)) - float(b_dict.get(p, 0)) for p in params}


@st.cache_data(show_spinner=False)
def train_fft_cached(decisions_json, params_tuple, override_json):
    """Cached FFT training. override_json is the JSON of a committed edited tree, or ''."""
    return train_fft(decisions_json, list(params_tuple), override_json or None)


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
        st.markdown(f"**{n_ans}** scenarios answered.")
        if st.button("See Model & Results",
                     type="primary", use_container_width=False):
            st.session_state.page = "model"
            st.rerun()
        st.stop()

    current_idx = st.session_state.sc_index
    sc          = st.session_state.scenarios[current_idx]
    params      = st.session_state.params

    st.markdown(
        f"<div style='font-size:14px;color:{TEXT_DIM};margin-bottom:6px'>"
        f"Scenario {current_idx + 1} of {n_total}</div>",
        unsafe_allow_html=True,
    )
    st.progress(current_idx / n_total)
    st.markdown(
        "<h2 style='font-size:28px;font-weight:600;margin:28px 0 24px'>"
        "Which option should be preferred?</h2>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2, gap="large")
    for col, label, data, color in [
        (col_a, "A", sc["A"], COL_A),
        (col_b, "B", sc["B"], COL_B),
    ]:
        with col:
            st.markdown(
                f"<div style='border-left:4px solid {color};padding:1.6rem 2rem'>"
                f"<p style='color:{color};font-weight:700;font-size:20px;"
                f"letter-spacing:.06em;margin:0 0 24px'>OPTION {label}</p>",
                unsafe_allow_html=True,
            )
            for p in params:
                st.markdown(
                    f"<div style='margin-bottom:22px'>"
                    f"<div style='font-size:13px;color:{get_theme()['TEXT_DIM']};text-transform:uppercase;"
                    f"letter-spacing:.07em;margin-bottom:4px'>{p.replace('_', ' ')}</div>"
                    f"<div style='font-size:38px;font-weight:600;"
                    f"font-family:monospace;line-height:1.1'>{data[p]:g}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Option A is better", use_container_width=True,
                     type="primary", key=f"btn_A_{current_idx}"):
            record_decision("A")
            st.rerun()
    with c2:
        if st.button("Option B is better", use_container_width=True,
                     type="primary", key=f"btn_B_{current_idx}"):
            record_decision("B")
            st.rerun()

    n_done = len([d for d in st.session_state.decisions
                  if d.get("choice") in ("A", "B")])
    st.caption(f"{n_done} answered so far")


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

    # ── Training Data Configuration Panel (EXMOS-style) ─────────────────────
    _all_decisions = decisions   # keep the full answered set for display tabs

    if "training_config" not in st.session_state:
        st.session_state["training_config"] = {}

    BG_cfg, BG2_cfg, BG3_cfg, BORDER_cfg, TEXT_cfg, TEXT_DIM_cfg, \
        TEXT_MUTED_cfg, A_WIN_cfg, B_WIN_cfg, COL_A_cfg, COL_B_cfg, ACCENT_cfg \
        = _setup_colours()
    T_cfg = get_theme()  # used for card wrapper HTML colours

    with st.expander("⚙️ Configure Training Data", expanded=False):
        st.caption(
            "Set a value range for each parameter to filter which of your answered "
            "decisions are used to train the model.  "
            "Decisions where **either** patient falls outside a range are excluded.  "
            "Uncheck a parameter to remove its filter entirely.  "
            "Click **💾 Save & Re-train** to apply."
        )

        _pdata   = compute_param_data(_all_decisions, params)
        _pending = {}

        # ── Parameter grid — 2 cards per row ─────────────────────────────────
        _plist    = list(params)
        _row_size = 2
        _grid     = [_plist[i:i + _row_size]
                     for i in range(0, len(_plist), _row_size)]

        for _row in _grid:
            _cols = st.columns(_row_size)
            for _col, _p in zip(_cols, _row):
                with _col:
                    _pn   = _p.replace("_", " ").title()
                    _pd   = _pdata[_p]
                    _lo_d = _pd["min"]
                    _hi_d = _pd["max"]

                    # Card wrapper
                    st.markdown(
                        f"<div style='border:1px solid {T_cfg['CARD_BORDER']};"
                        f"border-radius:10px;padding:12px 14px 10px;"
                        f"background:{T_cfg['CARD_BG']};margin-bottom:6px'>",
                        unsafe_allow_html=True,
                    )

                    # Checkbox — enable/disable this parameter's range filter
                    _enabled = st.checkbox(
                        f"**{_pn}**",
                        value=st.session_state["training_config"].get(
                            f"{_p}_enabled", True
                        ),
                        key=f"cfg_en_{_p}",
                        help="Uncheck to include all values for this parameter",
                    )
                    _pending[f"{_p}_enabled"] = _enabled

                    if _enabled and _lo_d < _hi_d:
                        # Restore previous range or default to full range
                        _cur_lo = float(
                            st.session_state["training_config"].get(f"{_p}_lo", _lo_d)
                        )
                        _cur_hi = float(
                            st.session_state["training_config"].get(f"{_p}_hi", _hi_d)
                        )
                        _cur_lo = max(_cur_lo, _lo_d)
                        _cur_hi = min(_cur_hi, _hi_d)

                        # Step: integer step for small integer-like ranges
                        _rng  = _hi_d - _lo_d
                        _step = 1.0 if _rng <= 100 else round(_rng / 100, 1)

                        _sel = st.slider(
                            f"{_pn} range",
                            min_value=float(_lo_d),
                            max_value=float(_hi_d),
                            value=(float(_cur_lo), float(_cur_hi)),
                            step=float(_step),
                            key=f"cfg_sl_{_p}",
                            label_visibility="collapsed",
                        )
                        _pending[f"{_p}_lo"] = _sel[0]
                        _pending[f"{_p}_hi"] = _sel[1]

                        # Histogram chart showing distribution + selected range
                        _fig = chart_param_histogram(_pd["values"], _sel[0], _sel[1])
                        if _fig is not None:
                            st.pyplot(_fig, use_container_width=True)
                            plt.close(_fig)

                        # Value range + coverage caption
                        _in  = sum(
                            1 for v in _pd["values"]
                            if _sel[0] <= v <= _sel[1]
                        )
                        _tot = len(_pd["values"])
                        _pct = _in / _tot * 100 if _tot else 0
                        st.caption(
                            f"{_sel[0]:.1f} – {_sel[1]:.1f}  "
                            f"· {_in}/{_tot} values ({_pct:.0f}%)"
                        )

                    else:
                        # Filter disabled or constant parameter
                        _pending[f"{_p}_lo"] = _lo_d
                        _pending[f"{_p}_hi"] = _hi_d
                        if _lo_d == _hi_d:
                            st.caption(f"Constant value: {_lo_d:.1f}")
                        else:
                            # Show full-range histogram (no selection lines)
                            _fig = chart_param_histogram(_pd["values"], _lo_d, _hi_d)
                            if _fig is not None:
                                st.pyplot(_fig, use_container_width=True)
                                plt.close(_fig)
                            st.caption(
                                f"No filter  "
                                f"· {_lo_d:.1f} – {_hi_d:.1f}"
                            )

                    st.markdown("</div>", unsafe_allow_html=True)

        # ── Live preview count ────────────────────────────────────────────────
        _preview = filter_decisions_by_config(_all_decisions, params, _pending)
        _n_prev  = len(_preview)
        _n_all   = len(_all_decisions)

        st.divider()
        _ci, _cr, _cs = st.columns([3, 1, 1])
        with _ci:
            if _n_prev == _n_all:
                st.success(
                    f"✅ **{_n_prev}** decisions selected — all decisions included."
                )
            elif _n_prev >= 6:
                st.info(
                    f"🔍 **{_n_prev}** of **{_n_all}** decisions match "
                    f"({_n_prev / _n_all * 100:.0f}%) — ready to train."
                )
            else:
                st.warning(
                    f"⚠️ Only **{_n_prev}** decisions match — "
                    "need ≥ 6 to train. Widen the ranges."
                )
        with _cr:
            if st.button(
                "↺ Reset", key="cfg_reset",
                help="Reset all filters to full data range",
                use_container_width=True,
            ):
                st.session_state["training_config"] = {}
                train_fft_cached.clear()
                st.rerun()
        with _cs:
            _save_disabled = _n_prev < 6
            if st.button(
                "💾 Save & Re-train",
                type="primary",
                key="cfg_save",
                disabled=_save_disabled,
                use_container_width=True,
                help="Apply filters and retrain the model" if not _save_disabled
                     else "Need at least 6 matching decisions",
            ):
                st.session_state["training_config"] = dict(_pending)
                train_fft_cached.clear()
                st.rerun()

    # ── Apply saved config to select training decisions ───────────────────────
    _cfg = st.session_state.get("training_config", {})
    if _cfg:
        _train_ds = filter_decisions_by_config(_all_decisions, params, _cfg)
        if len(_train_ds) >= 6:
            decisions = _train_ds
            if len(decisions) < len(_all_decisions):
                st.info(
                    f"🔍 Training on **{len(decisions)}** filtered decisions "
                    f"({len(decisions)}/{len(_all_decisions)}).  "
                    "Open ⚙️ Configure Training Data to adjust."
                )
        else:
            st.warning(
                f"Saved filters match only {len(_train_ds)} decisions (need ≥ 6). "
                "Using all decisions for training. Adjust or reset in ⚙️ Configure Training Data."
            )
            decisions = _all_decisions
    else:
        decisions = _all_decisions

    # ── Train the Fast-and-Frugal Tree ────────────────────────────────────────
    # If the user has committed manual edits to their tree, those take over as the
    # active model; otherwise a fresh FFT is learned from their decisions.
    decisions_json = json.dumps(decisions)
    _override = load_fft_override(st.session_state.username)
    _override_json = json.dumps(_override) if _override else ""
    with st.spinner("Training model…"):
        fft, nodes_df, fft_stats, feat_names, fft_error = train_fft_cached(
            decisions_json, tuple(params), _override_json
        )

    if _override and fft_error is None:
        st.info(
            "✏️ You are viewing an **edited tree** you committed earlier. "
            "Open the 🌳 Decision Tree tab to adjust it or reset to the learned tree."
        )

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decisions answered", len(decisions))
    c2.metric("Parameters",         len(params))
    if fft_error is None and fft_stats:
        c3.metric("Model accuracy",  f"{fft_stats['acc']*100:.0f}%")
        c4.metric("Symmetry",        f"{fft_stats['sym']*100:.0f}%",
                  help="% of predictions that correctly flip when A and B are swapped")
    st.divider()

    tab2, tab3, tab4 = st.tabs(["🌳 Decision Tree", "🎯 Predict New Pair", "🔍 Examples"])


    # ── Tab 2: Decision Tree ──────────────────────────────────────────────────
    with tab2:
        st.markdown("### 🌳 Your Fast-and-Frugal Tree")
        if fft_error or nodes_df is None or len(nodes_df) == 0:
            st.warning("No tree to show yet — answer at least 6 scenarios.")
        else:
            st.caption(
                "Read the tree top to bottom: at each step one cue is checked. "
                "If it holds, the tree exits immediately to that patient; otherwise it "
                "drops to the next step. **Edit any threshold, direction or outcome below** "
                "and the diagram + prediction update instantly. Commit your edits to make "
                "them your saved model."
            )

            active_tree = fft_stats["tree"]                 # override if committed, else learned
            tree_nodes  = active_tree["nodes"]

            # difference ranges (for slider bounds)
            diff_vals = {
                p: [float(d.get(f"A_{p}", 0)) - float(d.get(f"B_{p}", 0)) for d in decisions]
                for p in params
            }

            # ── seed editable widget state from the active tree (once) ─────────
            for i, nd in enumerate(tree_nodes):
                st.session_state.setdefault(f"e_thr_{i}", float(nd["threshold"]))
                st.session_state.setdefault(f"e_op_{i}",  nd["op"])
                st.session_state.setdefault(f"e_cls_{i}", int(nd["exit_class"]))
            st.session_state.setdefault("e_def", int(active_tree["default_class"]))
            for p in params:
                st.session_state.setdefault(f"e_test_{p}", 0.0)

            # ── build the working tree from current widget state ───────────────
            working = json.loads(json.dumps(active_tree))   # deep copy
            for i, nd in enumerate(working["nodes"]):
                nd["threshold"]  = float(st.session_state[f"e_thr_{i}"])
                nd["op"]         = st.session_state[f"e_op_{i}"]
                nd["exit_class"] = int(st.session_state[f"e_cls_{i}"])
            working["default_class"] = int(st.session_state["e_def"])

            # ── current test pair (as differences) ─────────────────────────────
            test_diffs = {f"{p}_diff": float(st.session_state[f"e_test_{p}"]) for p in params}

            # ── live prediction + accuracy of the working tree ─────────────────
            work_tree = FastFrugalTree.from_dict(working, feature_names=feat_names)
            F_train, _ = build_difference_features(decisions, params)
            y_train = np.array([1 if d["choice"] == "A" else 0 for d in decisions])
            work_acc = (work_tree.predict(F_train.values) == y_train).mean()

            col_svg, col_test = st.columns([3, 2], gap="large")

            with col_svg:
                svg = fft_svg(working, fft_palette(), test_diffs=test_diffs)
                # height from node count (matches fft_component geometry)
                _h = 70 + len(working["nodes"]) * 122 + 160
                components.html(
                    f"<div style='width:100%;overflow-x:auto'>{svg}</div>",
                    height=_h + 10, scrolling=False,
                )

            with col_test:
                st.markdown("**🧪 Test a case**")
                st.caption("Set how much higher A is than B on each factor "
                           "(positive = A higher). The path lights up live.")
                for p in params:
                    lo = min(diff_vals[p] + [0.0]); hi = max(diff_vals[p] + [0.0])
                    pad = max(2.0, (hi - lo) * 0.25)
                    st.slider(
                        f"{p.replace('_',' ').title()}  (A − B)",
                        min_value=float(np.floor(lo - pad)),
                        max_value=float(np.ceil(hi + pad)),
                        step=1.0, key=f"e_test_{p}",
                    )
                if st.button("↺ Clear test case", key="clear_test", use_container_width=True):
                    for p in params:
                        st.session_state[f"e_test_{p}"] = 0.0
                    st.rerun()

            st.divider()

            # ── editor ─────────────────────────────────────────────────────────
            ec1, ec2 = st.columns([3, 1])
            with ec1:
                st.markdown("#### ✏️ Edit the cues")
            with ec2:
                _wcol = "#22c55e" if work_acc >= 0.75 else "#f59e0b" if work_acc >= 0.6 else "#ef4444"
                st.markdown(
                    f"<div style='text-align:right;font-size:13px'>Training accuracy "
                    f"<b style='color:{_wcol};font-size:18px'>{work_acc*100:.0f}%</b></div>",
                    unsafe_allow_html=True,
                )

            for i, nd in enumerate(tree_nodes):
                p_name = nd["feature"].replace("_diff", "")
                lo = min(diff_vals.get(p_name, [0.0]) + [0.0])
                hi = max(diff_vals.get(p_name, [0.0]) + [0.0])
                pad = max(2.0, (hi - lo) * 0.3)
                with st.container():
                    st.markdown(
                        f"<span style='font-weight:600'>Step {i+1} · "
                        f"{pretty_feature(nd['feature'])}</span>",
                        unsafe_allow_html=True,
                    )
                    gc1, gc2, gc3 = st.columns([2, 1, 1])
                    with gc1:
                        st.slider(
                            "threshold", min_value=float(np.floor(lo - pad)),
                            max_value=float(np.ceil(hi + pad)), step=0.5,
                            key=f"e_thr_{i}", label_visibility="collapsed",
                        )
                    with gc2:
                        st.radio(
                            "direction", options=[">=", "<="],
                            key=f"e_op_{i}", horizontal=True,
                            label_visibility="collapsed",
                        )
                    with gc3:
                        st.radio(
                            "outcome", options=[1, 0],
                            format_func=lambda v: "→ A" if v == 1 else "→ B",
                            key=f"e_cls_{i}", horizontal=True,
                            label_visibility="collapsed",
                        )
            st.markdown("**Default** (when no cue above applies):")
            st.radio(
                "default", options=[1, 0],
                format_func=lambda v: "Prefer A" if v == 1 else "Prefer B",
                key="e_def", horizontal=True, label_visibility="collapsed",
            )

            st.divider()
            bc1, bc2, bc3 = st.columns([2, 2, 3])
            with bc1:
                if st.button("💾 Commit & persist", type="primary",
                             use_container_width=True, key="fft_commit"):
                    save_fft_override(st.session_state.username, working)
                    train_fft_cached.clear()
                    st.success("Saved — this edited tree is now your active model.")
                    st.rerun()
            with bc2:
                if st.button("↺ Reset to learned", use_container_width=True,
                             key="fft_reset"):
                    clear_fft_override(st.session_state.username)
                    for i in range(len(tree_nodes)):
                        for pre in ("e_thr_", "e_op_", "e_cls_"):
                            st.session_state.pop(f"{pre}{i}", None)
                    st.session_state.pop("e_def", None)
                    train_fft_cached.clear()
                    st.rerun()
            with bc3:
                if fft_stats.get("edited"):
                    st.caption("✏️ A committed edited tree is currently active.")
                else:
                    st.caption("Showing the learned tree. Edits apply live; commit to save.")

    # ── Tab 3: Predict new pair ───────────────────────────────────────────────
    with tab3:
        st.markdown("#### Predict outcome for a new patient pair")
        st.caption(
            "Enter values for two patients — the model predicts which is preferred."
        )

        if fft_error or fft is None:
            st.warning("No model yet — answer at least 6 scenarios first.")
        else:
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

            # ── Helper: build single difference-feature row ───────────────────────
            def _make_F(pa, pb):
                return feature_row(params, pa, pb, feat_names)


            # ── Predict button — results shown after click ────────────────────────
            predict_clicked = st.button(
                "🎯 Predict", type="primary",
                use_container_width=False, key="predict_btn"
            )

            # Store prediction in session state so results survive rerenders
            # (slider moves, selectbox changes) without requiring re-click
            if predict_clicked:
                F_tmp = _make_F(patient_a_input, patient_b_input)
                p_tmp = fft.predict(F_tmp.values)[0]
                pr_tmp = fft.predict_proba(F_tmp.values)[0]
                st.session_state["pred_result"] = {
                    "pred":  p_tmp,
                    "prob":  pr_tmp.tolist(),
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

                # ── How the tree decided (cue-by-cue path) ────────────────────────────
                st.markdown("**How the tree reached this decision:**")
                fv_arr  = F_new.values[0]
                exit_i  = fft.exit_index(fv_arr)
                for i, node in enumerate(fft.nodes):
                    x = fv_arr[node["feature_idx"]]
                    cond = (x >= node["threshold"]) if node["op"] == ">=" else (x <= node["threshold"])
                    exited = (i == exit_i)
                    future = (exit_i >= 0 and i > exit_i)
                    ex_cls = "A" if node["exit_class"] == 1 else "B"
                    color  = (COL_A if node["exit_class"] == 1 else COL_B) if exited else get_theme()["TEXT_MUTED"]
                    if future:
                        status = "not reached"
                    elif cond:
                        status = f"✓ holds → exit to Patient {ex_cls}"
                    else:
                        status = "✗ doesn't hold → check next cue"
                    op_disp = node["op"].replace("<", "&lt;").replace(">", "&gt;")
                    st.markdown(
                        f"<div style='border-left:3px solid {color};"
                        f"padding:6px 14px;margin:4px 0;opacity:{0.45 if future else 1.0};"
                        f"background:{get_theme()['BG2']};border-radius:4px;font-size:13px'>"
                        f"<b>Step {i+1}</b> &nbsp;<code>{pretty_feature(node['feature'])} "
                        f"{op_disp} {round(node['threshold'],2)}</code> "
                        f"&nbsp;(your case: {round(float(x),1)}) &nbsp;→ "
                        f"<span style='color:{color};font-weight:600'>{status}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                if exit_i < 0:
                    dcls = "A" if fft.default_class == 1 else "B"
                    st.markdown(
                        f"<div style='border-left:3px solid {COL_A if fft.default_class==1 else COL_B};"
                        f"padding:6px 14px;margin:4px 0;"
                        f"background:{get_theme()['BG2']};border-radius:4px;font-size:13px'>"
                        f"No cue applied → <b>default: prefer Patient {dcls}</b></div>",
                        unsafe_allow_html=True,
                    )

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
                        if fft.predict(F2.values)[0] != pred:
                            ds = f"+{delta}" if delta > 0 else str(delta)
                            fc = COL_A if fft.predict(F2.values)[0] == 1 else COL_B
                            np_lbl = "A preferred" if fft.predict(F2.values)[0] == 1 else "B preferred"
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

                # ── Plain-English explanation of this decision ────────────────────────
                st.markdown("#### 📝 In plain English")
                _expl = fft_explain_prediction(fft, params, feat_names,
                                               {**{f"A_{p}": _res["pa"][p] for p in params},
                                                **{f"B_{p}": _res["pb"][p] for p in params},
                                                "choice": "A"}, pred)
                st.markdown(
                    f"<div style='background:{get_theme()['BG2']};border:1px solid "
                    f"{get_theme()['CARD_BORDER']};border-radius:8px;padding:14px 18px;"
                    f"font-size:14px;line-height:1.6;color:{get_theme()['TEXT']}'>{_expl}</div>",
                    unsafe_allow_html=True,
                )

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
                sweep_probs = fft.predict_proba(F_sweep)[:, 1].tolist()

                ci_sweep = int(np.argmin(np.abs(sweep_vals - sens_val)))
                cp_a     = sweep_probs[ci_sweep]

                theme_rcparams()
                fig_s, ax_s = plt.subplots(figsize=(9, 3.5))
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
        if fft_error or fft is None or fft_stats is None:
            st.warning("No model yet — answer at least 6 scenarios first.")
        else:
            per_dec    = fft_stats.get("per_decision", [])
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

                # Build cue-based fallback first
                fallback_exp = fft_explain_prediction(
                    fft, params, feat_names, d, pred_int
                )
                # Try LLM explanation
                explanation = explain_prediction_llm(
                    d, params, pred_int, nodes_df, fallback_exp
                )

                # Confidence
                _, prob_a = decision_prob(fft, params, feat_names, d)
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

