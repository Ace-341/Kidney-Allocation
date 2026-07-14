"""
Preference Elicitation Portal
SURA 2026 · IIT Delhi

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
import os
import random
from datetime import datetime

from fft_model import train_fft, summarize_model_changes
from fft_viz import fft_viz
from fft_component import fft_svg, fft_svg_explained, DEFAULT_FFT_PALETTE

<<<<<<< HEAD
=======
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


def explain_rule_llm(rule_str, coef, support, params, fallback_text):
    """
    Generate a plain-English explanation of a single RuleFit rule
    for a medical-ethics / non-technical audience.

    Covers:
    - What patient profile triggers the rule
    - Why it might represent a coherent allocation preference
    - Fallback to a structured text version if the API is unavailable
    """
    pref     = "A" if coef > 0 else "B"
    opp      = "B" if coef > 0 else "A"
    strength = (
        "very strongly" if abs(coef) > 1.5 else
        "strongly"      if abs(coef) > 1.0 else
        "moderately"    if abs(coef) > 0.4 else
        "weakly"
    )

    # Feature-name glossary injected into the prompt so the LLM can translate
    # the raw feature strings without hallucinating their meaning.
    glossary = """Feature name glossary (use this to interpret the rule):
- "X difference (%)" is positive when Patient A scores higher than B on X, negative when B is higher.
- "Life years difference (%)" is positive when Patient A is YOUNGER (more remaining life years).
- "X similarity (%)" is high (near 100) when A and B are close in value on X.
- "X higher patient" = the larger of the two patients' values on X.
- "X lower patient"  = the smaller of the two patients' values on X.
- "Expected treatment benefit difference (%)" = (health_score × remaining_life_years)_A  minus the same for B.
- "Social responsibility difference (%)" = (dependents × remaining_life_years)_A  minus B.
- "Vulnerability index difference (%)" = (urgency_score × years_waiting)_A  minus B."""

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



>>>>>>> cc1120d344b8eb2aa7bd5a5d00c2bea509caf1af
st.set_page_config(
    page_title="Organ Allocation Preference Study",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ────────────────────────────────────────────────────────────
# Single source of truth for every color used in the app's own CSS/markup.
# fft_component.DEFAULT_FFT_PALETTE (used by the tree SVGs) is kept value-
# identical to the entries below — same hex, different dict shape — so the
# interactive tree, the "explain simply" view, and the rest of the page never
# drift out of sync. Light mode only, by design: this app has no dark theme.
COLORS = {
    "bg":             "#ffffff",
    "surface":        "#f8fafc",
    "surface_alt":    "#f1f5f9",
    "border":         "#e2e8f0",
    "border_strong":  "#cbd5e1",
    "text":           "#0f172a",
    "text_secondary": "#475569",
    "text_muted":     "#94a3b8",
    "accent":         "#2563eb",
    "accent_hover":   "#1d4ed8",
    "accent_soft":    "#eff6ff",
    "a":              "#b91c1c",
    "a_soft":         "#fef2f2",
    "b":              "#1d4ed8",
    "b_soft":         "#eff6ff",
    "success":        "#15803d",
    "success_soft":   "#f0fdf4",
    "warning":        "#b45309",
    "warning_soft":   "#fffbeb",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* Hide Streamlit chrome */
#MainMenu, footer {{ display: none !important; }}
[data-testid="stToolbar"]    {{ display: none !important; }}
[data-testid="stSidebar"]    {{ display: none !important; }}
[data-testid="stHeader"]     {{ display: none !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}

/* Force light mode regardless of the visitor's OS/browser theme */
:root {{ color-scheme: light !important; }}
html, body, .stApp {{ background-color: {COLORS["bg"]} !important; }}

/* Main container — full width, tight vertical padding */
.main .block-container {{
  padding-top: 2.25rem !important;
  padding-bottom: 3rem !important;
  padding-left: 3rem !important;
  padding-right: 3rem !important;
  max-width: 1400px !important;
}}

/* Base font */
html, body, [class*="css"] {{
  font-size: 16px;
  color: {COLORS["text"]};
  font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
}}

/* Headings */
h1 {{ font-size: 2rem !important;   font-weight: 700 !important; line-height: 1.2 !important; color: {COLORS["text"]} !important; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.5rem !important; font-weight: 600 !important; line-height: 1.3 !important; color: {COLORS["text"]} !important; letter-spacing: -0.01em; }}
h3 {{ font-size: 1.1rem !important; font-weight: 600 !important; line-height: 1.4 !important; color: {COLORS["text"]} !important; }}

/* Paragraph & list text */
.stMarkdown p  {{ font-size: 16px !important; line-height: 1.7 !important; color: {COLORS["text"]}; }}
.stMarkdown li {{ font-size: 16px !important; line-height: 1.7 !important; color: {COLORS["text"]}; }}
.stMarkdown strong {{ color: {COLORS["text"]}; }}

/* All buttons */
.stButton > button {{
  font-size: 16px !important;
  padding: 11px 22px !important;
  min-height: 46px !important;
  border-radius: 10px !important;
  font-weight: 500 !important;
  transition: background-color 0.15s ease, border-color 0.15s ease, transform 0.05s ease !important;
}}
.stButton > button:active {{ transform: scale(0.99); }}
.stButton > button[kind="primary"] {{
  background-color: {COLORS["accent"]} !important;
  color: {COLORS["bg"]} !important;
  border: 1px solid {COLORS["accent"]} !important;
}}
.stButton > button[kind="primary"]:hover  {{ background-color: {COLORS["accent_hover"]} !important; border-color: {COLORS["accent_hover"]} !important; }}
.stButton > button[kind="secondary"] {{
  background-color: {COLORS["bg"]} !important;
  color: {COLORS["text"]} !important;
  border: 1px solid {COLORS["border"]} !important;
}}
.stButton > button[kind="secondary"]:hover {{
  background-color: {COLORS["surface"]} !important;
  border-color: {COLORS["border_strong"]} !important;
}}

/* Text input / text area */
.stTextInput > div > div > input,
.stTextArea textarea {{
  font-size: 16px !important;
  padding: 13px 16px !important;
  border-radius: 10px !important;
  border: 1.5px solid {COLORS["border"]} !important;
  background: {COLORS["surface"]} !important;
  color: {COLORS["text"]} !important;
}}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {{
  border-color: {COLORS["accent"]} !important;
  box-shadow: 0 0 0 3px {COLORS["accent_soft"]} !important;
}}

/* Selectbox */
.stSelectbox > div > div {{
  border-radius: 10px !important;
  border: 1.5px solid {COLORS["border"]} !important;
  background: {COLORS["surface"]} !important;
}}

/* Number input */
.stNumberInput > div > div > input {{
  border-radius: 10px !important;
  border: 1.5px solid {COLORS["border"]} !important;
  background: {COLORS["surface"]} !important;
}}

/* Progress bar */
[data-testid="stProgressBar"] > div {{
  height: 6px !important;
  border-radius: 3px !important;
  background: {COLORS["surface_alt"]} !important;
}}
[data-testid="stProgressBar"] > div > div {{ background: {COLORS["accent"]} !important; }}

/* Divider */
hr {{ border: none !important; border-top: 1px solid {COLORS["border"]} !important; margin: 1.5rem 0 !important; }}

/* Expander */
[data-testid="stExpander"] {{
  border: 1px solid {COLORS["border"]} !important;
  border-radius: 12px !important;
  background: {COLORS["bg"]} !important;
}}
[data-testid="stExpander"] summary {{ font-weight: 600 !important; color: {COLORS["text"]} !important; }}

/* Alerts (info / success / warning / error) — keep flat and on-brand */
[data-testid="stAlert"] {{ border-radius: 10px !important; }}

/* Caption */
.stCaption, [data-testid="stCaptionContainer"] {{ color: {COLORS["text_muted"]} !important; }}

/* Right-panel feedback card */
.fb-card {{
  background: {COLORS["surface"]};
  border: 1px solid {COLORS["border"]};
  border-radius: 14px;
  padding: 28px 26px;
}}

/* Generic content card, used for scenario/review panels */
.content-card {{
  background: {COLORS["surface"]};
  border: 1px solid {COLORS["border"]};
  border-radius: 12px;
  padding: 18px 20px;
}}

/* Small uppercase eyebrow label */
.eyebrow {{
  font-size: 12px;
  color: {COLORS["text_muted"]};
  text-transform: uppercase;
  letter-spacing: .08em;
  font-weight: 600;
}}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
APP_DIR       = os.path.dirname(os.path.abspath(__file__))
RESPONSES_DIR = os.path.join(APP_DIR, "responses")
USERS_FILE    = os.path.join(APP_DIR, "users.json")

# Human-readable descriptions for the landing page
PARAM_DESCRIPTIONS = {
    "age":               "The patient's age in years.",
    "years_waiting":     "How long the patient has been on the transplant waiting list.",
    "health_score":      "Overall medical health score (higher is better, scale 1–10).",
    "dependents":        "Number of people who depend on this patient (family, caregivers).",
    "prior_transplants": "Number of organ transplants the patient has previously received.",
    "urgency_score":     "Medical urgency level — how critical their need is (scale 1–10).",
}

def _pretty_feature_label(feature):
    """'urgency_score_diff' -> 'Urgency Score'."""
    base = feature[:-5] if feature.endswith("_diff") else feature
    return base.replace("_", " ").title()

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "page":            "login",
    "username":        "",
    "scenarios":       [],
    "params":          [],
    "sc_index":        0,
    "decisions":       [],
    "alignment_score": None,
    "wants_edit":      None,
    "_saved_msg":      False,
    "pending_tree":    None,
    # Part 2 — the 20-question follow-up survey
    "survey_scenarios":       [],
    "survey_index":           0,
    "survey_decisions":       [],
    "survey_alignment_score": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSV loading ───────────────────────────────────────────────────────────────
def load_csv():
    path = os.path.join(APP_DIR, "organ_allocation_scenarios.csv")
    if not os.path.exists(path):
        st.error(f"Missing data file: organ_allocation_scenarios.csv")
        st.stop()
    df     = pd.read_csv(path)
    a_cols = [c for c in df.columns if str(c).startswith("A_")]
    params = [c[2:] for c in a_cols]
    for col in a_cols + [f"B_{p}" for p in params]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    scenarios = [
        {
            "A": {p: float(row[f"A_{p}"]) for p in params},
            "B": {p: float(row[f"B_{p}"]) for p in params},
        }
        for _, row in df.iterrows()
    ]
    return params, scenarios

if not st.session_state.scenarios:
    st.session_state.params, st.session_state.scenarios = load_csv()

def generate_survey_scenarios(params, base_scenarios, n=20, seed=None):
    """
    Build a second batch of pairwise comparisons for the Part 2 follow-up
    survey ("does the model still feel right after a few more?"). Sampled
    uniformly within the same per-parameter ranges seen in the main CSV
    (integer-valued parameters stay integers), so they feel consistent with
    the original scenarios without needing extra input data.
    """
    rng = random.Random(seed)
    ranges = {}
    for p in params:
        vals = [s["A"][p] for s in base_scenarios] + [s["B"][p] for s in base_scenarios]
        lo, hi = min(vals), max(vals)
        is_int = all(float(v).is_integer() for v in vals)
        ranges[p] = (lo, hi, is_int)
    out = []
    for _ in range(n):
        a, b = {}, {}
        for p in params:
            lo, hi, is_int = ranges[p]
            if is_int:
                a[p] = float(rng.randint(int(lo), int(hi)))
                b[p] = float(rng.randint(int(lo), int(hi)))
            else:
                a[p] = round(rng.uniform(lo, hi), 1)
                b[p] = round(rng.uniform(lo, hi), 1)
        out.append({"A": a, "B": b})
    return out

# ── User persistence ──────────────────────────────────────────────────────────
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
    u.setdefault(st.session_state.username, {})
    u[st.session_state.username].update({
        "sc_index":         st.session_state.sc_index,
        "decisions":        st.session_state.decisions,
        # Persisting the actual generated survey_scenarios (not just a seed) so a
        # resumed session replays the *same* pairs the participant already saw —
        # regenerating from a seed isn't safe since Python's string hashing is
        # randomized per process.
        "survey_scenarios": st.session_state.survey_scenarios,
        "survey_index":     st.session_state.survey_index,
        "survey_decisions": st.session_state.survey_decisions,
    })
    save_users(u)

def save_fft_override(username, tree_dict):
    u = load_users()
    u.setdefault(username, {})
    u[username]["fft_override"] = tree_dict
    save_users(u)

def load_fft_override(username):
    return load_users().get(username, {}).get("fft_override")

def record_decision(choice):
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

    user_file = os.path.join(RESPONSES_DIR, f"{st.session_state.username}_responses.csv")
    new_df    = pd.DataFrame([row])
    if os.path.exists(user_file):
        combined = pd.concat([pd.read_csv(user_file), new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(user_file, index=False)

    st.session_state.decisions.append(row)
    st.session_state.sc_index += 1
    save_session()

def record_survey_decision(choice):
    os.makedirs(RESPONSES_DIR, exist_ok=True)
    idx = st.session_state.survey_index
    sc  = st.session_state.survey_scenarios[idx]
    row = {
        "username":  st.session_state.username,
        "scenario":  idx + 1,
        "choice":    choice,
        "timestamp": datetime.now().isoformat(),
    }
    for p in st.session_state.params:
        row[f"A_{p}"] = sc["A"][p]
        row[f"B_{p}"] = sc["B"][p]

    user_file = os.path.join(RESPONSES_DIR, f"{st.session_state.username}_survey_responses.csv")
    new_df    = pd.DataFrame([row])
    if os.path.exists(user_file):
        combined = pd.concat([pd.read_csv(user_file), new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(user_file, index=False)

<<<<<<< HEAD
    st.session_state.survey_decisions.append(row)
    st.session_state.survey_index += 1
    save_session()
=======

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
>>>>>>> cc1120d344b8eb2aa7bd5a5d00c2bea509caf1af

@st.cache_data(show_spinner=False)
def train_fft_cached(decisions_json, params_tuple, override_json):
    return train_fft(decisions_json, list(params_tuple), override_json or None)


def render_pairwise_choice_page(scenarios, idx, n_total, qparams, heading,
                                 subheading, on_record, key_prefix):
    """
    Shared UI for a single pairwise A-vs-B comparison screen. Used by both the
    Part 1 questionnaire and the Part 2 follow-up survey so the two stay
    pixel-identical and any future tweak only needs to happen in one place.
    """
    _, qcol, _ = st.columns([1, 5, 1])
    with qcol:
        st.progress(idx / n_total)
        st.markdown(
            f"<div style='color:{COLORS['text_muted']};font-size:13px;font-weight:500;"
            f"margin-top:8px;margin-bottom:26px'>Scenario {idx + 1} of {n_total}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"## {heading}")
        if subheading:
            st.markdown(
                f"<div style='color:{COLORS['text_secondary']};font-size:15px;"
                f"margin-bottom:16px;line-height:1.6'>{subheading}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        sc = scenarios[idx]
        col_a, col_b = st.columns(2, gap="large")
        for col, label, data, color, soft in [
            (col_a, "Patient A", sc["A"], COLORS["a"], COLORS["a_soft"]),
            (col_b, "Patient B", sc["B"], COLORS["b"], COLORS["b_soft"]),
        ]:
            with col:
                st.markdown(
                    f"<div style='background:{soft};border-radius:12px;"
                    f"border-left:4px solid {color};padding:18px 20px 20px 18px'>"
                    f"<div style='color:{color};font-weight:700;font-size:15px;"
                    f"letter-spacing:.04em;text-transform:uppercase;margin-bottom:18px'>"
                    f"{label}</div>",
                    unsafe_allow_html=True,
                )
                for p in qparams:
                    pname = p.replace("_", " ").title()
                    val   = data[p]
                    st.markdown(
                        f"<div style='margin-bottom:14px'>"
                        f"<div style='font-size:11.5px;color:{COLORS['text_secondary']};"
                        f"text-transform:uppercase;letter-spacing:.07em;margin-bottom:2px'>"
                        f"{pname}</div>"
                        f"<div style='font-size:32px;font-weight:600;font-family:monospace;"
                        f"line-height:1.15;color:{COLORS['text']}'>{val:g}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        btn_a, btn_b = st.columns(2)
        with btn_a:
            if st.button("Choose Patient A", use_container_width=True,
                         type="primary", key=f"{key_prefix}_A_{idx}"):
                on_record("A"); st.rerun()
        with btn_b:
            if st.button("Choose Patient B", use_container_width=True,
                         type="primary", key=f"{key_prefix}_B_{idx}"):
                on_record("B"); st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LOGIN
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "login":
    # Centre the card on the wide canvas
    _, card, _ = st.columns([1, 2, 1])
    with card:
        st.markdown(
            f"<div class='eyebrow' style='margin-bottom:10px'>SURA 2026 · IIT Delhi</div>",
            unsafe_allow_html=True,
        )
        st.markdown("## Organ allocation preference study")
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        st.markdown(
            "You'll see **pairs of patients** who each need an organ transplant, and choose "
            "who you think should receive it. There's no right or wrong answer here — we're "
            "learning how *you* reason through it. Afterwards, we'll show you a simple model "
            "that captures the pattern in your choices, and you can tell us if it feels right."
        )

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='eyebrow'>Each patient is described by six factors</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        lp = st.session_state.params
        gc1, gc2 = st.columns(2, gap="large")
        for i, p in enumerate(lp):
            desc  = PARAM_DESCRIPTIONS.get(p, p.replace("_", " ").title() + ".")
            label = p.replace("_", " ").title()
            (gc1 if i % 2 == 0 else gc2).markdown(
                f"<div style='margin-bottom:10px'><strong>{label}</strong> — "
                f"<span style='color:{COLORS['text_secondary']}'>{desc}</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.markdown("**What should we call you?**")
        uname = st.text_input(
            "Name", placeholder="Enter your name to begin",
            label_visibility="collapsed", key="login_name_input",
        )
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Start →", type="primary", use_container_width=True, key="login_btn"):
            raw = uname.strip()
            if len(raw) < 2:
                st.error("Enter at least 2 characters to continue.")
            else:
                users = load_users()
                st.session_state.username = raw
                if raw in users:
                    ud = users[raw]
                    st.session_state.sc_index         = ud.get("sc_index", 0)
                    st.session_state.decisions        = ud.get("decisions", [])
                    st.session_state.survey_scenarios = ud.get("survey_scenarios", [])
                    st.session_state.survey_index     = ud.get("survey_index", 0)
                    st.session_state.survey_decisions = ud.get("survey_decisions", [])
                    if "alignment_score" in ud:
                        st.session_state.alignment_score = ud["alignment_score"]
                    if "survey_alignment_score" in ud:
                        st.session_state.survey_alignment_score = ud["survey_alignment_score"]
                else:
                    st.session_state.sc_index  = 0
                    st.session_state.decisions = []
                    users[raw] = {"sc_index": 0, "decisions": []}
                    save_users(users)
                st.session_state.page = "questionnaire"
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: QUESTIONNAIRE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "questionnaire":
    qparams = st.session_state.params
    n_total = len(st.session_state.scenarios)

    if st.session_state.sc_index >= n_total:
        st.session_state.page = "results"
        st.rerun()

    render_pairwise_choice_page(
        scenarios=st.session_state.scenarios,
        idx=st.session_state.sc_index,
        n_total=n_total,
        qparams=qparams,
        heading="Which patient should receive the organ?",
        subheading=None,
        on_record=record_decision,
        key_prefix="btn",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SURVEY QUESTIONNAIRE  (Part 2 — 20 more, after seeing the model)
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "survey_questionnaire":
    qparams = st.session_state.params
    n_total = len(st.session_state.survey_scenarios)

    if st.session_state.survey_index >= n_total:
        st.session_state.page = "survey_results"
        st.rerun()

    render_pairwise_choice_page(
        scenarios=st.session_state.survey_scenarios,
        idx=st.session_state.survey_index,
        n_total=n_total,
        qparams=qparams,
        heading="Part 2: A few more scenarios",
        subheading=(
            "Now that you've seen your model, here are 20 more pairs in the same "
            "format. This checks whether the model still matches how you'd actually "
            "choose."
        ),
        on_record=record_survey_decision,
        key_prefix="svy",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "results":
    rparams   = st.session_state.params
    decisions = [d for d in st.session_state.decisions if d.get("choice") in ("A", "B")]

    if len(decisions) < 6:
        st.warning(
            f"We need at least 6 answered scenarios to build a model — you've answered "
            f"{len(decisions)} so far."
        )
        if st.button("← Back to scenarios"):
            st.session_state.page = "questionnaire"
            st.rerun()
        st.stop()

<<<<<<< HEAD
    # Train the FFT
    rdecisions_json = json.dumps(decisions)
    _override       = load_fft_override(st.session_state.username)
    _override_json  = json.dumps(_override) if _override else ""

    with st.spinner("Building your preference model…"):
        fft, nodes_df, fft_stats, feat_names, fft_error = train_fft_cached(
            rdecisions_json, tuple(rparams), _override_json
=======
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
                train_rulefit.clear()
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
                train_rulefit.clear()
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

    # Train RuleFit
    decisions_json = json.dumps(decisions)
    with st.spinner("Training model…"):
        rulefit, rules_df, rf_stats, feat_names, rf_error = train_rulefit(
            decisions_json, params
>>>>>>> cc1120d344b8eb2aa7bd5a5d00c2bea509caf1af
        )

    if fft_error or fft_stats is None:
        st.warning(
            "We couldn't build a model from these answers — try going back and answering "
            "with a bit more variety."
        )
        if st.button("← Back to scenarios"):
            st.session_state.page = "questionnaire"
            st.rerun()
        st.stop()

<<<<<<< HEAD
    rtree_dict = fft_stats["tree"]
    editing    = (st.session_state.wants_edit is True)
=======
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Model Card", "📋 Rules", "📊 Charts", "🎯 Predict New Pair", "🔍 Examples", "🧠 Analysis"])
>>>>>>> cc1120d344b8eb2aa7bd5a5d00c2bea509caf1af

    # ── Two-column layout: visualization left, feedback right ─────────────────
    col_viz, col_fb = st.columns([5, 3], gap="large")

<<<<<<< HEAD
    # ── LEFT: heading + FFT ───────────────────────────────────────────────────
    with col_viz:
        st.markdown("## Your preference model")
        st.markdown(
            f"<div style='color:{COLORS['text_secondary']};font-size:15px;margin-bottom:20px;"
            f"line-height:1.6'>Based on your decisions, here's the decision tree that captures "
            f"your thinking. Each step is a single check the tree applies, in order.</div>",
            unsafe_allow_html=True,
=======
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

    # ── Tab 1: Rules (with per-rule LLM explanations) ─────────────────────────
    with tab1:
        st.markdown("### 📋 Learned Rules")
        if rf_error or rules_df is None or len(rules_df) == 0:
            st.warning(
                "No rules available. RuleFit requires Python 3.9–3.11 "
                "and at least 6 answered decisions."
            )
        else:
            BG_r,BG2_r,BG3_r,BORDER_r,TEXT_r,TEXT_DIM_r,TEXT_MUTED_r, \
                A_WIN_r,B_WIN_r,COL_A_r,COL_B_r,ACCENT_r = _setup_colours()
            T_r = get_theme()

            st.caption(
                f"The model distilled your {rf_stats['n_decisions']} decisions into "
                f"**{len(rules_df)} IF-THEN rules**. "
                "Each card shows the raw condition, how often it fires, how strongly it "
                "pushes toward a patient, and a plain-English explanation of what it means."
            )

            has_groq = bool(os.environ.get("GROQ_API_KEY", ""))
            if not has_groq:
                st.info(
                    "💡 Add a `GROQ_API_KEY` to your `.env` file to get LLM explanations. "
                    "Without it, the rule-based fallback text is shown instead."
                )

            for i, (_, row) in enumerate(rules_df.iterrows()):
                pref       = "A" if row["coef"] > 0 else "B"
                opp        = "B" if row["coef"] > 0 else "A"
                card_color = COL_A_r if pref == "A" else COL_B_r
                imp        = abs(row["coef"])

                # Strength label + colour
                if imp > 1.5:
                    strength_lbl = "Very Strong";  strength_col = "#16a34a"
                elif imp > 1.0:
                    strength_lbl = "Strong";       strength_col = "#22c55e"
                elif imp > 0.4:
                    strength_lbl = "Moderate";     strength_col = "#f59e0b"
                else:
                    strength_lbl = "Weak";         strength_col = "#94a3b8"

                # Coverage tier
                sup = row["support"]
                if sup >= 0.60:
                    cov_lbl = "Broad";   cov_col = ACCENT_r
                elif sup >= 0.30:
                    cov_lbl = "Common";  cov_col = TEXT_DIM_r
                else:
                    cov_lbl = "Narrow";  cov_col = TEXT_MUTED_r

                # Rule-based fallback
                conf_word = (
                    "strongly"   if imp > 1.0 else
                    "moderately" if imp > 0.4 else
                    "weakly"
                )
                fallback = (
                    f"When {row['rule']}, prefer Patient {pref} "
                    f"({conf_word}, covers {sup:.0%} of decisions). "
                    f"This pattern {conf_word} suggests {pref} should be prioritised "
                    f"in scenarios matching these conditions."
                )

                # Call LLM (cached by prompt hash)
                with st.spinner(f"Explaining rule {i + 1} of {len(rules_df)}…"):
                    explanation = explain_rule_llm(
                        row["rule"], row["coef"], row["support"], params, fallback
                    )

                # ── Rule card ─────────────────────────────────────────────────
                st.markdown(
                    f"<div style='"
                    f"border:1px solid {T_r['CARD_BORDER']};"
                    f"border-left:4px solid {card_color};"
                    f"border-radius:8px;"
                    f"padding:16px 18px 14px;"
                    f"margin-bottom:14px;"
                    f"background:{T_r['CARD_BG']}'>"

                    # ── Header row ────────────────────────────────────────────
                    f"<div style='display:flex;justify-content:space-between;"
                    f"align-items:center;margin-bottom:10px'>"

                    f"<span style='font-size:14px;font-weight:700;"
                    f"color:{card_color}'>Rule {i + 1} &nbsp;→&nbsp; "
                    f"Prefer Patient {pref}</span>"

                    f"<span style='display:flex;gap:6px;flex-shrink:0'>"
                    f"<span style='font-size:11px;padding:2px 9px;"
                    f"border-radius:12px;font-weight:600;"
                    f"background:{T_r['BG3']};color:{strength_col}'>"
                    f"{strength_lbl}</span>"
                    f"<span style='font-size:11px;padding:2px 9px;"
                    f"border-radius:12px;"
                    f"background:{T_r['BG3']};color:{cov_col}'>"
                    f"{cov_lbl} · {sup:.0%}</span>"
                    f"</span>"

                    f"</div>"  # end header row

                    # ── Raw rule block ────────────────────────────────────────
                    f"<div style='font-family:monospace;font-size:11.5px;"
                    f"color:{T_r['TEXT_DIM']};"
                    f"background:{T_r['BG3']};"
                    f"padding:8px 12px;border-radius:6px;"
                    f"margin-bottom:12px;"
                    f"white-space:pre-wrap;word-break:break-word'>"
                    f"IF &nbsp; {row['rule']}<br>"
                    f"THEN prefer Patient {pref}</div>"

                    # ── LLM explanation ───────────────────────────────────────
                    f"<div style='font-size:13.5px;line-height:1.65;"
                    f"color:{T_r['TEXT']}'>{explanation}</div>"

                    # ── Footer: coefficient bar ───────────────────────────────
                    f"<div style='margin-top:10px;display:flex;"
                    f"align-items:center;gap:8px'>"
                    f"<span style='font-size:11px;color:{T_r['TEXT_MUTED']}'>"
                    f"Strength</span>"
                    f"<div style='flex:1;height:4px;"
                    f"background:{T_r['BG3']};border-radius:2px'>"
                    f"<div style='width:{min(100, int(imp / 2.0 * 100))}%;"
                    f"height:4px;background:{card_color};"
                    f"border-radius:2px'></div>"
                    f"</div>"
                    f"<span style='font-size:11px;color:{T_r['TEXT_MUTED']}'>"
                    f"{imp:.3f}</span>"
                    f"</div>"  # end footer

                    f"</div>",  # end card
                    unsafe_allow_html=True,
                )

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
>>>>>>> cc1120d344b8eb2aa7bd5a5d00c2bea509caf1af
        )

        # When editing, use pending_tree if one exists (carries added nodes)
        if editing:
            if st.session_state.pending_tree is None:
                st.session_state.pending_tree = {
                    **rtree_dict,
                    "nodes": list(rtree_dict["nodes"]),
                }
            working_tree = st.session_state.pending_tree
        else:
            working_tree = rtree_dict
            st.session_state.pending_tree = None   # clear when not editing

        # Add decision step button (native Streamlit — reliable)
        if editing:
            if st.button("＋  Add decision step", key="add_step"):
                wt   = st.session_state.pending_tree
                used = {n["feature"] for n in wt["nodes"]}
                feat = next(
                    (p + "_diff" for p in rparams if p + "_diff" not in used),
                    rparams[0] + "_diff" if rparams else "age_diff",
                )
                new_node = {
                    "feature": feat, "op": ">=", "threshold": 0.0,
                    "exit_class": 1, "support": 0.0, "purity": 0.5,
                }
                st.session_state.pending_tree = {
                    **wt, "nodes": wt["nodes"] + [new_node]
                }
                st.rerun()

        # Per-node "add a node to the right" — a manual tie-breaker, same shape
        # as the automatic near-tie refine in fft_model, but available on any
        # step by choice rather than only when the model auto-detects a close
        # call. The live JS tree editor (fft_viz) doesn't know how to draw
        # this, so the small SVG preview right below shows it directly.
        if editing:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.markdown("**Add a node to the right of a step**")
            st.markdown(
                f"<div style='color:{COLORS['text_secondary']};font-size:13px;"
                f"margin-bottom:12px;line-height:1.5'>Turn any step's end point into one "
                f"more check, instead of exiting straight to Prefer A / Prefer B.</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div class='content-card'>", unsafe_allow_html=True)
            wt = st.session_state.pending_tree
            feat_options = [p + "_diff" for p in rparams] or ["age_diff"]
            for i, node in enumerate(wt["nodes"]):
                has_refine = bool(node.get("refine"))
                if i > 0:
                    st.markdown(
                        f"<div style='border-top:1px solid {COLORS['border']};"
                        f"margin:12px 0'></div>", unsafe_allow_html=True,
                    )
                rc1, rc2 = st.columns([4, 1.4])
                with rc1:
                    st.markdown(
                        f"<div style='padding-top:9px;font-size:14px;font-weight:500;"
                        f"color:{COLORS['text']}'>Step {i + 1} · "
                        f"{_pretty_feature_label(node['feature'])}</div>",
                        unsafe_allow_html=True,
                    )
                with rc2:
                    if has_refine:
                        if st.button("Remove node", key=f"rm_refine_{i}", use_container_width=True):
                            new_nodes = list(wt["nodes"])
                            new_nodes[i] = {k: v for k, v in new_nodes[i].items() if k != "refine"}
                            st.session_state.pending_tree = {**wt, "nodes": new_nodes}
                            st.rerun()
                    else:
                        if st.button("＋ Add node →", key=f"add_refine_{i}", use_container_width=True):
                            used = {node["feature"]}
                            feat = next((f for f in feat_options if f not in used), feat_options[0])
                            new_nodes = list(wt["nodes"])
                            new_nodes[i] = {
                                **new_nodes[i],
                                "refine": {
                                    "feature": feat, "op": ">=", "threshold": 0.0,
                                    "true_class": 1, "false_class": 0,
                                    "support": 0.0, "false_support": 0.0,
                                    "purity": 0.5, "false_purity": 0.5,
                                    "manual": True,
                                },
                            }
                            st.session_state.pending_tree = {**wt, "nodes": new_nodes}
                            st.rerun()

                if node.get("refine"):
                    r = node["refine"]
                    st.markdown(
                        f"<div style='color:{COLORS['text_muted']};font-size:12px;"
                        f"margin:8px 0 6px'>If close on this step, check —</div>",
                        unsafe_allow_html=True,
                    )
                    ec1, ec2, ec3, ec4 = st.columns([2, 1, 1, 1.3])
                    with ec1:
                        new_feat = st.selectbox(
                            "Factor", feat_options,
                            index=feat_options.index(r["feature"]) if r["feature"] in feat_options else 0,
                            key=f"refine_feat_{i}", label_visibility="collapsed",
                            format_func=_pretty_feature_label,
                        )
                    with ec2:
                        new_op = st.selectbox(
                            "Op", [">=", "<="], index=0 if r["op"] == ">=" else 1,
                            key=f"refine_op_{i}", label_visibility="collapsed",
                        )
                    with ec3:
                        new_thr = st.number_input(
                            "Threshold", value=float(r["threshold"]), step=0.5,
                            key=f"refine_thr_{i}", label_visibility="collapsed",
                        )
                    with ec4:
                        new_pref = st.selectbox(
                            "If true", ["Prefer A", "Prefer B"],
                            index=0 if r["true_class"] == 1 else 1,
                            key=f"refine_pref_{i}", label_visibility="collapsed",
                        )
                    updated = {
                        "feature": new_feat, "op": new_op, "threshold": float(new_thr),
                        "true_class": 1 if new_pref == "Prefer A" else 0,
                        "false_class": 0 if new_pref == "Prefer A" else 1,
                        "support": r.get("support", 0.0), "false_support": r.get("false_support", 0.0),
                        "purity": r.get("purity", 0.5), "false_purity": r.get("false_purity", 0.5),
                        "manual": True,
                    }
                    if updated != r:
                        new_nodes = list(wt["nodes"])
                        new_nodes[i] = {**new_nodes[i], "refine": updated}
                        st.session_state.pending_tree = {**wt, "nodes": new_nodes}
                        wt = st.session_state.pending_tree
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.caption(
                "Preview — the tree editor above doesn't draw these added nodes yet, "
                "so here's what your changes actually look like:"
            )
            preview_svg = fft_svg(st.session_state.pending_tree, DEFAULT_FFT_PALETTE)
            st.markdown(preview_svg, unsafe_allow_html=True)
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        edited_tree = fft_viz(
            tree=working_tree, editing=editing, params=rparams, key="fft_main"
        )

        if edited_tree is not None:
            save_fft_override(st.session_state.username, edited_tree)
            train_fft_cached.clear()
            st.session_state.wants_edit  = None
            st.session_state.pending_tree = None
            st.session_state._saved_msg  = True
            st.rerun()

        if st.session_state.get("_saved_msg", False):
            st.success("Changes saved. The model now reflects your edits.")
            st.session_state._saved_msg = False

        # Plain-English view: same tree as above, but every step is captioned
        # in simple language, close calls get their own tie-breaker box, and
        # there's a short summary of what the participant seems to value.
        # Always shows the last *trained* tree (rtree_dict), not an in-progress
        # edit, since explanations are only generated at training time.
        with st.expander("💬 Explain this model simply", expanded=not editing):
            explained_svg = fft_svg_explained(
                tree=rtree_dict,
                palette=DEFAULT_FFT_PALETTE,
                node_explanations=fft_stats.get("node_explanations"),
                summary_explanation=fft_stats.get("summary_explanation"),
            )
            st.markdown(explained_svg, unsafe_allow_html=True)

    # ── RIGHT: alignment rating + edit prompt ─────────────────────────────────
    with col_fb:
        st.markdown("<div class='fb-card'>", unsafe_allow_html=True)

        st.markdown("### How well does this match your thinking?")
        st.markdown(
            f"<div style='color:{COLORS['text_secondary']};font-size:14px;margin-bottom:16px'>"
            f"1 = not at all &nbsp;&nbsp;·&nbsp;&nbsp; 5 = perfectly</div>",
            unsafe_allow_html=True,
        )

        score_cols    = st.columns(5)
        current_score = st.session_state.alignment_score
        for i, rcol in enumerate(score_cols):
            with rcol:
                score    = i + 1
                btn_type = "primary" if current_score == score else "secondary"
                if st.button(str(score), key=f"align_{score}",
                             type=btn_type, use_container_width=True):
                    st.session_state.alignment_score = score
                    ru = load_users()
                    ru.setdefault(st.session_state.username, {})
                    ru[st.session_state.username]["alignment_score"] = score
                    save_users(ru)
                    st.rerun()

        # Editing status note
        if editing:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.info("Editing mode is on. Make your changes on the left, then click **Apply changes**.")

        # Correction prompt — only after rating, only when not yet decided
        elif st.session_state.alignment_score is not None and st.session_state.wants_edit is None:
            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("### Want to correct anything?")
            cy, cn = st.columns(2)
            with cy:
                if st.button("Yes, edit it", type="primary",
                             use_container_width=True, key="edit_yes"):
                    st.session_state.wants_edit = True
                    st.rerun()
            with cn:
                if st.button("No, it's good", use_container_width=True, key="edit_no"):
                    st.session_state.wants_edit   = False
                    st.session_state.pending_tree = None
                    st.rerun()

        if st.session_state.wants_edit is False:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.success("Part 1 is done and saved.")
            st.markdown(
                f"<div style='color:{COLORS['text_secondary']};font-size:14px;margin:10px 0 16px;"
                f"line-height:1.6'>One more thing — we'd like to show you 20 more scenarios, "
                f"then ask once more whether the model still feels right.</div>",
                unsafe_allow_html=True,
            )
            if st.button("Continue to Part 2 →", type="primary",
                         use_container_width=True, key="go_survey"):
                if not st.session_state.survey_scenarios:
                    st.session_state.survey_scenarios = generate_survey_scenarios(
                        rparams, st.session_state.scenarios, n=20
                    )
                    save_session()
                st.session_state.page = "survey_questionnaire"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SURVEY RESULTS  (Part 2 — final check, after 40 total decisions)
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "survey_results":
    rparams = st.session_state.params
    all_decisions = (
        [d for d in st.session_state.decisions if d.get("choice") in ("A", "B")]
        + [d for d in st.session_state.survey_decisions if d.get("choice") in ("A", "B")]
    )

    if len(all_decisions) < 6:
        st.warning("Not enough answered scenarios to rebuild the model yet.")
        st.stop()

    _override      = load_fft_override(st.session_state.username)
    _override_json = json.dumps(_override) if _override else ""

    with st.spinner("Updating your preference model with the new scenarios…"):
        fft, nodes_df, fft_stats, feat_names, fft_error = train_fft_cached(
            json.dumps(all_decisions), tuple(rparams), _override_json
        )

    if fft_error or fft_stats is None:
        st.warning("We couldn't rebuild the model — your Part 1 model is still saved.")
        st.stop()

    rtree_dict = fft_stats["tree"]

    # Part 1's tree, for the "what changed" review below. Same cache key as the
    # results page used, so this is a cache hit, not a second real train.
    part1_decisions = [d for d in st.session_state.decisions if d.get("choice") in ("A", "B")]
    _, _, part1_stats, _, part1_error = train_fft_cached(
        json.dumps(part1_decisions), tuple(rparams), _override_json
    )
    review_text = (
        summarize_model_changes(part1_stats["tree"], rtree_dict)
        if (part1_stats and not part1_error) else None
    )

    _, mcol, _ = st.columns([1, 5, 1])
    with mcol:
        st.markdown(f"<div class='eyebrow' style='margin-bottom:8px'>Part 2 · complete</div>",
                    unsafe_allow_html=True)
        st.markdown("## Here's your updated model")
        st.markdown(
            f"<div style='color:{COLORS['text_secondary']};font-size:15px;margin-bottom:20px;"
            f"line-height:1.6'>This reflects all 40 of your choices — the original 20 plus "
            f"the 20 you just answered. Take a look, then let us know if it still feels right."
            f"</div>",
            unsafe_allow_html=True,
        )

        # Read-only view — no re-editing in Part 2, keeps this a single final check.
        fft_viz(tree=rtree_dict, editing=False, params=rparams, key="fft_survey")

        with st.expander("💬 Explain this model simply", expanded=True):
            explained_svg = fft_svg_explained(
                tree=rtree_dict,
                palette=DEFAULT_FFT_PALETTE,
                node_explanations=fft_stats.get("node_explanations"),
                summary_explanation=fft_stats.get("summary_explanation"),
            )
            st.markdown(explained_svg, unsafe_allow_html=True)

        if review_text:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='content-card'>"
                f"<div style='font-size:14px;font-weight:600;color:{COLORS['text']};"
                f"margin-bottom:6px'>📋 Model review — what changed since Part 1</div>"
                f"<div style='font-size:14px;color:{COLORS['text_secondary']};line-height:1.6'>"
                f"{review_text}</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### After the extra scenarios, does this still feel right?")
        st.markdown(
            f"<div style='color:{COLORS['text_secondary']};font-size:14px;margin-bottom:16px'>"
            f"1 = not at all &nbsp;&nbsp;·&nbsp;&nbsp; 5 = perfectly</div>",
            unsafe_allow_html=True,
        )

        score_cols    = st.columns(5)
        current_score = st.session_state.survey_alignment_score
        for i, rcol in enumerate(score_cols):
            with rcol:
                score    = i + 1
                btn_type = "primary" if current_score == score else "secondary"
                if st.button(str(score), key=f"svy_align_{score}",
                             type=btn_type, use_container_width=True):
                    st.session_state.survey_alignment_score = score
                    ru = load_users()
                    ru.setdefault(st.session_state.username, {})
                    ru[st.session_state.username]["survey_alignment_score"] = score
                    save_users(ru)
                    st.rerun()

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        comment = st.text_area(
            "Anything you'd add? (optional)",
            key="survey_comment_input",
            placeholder="e.g. which step feels off, or what you'd change",
        )
        if st.button("Submit review", key="submit_survey_comment"):
            ru = load_users()
            ru.setdefault(st.session_state.username, {})
            ru[st.session_state.username]["survey_comment"] = comment
            save_users(ru)
            st.success("Thanks — your comments were saved.")

        if st.session_state.survey_alignment_score is not None:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.success(
                "You've completed both parts of the study — thank you. "
                "Your responses have been saved."
            )
