"""
Preference Elicitation Portal
SURA 2026 · IIT Delhi

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

from fft_model import train_fft
from fft_viz import fft_viz

st.set_page_config(
    page_title="Organ Allocation Preference Study",
    page_icon="🫘",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit chrome */
#MainMenu, footer { display: none !important; }
[data-testid="stToolbar"]  { display: none !important; }
[data-testid="stSidebar"]  { display: none !important; }
[data-testid="stHeader"]   { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* Content width */
.main .block-container {
  max-width: 780px !important;
  padding-top: 3rem !important;
  padding-bottom: 4rem !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
}

/* Base font */
html, body, [class*="css"] { font-size: 16px; }

/* Headings */
h1 { font-size: 2rem !important; font-weight: 700 !important; line-height: 1.2 !important; }
h2 { font-size: 1.5rem !important; font-weight: 600 !important; line-height: 1.3 !important; }
h3 { font-size: 1.2rem !important; font-weight: 600 !important; }

/* Paragraph text */
.stMarkdown p { font-size: 16px !important; line-height: 1.7 !important; }
.stMarkdown li { font-size: 16px !important; line-height: 1.7 !important; }

/* Buttons */
.stButton > button {
  font-size: 16px !important;
  padding: 12px 24px !important;
  min-height: 48px !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
  transition: all 0.15s !important;
}
.stButton > button[kind="primary"] {
  background-color: #2563eb !important;
  color: #ffffff !important;
  border: none !important;
}
.stButton > button[kind="primary"]:hover { background-color: #1d4ed8 !important; }
.stButton > button[kind="secondary"] {
  background-color: #f8fafc !important;
  color: #0f172a !important;
  border: 1px solid #e2e8f0 !important;
}

/* Text input */
.stTextInput > div > div > input {
  font-size: 18px !important;
  padding: 14px 16px !important;
  border-radius: 8px !important;
  border: 1.5px solid #e2e8f0 !important;
  background: #f8fafc !important;
}
.stTextInput > div > div > input:focus {
  border-color: #2563eb !important;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
}

/* Progress bar */
[data-testid="stProgressBar"] > div {
  height: 4px !important;
  border-radius: 2px !important;
  background: #e2e8f0 !important;
}
[data-testid="stProgressBar"] > div > div {
  background: #2563eb !important;
}

/* Divider */
hr { border-color: #e2e8f0 !important; margin: 2.5rem 0 !important; }
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
        "sc_index":  st.session_state.sc_index,
        "decisions": st.session_state.decisions,
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

@st.cache_data(show_spinner=False)
def train_fft_cached(decisions_json, params_tuple, override_json):
    return train_fft(decisions_json, list(params_tuple), override_json or None)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LOGIN
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "login":

    st.markdown("## Organ Allocation Preference Study")
    st.markdown("*SURA 2026 · IIT Delhi*")

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    st.markdown(
        "You will be shown **pairs of patients** who need an organ transplant. "
        "For each pair, choose which patient you think should receive it. "
        "There are no right or wrong answers — we are learning about *your* reasoning."
    )
    st.markdown(
        "After all scenarios, you will see a simple decision model that captures "
        "the pattern in your choices."
    )

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
    st.markdown("**Each patient is described by six factors:**")

    params = st.session_state.params
    for p in params:
        desc = PARAM_DESCRIPTIONS.get(p, p.replace("_", " ").title() + ".")
        label = p.replace("_", " ").title()
        st.markdown(f"- **{label}** — {desc}")

    st.markdown("<div style='height: 32px'></div>", unsafe_allow_html=True)
    st.markdown("**Enter your name to start:**")

    uname = st.text_input(
        "Name",
        placeholder="e.g. Participant 1",
        label_visibility="collapsed",
        key="login_name_input",
    )

    st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)

    if st.button("Continue →", type="primary", use_container_width=True, key="login_btn"):
        raw = uname.strip()
        if len(raw) < 2:
            st.error("Please enter at least 2 characters.")
        else:
            users = load_users()
            st.session_state.username = raw
            if raw in users:
                ud = users[raw]
                st.session_state.sc_index  = ud.get("sc_index",  0)
                st.session_state.decisions = ud.get("decisions", [])
                # Restore alignment score if previously set
                if "alignment_score" in ud:
                    st.session_state.alignment_score = ud["alignment_score"]
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
    params  = st.session_state.params
    n_total = len(st.session_state.scenarios)

    if st.session_state.sc_index >= n_total:
        st.session_state.page = "results"
        st.rerun()

    current_idx = st.session_state.sc_index
    sc          = st.session_state.scenarios[current_idx]

    # Progress indicator
    st.progress(current_idx / n_total)
    st.markdown(
        f"<div style='color: #94a3b8; font-size: 14px; margin-top: 6px; margin-bottom: 32px;'>"
        f"Scenario {current_idx + 1} of {n_total}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("## Which patient should receive the organ?")
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    for col, label, data, color in [
        (col_a, "Patient A", sc["A"], "#b91c1c"),
        (col_b, "Patient B", sc["B"], "#1d4ed8"),
    ]:
        with col:
            # Patient header
            st.markdown(
                f"<div style='border-left: 4px solid {color}; padding: 20px 22px 20px 18px;'>"
                f"<div style='color: {color}; font-weight: 700; font-size: 17px; "
                f"letter-spacing: 0.04em; margin-bottom: 22px;'>{label}</div>",
                unsafe_allow_html=True,
            )
            # Parameters
            for p in params:
                pname = p.replace("_", " ").title()
                val   = data[p]
                st.markdown(
                    f"<div style='margin-bottom: 18px;'>"
                    f"<div style='font-size: 12px; color: #94a3b8; text-transform: uppercase; "
                    f"letter-spacing: 0.08em; margin-bottom: 4px;'>{pname}</div>"
                    f"<div style='font-size: 36px; font-weight: 600; font-family: monospace; "
                    f"line-height: 1; color: #0f172a;'>{val:g}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)

    btn_a, btn_b = st.columns(2)
    with btn_a:
        if st.button(
            "Patient A should receive it",
            use_container_width=True,
            type="primary",
            key=f"btn_A_{current_idx}",
        ):
            record_decision("A")
            st.rerun()
    with btn_b:
        if st.button(
            "Patient B should receive it",
            use_container_width=True,
            type="primary",
            key=f"btn_B_{current_idx}",
        ):
            record_decision("B")
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "results":
    params    = st.session_state.params
    decisions = [d for d in st.session_state.decisions if d.get("choice") in ("A", "B")]

    if len(decisions) < 6:
        st.warning(f"Need at least 6 answered scenarios to build a model (you have {len(decisions)}).")
        if st.button("← Back to scenarios"):
            st.session_state.page = "questionnaire"
            st.rerun()
        st.stop()

    # Train the FFT
    decisions_json = json.dumps(decisions)
    _override      = load_fft_override(st.session_state.username)
    _override_json = json.dumps(_override) if _override else ""

    with st.spinner("Building your preference model…"):
        fft, nodes_df, fft_stats, feat_names, fft_error = train_fft_cached(
            decisions_json, tuple(params), _override_json
        )

    if fft_error or fft_stats is None:
        st.warning(
            "Could not build a model from your decisions. "
            "Try answering more scenarios with varied choices."
        )
        if st.button("← Back to scenarios"):
            st.session_state.page = "questionnaire"
            st.rerun()
        st.stop()

    # ── Heading ───────────────────────────────────────────────────────────────
    st.markdown("## Your Preference Model")
    st.markdown(
        "<div style='color: #64748b; font-size: 16px; margin-bottom: 36px; line-height: 1.6;'>"
        "Based on your decisions, here is the decision tree that best describes your thinking. "
        "Each step is a single question the tree asks about the two patients."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── FFT Visualization ─────────────────────────────────────────────────────
    tree_dict = fft_stats["tree"]
    editing   = (st.session_state.wants_edit is True)

    edited_tree = fft_viz(
        tree=tree_dict,
        editing=editing,
        params=params,
        key="fft_main",
    )

    if edited_tree is not None:
        save_fft_override(st.session_state.username, edited_tree)
        train_fft_cached.clear()
        st.session_state.wants_edit = None
        st.session_state._saved_msg = True
        st.rerun()

    # Show one-time save confirmation
    if st.session_state.get("_saved_msg", False):
        st.success("Your corrections have been saved. The model now reflects your edits.")
        st.session_state._saved_msg = False

    # ── Feedback (hidden while in editing mode) ───────────────────────────────
    if not editing:
        st.markdown("<div style='height: 48px'></div>", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### How much do you think this model aligns with your thinking?")
        st.markdown(
            "<div style='color: #64748b; font-size: 15px; margin-bottom: 20px;'>"
            "1 = not at all &nbsp;&nbsp; 5 = perfectly"
            "</div>",
            unsafe_allow_html=True,
        )

        score_cols    = st.columns(5)
        current_score = st.session_state.alignment_score

        for i, col in enumerate(score_cols):
            with col:
                score    = i + 1
                btn_type = "primary" if current_score == score else "secondary"
                if st.button(str(score), key=f"align_{score}", type=btn_type, use_container_width=True):
                    st.session_state.alignment_score = score
                    u = load_users()
                    u.setdefault(st.session_state.username, {})
                    u[st.session_state.username]["alignment_score"] = score
                    save_users(u)
                    st.rerun()

        # Editing prompt — only after rating, and when not already decided
        if st.session_state.alignment_score is not None and st.session_state.wants_edit is None:
            st.markdown("<div style='height: 32px'></div>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("### Would you like to correct the model?")

            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, let me edit it", type="primary", use_container_width=True, key="edit_yes"):
                    st.session_state.wants_edit = True
                    st.rerun()
            with col_no:
                if st.button("No, I'm done", use_container_width=True, key="edit_no"):
                    st.session_state.wants_edit = False
                    st.rerun()

        if st.session_state.wants_edit is False:
            st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
            st.success("Thank you for participating. Your responses have been saved.")
