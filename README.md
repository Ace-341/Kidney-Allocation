# Cognitively Aware AI — Preference Elicitation Portal

**SURA 2026 · IIT Delhi · Department of Computer Science**

A research web application for collecting human pairwise preferences and learning an interpretable machine learning model that captures how people reason about resource allocation decisions.

---

## Research Context

When a scarce resource must be allocated between two people — such as an organ for transplant — the decision involves competing ethical principles: medical urgency, fairness, life years gained, social responsibility, and prior access history. Different people weigh these principles differently, and those differences are rarely made explicit.

This project addresses the question: **can we learn a simple, transparent model of human preference from a small number of pairwise comparisons?**

The portal collects decisions from participants, trains an interpretable rule-based model on their choices, and explains what the model learned — in plain English that a doctor, ethicist, or policy-maker can understand.

---

## What It Does

```
Load scenarios CSV  →  Login  →  Answer 20 pairwise comparisons
        ↓
Train RuleFit model on symmetric feature library
        ↓
Show interpretable IF-THEN rules + Symphony-inspired visualisations
        ↓
LLM (Groq / Llama) generates plain-English explanations
        ↓
Detect inconsistencies · Predict new cases · Sensitivity analysis
```

---

## Project Structure

```
Front_end/
├── app.py                          ← Main Streamlit application (2400+ lines)
├── run.py                          ← Cross-platform launcher
├── requirements.txt                ← pip dependencies
├── environment.yml                 ← conda environment (Python 3.11)
├── Procfile                        ← Railway deployment config
├── runtime.txt                     ← Python version for Railway
├── organ_allocation_scenarios.csv  ← 20 curated scenarios (auto-loaded)
├── .env                            ← API keys (not committed to git)
├── users.json                      ← Session state per user (auto-created)
├── responses/                      ← Per-user response files (auto-created)
│   └── <username>_responses.csv
├── venv/                           ← Virtual environment (not committed)
└── olds/                           ← Previous app versions
```

---

## Technical Stack

### Frontend
| Library | Version | Role |
|---------|---------|------|
| `streamlit` | ≥1.32.0 | Web UI — pages, tabs, widgets, session state |
| `matplotlib` | ≥3.7.0 | All charts and visualisations |

### Machine Learning
| Library | Version | Role |
|---------|---------|------|
| `scikit-learn` | ≥1.3.0 | Feature engineering, evaluation, cross-validation |
| `imodels` | ≥1.3.0 | **RuleFitClassifier** — learns interpretable IF-THEN rules. Requires Python 3.9–3.11 |

### Data
| Library | Version | Role |
|---------|---------|------|
| `pandas` | ≥1.5.0 | CSV handling, response saving, data manipulation |
| `numpy` | ≥1.24.0 | Feature computation, matrix operations |

### LLM Integration
| Library | Version | Role |
|---------|---------|------|
| `groq` | ≥0.9.0 | Groq API client for LLM-generated explanations |
| `python-dotenv` | ≥1.0.0 | Load `.env` file for local API key management |

**LLM model used:** `llama-3.1-8b-instant` via Groq (free tier: 14,400 req/day)

---

## Feature Engineering

The model is trained on a symmetric feature library — 20 interpretable features computed from the 6 patient parameters:

### Per-parameter features (×6 parameters)

| Feature | Formula | Type | What it captures |
|---------|---------|------|-----------------|
| `[param] difference (%)` | `(A−B) / mean × 100` | Antisymmetric | Who leads and by how much (%) |
| `[param] similarity (%)` | `(1 − |A−B|/max) × 100` | Symmetric | How similar A and B are on this parameter |
| `[param] higher patient` | `max(A, B)` | Symmetric | The better-scoring patient's value |
| `[param] lower patient` | `min(A, B)` | Symmetric | The worse-scoring patient's value |

**Note:** Age uses `Life years difference (%)` = `(85−A − 85−B) / mean × 100`, so younger patients get a positive value. Prior transplants has no similarity feature (values are 0/1/2 only).

### Composite features

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `Expected treatment benefit difference (%)` | `(health × (85−age))_A − (health × (85−age))_B` | Medical utility — will the transplant actually help? |
| `Social responsibility difference (%)` | `(dependents × (85−age))_A − ... _B` | Social impact — how many people depend on them for how long? |
| `Vulnerability index difference (%)` | `(urgency × years_waiting)_A − ... _B` | Combined need — urgent AND long-ignored |

All difference features are **antisymmetric**: swapping A↔B negates them. This, combined with data augmentation, ensures the model is symmetric.

---

## Model Architecture

### RuleFit (imodels)
1. **Grow trees** — 10 shallow decision trees (max 4 leaf nodes each) on augmented symmetric features
2. **Extract rules** — every tree path becomes a binary feature (does this patient pair satisfy the rule?)
3. **Sparse logistic regression** — L1 regularisation zeros out weak rules, leaving only the most predictive ones
4. **Filter** — rules are filtered by importance (≥0.05) and support (≥10% of decisions), deduplicated by parameter, capped at 10

### Data Augmentation
Every decision `(A, B, y)` is duplicated as `(B, A, 1−y)`. This forces the model to learn symmetric patterns — it cannot treat "which column is A" as a signal.

### Why RuleFit?
Rules like `IF Life years difference (%) > 25 AND Urgency difference (%) > 30 → prefer A` are immediately readable by a doctor or ethicist without any ML background.

---

## Visualisations (Symphony-inspired)

The Charts tab implements four visualisations, each answering a different question (following Symphony's principle of task-specific components):

| Chart | Question answered |
|-------|------------------|
| **Rule Coefficients** | Which rules does the model use and how strongly? |
| **Importance Ranking + Confidence Strip** | Which rules matter most, and which decisions was the model uncertain about? |
| **Decision Boundary Heatmap** | Where exactly does the model draw the line between A and B? |
| **Parameter Importance Radar** | Which parameters dominate the model overall? |

**Symphony reference:** Bäuerle et al. (2022). *Symphony: Composing Interactive Interfaces for Machine Learning.* CHI 2022.

---

## LLM Explanations

Three sections use Groq (Llama 3.1 8B) for plain-English explanations:

| Section | What the LLM explains |
|---------|-----------------------|
| **Model Card → What the model learned** | Summarises the participant's values in 2-3 sentences for a medical ethics audience |
| **Examples tab** | Why the model agreed or disagreed with each specific decision |
| **Analysis → Consistency Check** | Why the model predicted differently for two similar scenarios, and what value tension it reveals |

The LLM falls back to rule-based text if the API key is missing or Groq is unavailable.

---

## Model & Results Tabs

| Tab | Contents |
|-----|----------|
| 📄 **Model Card** | Training summary, accuracy, symmetry %, data sufficiency warning, plain-English rule summary, limitations |
| 📋 **Rules** | IF-THEN rules with coverage % and strength |
| 📊 **Charts** | Four Symphony-inspired visualisations |
| 🎯 **Predict New Pair** | Enter two patients → prediction + confidence tier + counterfactual + parameter contributions waterfall + sensitivity slider |
| 🔍 **Examples** | 3 agreements and 3 disagreements, each with LLM explanation |
| 🧠 **Analysis** | Scenario difficulty bar chart + consistency checker with per-conflict model explanations |

---

## CSV Format

The app auto-loads `organ_allocation_scenarios.csv` from the same folder as `app.py`.

Columns must be `A_<param>` and `B_<param>` pairs:

```
A_age, A_years_waiting, A_health_score, A_dependents, A_prior_transplants, A_urgency_score,
B_age, B_years_waiting, B_health_score, B_dependents, B_prior_transplants, B_urgency_score
35, 3, 8, 1, 0, 6,   60, 9, 5, 2, 0, 7
```

- Any number of parameters · Any parameter names · All values must be numeric

The included `organ_allocation_scenarios.csv` has 20 scenarios designed using **optimal design principles** (60% near-tie, 20% clear-winner, 20% edge-case) to maximise information gained per comparison.

---

## Setup

### ⚠️ Python Version

| Feature | Python requirement |
|---------|-------------------|
| App, UI, response saving | Any Python 3.9+ |
| RuleFit model (imodels) | **Python 3.9–3.11 only** (not 3.12) |

---

### Mac / Linux — venv

```bash
# Install Python 3.11
brew install python@3.11

# Create and activate venv
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install imodels groq python-dotenv

# Add Groq API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# Run
streamlit run app.py
```

### Windows

```cmd
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install imodels groq python-dotenv
echo GROQ_API_KEY=gsk_your_key_here > .env
streamlit run app.py
```

### Conda (alternative)

```bash
conda env create -f environment.yml
conda activate imodels_env
pip install groq python-dotenv
echo "GROQ_API_KEY=gsk_your_key_here" > .env
streamlit run app.py
```

---

## Getting a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up with Google or GitHub — no credit card needed
3. API Keys → Create API Key
4. Copy the key (starts with `gsk_...`) → paste into `.env`

Free tier: 14,400 requests/day, 6,000 tokens/minute — sufficient for research use.

---

## Deployment (Railway)

```
Procfile:   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
runtime.txt: python-3.11.8
```

**Railway setup:**
1. Push to GitHub
2. railway.app → New Project → Deploy from GitHub
3. Settings → Root Directory → `RoughWork/Front_end`
4. Variables → add `GROQ_API_KEY` and `MISE_PYTHON_GITHUB_ATTESTATIONS=false`
5. Networking → Generate Domain (port 8080)

---

## Theme System

The app supports light and dark modes via a toggle button (🌙/☀️) in the sidebar.

- Always starts in **light mode**
- Theme is stored in `st.session_state["theme_mode"]`
- `THEMES` dict contains two complete palettes (18 named colours each)
- `inject_theme_css()` generates CSS overrides for all Streamlit components
- All charts call `_setup_colours()` to unpack the active theme's colours as local variables
- No hardcoded colours anywhere — adding a new UI element means only using the named variables

---

## Output Files

| File | Description |
|------|-------------|
| `responses/<username>_responses.csv` | One row per decision — auto-appended on each answer |
| `users.json` | Session progress — allows users to resume incomplete sessions |

---

## .gitignore

```
venv/
__pycache__/
*.pyc
.DS_Store
.env
responses/
users.json
```

---

## References

1. **RuleFit:** Friedman & Popescu (2008). *Predictive Learning via Rule Ensembles.* Annals of Applied Statistics.
2. **imodels:** [github.com/csinva/imodels](https://github.com/csinva/imodels)
3. **Symphony:** Bäuerle et al. (2022). *Symphony: Composing Interactive Interfaces for Machine Learning.* CHI 2022. [apple.github.io/ml-symphony](https://apple.github.io/ml-symphony)
4. **Optimal preference elicitation:** [arxiv.org/abs/2404.13895](https://arxiv.org/abs/2404.13895)
5. **Groq:** [console.groq.com](https://console.groq.com)
6. **Streamlit:** [streamlit.io](https://streamlit.io)
