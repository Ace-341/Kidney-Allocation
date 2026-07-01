"""
Fast-and-Frugal Tree (FFT) model for the Preference Elicitation Portal.
SURA 2026 · IIT Delhi

This module is the *only* predictive model used by the application. It is kept
deliberately self-contained (pure numpy / pandas / scikit-learn metric) so it
can be retrained, unit-tested and extended independently of the Streamlit UI.

Design
------
* Features are ONLY the per-parameter differences between the two candidates:
      diff(param) = value(param, A) − value(param, B)
  Nothing else is engineered. Direction ("is higher better?") is learned by the
  tree itself through the split operator, so no hand-tuned transforms are needed.

* A Fast-and-Frugal Tree is an ordered checklist of single-cue tests. At every
  level exactly one cue (one difference feature) is tested; the TRUE branch is an
  immediate exit (a leaf that decides A or B) and the FALSE branch falls through
  to the next cue. After the last cue, a single default leaf catches everything
  that fell through. For k cues there are k+1 exits.

      node i :  if  (feature_i  op_i  threshold_i)   ->  exit to exit_class_i
                else                                  ->  continue to node i+1
      fall-through after last node                    ->  default_class

  This structure is maximally interpretable and trivially editable: every node is
  one feature, one operator, one threshold and one outcome.

Public API
----------
    build_difference_features(decisions, params) -> (DataFrame, feature_names)
    augment(F, y)                                -> (DataFrame, y)
    FastFrugalTree                               -> .fit / .predict / .predict_proba
                                                    .to_dict / .from_dict / .evaluate
    train_fft(decisions_json, params, override_json=None)
                                                 -> (tree, nodes_df, stats, feat_names, error)
    feature_row(...) / decision_prob(...) / explain_prediction(...)
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


# ════════════════════════════════════════════════════════════════════════════
# FEATURES  —  only pairwise differences, nothing else
# ════════════════════════════════════════════════════════════════════════════

def build_difference_features(decisions, params):
    """
    Build the difference feature matrix.

    For every decision and every parameter p:
        <p>_diff = A_<p> - B_<p>

    Returns (F, feature_names) where F is a DataFrame with one column per param.
    These features are inherently antisymmetric: swapping A and B negates them.
    """
    a = np.array([[float(d.get(f"A_{p}", 0.0)) for p in params] for d in decisions],
                 dtype=float)
    b = np.array([[float(d.get(f"B_{p}", 0.0)) for p in params] for d in decisions],
                 dtype=float)
    if a.size == 0:
        cols = [f"{p}_diff" for p in params]
        return pd.DataFrame(columns=cols), cols
    diff = a - b
    cols = [f"{p}_diff" for p in params]
    return pd.DataFrame(diff, columns=cols), cols


def augment(F, y):
    """
    Double the dataset by swapping A and B. Because every feature is a difference
    (antisymmetric), the swap simply negates the whole matrix and flips the label.
    This forces the learned tree to be (near-)symmetric.
    """
    y = np.asarray(y).astype(int)
    F_swap = -F
    return (
        pd.concat([F, F_swap], ignore_index=True),
        np.concatenate([y, 1 - y]),
    )


# ════════════════════════════════════════════════════════════════════════════
# FAST-AND-FRUGAL TREE
# ════════════════════════════════════════════════════════════════════════════

class FastFrugalTree:
    """
    A depth-limited Fast-and-Frugal Tree over difference features.

    Each node is a dict:
        {feature, feature_idx, op ('>=' | '<='), threshold,
         exit_class (1=A, 0=B), support, purity}

    Class 1 == "prefer candidate A", class 0 == "prefer candidate B".
    """

    def __init__(self, max_depth=4):
        self.max_depth = int(max_depth)
        self.nodes = []
        self.default_class = 0
        self.default_support = 0.0
        self.default_purity = 0.5
        self.feature_names = []

    # ── threshold candidates ────────────────────────────────────────────────
    @staticmethod
    def _candidate_thresholds(x, cap=60):
        u = np.unique(x)
        if len(u) <= 1:
            return np.array([float(u[0]) if len(u) else 0.0])
        mids = (u[:-1] + u[1:]) / 2.0
        cands = np.unique(np.concatenate([mids, [0.0]]))  # 0 is a natural split
        if len(cands) > cap:
            idx = np.linspace(0, len(cands) - 1, cap).astype(int)
            cands = cands[idx]
        return cands

    # ── fit ─────────────────────────────────────────────────────────────────
    def fit(self, F, y, feature_names=None):
        X = np.asarray(F, dtype=float)
        y = np.asarray(y).astype(int)
        n_total = len(y)
        if X.ndim != 2 or n_total == 0:
            return self

        self.feature_names = (list(feature_names) if feature_names is not None
                              else [f"f{i}" for i in range(X.shape[1])])

        remaining = np.ones(n_total, dtype=bool)
        used = set()
        self.nodes = []

        for _level in range(self.max_depth):
            idx = np.where(remaining)[0]
            if len(idx) == 0:
                break
            yr = y[idx]
            if len(np.unique(yr)) == 1:
                break  # already pure — the default leaf will cover it

            best = None  # (score, purity, n_exit, fidx, op, thr, exit_class, mask)
            for fidx in range(X.shape[1]):
                if fidx in used:
                    continue
                xr = X[idx, fidx]
                for thr in self._candidate_thresholds(xr):
                    for op in (">=", "<="):
                        M = (xr >= thr) if op == ">=" else (xr <= thr)
                        n_exit = int(M.sum())
                        if n_exit == 0 or n_exit == len(idx):
                            continue  # intermediate node needs a real split
                        yexit = yr[M]
                        exit_class = int(round(yexit.mean()))
                        n_correct = int((yexit == exit_class).sum())
                        score = n_correct - (n_exit - n_correct)  # net correct exits
                        purity = n_correct / n_exit
                        key = (score, purity, n_exit)
                        if best is None or key > (best[0], best[1], best[2]):
                            best = (score, purity, n_exit, fidx, op,
                                    float(thr), exit_class, M)
            if best is None:
                break

            score, purity, n_exit, fidx, op, thr, exit_class, M = best
            global_exit = idx[M]
            self.nodes.append({
                "feature":     self.feature_names[fidx],
                "feature_idx": fidx,
                "op":          op,
                "threshold":   thr,
                "exit_class":  exit_class,
                "support":     len(global_exit) / n_total,
                "purity":      purity,
            })
            used.add(fidx)
            remaining[global_exit] = False

            idx2 = np.where(remaining)[0]
            if len(idx2) == 0 or len(np.unique(y[idx2])) == 1:
                break

        # default leaf for whatever fell through
        idx = np.where(remaining)[0]
        if len(idx) > 0:
            yr = y[idx]
            self.default_class = int(round(yr.mean()))
            self.default_support = len(idx) / n_total
            self.default_purity = float((yr == self.default_class).mean())
        elif self.nodes:
            self.default_class = 1 - self.nodes[-1]["exit_class"]
            self.default_support = 0.0
            self.default_purity = 0.5
        return self

    # ── single-row evaluation ───────────────────────────────────────────────
    def _row_predict(self, row):
        """Return (predicted_class, exit_node_index, purity). exit_index -1 = default."""
        for i, node in enumerate(self.nodes):
            x = row[node["feature_idx"]]
            cond = (x >= node["threshold"]) if node["op"] == ">=" else (x <= node["threshold"])
            if cond:
                return node["exit_class"], i, node["purity"]
        return self.default_class, -1, self.default_purity

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._row_predict(r)[0] for r in X])

    def predict_proba(self, X):
        """Confidence derived from the purity of the exit each row lands in."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = []
        for r in X:
            cls, _i, purity = self._row_predict(r)
            conf = min(max(float(purity), 0.5), 1.0)
            p1 = conf if cls == 1 else 1.0 - conf
            out.append([1.0 - p1, p1])
        return np.array(out)

    def exit_index(self, row):
        """Index of the node a single row exits at (-1 = default leaf)."""
        return self._row_predict(np.asarray(row, dtype=float))[1]

    def evaluate(self, F, y):
        preds = self.predict(np.asarray(F, dtype=float))
        return float(balanced_accuracy_score(np.asarray(y).astype(int), preds))

    def recompute_stats(self, F, y):
        """
        Recompute per-node support/purity and the default leaf stats against data.
        Used after a user edits an existing tree so the displayed reliability and
        coverage reflect the edited thresholds rather than the original ones.
        """
        X = np.asarray(F, dtype=float)
        y = np.asarray(y).astype(int)
        n_total = max(1, len(y))
        reached = np.ones(len(y), dtype=bool)
        for node in self.nodes:
            xs = X[:, node["feature_idx"]]
            cond = (xs >= node["threshold"]) if node["op"] == ">=" else (xs <= node["threshold"])
            exit_here = reached & cond
            n_exit = int(exit_here.sum())
            node["support"] = n_exit / n_total
            if n_exit > 0:
                node["purity"] = float((y[exit_here] == node["exit_class"]).mean())
            else:
                node["purity"] = 0.5
            reached = reached & ~cond
        n_def = int(reached.sum())
        self.default_support = n_def / n_total
        self.default_purity = (float((y[reached] == self.default_class).mean())
                               if n_def > 0 else 0.5)
        return self

    # ── (de)serialisation ───────────────────────────────────────────────────
    def to_dict(self):
        return {
            "nodes": [
                {"feature": n["feature"], "op": n["op"],
                 "threshold": float(n["threshold"]), "exit_class": int(n["exit_class"]),
                 "support": float(n.get("support", 0.0)), "purity": float(n.get("purity", 0.5))}
                for n in self.nodes
            ],
            "default_class":   int(self.default_class),
            "default_support": float(self.default_support),
            "default_purity":  float(self.default_purity),
            "feature_names":   list(self.feature_names),
            "max_depth":       self.max_depth,
        }

    @classmethod
    def from_dict(cls, d, feature_names=None):
        t = cls(max_depth=d.get("max_depth", 4))
        t.feature_names = list(feature_names or d.get("feature_names") or [])
        name_to_idx = {n: i for i, n in enumerate(t.feature_names)}
        t.nodes = []
        for n in d.get("nodes", []):
            t.nodes.append({
                "feature":     n["feature"],
                "feature_idx": name_to_idx.get(n["feature"], 0),
                "op":          n["op"],
                "threshold":   float(n["threshold"]),
                "exit_class":  int(n["exit_class"]),
                "support":     float(n.get("support", 0.0)),
                "purity":      float(n.get("purity", 0.5)),
            })
        t.default_class   = int(d.get("default_class", 0))
        t.default_support = float(d.get("default_support", 0.0))
        t.default_purity  = float(d.get("default_purity", 0.5))
        return t


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS  (single pair)
# ════════════════════════════════════════════════════════════════════════════

def feature_row(params, a_dict, b_dict, feat_names):
    """Single-row difference DataFrame aligned to feat_names."""
    row = {f"{p}_diff": float(a_dict.get(p, 0)) - float(b_dict.get(p, 0)) for p in params}
    F = pd.DataFrame([row])
    return F.reindex(columns=feat_names, fill_value=0.0)


def decision_prob(tree, params, feat_names, d):
    """Return (pred_int, prob_A) for a decision dict."""
    a = {p: float(d.get(f"A_{p}", 0)) for p in params}
    b = {p: float(d.get(f"B_{p}", 0)) for p in params}
    F = feature_row(params, a, b, feat_names)
    pred = int(tree.predict(F.values)[0])
    prob = float(tree.predict_proba(F.values)[0][1])
    return pred, prob


def _pretty(feature):
    return feature.replace("_diff", "").replace("_", " ").title()


def explain_prediction(tree, params, feat_names, d, pred):
    """Plain-English explanation of which cue decided this pair."""
    a = {p: float(d.get(f"A_{p}", 0)) for p in params}
    b = {p: float(d.get(f"B_{p}", 0)) for p in params}
    F = feature_row(params, a, b, feat_names)
    cls, exit_i, purity = tree._row_predict(F.values[0])
    pred_label = "A" if pred == 1 else "B"

    if exit_i >= 0:
        n = tree.nodes[exit_i]
        return (
            f"The tree decided **Option {pred_label}** at step {exit_i + 1}: "
            f"the condition `{_pretty(n['feature'])} ({n['feature']}) "
            f"{n['op']} {round(n['threshold'], 2)}` held, which exits straight to "
            f"Patient {pred_label}. This cue agreed with your choices "
            f"{purity:.0%} of the time in training."
        )
    return (
        f"No cue triggered for this pair, so the tree fell through to its "
        f"default preference for **Option {pred_label}** "
        f"({tree.default_purity:.0%} reliable on training)."
    )


# ════════════════════════════════════════════════════════════════════════════
# TRAINING ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def train_fft(decisions_json, params, override_json=None, max_depth=4):
    """
    Train (or load an edited) FFT.

    Returns (tree, nodes_df, stats, feat_names, error_msg).
    The tuple shape mirrors the old train_rulefit() so the UI swaps cleanly.

    If override_json is given, that edited tree is used verbatim (the user has
    committed manual edits) instead of learning a fresh one — but all statistics
    are still recomputed against the real training data.
    """
    decisions = json.loads(decisions_json) if isinstance(decisions_json, str) else decisions_json
    F, feat_names = build_difference_features(decisions, params)

    if len(decisions) < 6:
        return None, None, None, feat_names, "Need at least 6 answered decisions to train."

    y = np.array([1 if d["choice"] == "A" else 0 for d in decisions])
    F_aug, y_aug = augment(F, y)

    if override_json:
        tree = FastFrugalTree.from_dict(
            json.loads(override_json) if isinstance(override_json, str) else override_json,
            feature_names=feat_names,
        )
        tree.recompute_stats(F_aug, y_aug)   # refresh support/purity for edited thresholds
        edited = True
    else:
        tree = FastFrugalTree(max_depth=max_depth).fit(F_aug, y_aug, feature_names=feat_names)
        edited = False

    if not tree.nodes:
        return None, None, None, feat_names, (
            "Could not build a tree from these decisions — answer a few more, "
            "more varied scenarios."
        )

    # ── metrics ─────────────────────────────────────────────────────────────
    acc = float(balanced_accuracy_score(y_aug, tree.predict(F_aug.values)))
    p_orig = tree.predict(F.values)
    p_swap = tree.predict((-F).values)
    sym = float(((p_orig + p_swap) == 1).mean())

    # ── nodes_df (drives the readable Rules tab + LLM, mirrors old rules_df) ──
    rows = []
    for n in tree.nodes:
        direction = 1 if n["exit_class"] == 1 else -1   # +A / -B
        strength = max(0.0, (n["purity"] - 0.5) * 2.0)  # 0..1
        rows.append({
            "feature":    n["feature"],
            "rule":       f'{n["feature"]} {n["op"]} {round(n["threshold"], 2)}',
            "op":         n["op"],
            "threshold":  float(n["threshold"]),
            "exit_class": int(n["exit_class"]),
            "support":    float(n["support"]),
            "purity":     float(n["purity"]),
            "coef":       float(direction * strength),
            "importance": float(n["support"] * strength),
        })
    nodes_df = pd.DataFrame(rows)

    # ── per-decision agreement (drives Examples tab) ─────────────────────────
    preds_o = tree.predict(F.values)
    per_decision = []
    for i, (d, actual, predicted) in enumerate(zip(decisions, y, preds_o)):
        per_decision.append({
            "scenario":  d.get("scenario", i + 1),
            "actual":    "A" if actual == 1 else "B",
            "predicted": "A" if predicted == 1 else "B",
            "match":     bool(actual == predicted),
            "decision":  d,
        })

    stats = {
        "acc":          acc,
        "sym":          sym,
        "n_rules":      len(tree.nodes),
        "n_decisions":  len(decisions),
        "top_param":    _pretty(tree.nodes[0]["feature"]),
        "edited":       edited,
        "per_decision": per_decision,
        "tree":         tree.to_dict(),
    }
    return tree, nodes_df, stats, feat_names, None
