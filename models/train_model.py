"""
FIFA Oracle — Match Prediction Engine
XGBoost-powered model that predicts match outcomes, scorelines,
and expected goals using team ratings, form, and historical data.
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ── Load Data ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data/raw")
MODEL_DIR = os.path.join(BASE_DIR, "../models")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    matches = pd.read_csv(os.path.join(DATA_DIR, "historical_matches.csv"))
    features = pd.read_csv(os.path.join(DATA_DIR, "team_features.csv"))
    with open(os.path.join(DATA_DIR, "team_ratings.json")) as f:
        ratings = json.load(f)
    return matches, features, ratings

# ── Feature Engineering ──────────────────────────────────────────────────────
def build_features(matches, team_features):
    """Build ML feature matrix from match + team data."""
    tf = team_features.set_index("team")

    rows = []
    for _, row in matches.iterrows():
        ta, tb = row["team_a"], row["team_b"]
        if ta not in tf.index or tb not in tf.index:
            continue

        ra = tf.loc[ta]
        rb = tf.loc[tb]

        stage_map = {"friendly": 0, "group": 1, "round_of_16": 2,
                     "quarter": 3, "semi": 4, "final": 5}

        feat = {
            # Rating differential
            "elo_diff":          ra["elo_rating"] - rb["elo_rating"],
            "elo_a":             ra["elo_rating"],
            "elo_b":             rb["elo_rating"],

            # Form
            "form_diff":         ra["form_points"] - rb["form_points"],
            "form_a":            ra["form_points"],
            "form_b":            rb["form_points"],

            # Goal threat
            "goals_scored_diff": ra["avg_goals_scored"] - rb["avg_goals_scored"],
            "goals_conceded_diff": ra["avg_goals_conceded"] - rb["avg_goals_conceded"],
            "goals_scored_a":    ra["avg_goals_scored"],
            "goals_scored_b":    rb["avg_goals_scored"],
            "goals_conceded_a":  ra["avg_goals_conceded"],
            "goals_conceded_b":  rb["avg_goals_conceded"],

            # Experience
            "wc_titles_diff":    ra["wc_titles"] - rb["wc_titles"],
            "ranking_diff":      rb["world_ranking"] - ra["world_ranking"],
            "squad_value_diff":  ra["squad_value_m"] - rb["squad_value_m"],

            # Match context
            "neutral":           int(row["neutral"]),
            "stage":             stage_map.get(row.get("stage", "group"), 1),

            # Target
            "outcome":           row["outcome"],
            "goals_a":           row["goals_a"],
            "goals_b":           row["goals_b"],
        }
        rows.append(feat)

    return pd.DataFrame(rows)

# ── Train Model ──────────────────────────────────────────────────────────────
def train_model(df):
    """Train XGBoost classifier for match outcome prediction."""
    feature_cols = [c for c in df.columns if c not in ["outcome", "goals_a", "goals_b"]]

    le = LabelEncoder()
    y = le.fit_transform(df["outcome"])  # 0=away_win, 1=draw, 2=home_win

    X = df[feature_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Model accuracy: {acc:.2%}")
    print(f"  Classes: {le.classes_}")

    return model, le, feature_cols, acc

# ── Prediction Function ───────────────────────────────────────────────────────
def predict_match(team_a, team_b, team_features, model, le, feature_cols,
                  neutral=True, stage="group"):
    """
    Predict match outcome between two teams.
    Returns dict with probabilities, predicted score, xG, confidence.
    """
    tf = team_features.set_index("team")

    if team_a not in tf.index or team_b not in tf.index:
        return {"error": f"Team not found. Available: {list(tf.index)}"}

    ra = tf.loc[team_a]
    rb = tf.loc[team_b]

    stage_map = {"friendly": 0, "group": 1, "round_of_16": 2,
                 "quarter": 3, "semi": 4, "final": 5}

    feat = {
        "elo_diff":          ra["elo_rating"] - rb["elo_rating"],
        "elo_a":             ra["elo_rating"],
        "elo_b":             rb["elo_rating"],
        "form_diff":         ra["form_points"] - rb["form_points"],
        "form_a":            ra["form_points"],
        "form_b":            rb["form_points"],
        "goals_scored_diff": ra["avg_goals_scored"] - rb["avg_goals_scored"],
        "goals_conceded_diff": ra["avg_goals_conceded"] - rb["avg_goals_conceded"],
        "goals_scored_a":    ra["avg_goals_scored"],
        "goals_scored_b":    rb["avg_goals_scored"],
        "goals_conceded_a":  ra["avg_goals_conceded"],
        "goals_conceded_b":  rb["avg_goals_conceded"],
        "wc_titles_diff":    ra["wc_titles"] - rb["wc_titles"],
        "ranking_diff":      rb["world_ranking"] - ra["world_ranking"],
        "squad_value_diff":  ra["squad_value_m"] - rb["squad_value_m"],
        "neutral":           int(neutral),
        "stage":             stage_map.get(stage, 1),
    }

    X = pd.DataFrame([feat])[feature_cols]
    probs = model.predict_proba(X)[0]
    classes = le.classes_  # ['away_win', 'draw', 'home_win']

    prob_dict = dict(zip(classes, probs))
    prob_a_win = prob_dict.get("home_win", 0)
    prob_draw  = prob_dict.get("draw", 0)
    prob_b_win = prob_dict.get("away_win", 0)

    # xG estimates
    xg_a = round(ra["avg_goals_scored"] * (1 + feat["elo_diff"] / 2000), 2)
    xg_b = round(rb["avg_goals_scored"] * (1 - feat["elo_diff"] / 2000), 2)
    xg_a = max(0.3, xg_a)
    xg_b = max(0.3, xg_b)

    # Predicted scoreline (most likely)
    pred_score_a = int(round(xg_a))
    pred_score_b = int(round(xg_b))

    # Confidence = max probability * form alignment bonus
    confidence = round(max(prob_a_win, prob_draw, prob_b_win) * 100, 1)

    return {
        "team_a": team_a,
        "team_b": team_b,
        "prob_a_win": round(prob_a_win * 100, 1),
        "prob_draw":  round(prob_draw * 100, 1),
        "prob_b_win": round(prob_b_win * 100, 1),
        "predicted_score": f"{pred_score_a}–{pred_score_b}",
        "xg_a": xg_a,
        "xg_b": xg_b,
        "confidence": confidence,
        "rating_a": int(ra["elo_rating"]),
        "rating_b": int(rb["elo_rating"]),
        "ranking_a": int(ra["world_ranking"]),
        "ranking_b": int(rb["world_ranking"]),
        "stage": stage,
    }

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("FIFA Oracle — Training Prediction Model")
    print("=" * 45)

    matches, features, ratings = load_data()
    print(f"Loaded {len(matches)} matches, {len(features)} teams")

    print("Building feature matrix...")
    df = build_features(matches, features)
    print(f"  ✓ {len(df)} training samples, {len(df.columns)-3} features")

    print("Training XGBoost model...")
    model, le, feature_cols, acc = train_model(df)

    # Save model
    with open(os.path.join(MODEL_DIR, "match_predictor.pkl"), "wb") as f:
        pickle.dump({"model": model, "le": le, "feature_cols": feature_cols, "accuracy": acc}, f)
    print(f"  ✓ Model saved to models/match_predictor.pkl")

    # Test predictions
    print("\n── Test Predictions ──")
    test_matches = [
        ("Argentina", "France", True, "final"),
        ("Brazil", "England", True, "semi"),
        ("Spain", "Germany", True, "quarter"),
        ("Morocco", "Portugal", True, "round_of_16"),
    ]

    for ta, tb, neutral, stage in test_matches:
        result = predict_match(ta, tb, features, model, le, feature_cols, neutral, stage)
        print(f"\n  {ta} vs {tb} ({stage.upper()})")
        print(f"  {ta} Win: {result['prob_a_win']}%  |  Draw: {result['prob_draw']}%  |  {tb} Win: {result['prob_b_win']}%")
        print(f"  Predicted: {result['predicted_score']}  |  xG: {result['xg_a']} – {result['xg_b']}  |  Confidence: {result['confidence']}%")

    print("\n✓ Model training complete!")
