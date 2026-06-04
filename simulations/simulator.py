"""
FIFA Oracle — Tournament Simulator
Simulates the FIFA World Cup 10,000+ times using the prediction model.
Outputs win probabilities, final/semi appearances, and dark horse teams.
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
from collections import defaultdict
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../models"))
from train_model import predict_match, load_data

MODEL_PATH = os.path.join(BASE_DIR, "../models/match_predictor.pkl")
DATA_DIR   = os.path.join(BASE_DIR, "../data/raw")

# 2026 World Cup Groups (projected)
WC_2026_GROUPS = {
    "A": ["USA", "Mexico", "Canada", "Uruguay"],
    "B": ["Brazil", "Argentina", "Ecuador", "Japan"],
    "C": ["France", "Belgium", "Morocco", "Senegal"],
    "D": ["Spain", "Portugal", "Croatia", "Serbia"],
    "E": ["England", "Netherlands", "Denmark", "Poland"],
    "F": ["Germany", "Switzerland", "Qatar", "Iran"],
    "G": ["Italy", "South Korea", "Australia", "Ghana"],
    "H": ["Saudi Arabia", "Tunisia", "Cameroon", "Costa Rica"],
}

def load_model_and_data():
    with open(MODEL_PATH, "rb") as f:
        pkg = pickle.load(f)
    _, features, _ = load_data()
    return pkg["model"], pkg["le"], pkg["feature_cols"], features

def simulate_match(team_a, team_b, model, le, feature_cols, features, stage="group"):
    """Simulate a single match, return winner (or draw for group stage)."""
    result = predict_match(team_a, team_b, features, model, le, feature_cols,
                           neutral=True, stage=stage)
    rand = np.random.random() * 100
    if rand < float(result["prob_a_win"]):
        return team_a, "a_win"
    elif rand < float(result["prob_a_win"]) + result["prob_draw"]:
        return None, "draw"  # group stage draw
    else:
        return team_b, "b_win"

def simulate_ko_match(team_a, team_b, model, le, feature_cols, features, stage):
    """KO match — must have a winner (no draws allowed)."""
    result = predict_match(team_a, team_b, features, model, le, feature_cols,
                           neutral=True, stage=stage)
    if "error" in result:
        return team_a if np.random.random() < 0.5 else team_b
    # In KO rounds adjust draw prob to penalties (roughly 50/50 added to both)
    adj_a = float(result["prob_a_win"]) + float(result["prob_draw"]) * 0.5
    adj_b = float(result["prob_b_win"]) + float(result["prob_draw"]) * 0.5
    return team_a if np.random.random() * 100 < adj_a else team_b

def simulate_group_stage(groups, model, le, feature_cols, features):
    """Simulate all group games, return top 2 from each group."""
    qualifiers = {}
    for group, teams in groups.items():
        points = defaultdict(int)
        gd = defaultdict(int)

        for i in range(len(teams)):
            for j in range(i+1, len(teams)):
                ta, tb = teams[i], teams[j]
                winner, outcome = simulate_match(ta, tb, model, le, feature_cols, features, "group")
                if outcome == "a_win":
                    points[ta] += 3
                    gd[ta] += 1; gd[tb] -= 1
                elif outcome == "b_win":
                    points[tb] += 3
                    gd[tb] += 1; gd[ta] -= 1
                else:
                    points[ta] += 1; points[tb] += 1

        # Sort by points, then goal difference
        standings = sorted(teams, key=lambda t: (points[t], gd[t]), reverse=True)
        qualifiers[group] = standings[:2]

    return qualifiers

def simulate_knockout(qualifiers):
    """Simulate R16 → QF → SF → Final."""
    groups = list(qualifiers.keys())
    # R16 pairings: A1 vs B2, B1 vs A2, etc.
    r16_pairs = []
    for i in range(0, len(groups), 2):
        g1, g2 = groups[i], groups[i+1]
        r16_pairs.append((qualifiers[g1][0], qualifiers[g2][1]))
        r16_pairs.append((qualifiers[g2][0], qualifiers[g1][1]))

    return r16_pairs

def simulate_tournament(groups, model, le, feature_cols, features):
    """Full tournament simulation. Returns champion and stage appearances."""
    appearances = defaultdict(lambda: {"r16": 0, "qf": 0, "sf": 0, "final": 0, "champion": 0})

    qualifiers = simulate_group_stage(groups, model, le, feature_cols, features)

    # Build R16
    groups_list = list(qualifiers.keys())
    r16 = []
    for i in range(0, len(groups_list), 2):
        g1, g2 = groups_list[i], groups_list[i+1]
        r16.append((qualifiers[g1][0], qualifiers[g2][1]))
        r16.append((qualifiers[g2][0], qualifiers[g1][1]))

    # R16
    qf_teams = []
    for ta, tb in r16:
        appearances[ta]["r16"] += 1
        appearances[tb]["r16"] += 1
        winner = simulate_ko_match(ta, tb, model, le, feature_cols, features, "round_of_16")
        qf_teams.append(winner)

    # QF
    sf_teams = []
    for i in range(0, len(qf_teams), 2):
        ta, tb = qf_teams[i], qf_teams[i+1]
        appearances[ta]["qf"] += 1
        appearances[tb]["qf"] += 1
        winner = simulate_ko_match(ta, tb, model, le, feature_cols, features, "quarter")
        sf_teams.append(winner)

    # SF
    final_teams = []
    for i in range(0, len(sf_teams), 2):
        ta, tb = sf_teams[i], sf_teams[i+1]
        appearances[ta]["sf"] += 1
        appearances[tb]["sf"] += 1
        winner = simulate_ko_match(ta, tb, model, le, feature_cols, features, "semi")
        final_teams.append(winner)

    # Final
    champion = simulate_ko_match(final_teams[0], final_teams[1],
                                  model, le, feature_cols, features, "final")
    appearances[final_teams[0]]["final"] += 1
    appearances[final_teams[1]]["final"] += 1
    appearances[champion]["champion"] += 1

    return appearances

def run_simulation(n=10000):
    """Run n tournament simulations and aggregate results."""
    print(f"FIFA Oracle — Running {n:,} tournament simulations...")
    model, le, feature_cols, features = load_model_and_data()

    all_teams = [t for g in WC_2026_GROUPS.values() for t in g]
    totals = defaultdict(lambda: {"r16": 0, "qf": 0, "sf": 0, "final": 0, "champion": 0})

    for i in range(n):
        if i % 1000 == 0:
            print(f"  Simulating... {i:,}/{n:,}")
        result = simulate_tournament(WC_2026_GROUPS, model, le, feature_cols, features)
        for team, stats in result.items():
            for k, v in stats.items():
                totals[team][k] += v

    # Convert to percentages
    results = []
    for team in all_teams:
        s = totals[team]
        results.append({
            "team": team,
            "champion_pct":  round(s["champion"] / n * 100, 1),
            "final_pct":     round(s["final"] / n * 100, 1),
            "semifinal_pct": round(s["sf"] / n * 100, 1),
            "quarterfinal_pct": round(s["qf"] / n * 100, 1),
            "r16_pct":       round(s["r16"] / n * 100, 1),
        })

    df = pd.DataFrame(results).sort_values("champion_pct", ascending=False)
    df.to_csv(os.path.join(BASE_DIR, "../data/processed/simulation_results.csv"), index=False)
    return df

if __name__ == "__main__":
    results = run_simulation(n=10000)

    print("\n── 2026 World Cup Simulation Results (10,000 runs) ──")
    print(f"{'Team':<18} {'Champion':>10} {'Final':>8} {'Semi':>8} {'QF':>8}")
    print("-" * 55)
    for _, row in results.iterrows():
        print(f"  {row['team']:<16} {row['champion_pct']:>8}%  {row['final_pct']:>6}%  {row['semifinal_pct']:>6}%  {row['quarterfinal_pct']:>6}%")

    # Dark horses: high semifinal% relative to champion%
    dark_horses = results[results["semifinal_pct"] > 20].nsmallest(3, "champion_pct")
    print(f"\n── Dark Horse Teams ──")
    for _, row in dark_horses.iterrows():
        print(f"  {row['team']}: {row['semifinal_pct']}% semi chance, only {row['champion_pct']}% to win it all")
