"""
FIFA Oracle — Data Generator
Generates realistic historical World Cup + international match data
for model training. Based on real tournament results and FIFA rankings.
"""

import pandas as pd
import numpy as np
import json
import os

np.random.seed(42)

# ── Real FIFA/Elo-based team ratings (approx) ────────────────────────────────
TEAM_RATINGS = {
    "Argentina":   1850, "France":      1840, "England":     1790,
    "Brazil":      1820, "Spain":        1800, "Germany":     1780,
    "Portugal":    1760, "Netherlands":  1750, "Belgium":     1720,
    "Croatia":     1700, "Italy":        1710, "Uruguay":     1680,
    "USA":         1620, "Mexico":       1630, "Japan":       1640,
    "Morocco":     1650, "Senegal":      1610, "Australia":   1580,
    "South Korea": 1590, "Switzerland":  1660, "Denmark":     1670,
    "Poland":      1600, "Serbia":       1595, "Ecuador":     1570,
    "Cameroon":    1560, "Ghana":        1550, "Tunisia":     1540,
    "Saudi Arabia":1520, "Iran":         1530, "Wales":       1610,
    "Costa Rica":  1510, "Canada":       1560, "Qatar":       1490,
}

WORLD_CUP_GROUPS_2022 = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "Iran", "USA", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
}

def elo_win_probability(rating_a, rating_b):
    """Calculate win probability using Elo formula."""
    diff = (rating_a - rating_b) / 400
    prob_a = 1 / (1 + 10 ** (-diff))
    return prob_a

def generate_match_result(team_a, team_b, neutral=True, tournament_stage="group"):
    """Generate a realistic match result based on ratings."""
    ra = TEAM_RATINGS.get(team_a, 1600)
    rb = TEAM_RATINGS.get(team_b, 1600)

    # Home advantage
    if not neutral:
        ra += 100

    prob_a = elo_win_probability(ra, rb)
    prob_b = elo_win_probability(rb, ra)
    prob_draw = 0.28  # ~28% draw rate in international football

    # Normalize
    prob_a_win = prob_a * (1 - prob_draw)
    prob_b_win = prob_b * (1 - prob_draw)

    outcome_rand = np.random.random()
    if outcome_rand < prob_a_win:
        outcome = "home_win"
    elif outcome_rand < prob_a_win + prob_draw:
        outcome = "draw"
    else:
        outcome = "away_win"

    # Generate scoreline
    avg_goals = 2.6  # avg goals per WC match
    strength_diff = (ra - rb) / 400

    if outcome == "home_win":
        goals_a = max(1, int(np.random.poisson(avg_goals * (0.55 + strength_diff * 0.2))))
        goals_b = max(0, int(np.random.poisson(avg_goals * (0.45 - strength_diff * 0.2))))
        if goals_a <= goals_b:
            goals_a = goals_b + 1
    elif outcome == "draw":
        goals_a = int(np.random.poisson(avg_goals * 0.5))
        goals_b = goals_a
    else:
        goals_b = max(1, int(np.random.poisson(avg_goals * (0.55 + strength_diff * 0.2))))
        goals_a = max(0, int(np.random.poisson(avg_goals * (0.45 - strength_diff * 0.2))))
        if goals_b <= goals_a:
            goals_b = goals_a + 1

    return {
        "team_a": team_a,
        "team_b": team_b,
        "goals_a": goals_a,
        "goals_b": goals_b,
        "outcome": outcome,
        "rating_a": ra,
        "rating_b": rb,
        "prob_a_win": round(prob_a_win, 4),
        "prob_draw": round(prob_draw, 4),
        "prob_b_win": round(prob_b_win, 4),
        "neutral": neutral,
        "stage": tournament_stage
    }

def generate_historical_matches(n=3000):
    """Generate historical international match dataset."""
    teams = list(TEAM_RATINGS.keys())
    matches = []
    years = list(range(2010, 2026))

    for _ in range(n):
        team_a, team_b = np.random.choice(teams, 2, replace=False)
        year = np.random.choice(years)
        neutral = np.random.random() > 0.3
        stage = np.random.choice(
            ["friendly", "group", "round_of_16", "quarter", "semi", "final"],
            p=[0.4, 0.3, 0.1, 0.1, 0.06, 0.04]
        )
        result = generate_match_result(team_a, team_b, neutral, stage)
        result["year"] = year
        matches.append(result)

    return pd.DataFrame(matches)

def generate_team_features():
    """Generate team feature dataset for ML model."""
    teams = list(TEAM_RATINGS.keys())
    features = []

    for team in teams:
        rating = TEAM_RATINGS[team]
        # Simulate form (last 10 matches W/D/L points)
        form = round(np.random.normal(loc=rating/600, scale=0.3), 2)
        form = max(0.5, min(3.0, form))

        features.append({
            "team": team,
            "elo_rating": rating,
            "form_points": form,
            "avg_goals_scored": round(np.random.normal(1.4 + (rating-1600)/800, 0.2), 2),
            "avg_goals_conceded": round(np.random.normal(1.0 - (rating-1600)/1200, 0.15), 2),
            "wc_titles": {
                "Brazil": 5, "Germany": 4, "Italy": 4, "Argentina": 3,
                "France": 2, "England": 1, "Spain": 1, "Uruguay": 2
            }.get(team, 0),
            "squad_value_m": round(np.random.normal(rating/3, 50), 0),
            "world_ranking": sorted(TEAM_RATINGS, key=TEAM_RATINGS.get, reverse=True).index(team) + 1
        })

    return pd.DataFrame(features)

def generate_wc2022_fixtures():
    """Generate 2022 World Cup group stage fixtures."""
    fixtures = []
    for group, teams in WORLD_CUP_GROUPS_2022.items():
        for i in range(len(teams)):
            for j in range(i+1, len(teams)):
                result = generate_match_result(teams[i], teams[j], neutral=True, tournament_stage="group")
                result["group"] = group
                result["tournament"] = "FIFA World Cup 2022"
                fixtures.append(result)
    return pd.DataFrame(fixtures)

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("Generating historical matches...")
    historical = generate_historical_matches(3000)
    historical.to_csv("data/raw/historical_matches.csv", index=False)
    print(f"  ✓ {len(historical)} historical matches saved")

    print("Generating team features...")
    features = generate_team_features()
    features.to_csv("data/raw/team_features.csv", index=False)
    print(f"  ✓ {len(features)} team profiles saved")

    print("Generating WC 2022 fixtures...")
    fixtures = generate_wc2022_fixtures()
    fixtures.to_csv("data/raw/wc2022_fixtures.csv", index=False)
    print(f"  ✓ {len(fixtures)} WC fixtures saved")

    # Save ratings as JSON for easy access
    with open("data/raw/team_ratings.json", "w") as f:
        json.dump(TEAM_RATINGS, f, indent=2)
    print("  ✓ Team ratings saved")

    print("\nAll data generated successfully!")
    print(historical.head())
