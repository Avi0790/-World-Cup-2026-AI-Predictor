import pandas as pd
import numpy as np
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WC_2026_GROUPS = {
    "A":["Mexico","South Korea","South Africa","Czechia"],
    "B":["Canada","Switzerland","Qatar","Bosnia and Herzegovina"],
    "C":["Brazil","Morocco","Scotland","Haiti"],
    "D":["USA","Australia","Paraguay","Turkey"],
    "E":["Germany","Ecuador","Ivory Coast","Curacao"],
    "F":["Netherlands","Japan","Tunisia","Sweden"],
    "G":["Belgium","Iran","Egypt","New Zealand"],
    "H":["Spain","Uruguay","Saudi Arabia","Cape Verde"],
    "I":["France","Senegal","Norway","Iraq"],
    "J":["Argentina","Austria","Algeria","Jordan"],
    "K":["Portugal","Colombia","Uzbekistan","DR Congo"],
    "L":["England","Croatia","Panama","Ghana"],
}

RATINGS = {
    "France":1870,"Spain":1855,"Argentina":1850,"England":1820,"Portugal":1810,
    "Brazil":1800,"Netherlands":1760,"Morocco":1740,"Belgium":1730,"Germany":1720,
    "Croatia":1700,"Colombia":1680,"Senegal":1660,"Mexico":1640,"USA":1630,
    "Uruguay":1680,"Japan":1640,"Switzerland":1650,"Norway":1630,"Austria":1600,
    "Australia":1580,"South Korea":1590,"Ecuador":1570,"Tunisia":1540,"Ghana":1550,
    "Saudi Arabia":1520,"Iran":1530,"Ivory Coast":1580,"South Africa":1510,
    "Czechia":1570,"Scotland":1600,"Bosnia and Herzegovina":1540,"Sweden":1580,
    "Turkey":1590,"Canada":1560,"Qatar":1490,"Haiti":1420,"Paraguay":1540,
    "Algeria":1540,"Jordan":1490,"Uzbekistan":1470,"DR Congo":1520,"Egypt":1550,
    "Iraq":1500,"New Zealand":1450,"Panama":1470,"Cape Verde":1480,"Curacao":1400,
}


def ko(team_a, team_b):
    ra = RATINGS.get(team_a, 1500)
    rb = RATINGS.get(team_b, 1500)

    p_a = 1 / (1 + 10 ** (-(ra - rb) / 400))

    return team_a if np.random.random() < p_a else team_b


def sim_group(groups):
    qualified = {}

    for group, teams in groups.items():
        pts = defaultdict(int)
        gd = defaultdict(int)

        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                ta, tb = teams[i], teams[j]

                ra = RATINGS.get(ta, 1500)
                rb = RATINGS.get(tb, 1500)

                p_win = (1 / (1 + 10 ** (-(ra - rb) / 400))) * 0.72
                p_draw = 0.27

                r = np.random.random()

                if r < p_win:
                    pts[ta] += 3
                    gd[ta] += 1
                    gd[tb] -= 1

                elif r < p_win + p_draw:
                    pts[ta] += 1
                    pts[tb] += 1

                else:
                    pts[tb] += 3
                    gd[tb] += 1
                    gd[ta] -= 1

        qualified[group] = sorted(
            teams,
            key=lambda t: (pts[t], gd[t]),
            reverse=True
        )[:2]

    return qualified


def run_simulation(n=500):
    print(f"Running {n} simulations with corrected 2026 teams...")

    champion_count = defaultdict(int)

    for _ in range(n):

        # Simulate groups
        q = sim_group(WC_2026_GROUPS)

        # Collect 24 qualified teams
        qualified_teams = []

        for group in q:
            qualified_teams.extend(q[group])

        # Shuffle bracket for fairness
        np.random.shuffle(qualified_teams)

        # Round of 24 → 12 winners
        r24 = [
            ko(qualified_teams[i], qualified_teams[i + 1])
            for i in range(0, len(qualified_teams), 2)
        ]

        # Give top-rated 4 teams a bye (simulate realistic bracket)
        r24_sorted = sorted(
            r24,
            key=lambda t: RATINGS.get(t, 1500),
            reverse=True
        )

        quarterfinalists = r24_sorted[:8]

        # Quarterfinals → 4
        sf = [
            ko(quarterfinalists[i], quarterfinalists[i + 1])
            for i in range(0, len(quarterfinalists), 2)
        ]

        # Finalists
        finalists = [
            ko(sf[i], sf[i + 1])
            for i in range(0, len(sf), 2)
        ]

        champion = ko(finalists[0], finalists[1])

        champion_count[champion] += 1

    all_teams = [
        team
        for group in WC_2026_GROUPS.values()
        for team in group
    ]

    rows = [
        {
            "team": team,
            "champion_pct": round(
                champion_count.get(team, 0) / n * 100,
                1
            )
        }
        for team in all_teams
    ]

    df = pd.DataFrame(rows).sort_values(
        "champion_pct",
        ascending=False
    )

    os.makedirs(
        os.path.join(BASE_DIR, "../data/processed"),
        exist_ok=True
    )

    df.to_csv(
        os.path.join(
            BASE_DIR,
            "../data/processed/simulation_results.csv"
        ),
        index=False
    )

    print("Top 10 champion probabilities:")

    for _, row in df.head(10).iterrows():
        print(
            f"  {row['team']:<30}"
            f"{row['champion_pct']}%"
        )

    return df


if __name__ == "__main__":
    run_simulation(500)