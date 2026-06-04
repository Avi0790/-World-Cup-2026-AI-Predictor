# ⚽ World Cup 2026 AI Predictor

> An AI-powered FIFA World Cup 2026 prediction platform that predicts match outcomes, simulates the tournament, analyzes odds, and provides deep team intelligence for all 48 nations.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML_Model-orange?style=flat)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat&logo=streamlit)](https://streamlit.io)
[![Firebase](https://img.shields.io/badge/Firebase-Auth_%26_DB-yellow?style=flat&logo=firebase)](https://firebase.google.com)

🔗 **Live Demo:** [wc2026-predictor.streamlit.app](https://wc2026-predictor.streamlit.app)

---

## 🎯 What It Does

A full-stack AI analytics platform with 6 core features:

| Feature | Description |
|---|---|
| **Match Predictor** | XGBoost model predicts win %, draw %, scoreline, xG, and confidence |
| **Tournament Simulator** | Monte Carlo simulation of the full 48-team bracket |
| **Odds Analyzer** | Compares AI model probability vs bookmaker implied probability |
| **Team Intelligence** | Radar charts, rankings, H2H analytics for all 48 nations |
| **AI Match Narrator** | Auto-generated match previews with tactical analysis |
| **My Predictions** | Save and track your predictions — powered by Firebase |

---

## 🖥️ Demo

### Match Predictor
```
🇦🇷 Argentina vs 🇫🇷 France — World Cup Final

Argentina Win: 52.3%  |  Draw: 19.1%  |  France Win: 28.6%
Predicted Score: 2–1
Expected Goals: Argentina 1.84 — France 1.52
Confidence: 🟢 52.3%
```

### Tournament Simulator (500 runs)
```
Champion Probabilities:
  🇫🇷 France        18.6%
  🏴󠁧󠁢󠁥󠁮󠁧󠁿 England       13.6%
  🇩🇰 Denmark       10.2%
  🇦🇷 Argentina      7.4%
  🇧🇷 Brazil         7.2%
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| ML Model | XGBoost, scikit-learn |
| Data Processing | Pandas, NumPy |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Authentication | Firebase Auth |
| Database | Firebase Firestore |
| Backend | Python 3.10+ |

---

## 📁 Project Structure

```
fifa-oracle/
│
├── data/
│   ├── raw/                    # Historical matches, team ratings, features
│   └── processed/              # Simulation results
│
├── models/
│   ├── train_model.py          # XGBoost training + prediction engine
│
├── simulations/
│   └── simulator.py            # Monte Carlo tournament simulator
│
├── firebase_auth.py            # Firebase Auth + Firestore integration
├── app.py                      # Streamlit dashboard (main entry point)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/Avi0790/-World-Cup-2026-AI-Predictor
cd -World-Cup-2026-AI-Predictor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate data + train model
python3 data/generate_data.py
python3 models/train_model.py
python3 simulations/simulator.py

# Launch dashboard
streamlit run app.py
```

---

## 🧠 ML Model Details

The prediction engine uses **XGBoost** trained on 3,000+ historical international matches with 17 engineered features including Elo ratings, recent form, goals scored/conceded, World Cup experience, squad value, and tournament stage pressure.

---

## 🗺️ Roadmap

- [x] XGBoost match prediction model
- [x] Monte Carlo tournament simulator
- [x] Streamlit dashboard with 6 modules
- [x] Firebase authentication + Firestore
- [x] Save & track prediction history
- [x] All 48 WC 2026 teams with flags + star players
- [x] Live countdown to June 11, 2026
- [ ] Live scores API integration
- [ ] Friend prediction leaderboard
- [ ] Player injury tracker

---

## 👤 Author

**Avinaya Khadka**
CS Student @ Arkansas State University (Graduating Dec 2026)
[LinkedIn](https://www.linkedin.com/in/avinaya-khadka-cs/) · [GitHub](https://github.com/Avi0790)

---

*Built as a portfolio project demonstrating applied machine learning, data engineering, Firebase integration, and dashboard development.*

