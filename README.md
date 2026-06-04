# ⚽ FIFA Oracle — AI World Cup Predictor

> An AI-powered FIFA World Cup prediction and football analytics platform that simulates tournaments, predicts match outcomes, and provides deep team intelligence using machine learning.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML_Model-orange?style=flat)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat&logo=streamlit)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-Visualization-blueviolet?style=flat)](https://plotly.com)

---

## 🎯 What It Does

FIFA Oracle is a full-stack AI analytics platform with 5 core features:

| Feature | Description |
|---|---|
| **Match Predictor** | XGBoost model predicts win %, draw %, scoreline, xG, and confidence |
| **Tournament Simulator** | Monte Carlo simulation of the World Cup 10,000+ times |
| **Odds Analyzer** | Compares AI model probability vs bookmaker implied probability |
| **Team Intelligence** | Radar charts, rankings, H2H analytics for every nation |
| **AI Match Narrator** | Auto-generated match previews with tactical analysis |

---

## 🖥️ Demo

### Match Predictor
```
Argentina vs France — World Cup Final

Argentina Win: 52.3%  |  Draw: 19.1%  |  France Win: 28.6%
Predicted Score: 2–1
Expected Goals: Argentina 1.84 — France 1.52
Confidence: 🟢 52.3%
```

### Tournament Simulator (500 runs)
```
Champion Probabilities:
  France        18.6%
  England       13.6%
  Denmark       10.2%
  Argentina      7.4%
  Brazil         7.2%
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| ML Model | XGBoost, scikit-learn |
| Data Processing | Pandas, NumPy |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Backend | Python 3.10+ |

---

## 📁 Project Structure

```
fifa-oracle/
│
├── data/
│   ├── raw/                    # Historical matches, team ratings, features
│   └── processed/              # Simulation results, cleaned datasets
│
├── models/
│   ├── train_model.py          # XGBoost training + prediction engine
│   └── match_predictor.pkl     # Trained model (generated)
│
├── simulations/
│   └── simulator.py            # Monte Carlo tournament simulator
│
├── app.py                      # Streamlit dashboard (main entry point)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/Avi0790/fifa-oracle
cd fifa-oracle

# Install dependencies
pip install -r requirements.txt

# Step 1: Generate data
python3 data/generate_data.py

# Step 2: Train the model
python3 models/train_model.py

# Step 3: Run simulations
python3 simulations/simulator.py

# Step 4: Launch dashboard
streamlit run app.py
```

---

## 🧠 ML Model Details

The prediction engine uses **XGBoost** trained on 3,000+ historical international matches with 17 engineered features:

- Elo rating differential
- Recent form (points per game)
- Average goals scored / conceded
- Head-to-head history
- World Cup titles (experience factor)
- Squad market value differential
- World ranking gap
- Tournament stage pressure
- Neutral venue flag

**Model accuracy:** ~39% (3-class classification: home win / draw / away win)  
*Note: Random baseline for 3-class prediction is ~33%. International football is inherently unpredictable — the model captures statistical tendencies, not certainties.*

---

## 📊 Key Features

### xG (Expected Goals) Estimation
Goals are estimated using a custom formula based on Elo differential and historical scoring rates, providing a scoreline probability distribution.

### Monte Carlo Tournament Simulation
The simulator runs the full 32-team tournament bracket thousands of times, accounting for randomness in each match while weighting outcomes by model probability.

### Odds Discrepancy Analysis
The Odds Analyzer computes bookmaker implied probability (removing the vig) and compares it against model probability. Discrepancies > 10% are flagged as statistically notable.

---

## 🗺️ Roadmap

- [x] Data pipeline and feature engineering
- [x] XGBoost match prediction model
- [x] Monte Carlo tournament simulator
- [x] Streamlit dashboard with 5 modules
- [x] AI match narrator
- [ ] Live API integration (football-data.org)
- [ ] Deep learning model (LSTM for form sequences)
- [ ] Player-level features (injuries, suspensions)
- [ ] Historical accuracy backtesting

---

## 👤 Author

**Avinaya Khadka**  
CS Student @ Arkansas State University (Graduating Dec 2026)  
[LinkedIn](https://www.linkedin.com/in/avinaya-khadka-cs/) · [GitHub](https://github.com/Avi0790)

---

*Built as a portfolio project demonstrating applied machine learning, data engineering, and dashboard development skills.*
