"""
World Cup 2026 AI Predictor — Premium Edition
Full Streamlit Dashboard with Firebase Auth, all 48 WC 2026 teams
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle, os, sys
from datetime import datetime, timezone
from firebase_auth import (firebase_sign_up, firebase_sign_in,
                           save_prediction, get_user_predictions)

sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
from train_model import predict_match, load_data

st.set_page_config(
    page_title="World Cup 2026 AI Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# FLAGS & TEAM DATA
# ══════════════════════════════════════════════════════════════════════════════
FLAGS = {
    "England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","France":"🇫🇷","Croatia":"🇭🇷","Norway":"🇳🇴",
    "Portugal":"🇵🇹","Germany":"🇩🇪","Netherlands":"🇳🇱","Switzerland":"🇨🇭",
    "Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Spain":"🇪🇸","Austria":"🇦🇹","Belgium":"🇧🇪",
    "Bosnia and Herzegovina":"🇧🇦","Sweden":"🇸🇪","Turkey":"🇹🇷","Czechia":"🇨🇿",
    "Argentina":"🇦🇷","Brazil":"🇧🇷","Ecuador":"🇪🇨","Uruguay":"🇺🇾",
    "Colombia":"🇨🇴","Paraguay":"🇵🇾","USA":"🇺🇸","Mexico":"🇲🇽","Canada":"🇨🇦",
    "Jamaica":"🇯🇲","Suriname":"🇸🇷","Panama":"🇵🇦","Honduras":"🇭🇳",
    "Costa Rica":"🇨🇷","Japan":"🇯🇵","South Korea":"🇰🇷","Australia":"🇦🇺",
    "Iran":"🇮🇷","Jordan":"🇯🇴","Uzbekistan":"🇺🇿","Morocco":"🇲🇦",
    "Senegal":"🇸🇳","Egypt":"🇪🇬","Algeria":"🇩🇿","Tunisia":"🇹🇳",
    "Ghana":"🇬🇭","Cameroon":"🇨🇲","DR Congo":"🇨🇩","South Africa":"🇿🇦",
    "New Zealand":"🇳🇿","Bolivia":"🇧🇴","New Caledonia":"🇳🇨",
}

TEAM_RATINGS = {
    "France":1840,"Argentina":1850,"England":1790,"Brazil":1820,"Spain":1800,
    "Germany":1780,"Portugal":1760,"Netherlands":1750,"Belgium":1720,"Croatia":1700,
    "Uruguay":1680,"USA":1620,"Mexico":1630,"Japan":1640,"Morocco":1650,
    "Senegal":1610,"Australia":1580,"South Korea":1590,"Switzerland":1660,
    "Colombia":1660,"Norway":1630,"Scotland":1600,"Austria":1580,"Turkey":1590,
    "Czechia":1570,"Sweden":1580,"Bosnia and Herzegovina":1540,"Ecuador":1570,
    "Paraguay":1540,"Canada":1560,"Jamaica":1480,"Suriname":1420,"Panama":1470,
    "Honduras":1450,"Costa Rica":1510,"Iran":1530,"Jordan":1490,"Uzbekistan":1470,
    "Egypt":1550,"Algeria":1540,"Tunisia":1540,"Ghana":1550,"Cameroon":1560,
    "DR Congo":1520,"South Africa":1510,"New Zealand":1450,"Bolivia":1430,
    "New Caledonia":1300,
}

WC_2026_GROUPS = {
    "A":["USA","Mexico","Canada","Uruguay"],
    "B":["Brazil","Argentina","Ecuador","Bolivia"],
    "C":["France","Belgium","Morocco","Senegal"],
    "D":["Spain","Portugal","Croatia","Turkey"],
    "E":["England","Netherlands","Denmark","Panama"],
    "F":["Germany","Switzerland","Austria","Czechia"],
    "G":["Japan","South Korea","Australia","Jordan"],
    "H":["Algeria","Egypt","Ghana","Cameroon"],
    "I":["Colombia","Paraguay","Norway","New Zealand"],
    "J":["Scotland","Sweden","Jamaica","Suriname"],
    "K":["Iran","Uzbekistan","DR Congo","Honduras"],
    "L":["Tunisia","Costa Rica","South Africa","New Caledonia"],
}

STAR_PLAYERS = {
    "Argentina":[("Lionel Messi","Forward","Inter Miami","🐐 Defending champion captain"),
                 ("Julián Álvarez","Forward","Atlético Madrid","⚡ 2022 WC hero"),
                 ("Rodrigo De Paul","Midfielder","Atlético Madrid","🎯 Midfield engine")],
    "France":   [("Kylian Mbappé","Forward","Real Madrid","⚡ World's fastest striker"),
                 ("Antoine Griezmann","Midfielder","Atlético Madrid","🎯 117 caps"),
                 ("Aurélien Tchouaméni","Midfielder","Real Madrid","🛡️ Anchor")],
    "Brazil":   [("Vinicius Jr.","Forward","Real Madrid","🔥 Ballon d'Or"),
                 ("Rodrygo","Forward","Real Madrid","⚡ Big game player"),
                 ("Casemiro","Midfielder","Man United","🛡️ 5x UCL")],
    "England":  [("Jude Bellingham","Midfielder","Real Madrid","⭐ England's superstar"),
                 ("Harry Kane","Forward","Bayern Munich","🎯 All-time top scorer"),
                 ("Phil Foden","Midfielder","Man City","🏆 World-class")],
    "Spain":    [("Pedri","Midfielder","Barcelona","🎨 Creative genius"),
                 ("Lamine Yamal","Forward","Barcelona","🌟 Teen phenom"),
                 ("Rodri","Midfielder","Man City","🏆 Ballon d'Or")],
    "Germany":  [("Florian Wirtz","Midfielder","Bayer Leverkusen","⚡ Breakout star"),
                 ("Jamal Musiala","Midfielder","Bayern Munich","🌟 Future"),
                 ("Toni Kroos","Midfielder","Real Madrid","🎯 Maestro")],
    "Portugal": [("Cristiano Ronaldo","Forward","Al Nassr","🐐 900+ goals"),
                 ("Bruno Fernandes","Midfielder","Man United","🎯 Playmaker"),
                 ("Rafael Leão","Forward","AC Milan","⚡ Explosive")],
    "Norway":   [("Erling Haaland","Forward","Man City","🚀 60+ goals/season"),
                 ("Martin Ødegaard","Midfielder","Arsenal","🎨 Creative maestro"),
                 ("Alexander Sørloth","Forward","Atlético Madrid","💪 Powerful")],
    "Colombia": [("James Rodríguez","Midfielder","Rayo Vallecano","🌟 Golden Boot 2014"),
                 ("Luis Díaz","Forward","Liverpool","⚡ Explosive winger"),
                 ("Jhon Durán","Forward","Al-Qadsiah","🔥 Rising star")],
    "Morocco":  [("Achraf Hakimi","Defender","PSG","🚀 Best RB in world"),
                 ("Hakim Ziyech","Midfielder","Galatasaray","🎯 Creative"),
                 ("Youssef En-Nesyri","Forward","Fenerbahce","⚽ Clinical")],
    "USA":      [("Christian Pulisic","Forward","AC Milan","⭐ USMNT captain"),
                 ("Gio Reyna","Midfielder","Nottm Forest","🌟 Playmaker"),
                 ("Tyler Adams","Midfielder","Bournemouth","💪 Engine")],
    "Mexico":   [("Hirving Lozano","Forward","PSV","⚡ El Chucky"),
                 ("Edson Álvarez","Midfielder","West Ham","🛡️ Anchor"),
                 ("Santiago Giménez","Forward","AC Milan","🎯 Clinical")],
    "Japan":    [("Takefusa Kubo","Forward","Real Sociedad","⚡ Creative"),
                 ("Ritsu Doan","Forward","Freiburg","🎯 Winger"),
                 ("Wataru Endo","Midfielder","Liverpool","🛡️ Anchor")],
    "South Korea":[("Son Heung-min","Forward","Tottenham","⭐ Asian superstar"),
                   ("Kim Min-jae","Defender","Bayern Munich","🛡️ World-class"),
                   ("Lee Jae-sung","Midfielder","Mainz","🎯 Creative")],
    "Senegal":  [("Sadio Mané","Forward","Al Nassr","🏆 AFCON winner"),
                 ("Édouard Mendy","Goalkeeper","Al Ahli","🧤 Stopper"),
                 ("Idrissa Gueye","Midfielder","Everton","💪 Tireless")],
    "Uruguay":  [("Darwin Núñez","Forward","Liverpool","⚡ Raw pace"),
                 ("Federico Valverde","Midfielder","Real Madrid","🏆 Engine"),
                 ("Ronald Araújo","Defender","Barcelona","🛡️ Beast")],
}

def get_players(team):
    return STAR_PLAYERS.get(team,[
        ("Key Player 1","Forward","Club","⚽ Top attacker"),
        ("Key Player 2","Midfielder","Club","🎯 Creative force"),
        ("Key Player 3","Defender","Club","🛡️ Defensive rock"),
    ])

def flag(t): return FLAGS.get(t,"🏳️")
def ft(t):   return f"{flag(t)} {t}"
def strip_flag(s): return " ".join(s.split(" ")[1:]) if " " in s else s

# ══════════════════════════════════════════════════════════════════════════════
# PREMIUM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.stApp { background: #060b18; color: #ffffff; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }

/* Hero section */
.hero {
    background: linear-gradient(135deg, #0d1b3e 0%, #1a0a2e 50%, #0a1628 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 48px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(74,158,255,0.08) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(46,204,113,0.06) 0%, transparent 50%);
}
.hero-eyebrow {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: #4a9eff;
    margin-bottom: 12px;
}
.hero-title {
    font-size: 48px;
    font-weight: 800;
    line-height: 1.1;
    color: #ffffff;
    margin-bottom: 16px;
}
.hero-title span { color: #4a9eff; }
.hero-sub {
    font-size: 17px;
    color: rgba(255,255,255,0.55);
    line-height: 1.7;
    max-width: 560px;
    margin-bottom: 24px;
}
.hero-stats {
    display: flex;
    gap: 32px;
    flex-wrap: wrap;
}
.hero-stat-num  { font-size: 28px; font-weight: 700; color: #fff; }
.hero-stat-label { font-size: 12px; color: rgba(255,255,255,0.45); margin-top: 2px; }

/* Mobile sidebar fix */
@media (max-width: 768px) {
    section[data-testid="stSidebar"] {
        width: 85% !important;
    }

    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
}

/* Section headers */
.section-header {
    font-size: 22px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin-left: 12px;
}

/* Cards */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    transition: border-color .2s;
}
.glass-card:hover { border-color: rgba(74,158,255,0.3); }

/* Match card */
.match-hero {
    background: linear-gradient(135deg, #0d1b3e, #1a0a2e);
    border: 1px solid rgba(74,158,255,0.2);
    border-radius: 20px;
    padding: 40px 32px;
    text-align: center;
    margin-bottom: 24px;
}
.match-flags { font-size: 64px; line-height: 1; margin-bottom: 12px; }
.match-teams { font-size: 24px; font-weight: 700; color: #fff; margin-bottom: 8px; }
.match-stage { font-size: 13px; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing:.1em; margin-bottom: 20px; }
.match-score { font-size: 56px; font-weight: 800; color: #4a9eff; letter-spacing: .05em; }
.match-score-label { font-size: 12px; color: rgba(255,255,255,0.35); margin-top: 4px; }

/* Probability display */
.prob-row { display: flex; align-items: center; gap: 12px; margin: 8px 0; }
.prob-team { font-size: 14px; font-weight: 500; color: #fff; width: 140px; }
.prob-bar-bg { flex: 1; height: 8px; background: rgba(255,255,255,0.08); border-radius: 4px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 4px; }
.prob-num { font-size: 14px; font-weight: 600; color: #fff; width: 45px; text-align: right; }

/* Reason card */
.reason-card {
    background: linear-gradient(135deg, rgba(74,158,255,0.08), rgba(46,204,113,0.06));
    border: 1px solid rgba(74,158,255,0.2);
    border-radius: 12px;
    padding: 20px;
    margin: 16px 0;
}
.reason-title { font-size: 13px; font-weight: 600; color: #4a9eff; margin-bottom: 12px; letter-spacing:.05em; text-transform:uppercase; }
.reason-item { display: flex; align-items: center; gap: 10px; margin: 8px 0; font-size: 14px; color: rgba(255,255,255,0.8); }

/* Player card */
.player-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    transition: all .2s;
}
.player-card:hover { background: rgba(74,158,255,0.06); border-color: rgba(74,158,255,0.2); }
.player-emoji { font-size: 32px; margin-bottom: 8px; }
.player-name  { font-size: 14px; font-weight: 600; color: #fff; margin-bottom: 3px; }
.player-pos   { font-size: 12px; color: #4a9eff; margin-bottom: 2px; }
.player-club  { font-size: 11px; color: rgba(255,255,255,0.4); margin-bottom: 6px; }
.player-stat  { font-size: 11px; color: #f39c12; }

/* Confidence meter */
.conf-meter {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.conf-label { font-size: 11px; color: rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:.08em; margin-bottom: 6px; }
.conf-value { font-size: 32px; font-weight: 800; }
.conf-high   { color: #2ecc71; }
.conf-mid    { color: #f39c12; }
.conf-low    { color: #e74c3c; }

/* Metric override */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
div[data-testid="metric-container"] label { color: rgba(255,255,255,0.45) !important; font-size: 12px !important; }
div[data-testid="metric-container"] [data-testid="metric-value"] { color: #fff !important; font-size: 24px !important; font-weight: 700 !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0a0f1e !important; border-right: 1px solid rgba(255,255,255,0.06) !important; }
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.75) !important; }

/* Countdown */
.countdown {
    background: linear-gradient(135deg, rgba(46,204,113,0.1), rgba(46,204,113,0.05));
    border: 1px solid rgba(46,204,113,0.25);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    margin: 12px 0;
}
.countdown-label { font-size: 10px; font-weight: 600; letter-spacing:.1em; color: #2ecc71; text-transform:uppercase; }
.countdown-time  { font-size: 24px; font-weight: 800; color: #fff; margin: 4px 0; }
.countdown-sub   { font-size: 11px; color: rgba(255,255,255,0.35); }

/* User badge */
.user-badge {
    background: rgba(46,204,113,0.1);
    border: 1px solid rgba(46,204,113,0.3);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 13px;
    color: #2ecc71;
    font-weight: 600;
    display: inline-block;
}

/* Login */
.login-hero { text-align: center; padding: 40px 0 20px; }
.login-icon  { font-size: 72px; margin-bottom: 12px; }
.login-title { font-size: 34px; font-weight: 800; color: #fff; margin-bottom: 8px; }
.login-sub   { font-size: 15px; color: rgba(255,255,255,0.45); }

/* How it works */
.how-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.how-num   { font-size: 28px; font-weight: 800; color: #4a9eff; margin-bottom: 8px; }
.how-title { font-size: 14px; font-weight: 600; color: #fff; margin-bottom: 6px; }
.how-desc  { font-size: 12px; color: rgba(255,255,255,0.45); line-height: 1.6; }

/* Upset badge */
.upset-badge {
    background: rgba(231,76,60,0.1);
    border: 1px solid rgba(231,76,60,0.3);
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 11px;
    color: #e74c3c;
    font-weight: 600;
    display: inline-block;
}

/* Footer */
.footer {
    background: rgba(255,255,255,0.02);
    border-top: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-top: 40px;
    font-size: 13px;
    color: rgba(255,255,255,0.3);
}
.footer a { color: #4a9eff; text-decoration: none; }

h1,h2,h3,h4 { color: #ffffff !important; font-weight: 700 !important; }
.stSelectbox label, .stTextInput label, .stRadio label, .stCheckbox label
    { color: rgba(255,255,255,0.5) !important; font-size: 12px !important; }
.stTabs [data-baseweb="tab"] { color: rgba(255,255,255,0.5) !important; }
.stTabs [aria-selected="true"] { color: #fff !important; }
.stDivider { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username  = ""
    st.session_state.user_data = {}

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="login-hero">
            <div class="login-icon">🏆</div>
            <div class="login-title">World Cup 2026</div>
            <div class="login-title" style="color:#4a9eff;margin-top:-8px">AI Predictor</div>
            <div class="login-sub" style="margin-top:12px">
                AI-powered predictions for all 48 nations.<br>
                Powered by XGBoost · Firebase · Python
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Countdown on login page too
        kickoff = datetime(2026, 6, 11, 18, 0, 0, tzinfo=timezone.utc)
        delta   = kickoff - datetime.now(timezone.utc)
        if delta.total_seconds() > 0:
            d = delta.days; h = delta.seconds//3600; m = (delta.seconds%3600)//60
            st.markdown(f"""
            <div class="countdown">
                <div class="countdown-label">⏱️ Kickoff Countdown</div>
                <div class="countdown-time">{d}d {h:02d}h {m:02d}m</div>
                <div class="countdown-sub">June 11, 2026 · Mexico City</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔑  Login", "📝  Create Account"])

        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            email = st.text_input("Email", placeholder="your@email.com", key="li_email")
            pw    = st.text_input("Password", type="password", placeholder="Your password", key="li_pw")
            if st.button("Sign In →", use_container_width=True, type="primary"):
                if email and pw:
                    with st.spinner("Signing in..."):
                        ok, res = firebase_sign_in(email, pw)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.username  = res["username"]
                        st.session_state.user_data = res
                        st.rerun()
                    else:
                        st.error(res)
                else:
                    st.warning("Enter your email and password.")
            st.caption("🔒 Secured by Firebase · Google Cloud")

        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            nu = st.text_input("Username", placeholder="football_fan99", key="ru")
            ne = st.text_input("Email",    placeholder="your@email.com",  key="re")
            np1= st.text_input("Password", type="password", placeholder="Min 6 characters", key="rp1")
            np2= st.text_input("Confirm",  type="password", placeholder="Repeat password",  key="rp2")
            if st.button("Create Account →", use_container_width=True, type="primary"):
                if not nu or not ne or not np1: st.warning("Fill in all fields.")
                elif len(np1) < 6:             st.error("Password min 6 characters.")
                elif np1 != np2:               st.error("Passwords don't match.")
                else:
                    with st.spinner("Creating account..."):
                        ok, res = firebase_sign_up(ne, np1, nu)
                    if ok: st.success("✅ Account created! Log in now.")
                    else:  st.error(res)

    # How it works
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🤖 How the AI Works</div>", unsafe_allow_html=True)
    hc = st.columns(4)
    steps = [
        ("01","Historical Data","49,000+ international matches used to train the model"),
        ("02","Team Intelligence","Elo ratings, form, goals scored/conceded, WC experience"),
        ("03","XGBoost Model","ML model predicts win %, draw %, xG, and scoreline"),
        ("04","Simulation","Monte Carlo runs the full tournament 500+ times"),
    ]
    for col,(num,title,desc) in zip(hc,steps):
        with col:
            st.markdown(f"""
            <div class="how-card">
                <div class="how-num">{num}</div>
                <div class="how-title">{title}</div>
                <div class="how-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.stop()


# ── Auto-generate data and train model if not present ──
import subprocess, sys as _sys
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_pkl_path = os.path.join(_BASE_DIR, 'models/match_predictor.pkl')
_sim_path = os.path.join(_BASE_DIR, 'data/processed/simulation_results.csv')
if not os.path.exists(_pkl_path):
    with st.spinner('🤖 First run: generating data and training AI model (~30 sec)...'):
        subprocess.run([_sys.executable, os.path.join(_BASE_DIR, 'data/generate_data.py')], cwd=_BASE_DIR)
        subprocess.run([_sys.executable, os.path.join(_BASE_DIR, 'models/train_model.py')], cwd=_BASE_DIR)
if not os.path.exists(_sim_path):
    with st.spinner('🔄 Running tournament simulations...'):
        subprocess.run([_sys.executable, os.path.join(_BASE_DIR, 'simulations/simulator.py')], cwd=_BASE_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model_pkg():
    BASE = os.path.dirname(__file__)
    with open(os.path.join(BASE, "models/match_predictor.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_datasets():
    BASE = os.path.dirname(__file__)
    _, features, ratings = load_data()
    sim_path = os.path.join(BASE, "data/processed/simulation_results.csv")
    sim = pd.read_csv(sim_path) if os.path.exists(sim_path) else pd.DataFrame()
    return features, ratings, sim

pkg = load_model_pkg()
model, le, feature_cols = pkg["model"], pkg["le"], pkg["feature_cols"]
features_df, ratings, sim_results = load_datasets()
MODEL_TEAMS = features_df["team"].tolist()
ALL_TEAMS   = sorted(FLAGS.keys())
TWF         = [ft(t) for t in ALL_TEAMS]

def safe_predict(ta, tb, stage="group", neutral=True):
    if ta in MODEL_TEAMS and tb in MODEL_TEAMS:
        return predict_match(ta, tb, features_df, model, le, feature_cols, neutral, stage)
    ra = TEAM_RATINGS.get(ta,1500); rb = TEAM_RATINGS.get(tb,1500)
    diff = ra-rb
    pa = round(1/(1+10**(-diff/400))*100*0.72,1)
    pb = round(1/(1+10**(diff/400))*100*0.72,1)
    pd_= round(100-pa-pb,1)
    ranks = sorted(TEAM_RATINGS, key=TEAM_RATINGS.get, reverse=True)
    return {"team_a":ta,"team_b":tb,"prob_a_win":pa,"prob_draw":pd_,"prob_b_win":pb,
            "predicted_score":f"{max(1,int(ra/800))}–{max(0,int(rb/900))}",
            "xg_a":round(ra/1400,2),"xg_b":round(rb/1400,2),"confidence":round(max(pa,pb),1),
            "rating_a":ra,"rating_b":rb,"ranking_a":ranks.index(ta)+1,"ranking_b":ranks.index(tb)+1,"stage":stage}

def why_prediction(ta, tb, r):
    """Generate human-readable reason for prediction."""
    ra_r = TEAM_RATINGS.get(ta,1500); rb_r = TEAM_RATINGS.get(tb,1500)
    tf   = features_df.set_index("team") if ta in MODEL_TEAMS else None
    reasons = []
    fav = ta if r["prob_a_win"] > r["prob_b_win"] else tb
    und = tb if fav==ta else ta

    elo_diff = abs(ra_r - rb_r)
    if elo_diff > 100:
        reasons.append(f"🏆 {fav} has a significantly higher Elo rating (+{elo_diff} pts)")
    elif elo_diff > 40:
        reasons.append(f"📊 {fav} holds a slight Elo rating advantage (+{elo_diff} pts)")
    else:
        reasons.append(f"⚖️ Both teams are closely matched in Elo ratings (diff: {elo_diff} pts)")

    wc = {"Brazil":5,"Germany":4,"Italy":4,"Argentina":3,"France":2,"England":1,"Spain":1,"Uruguay":2}
    fa_wc = wc.get(ta,0); fb_wc = wc.get(tb,0)
    if fa_wc > fb_wc:
        reasons.append(f"🥇 {ta} has more World Cup titles ({fa_wc} vs {fb_wc}) — proven tournament experience")
    elif fb_wc > fa_wc:
        reasons.append(f"🥇 {tb} has more World Cup titles ({fb_wc} vs {fa_wc}) — proven tournament experience")

    if tf is not None and ta in tf.index and tb in tf.index:
        gs_a = tf.loc[ta,"avg_goals_scored"]; gs_b = tf.loc[tb,"avg_goals_scored"]
        gc_a = tf.loc[ta,"avg_goals_conceded"]; gc_b = tf.loc[tb,"avg_goals_conceded"]
        if gs_a > gs_b + 0.2:
            reasons.append(f"⚡ {ta} averages more goals per game ({gs_a} vs {gs_b}) — stronger attack")
        elif gs_b > gs_a + 0.2:
            reasons.append(f"⚡ {tb} averages more goals per game ({gs_b} vs {gs_a}) — stronger attack")
        if gc_a < gc_b - 0.15:
            reasons.append(f"🛡️ {ta} concedes fewer goals ({gc_a} vs {gc_b}) — more solid defence")
        elif gc_b < gc_a - 0.15:
            reasons.append(f"🛡️ {tb} concedes fewer goals ({gc_b} vs {gc_a}) — more solid defence")

    if r["prob_draw"] > 28:
        reasons.append(f"🤝 High draw probability ({r['prob_draw']}%) — evenly matched contest expected")

    ranks = sorted(TEAM_RATINGS, key=TEAM_RATINGS.get, reverse=True)
    rank_a = ranks.index(ta)+1; rank_b = ranks.index(tb)+1
    if abs(rank_a-rank_b) > 8:
        higher = ta if rank_a < rank_b else tb
        reasons.append(f"📈 {higher} is ranked significantly higher (#{min(rank_a,rank_b)} vs #{max(rank_a,rank_b)})")

    return reasons[:4], fav

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"<div style='text-align:center;font-size:28px;margin-bottom:4px'>🏆</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;font-size:14px;font-weight:700;color:#fff;margin-bottom:2px'>World Cup 2026</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center;font-size:11px;color:#4a9eff;margin-bottom:12px'>AI Predictor</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:center'><span class='user-badge'>👤 {st.session_state.username}</span></div>", unsafe_allow_html=True)

    kickoff = datetime(2026,6,11,18,0,0,tzinfo=timezone.utc)
    delta   = kickoff - datetime.now(timezone.utc)
    if delta.total_seconds() > 0:
        d=delta.days; h=delta.seconds//3600; m=(delta.seconds%3600)//60
        st.markdown(f"""
        <div class="countdown" style="margin-top:12px">
            <div class="countdown-label">⏱️ To Kickoff</div>
            <div class="countdown-time">{d}d {h:02d}h {m:02d}m</div>
            <div class="countdown-sub">Jun 11 · Mexico City</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.success("🏆 The World Cup is LIVE!")

    st.divider()
    page = st.radio("", [
        "🏠  Home",
        "⚽  Match Predictor",
        "🌍  Tournament Simulator",
        "📊  Odds Analyzer",
        "🧠  Team Intelligence",
        "📰  AI Narrator",
        "📋  My Predictions",
    ])
    st.divider()
    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username  = ""
        st.rerun()
    st.markdown("""
    <div style='font-size:11px;color:rgba(255,255,255,0.25);text-align:center;margin-top:8px'>
        Built by Avinaya Khadka<br>
        XGBoost · Firebase · Streamlit
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    # Hero
    st.markdown(f"""
    <div class="hero">
        <div class="hero-eyebrow">⚽ FIFA World Cup 2026 · 48 Teams · 104 Matches</div>
        <div class="hero-title">AI-Powered<br><span>World Cup Predictions</span></div>
        <div class="hero-sub">
            Machine learning predictions for every match. Simulate the full tournament,
            analyze team intelligence, and track your predictions — all in one platform.
        </div>
        <div class="hero-stats">
            <div>
                <div class="hero-stat-num">48</div>
                <div class="hero-stat-label">Nations</div>
            </div>
            <div>
                <div class="hero-stat-num">104</div>
                <div class="hero-stat-label">Matches</div>
            </div>
            <div>
                <div class="hero-stat-num">49,000+</div>
                <div class="hero-stat-label">Training Matches</div>
            </div>
            <div>
                <div class="hero-stat-num">500+</div>
                <div class="hero-stat-label">Simulations</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick predictions
    st.markdown("<div class='section-header'>🔥 Featured Matchups</div>", unsafe_allow_html=True)
    featured = [("Argentina","France","final"),("Brazil","England","semi"),
                ("Spain","Germany","quarter"),("Norway","Colombia","group")]
    cols = st.columns(4)
    for col,(ta,tb,stage) in zip(cols,featured):
        r = safe_predict(ta,tb,stage)
        fav = ta if r["prob_a_win"]>r["prob_b_win"] else tb
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center">
                <div style="font-size:28px">{flag(ta)} <span style="color:#555;font-size:16px">vs</span> {flag(tb)}</div>
                <div style="font-size:13px;font-weight:600;color:#fff;margin:8px 0">{ta} vs {tb}</div>
                <div style="font-size:11px;color:#4a9eff;margin-bottom:8px">{stage.replace('_',' ').title()}</div>
                <div style="font-size:20px;font-weight:700;color:#4a9eff">{r['predicted_score']}</div>
                <div style="font-size:11px;color:rgba(255,255,255,.4);margin-top:4px">Fav: {flag(fav)} {fav} ({max(r['prob_a_win'],r['prob_b_win'])}%)</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Upset alerts
    st.markdown("<div class='section-header'>🚨 Upset Alerts</div>", unsafe_allow_html=True)
    st.caption("Matches where the underdog has >35% win probability — according to the AI model")
    upset_matches = [("Morocco","Spain","round_of_16"),("Japan","Germany","group"),
                     ("USA","Portugal","group"),("Senegal","Belgium","group")]
    ucols = st.columns(4)
    for col,(ta,tb,stage) in zip(ucols,upset_matches):
        r = safe_predict(ta,tb,stage)
        und = ta if r["prob_a_win"] < r["prob_b_win"] else tb
        und_prob = min(r["prob_a_win"], r["prob_b_win"])
        if und_prob > 30:
            with col:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center">
                    <div class="upset-badge">🚨 UPSET ALERT</div>
                    <div style="font-size:24px;margin:10px 0">{flag(und)}</div>
                    <div style="font-size:13px;font-weight:600;color:#fff">{und}</div>
                    <div style="font-size:12px;color:rgba(255,255,255,.4);margin:4px 0">{flag(ta)} {ta} vs {flag(tb)} {tb}</div>
                    <div style="font-size:18px;font-weight:700;color:#e74c3c;margin-top:8px">{und_prob}% chance</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works
    st.markdown("<div class='section-header'>🤖 How the AI Works</div>", unsafe_allow_html=True)
    hcols = st.columns(4)
    steps = [
        ("01","Historical Data","49,000+ international matches from 2010–2026 used for training"),
        ("02","17 Features","Elo ratings, recent form, goals scored/conceded, WC titles, squad value"),
        ("03","XGBoost Model","Gradient boosted trees predict win %, draw %, xG, and confidence"),
        ("04","Monte Carlo","Full tournament simulated 500+ times for probability distributions"),
    ]
    for col,(num,title,desc) in zip(hcols,steps):
        with col:
            st.markdown(f"""
            <div class="how-card">
                <div class="how-num">{num}</div>
                <div class="how-title">{title}</div>
                <div class="how-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        Built with
        <strong style="color:rgba(255,255,255,.6)">Python · XGBoost · Streamlit · Firebase · Plotly · Pandas</strong>
        &nbsp;·&nbsp;
        <a href="https://github.com/Avi0790/-World-Cup-2026-AI-Predictor" target="_blank">GitHub</a>
        &nbsp;·&nbsp;
        <a href="https://www.linkedin.com/in/avinaya-khadka-cs/" target="_blank">LinkedIn</a>
        &nbsp;·&nbsp; Built by Avinaya Khadka
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚽  Match Predictor":
    st.markdown("<h1>⚽ Match Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,.45);margin-top:-12px'>XGBoost model · Elo ratings · xG estimation · Why this prediction</p>", unsafe_allow_html=True)

    c1,c2,c3 = st.columns([2,1,2])
    with c1:
        sel_a  = st.selectbox("Team A", TWF, index=TWF.index(ft("Argentina")))
        team_a = strip_flag(sel_a)
    with c2:
        st.markdown("<br><br><div style='text-align:center;font-size:20px;color:rgba(255,255,255,.3)'>VS</div>", unsafe_allow_html=True)
    with c3:
        sel_b  = st.selectbox("Team B", TWF, index=TWF.index(ft("France")))
        team_b = strip_flag(sel_b)

    c4,c5 = st.columns(2)
    with c4: stage   = st.selectbox("Stage", ["group","round_of_16","quarter","semi","final"])
    with c5: neutral = st.checkbox("Neutral Venue", value=True)

    if st.button("🔮  Generate Prediction", use_container_width=True, type="primary"):
        if team_a == team_b:
            st.error("Select two different teams.")
        else:
            r = safe_predict(team_a, team_b, stage, neutral)

            # Match hero card
            st.markdown(f"""
            <div class="match-hero">
                <div class="match-flags">{flag(team_a)} &nbsp; {flag(team_b)}</div>
                <div class="match-teams">{team_a} &nbsp;vs&nbsp; {team_b}</div>
                <div class="match-stage">{stage.replace('_',' ').upper()} · {'NEUTRAL' if neutral else 'HOME ADVANTAGE'}</div>
                <div class="match-score">{r['predicted_score']}</div>
                <div class="match-score-label">AI Predicted Scoreline</div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown("<div class='section-header' style='margin-top:0'>📊 Win Probability</div>", unsafe_allow_html=True)
            probs = [(team_a, r["prob_a_win"], "#4a9eff"),
                     ("Draw",  r["prob_draw"],  "#f39c12"),
                     (team_b, r["prob_b_win"],  "#e74c3c")]
            for name, prob, color in probs:
                f_icon = flag(name) + " " if name not in ["Draw"] else "🤝 "
                st.markdown(f"""
                <div class="prob-row">
                    <div class="prob-team">{f_icon}{name}</div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{prob}%;background:{color}"></div>
                    </div>
                    <div class="prob-num">{prob}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # xG + confidence
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric(f"xG {flag(team_a)}", r['xg_a'])
            with c2: st.metric(f"xG {flag(team_b)}", r['xg_b'])
            with c3: st.metric("Model Confidence", f"{r['confidence']}%")
            with c4: st.metric("Stage", stage.replace('_',' ').title())

            # Confidence meter
            conf = r["confidence"]
            cclass = "conf-high" if conf>60 else "conf-mid" if conf>45 else "conf-low"
            clabel = "High Confidence" if conf>60 else "Moderate Confidence" if conf>45 else "Low Confidence"
            st.markdown(f"""
            <div class="conf-meter" style="margin:16px 0">
                <div class="conf-label">Prediction Confidence</div>
                <div class="conf-value {cclass}">{conf}%</div>
                <div style="font-size:12px;color:rgba(255,255,255,.35);margin-top:4px">{clabel}</div>
            </div>""", unsafe_allow_html=True)

            # Why this prediction
            reasons, fav = why_prediction(team_a, team_b, r)
            reason_html = "".join([f'<div class="reason-item">✦ {re}</div>' for re in reasons])
            st.markdown(f"""
            <div class="reason-card">
                <div class="reason-title">🧠 Why {flag(fav)} {fav} is favoured</div>
                {reason_html}
            </div>""", unsafe_allow_html=True)

            # Plotly bar chart
            fig = go.Figure(go.Bar(
                x=[f"{flag(team_a)} {team_a}", "Draw", f"{flag(team_b)} {team_b}"],
                y=[r['prob_a_win'], r['prob_draw'], r['prob_b_win']],
                marker_color=["#4a9eff","#f39c12","#e74c3c"],
                text=[f"{v}%" for v in [r['prob_a_win'],r['prob_draw'],r['prob_b_win']]],
                textposition="outside", textfont=dict(color="white", size=13)
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=300, showlegend=False,
                margin=dict(t=20,b=20,l=0,r=0),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True),
                xaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Team comparison chart
            st.markdown("<div class='section-header'>📈 Team Comparison</div>", unsafe_allow_html=True)
            if team_a in MODEL_TEAMS and team_b in MODEL_TEAMS:
                tf = features_df.set_index("team")
                categories = ["Elo Rating","Avg Goals","Clean Sheet","WC Titles","World Rank"]
                def normalize(val, min_v, max_v): return round((val-min_v)/(max_v-min_v)*100,1)
                ra = tf.loc[team_a]; rb = tf.loc[team_b]
                vals_a = [
                    normalize(ra["elo_rating"],1300,1860),
                    normalize(ra["avg_goals_scored"],0.5,2.5),
                    normalize(1/max(ra["avg_goals_conceded"],0.1),0.5,3),
                    normalize(ra["wc_titles"],0,5),
                    normalize(1/max(ra["world_ranking"],1),0,1)*100,
                ]
                vals_b = [
                    normalize(rb["elo_rating"],1300,1860),
                    normalize(rb["avg_goals_scored"],0.5,2.5),
                    normalize(1/max(rb["avg_goals_conceded"],0.1),0.5,3),
                    normalize(rb["wc_titles"],0,5),
                    normalize(1/max(rb["world_ranking"],1),0,1)*100,
                ]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatterpolar(r=vals_a+[vals_a[0]],
                    theta=categories+[categories[0]], name=team_a,
                    fill="toself", fillcolor="rgba(74,158,255,0.15)",
                    line=dict(color="#4a9eff",width=2)))
                fig2.add_trace(go.Scatterpolar(r=vals_b+[vals_b[0]],
                    theta=categories+[categories[0]], name=team_b,
                    fill="toself", fillcolor="rgba(231,76,60,0.15)",
                    line=dict(color="#e74c3c",width=2)))
                fig2.update_layout(
                    polar=dict(bgcolor="rgba(0,0,0,0)",
                               radialaxis=dict(visible=True,range=[0,100],
                                               gridcolor="rgba(255,255,255,0.08)",
                                               tickfont=dict(color="rgba(255,255,255,0.3)",size=9))),
                    paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                    height=380, legend=dict(bgcolor="rgba(0,0,0,0)"),
                    margin=dict(t=20,b=20)
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Star players
            st.markdown("<div class='section-header'>⭐ Key Players</div>", unsafe_allow_html=True)
            pc1,pc2 = st.columns(2)
            for col,team in [(pc1,team_a),(pc2,team_b)]:
                with col:
                    st.markdown(f"<div style='font-size:16px;font-weight:600;color:#fff;margin-bottom:12px'>{flag(team)} {team}</div>", unsafe_allow_html=True)
                    for i,(name,pos,club,stat) in enumerate(get_players(team)):
                        st.markdown(f"""
                        <div class="player-card" style="margin-bottom:8px">
                            <div class="player-emoji">{"🥇" if i==0 else "⭐"}</div>
                            <div class="player-name">{name}</div>
                            <div class="player-pos">{pos}</div>
                            <div class="player-club">🏟️ {club}</div>
                            <div class="player-stat">{stat}</div>
                        </div>""", unsafe_allow_html=True)

            # Save + export
            st.divider()
            cs,ce = st.columns(2)
            with cs:
                if st.button("💾  Save Prediction", use_container_width=True):
                    uid = st.session_state.user_data.get("uid","")
                    tok = st.session_state.user_data.get("idToken","")
                    if uid and tok:
                        ok = save_prediction(uid, tok, team_a, team_b, stage, r)
                        st.success("✅ Saved!") if ok else st.error("Save failed.")
                    else:
                        st.error("Log in again to save.")
            with ce:
                share = (f"⚽ WC 2026 Prediction\n{flag(team_a)} {team_a} vs {flag(team_b)} {team_b}\n"
                         f"Score: {r['predicted_score']}\n{team_a}: {r['prob_a_win']}% | Draw: {r['prob_draw']}% | {team_b}: {r['prob_b_win']}%\n"
                         f"Confidence: {r['confidence']}%\n🤖 World Cup 2026 AI Predictor")
                st.download_button("📤  Export Card", data=share,
                    file_name=f"{team_a}_vs_{team_b}.txt", mime="text/plain",
                    use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TOURNAMENT SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌍  Tournament Simulator":
    st.markdown("<h1>🌍 Tournament Simulator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,.45);margin-top:-12px'>Monte Carlo simulation · 500 tournament runs · All 48 teams</p>", unsafe_allow_html=True)

    if sim_results.empty:
        st.warning("Run simulations/simulator.py first.")
    else:
        sim = sim_results.copy()
        tc  = sim.columns[0]; cc = sim.columns[1]
        sim["display"] = sim[tc].apply(ft)

        top12 = sim.head(12)
        fig = go.Figure(go.Bar(
            y=top12["display"], x=top12[cc], orientation="h",
            marker=dict(color=top12[cc], colorscale="Blues", showscale=False),
            text=[f"{v}%" for v in top12[cc]], textposition="outside",
            textfont=dict(color="white",size=12)
        ))
        fig.update_layout(
            title=dict(text="🏆 World Cup 2026 Champion Probability", font=dict(color="white",size=18)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=480,
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)",title="Champion Probability (%)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.0)"),
            margin=dict(l=0,r=80,t=50,b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top 5
        st.markdown("<div class='section-header'>🏅 Top Favorites</div>", unsafe_allow_html=True)
        cols = st.columns(5)
        for i,(_,row) in enumerate(sim.head(5).iterrows()):
            with cols[i]:
                st.markdown(f"<div style='text-align:center;font-size:40px'>{flag(row[tc])}</div>", unsafe_allow_html=True)
                st.metric(row[tc], f"{row[cc]}%")

        # Dark horses
        st.markdown("<div class='section-header'>🐴 Dark Horse Watch</div>", unsafe_allow_html=True)
        dark = sim[(sim[cc]>0.5)&(sim[cc]<4)].head(4)
        if not dark.empty:
            dc = st.columns(min(4,len(dark)))
            for i,(_,row) in enumerate(dark.iterrows()):
                with dc[i]:
                    st.markdown(f"""
                    <div class="glass-card" style="text-align:center">
                        <div style="font-size:36px">{flag(row[tc])}</div>
                        <div style="font-size:14px;font-weight:600;color:#fff;margin:6px 0">{row[tc]}</div>
                        <div class="upset-badge">🐴 {row[cc]}% to win</div>
                    </div>""", unsafe_allow_html=True)

        # Groups
        st.markdown("<div class='section-header'>📋 Group Stage Draw</div>", unsafe_allow_html=True)
        gcols = st.columns(4)
        for i,(grp,teams) in enumerate(WC_2026_GROUPS.items()):
            with gcols[i%4]:
                team_rows = "".join([f"<div style='padding:4px 0;font-size:13px;border-bottom:1px solid rgba(255,255,255,0.05)'>{flag(t)} {t}</div>" for t in teams])
                st.markdown(f"""
                <div class="glass-card" style="margin-bottom:12px">
                    <div style="font-size:13px;font-weight:700;color:#4a9eff;margin-bottom:8px">GROUP {grp}</div>
                    {team_rows}
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ODDS ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Odds Analyzer":
    st.markdown("<h1>📊 Smart Odds Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,.45);margin-top:-12px'>Educational analytics only — not a betting recommendation tool</p>", unsafe_allow_html=True)
    st.info("⚠️ This tool is for **educational and analytical purposes only**.")

    c1,c2 = st.columns(2)
    with c1:
        sel_a=st.selectbox("Team A",TWF,index=TWF.index(ft("Brazil"))); team_a=strip_flag(sel_a)
    with c2:
        sel_b=st.selectbox("Team B",TWF,index=TWF.index(ft("Germany"))); team_b=strip_flag(sel_b)

    c3,c4,c5=st.columns(3)
    with c3: odds_a   =st.number_input(f"{flag(team_a)} {team_a}",min_value=1.01,value=2.10,step=0.05)
    with c4: odds_draw=st.number_input("Draw",min_value=1.01,value=3.40,step=0.05)
    with c5: odds_b   =st.number_input(f"{flag(team_b)} {team_b}",min_value=1.01,value=3.20,step=0.05)

    if st.button("🔍  Analyze", use_container_width=True, type="primary"):
        if team_a==team_b: st.error("Select two different teams.")
        else:
            ri=1/odds_a+1/odds_draw+1/odds_b
            ia,id_,ib=round(1/odds_a/ri*100,1),round(1/odds_draw/ri*100,1),round(1/odds_b/ri*100,1)
            r=safe_predict(team_a,team_b)
            ma,md,mb=r["prob_a_win"],r["prob_draw"],r["prob_b_win"]

            fig=go.Figure()
            lbs=[f"{flag(team_a)} {team_a}","Draw",f"{flag(team_b)} {team_b}"]
            fig.add_trace(go.Bar(name="Bookmaker",x=lbs,y=[ia,id_,ib],marker_color="#e74c3c"))
            fig.add_trace(go.Bar(name="AI Model", x=lbs,y=[ma,md,mb], marker_color="#4a9eff"))
            fig.update_layout(barmode="group",paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",font_color="white",
                              height=360,title="Model vs Market Probability",
                              legend=dict(bgcolor="rgba(0,0,0,0)"),
                              yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                              xaxis=dict(showgrid=False))
            st.plotly_chart(fig,use_container_width=True)

            c6,c7,c8=st.columns(3)
            for col,lbl,disc in zip([c6,c7,c8],lbs,[ma-ia,md-id_,mb-ib]):
                with col:
                    if abs(disc)>10: st.success(f"**{lbl}**\n\n🟢 Model discrepancy\n\nDiff: **{disc:+.1f}%**")
                    else:            st.info(f"**{lbl}**\n\nNo major discrepancy\n\nDiff: {disc:+.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TEAM INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠  Team Intelligence":
    st.markdown("<h1>🧠 Team Intelligence</h1>", unsafe_allow_html=True)

    sel=st.selectbox("Select a Team",TWF,index=TWF.index(ft("Brazil"))); team=strip_flag(sel)
    rating=TEAM_RATINGS.get(team,1500)
    ranks=sorted(TEAM_RATINGS,key=TEAM_RATINGS.get,reverse=True)
    rank=ranks.index(team)+1

    st.markdown(f"<div style='font-size:80px;text-align:center;margin:16px 0'>{flag(team)}</div>",unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align:center'>{team}</h2>",unsafe_allow_html=True)

    wc_titles={"Brazil":5,"Germany":4,"Italy":4,"Argentina":3,"France":2,"England":1,"Spain":1,"Uruguay":2}.get(team,0)
    grp=[g for g,t in WC_2026_GROUPS.items() if team in t]
    grp_str=f"Group {grp[0]}" if grp else "—"

    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Elo Rating",rating)
    with c2: st.metric("World Ranking",f"#{rank}")
    with c3: st.metric("WC Titles",wc_titles)
    with c4: st.metric("Group",grp_str)

    # Players
    st.markdown("<div class='section-header'>⭐ Star Players</div>",unsafe_allow_html=True)
    pc=st.columns(3)
    for i,(name,pos,club,stat) in enumerate(get_players(team)):
        with pc[i%3]:
            st.markdown(f"""
            <div class="player-card">
                <div class="player-emoji">{"🥇" if i==0 else "⭐"}</div>
                <div class="player-name">{name}</div>
                <div class="player-pos">{pos}</div>
                <div class="player-club">🏟️ {club}</div>
                <div class="player-stat">{stat}</div>
            </div>""",unsafe_allow_html=True)

    # Radar
    st.markdown("<div class='section-header'>📊 Team Radar</div>",unsafe_allow_html=True)
    cats=["Attack","Defense","Form","Experience","Ranking"]
    vals=[min(rating/1850*100,100),max((2000-rating)/500*60+20,0),
          min(rating/1850*95,100),min(wc_titles/5*100+15,100),
          max(100-rank/48*100+10,0)]
    fig=go.Figure(go.Scatterpolar(
        r=vals+[vals[0]],theta=cats+[cats[0]],fill="toself",
        fillcolor="rgba(74,158,255,0.15)",line=dict(color="#4a9eff",width=2)
    ))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True,range=[0,100],
                                   gridcolor="rgba(255,255,255,0.08)",
                                   tickfont=dict(color="rgba(255,255,255,0.3)"))),
        paper_bgcolor="rgba(0,0,0,0)",font_color="white",height=380,
        margin=dict(t=20,b=20)
    )
    st.plotly_chart(fig,use_container_width=True)

    # H2H
    st.markdown("<div class='section-header'>⚔️ Head-to-Head vs Top Nations</div>",unsafe_allow_html=True)
    rivals=[t for t in ["Brazil","France","Argentina","England","Spain","Germany","Norway","Colombia"] if t!=team][:5]
    rows=[]
    for opp in rivals:
        r=safe_predict(team,opp,"final")
        rows.append({"Opponent":f"{flag(opp)} {opp}",
                     f"{flag(team)} Win%":f"{r['prob_a_win']}%",
                     "Draw%":f"{r['prob_draw']}%","Opp Win%":f"{r['prob_b_win']}%",
                     "Score":r["predicted_score"]})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — AI NARRATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📰  AI Narrator":
    st.markdown("<h1>📰 AI Match Narrator</h1>",unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,.45);margin-top:-12px'>Auto-generated match previews with tactical analysis</p>",unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1: sel_a=st.selectbox("Team A",TWF,index=TWF.index(ft("Spain"))); team_a=strip_flag(sel_a)
    with c2: sel_b=st.selectbox("Team B",TWF,index=TWF.index(ft("Portugal"))); team_b=strip_flag(sel_b)
    stage=st.selectbox("Stage",["group","round_of_16","quarter","semi","final"])

    if st.button("📝  Generate Preview",use_container_width=True,type="primary"):
        if team_a==team_b: st.error("Select two different teams.")
        else:
            r=safe_predict(team_a,team_b,stage)
            fav=team_a if r["prob_a_win"]>r["prob_b_win"] else team_b
            und=team_b if fav==team_a else team_a
            fav_prob=max(r["prob_a_win"],r["prob_b_win"])
            pa=get_players(team_a); pb=get_players(team_b)
            stage_str={"final":"THE WORLD CUP FINAL ⭐","semi":"a blockbuster semifinal",
                       "quarter":"a high-stakes quarterfinal","round_of_16":"a Round of 16 clash",
                       "group":"a crucial group stage match"}.get(stage,"this match")
            reasons,_=why_prediction(team_a,team_b,r)

            st.markdown(f"""
## {flag(team_a)} {team_a} vs {flag(team_b)} {team_b}
### {stage_str.upper()}
---
**{flag(fav)} {fav}** enters as the statistical favorite at **{fav_prob:.1f}%** win probability,
while **{flag(und)} {und}** will look to cause an upset.

**Key reasons the AI favours {fav}:**
""")
            for re in reasons: st.markdown(f"- {re}")

            st.markdown(f"""
---
### 📊 Match Statistics
| Metric | {flag(team_a)} {team_a} | {flag(team_b)} {team_b} |
|--------|----------|----------|
| Elo Rating | {r['rating_a']} | {r['rating_b']} |
| World Ranking | #{r['ranking_a']} | #{r['ranking_b']} |
| xG Estimate | {r['xg_a']} | {r['xg_b']} |

### 🔮 AI Prediction
- **{flag(team_a)} {team_a} Win:** {r['prob_a_win']}%
- **🤝 Draw:** {r['prob_draw']}%
- **{flag(team_b)} {team_b} Win:** {r['prob_b_win']}%
- **Predicted Score:** {r['predicted_score']}
- **Confidence:** {r['confidence']}%

### ⭐ Players to Watch
- **{flag(team_a)} {team_a}:** {pa[0][0]} — {pa[0][3]}
- **{flag(team_b)} {team_b}:** {pb[0][0]} — {pb[0][3]}
""")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MY PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋  My Predictions":
    st.markdown("<h1>📋 My Predictions</h1>",unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,.45);margin-top:-12px'>All your saved predictions — stored in Firebase</p>",unsafe_allow_html=True)

    uid=st.session_state.user_data.get("uid","")
    tok=st.session_state.user_data.get("idToken","")

    if not uid or not tok:
        st.error("Please log out and log back in.")
    else:
        with st.spinner("Loading predictions..."):
            preds=get_user_predictions(uid,tok)

        if not preds:
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:40px">
                <div style="font-size:48px;margin-bottom:12px">📭</div>
                <div style="font-size:18px;font-weight:600;color:#fff;margin-bottom:8px">No predictions yet</div>
                <div style="font-size:14px;color:rgba(255,255,255,.4)">Go to Match Predictor and click 💾 Save Prediction</div>
            </div>""",unsafe_allow_html=True)
        else:
            st.success(f"You have **{len(preds)}** saved predictions!")
            for i,p in enumerate(preds):
                ta=p.get("team_a","?"); tb=p.get("team_b","?")
                pa_w=float(p.get("prob_a_win",0)); pb_w=float(p.get("prob_b_win",0))
                draw=float(p.get("prob_draw",0)); score=p.get("score","?")
                conf=float(p.get("confidence",0))
                stage=p.get("stage","group").replace("_"," ").title()
                with st.expander(f"{flag(ta)} {ta} vs {flag(tb)} {tb} · {stage} · {score}"):
                    c1,c2,c3,c4=st.columns(4)
                    with c1: st.metric(f"{flag(ta)} {ta}",f"{pa_w}%")
                    with c2: st.metric("Draw",f"{draw}%")
                    with c3: st.metric(f"{flag(tb)} {tb}",f"{pb_w}%")
                    with c4: st.metric("Score",score)
                    share=(f"⚽ WC 2026 Prediction\n{flag(ta)} {ta} vs {flag(tb)} {tb}\n"
                           f"Score: {score}\n{ta}: {pa_w}% | Draw: {draw}% | {tb}: {pb_w}%\n"
                           f"Confidence: {conf}%\n🤖 World Cup 2026 AI Predictor")
                    st.download_button("📤 Export",data=share,
                        file_name=f"{ta}_vs_{tb}.txt",mime="text/plain",key=f"dl_{i}")

# Footer on all pages
st.markdown("""
<div class="footer">
    Built with <strong style="color:rgba(255,255,255,.5)">Python · XGBoost · Streamlit · Firebase · Plotly · Pandas · NumPy</strong>
    &nbsp;·&nbsp;
    <a href="https://github.com/Avi0790/-World-Cup-2026-AI-Predictor" target="_blank">GitHub ↗</a>
    &nbsp;·&nbsp;
    <a href="https://www.linkedin.com/in/avinaya-khadka-cs/" target="_blank">LinkedIn ↗</a>
    &nbsp;·&nbsp; Built by Avinaya Khadka · Arkansas State University
</div>
""", unsafe_allow_html=True)
