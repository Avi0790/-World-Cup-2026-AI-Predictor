"""
World Cup 2026 AI Predictor — AI World Cup 2026 Predictor
Full Streamlit Dashboard with Login, all 48 WC teams, flags, player cards
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle, os, sys
from firebase_auth import firebase_sign_up, firebase_sign_in, increment_predictions, save_prediction, get_user_predictions

sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
from train_model import predict_match, load_data

st.set_page_config(
    page_title="World Cup 2026 AI Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# DATA — ALL 48 WC 2026 TEAMS
# ══════════════════════════════════════════════════════════════════════════════
FLAGS = {
    # Europe (16)
    "England":              "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "France":               "🇫🇷",
    "Croatia":              "🇭🇷",
    "Norway":               "🇳🇴",
    "Portugal":             "🇵🇹",
    "Germany":              "🇩🇪",
    "Netherlands":          "🇳🇱",
    "Switzerland":          "🇨🇭",
    "Scotland":             "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Spain":                "🇪🇸",
    "Austria":              "🇦🇹",
    "Belgium":              "🇧🇪",
    "Bosnia and Herzegovina":"🇧🇦",
    "Sweden":               "🇸🇪",
    "Turkey":               "🇹🇷",
    "Czechia":              "🇨🇿",
    # South America (6)
    "Argentina":            "🇦🇷",
    "Brazil":               "🇧🇷",
    "Ecuador":              "🇪🇨",
    "Uruguay":              "🇺🇾",
    "Colombia":             "🇨🇴",
    "Paraguay":             "🇵🇾",
    # CONCACAF hosts (3)
    "USA":                  "🇺🇸",
    "Mexico":               "🇲🇽",
    "Canada":               "🇨🇦",
    # CONCACAF others (5)
    "Jamaica":              "🇯🇲",
    "Suriname":             "🇸🇷",
    "Panama":               "🇵🇦",
    "Honduras":             "🇭🇳",
    "Costa Rica":           "🇨🇷",
    # Asia (6)
    "Japan":                "🇯🇵",
    "South Korea":          "🇰🇷",
    "Australia":            "🇦🇺",
    "Iran":                 "🇮🇷",
    "Jordan":               "🇯🇴",
    "Uzbekistan":           "🇺🇿",
    # Africa (9)
    "Morocco":              "🇲🇦",
    "Senegal":              "🇸🇳",
    "Egypt":                "🇪🇬",
    "Algeria":              "🇩🇿",
    "Tunisia":              "🇹🇳",
    "Ghana":                "🇬🇭",
    "Cameroon":             "🇨🇲",
    "DR Congo":             "🇨🇩",
    "South Africa":         "🇿🇦",
    # Oceania (1)
    "New Zealand":          "🇳🇿",
    # Intercontinental playoff (2)
    "Bolivia":              "🇧🇴",
    "New Caledonia":        "🇳🇨",
}

# Elo ratings for all 48 teams
TEAM_RATINGS_48 = {
    "France":1840,"Argentina":1850,"England":1790,"Brazil":1820,"Spain":1800,
    "Germany":1780,"Portugal":1760,"Netherlands":1750,"Belgium":1720,"Croatia":1700,
    "Uruguay":1680,"USA":1620,"Mexico":1630,"Japan":1640,"Morocco":1650,
    "Senegal":1610,"Australia":1580,"South Korea":1590,"Switzerland":1660,
    "Denmark":1670,"Colombia":1660,"Norway":1630,"Scotland":1600,"Austria":1580,
    "Turkey":1590,"Czechia":1570,"Sweden":1580,"Bosnia and Herzegovina":1540,
    "Ecuador":1570,"Paraguay":1540,"Canada":1560,"Jamaica":1480,"Suriname":1420,
    "Panama":1470,"Honduras":1450,"Costa Rica":1510,"Iran":1530,"Jordan":1490,
    "Uzbekistan":1470,"Egypt":1550,"Algeria":1540,"Tunisia":1540,"Ghana":1550,
    "Cameroon":1560,"DR Congo":1520,"South Africa":1510,"New Zealand":1450,
    "Bolivia":1430,"New Caledonia":1300,
}

WC_2026_GROUPS = {
    "A": ["USA",        "Mexico",    "Canada",    "Uruguay"],
    "B": ["Brazil",     "Argentina", "Ecuador",   "Bolivia"],
    "C": ["France",     "Belgium",   "Morocco",   "Senegal"],
    "D": ["Spain",      "Portugal",  "Croatia",   "Turkey"],
    "E": ["England",    "Netherlands","Denmark",  "Panama"],
    "F": ["Germany",    "Switzerland","Austria",  "Czechia"],
    "G": ["Japan",      "South Korea","Australia","Jordan"],
    "H": ["Algeria",    "Egypt",     "Ghana",     "Cameroon"],
    "I": ["Colombia",   "Paraguay",  "Norway",    "New Zealand"],
    "J": ["Scotland",   "Sweden",    "Jamaica",   "Suriname"],
    "K": ["Iran",       "Uzbekistan","DR Congo",  "Honduras"],
    "L": ["Tunisia",    "Costa Rica","South Africa","New Caledonia"],
}

STAR_PLAYERS = {
    "Argentina":   [("Lionel Messi","Forward","Inter Miami","🐐 Defending champion captain"),
                    ("Julián Álvarez","Forward","Atlético Madrid","⚡ 2022 WC hero"),
                    ("Rodrigo De Paul","Midfielder","Atlético Madrid","🎯 Midfield engine")],
    "France":      [("Kylian Mbappé","Forward","Real Madrid","⚡ World's fastest striker"),
                    ("Antoine Griezmann","Midfielder","Atlético Madrid","🎯 117 caps"),
                    ("Aurélien Tchouaméni","Midfielder","Real Madrid","🛡️ Defensive anchor")],
    "Brazil":      [("Vinicius Jr.","Forward","Real Madrid","🔥 Ballon d'Or winner"),
                    ("Rodrygo","Forward","Real Madrid","⚡ Big game performer"),
                    ("Casemiro","Midfielder","Man United","🛡️ 5x UCL winner")],
    "England":     [("Jude Bellingham","Midfielder","Real Madrid","⭐ England's superstar"),
                    ("Harry Kane","Forward","Bayern Munich","🎯 All-time top scorer"),
                    ("Phil Foden","Midfielder","Man City","🏆 World-class creator")],
    "Spain":       [("Pedri","Midfielder","Barcelona","🎨 Creative genius"),
                    ("Lamine Yamal","Forward","Barcelona","🌟 Teen phenom"),
                    ("Rodri","Midfielder","Man City","🏆 Ballon d'Or")],
    "Germany":     [("Florian Wirtz","Midfielder","Bayer Leverkusen","⚡ Breakout star"),
                    ("Jamal Musiala","Midfielder","Bayern Munich","🌟 Germany's future"),
                    ("Toni Kroos","Midfielder","Real Madrid","🎯 Midfield maestro")],
    "Portugal":    [("Cristiano Ronaldo","Forward","Al Nassr","🐐 900+ career goals"),
                    ("Bruno Fernandes","Midfielder","Man United","🎯 Creative playmaker"),
                    ("Rafael Leão","Forward","AC Milan","⚡ Explosive winger")],
    "Netherlands": [("Virgil van Dijk","Defender","Liverpool","🛡️ World-class CB"),
                    ("Cody Gakpo","Forward","Liverpool","⚡ 20+ goals"),
                    ("Frenkie de Jong","Midfielder","Barcelona","🎨 Ball-playing maestro")],
    "Morocco":     [("Achraf Hakimi","Defender","PSG","🚀 Best RB in the world"),
                    ("Hakim Ziyech","Midfielder","Galatasaray","🎯 Creative leader"),
                    ("Youssef En-Nesyri","Forward","Fenerbahce","⚽ Clinical finisher")],
    "Croatia":     [("Luka Modrić","Midfielder","Real Madrid","🏆 2018 Ballon d'Or"),
                    ("Joško Gvardiol","Defender","Man City","🛡️ World-class CB"),
                    ("Ivan Perišić","Forward","Hajduk Split","💪 Veteran leader")],
    "Belgium":     [("Kevin De Bruyne","Midfielder","Man City","🎯 World-class creator"),
                    ("Romelu Lukaku","Forward","Roma","💪 75+ international goals"),
                    ("Thibaut Courtois","Goalkeeper","Real Madrid","🧤 UCL Final hero")],
    "Colombia":    [("James Rodríguez","Midfielder","Rayo Vallecano","🌟 Golden Boot 2014"),
                    ("Luis Díaz","Forward","Liverpool","⚡ Explosive winger"),
                    ("Jhon Durán","Forward","Al-Qadsiah","🔥 Rising star")],
    "Uruguay":     [("Darwin Núñez","Forward","Liverpool","⚡ Raw pace and power"),
                    ("Federico Valverde","Midfielder","Real Madrid","🏆 Box-to-box engine"),
                    ("Ronald Araújo","Defender","Barcelona","🛡️ Physical beast")],
    "USA":         [("Christian Pulisic","Forward","AC Milan","⭐ USMNT captain"),
                    ("Gio Reyna","Midfielder","Nottm Forest","🌟 Tech. playmaker"),
                    ("Tyler Adams","Midfielder","Bournemouth","💪 Midfield engine")],
    "Mexico":      [("Hirving Lozano","Forward","PSV","⚡ El Chucky"),
                    ("Edson Álvarez","Midfielder","West Ham","🛡️ Midfield anchor"),
                    ("Santiago Giménez","Forward","AC Milan","🎯 Clinical finisher")],
    "Japan":       [("Takefusa Kubo","Forward","Real Sociedad","⚡ Most creative"),
                    ("Ritsu Doan","Forward","Freiburg","🎯 Clinical winger"),
                    ("Wataru Endo","Midfielder","Liverpool","🛡️ Defensive anchor")],
    "Senegal":     [("Sadio Mané","Forward","Al Nassr","🏆 AFCON winner"),
                    ("Édouard Mendy","Goalkeeper","Al Ahli","🧤 Reliable stopper"),
                    ("Idrissa Gueye","Midfielder","Everton","💪 Tireless runner")],
    "Norway":      [("Erling Haaland","Forward","Man City","🚀 60+ goals per season"),
                    ("Martin Ødegaard","Midfielder","Arsenal","🎨 Creative maestro"),
                    ("Alexander Sørloth","Forward","Atlético Madrid","💪 Powerful striker")],
    "Australia":   [("Mathew Ryan","Goalkeeper","Real Sociedad","🧤 Shot stopper"),
                    ("Miloš Degenek","Defender","Columbus Crew","🛡️ Defensive rock"),
                    ("Martin Boyle","Forward","Al-Faisaly","⚡ Pacey winger")],
    "South Korea": [("Son Heung-min","Forward","Tottenham","⭐ Asian superstar"),
                    ("Lee Jae-sung","Midfielder","Mainz","🎯 Creative midfielder"),
                    ("Kim Min-jae","Defender","Bayern Munich","🛡️ Top-class CB")],
}

def get_players(team):
    return STAR_PLAYERS.get(team, [
        ("Key Player 1","Forward","Club","⚽ Top attacker"),
        ("Key Player 2","Midfielder","Club","🎯 Creative force"),
        ("Key Player 3","Defender","Club","🛡️ Defensive rock"),
    ])

def flag(team): return FLAGS.get(team, "🏳️")
def ft(team):   return f"{flag(team)} {team}"
def strip_flag(s): return " ".join(s.split(" ")[1:]) if " " in s else s

# ══════════════════════════════════════════════════════════════════════════════
# AUTH — Firebase Authentication
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp { background-color: #0a0e1a; color: #ffffff; }
    h1,h2,h3 { color: #ffffff !important; }
    .stSelectbox label, .stCheckbox label, .stTextInput label, .stRadio label
        { color: #aaaaaa !important; }
    div[data-testid="metric-container"] {
        background: #1a2744; border: 1px solid #2a4080;
        border-radius: 10px; padding: 15px;
    }
    .login-box {
        max-width: 420px; margin: 60px auto;
        background: linear-gradient(135deg,#1a2744,#0d1929);
        border: 1px solid #2a4080; border-radius: 16px; padding: 40px;
    }
    .login-title { font-size:28px; font-weight:700; color:#fff; text-align:center; margin-bottom:8px; }
    .login-sub   { font-size:14px; color:#888; text-align:center; margin-bottom:28px; }
    .player-card {
        background: linear-gradient(135deg,#1a2744,#0d1929);
        border:1px solid #2a4080; border-radius:12px;
        padding:16px; margin-bottom:8px; text-align:center;
    }
    .player-name { font-size:15px; font-weight:700; color:#fff; margin-bottom:4px; }
    .player-pos  { font-size:12px; color:#4a9eff; margin-bottom:2px; }
    .player-club { font-size:12px; color:#888; margin-bottom:6px; }
    .player-stat { font-size:12px; color:#f39c12; }
    .match-card  {
        background:linear-gradient(135deg,#1a2744,#0d1929);
        border:1px solid #2a4080; border-radius:16px;
        padding:24px; text-align:center; margin-bottom:16px;
    }
    .score-pred { font-size:42px; font-weight:700; color:#4a9eff; }
    .user-badge {
        background:#1a4d2e; border:1px solid #2ecc71;
        border-radius:20px; padding:4px 14px;
        font-size:13px; color:#2ecc71; font-weight:600;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state.logged_in   = False
    st.session_state.username    = ""
    st.session_state.user_data   = {}
    st.session_state.predictions = 0

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div style="text-align:center;margin-top:40px;margin-bottom:20px">
            <div style="font-size:64px">⚽</div>
            <div style="font-size:32px;font-weight:700;color:#fff">World Cup 2026 AI Predictor</div>
            <div style="font-size:15px;color:#888;margin-top:6px">
                AI-Powered World Cup 2026 Analytics
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔑 Login", "📝 Create Account"])

        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            login_email = st.text_input("Email", placeholder="your@email.com", key="login_email")
            login_pass  = st.text_input("Password", type="password", placeholder="Your password", key="login_pass")

            if st.button("Login →", use_container_width=True, type="primary"):
                if login_email and login_pass:
                    with st.spinner("Signing in..."):
                        ok, result = firebase_sign_in(login_email, login_pass)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.username  = result["username"]
                        st.session_state.user_data = result
                        st.rerun()
                    else:
                        st.error(result)
                else:
                    st.warning("Please enter your email and password.")
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("🔒 Secured by Firebase Authentication · Google Cloud")

        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            new_username = st.text_input("Username", placeholder="e.g. football_fan99", key="reg_user")
            new_email    = st.text_input("Email", placeholder="your@email.com", key="reg_email")
            new_pass     = st.text_input("Password", type="password", placeholder="Min 6 characters", key="reg_pass")
            new_pass2    = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="reg_pass2")

            if st.button("Create Account →", use_container_width=True, type="primary"):
                if not new_username or not new_email or not new_pass:
                    st.warning("Please fill in all fields.")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                elif new_pass != new_pass2:
                    st.error("Passwords don't match.")
                else:
                    with st.spinner("Creating your account..."):
                        ok, result = firebase_sign_up(new_email, new_pass, new_username)
                    if ok:
                        st.success("✅ Account created! Switch to the Login tab to sign in.")
                    else:
                        st.error(result)

    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP — only shown when logged in
# ══════════════════════════════════════════════════════════════════════════════

# Load model + data
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

# Use all WC 2026 teams — fall back to model teams if not in features
MODEL_TEAMS = features_df["team"].tolist()
ALL_TEAMS   = sorted(FLAGS.keys())
TEAMS_WITH_FLAGS = [ft(t) for t in ALL_TEAMS]

def safe_predict(ta, tb, stage="group", neutral=True):
    """Predict match — falls back gracefully if team not in model."""
    if ta in MODEL_TEAMS and tb in MODEL_TEAMS:
        return predict_match(ta, tb, features_df, model, le, feature_cols, neutral, stage)
    # Fallback: use ratings directly
    ra = TEAM_RATINGS_48.get(ta, 1500)
    rb = TEAM_RATINGS_48.get(tb, 1500)
    diff = ra - rb
    pa = round(1/(1+10**(-diff/400))*100 * 0.72, 1)
    pb = round(1/(1+10**(-(-diff)/400))*100 * 0.72, 1)
    pd_ = round(100-pa-pb, 1)
    return {
        "team_a":ta,"team_b":tb,
        "prob_a_win":pa,"prob_draw":pd_,"prob_b_win":pb,
        "predicted_score": f"{max(1,int(ra/800))}–{max(0,int(rb/900))}",
        "xg_a":round(ra/1400,2),"xg_b":round(rb/1400,2),
        "confidence":round(max(pa,pb),1),
        "rating_a":ra,"rating_b":rb,
        "ranking_a":sorted(TEAM_RATINGS_48,key=TEAM_RATINGS_48.get,reverse=True).index(ta)+1,
        "ranking_b":sorted(TEAM_RATINGS_48,key=TEAM_RATINGS_48.get,reverse=True).index(tb)+1,
        "stage":stage,
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏆 World Cup 2026 AI Predictor")
    st.markdown(f"<span class='user-badge'>👤 {st.session_state.username}</span>", unsafe_allow_html=True)
    st.markdown(f"<small style='color:#666'>Member since {st.session_state.user_data.get('joined','2026')}</small>", unsafe_allow_html=True)

    # ── Live countdown in sidebar ──
    from datetime import datetime, timezone
    kickoff = datetime(2026, 6, 11, 18, 0, 0, tzinfo=timezone.utc)
    now     = datetime.now(timezone.utc)
    delta   = kickoff - now
    if delta.total_seconds() > 0:
        days    = delta.days
        hours   = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60
        seconds = delta.seconds % 60
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1a4d2e,#0d2918);
                    border:1px solid #2ecc71;border-radius:10px;
                    padding:10px;text-align:center;margin:10px 0'>
            <div style='font-size:11px;color:#2ecc71;font-weight:600;
                        letter-spacing:.08em;text-transform:uppercase'>
                ⏱️ Kickoff Countdown
            </div>
            <div style='font-size:22px;font-weight:700;color:#fff;margin-top:4px'>
                {days}d {hours:02d}h {minutes:02d}m
            </div>
            <div style='font-size:11px;color:#888;margin-top:2px'>
                Jun 11, 2026 · Mexico City
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("🏆 The World Cup is ON!")

    st.divider()
    page = st.radio("Navigate", [
        "🏆 Match Predictor",
        "🌍 Tournament Simulator",
        "📊 Odds Analyzer",
        "🧠 Team Intelligence",
        "📰 AI Match Narrator",
        "📋 My Predictions",
    ])
    st.divider()
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username  = ""
        st.rerun()
    st.caption("Built by Avinaya Khadka")
    st.caption("XGBoost · Python · Streamlit")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏆 Match Predictor":
    st.title("⚽ Match Prediction Engine")
    st.caption("AI-powered predictions for all 48 World Cup 2026 nations")

    c1,c2,c3 = st.columns([2,1,2])
    with c1:
        sel_a  = st.selectbox("🔵 Team A", TEAMS_WITH_FLAGS, index=TEAMS_WITH_FLAGS.index(ft("Argentina")))
        team_a = strip_flag(sel_a)
    with c2:
        st.markdown("<br><br><center style='color:#888;font-size:1.5rem'>VS</center>", unsafe_allow_html=True)
    with c3:
        sel_b  = st.selectbox("🔴 Team B", TEAMS_WITH_FLAGS, index=TEAMS_WITH_FLAGS.index(ft("France")))
        team_b = strip_flag(sel_b)

    c4,c5 = st.columns(2)
    with c4: stage   = st.selectbox("Stage", ["group","round_of_16","quarter","semi","final"])
    with c5: neutral = st.checkbox("Neutral Venue", value=True)

    if st.button("🔮 Predict Match", use_container_width=True, type="primary"):
        if team_a == team_b:
            st.error("Select two different teams.")
        else:
            r = safe_predict(team_a, team_b, stage, neutral)
            st.markdown(f"""
            <div class="match-card">
                <span style="font-size:52px">{flag(team_a)}</span>
                &nbsp;&nbsp;<span style="font-size:20px;color:#888">vs</span>&nbsp;&nbsp;
                <span style="font-size:52px">{flag(team_b)}</span><br>
                <span style="font-size:22px;font-weight:700">{team_a}</span>
                &nbsp;&nbsp;&nbsp;
                <span style="font-size:22px;font-weight:700">{team_b}</span><br><br>
                <span class="score-pred">{r['predicted_score']}</span><br>
                <small style="color:#888">AI Predicted Score</small>
            </div>
            """, unsafe_allow_html=True)

            c1,c2,c3 = st.columns(3)
            with c1: st.metric(f"{flag(team_a)} {team_a} Win", f"{r['prob_a_win']}%")
            with c2: st.metric("🤝 Draw", f"{r['prob_draw']}%")
            with c3: st.metric(f"{flag(team_b)} {team_b} Win", f"{r['prob_b_win']}%")

            fig = go.Figure(go.Bar(
                x=[f"{flag(team_a)} {team_a}", "Draw", f"{flag(team_b)} {team_b}"],
                y=[r['prob_a_win'], r['prob_draw'], r['prob_b_win']],
                marker_color=["#4a9eff","#f39c12","#e74c3c"],
                text=[f"{v}%" for v in [r['prob_a_win'],r['prob_draw'],r['prob_b_win']]],
                textposition="outside"
            ))
            fig.update_layout(paper_bgcolor="#0a0e1a",plot_bgcolor="#0a0e1a",
                              font_color="white",height=320,showlegend=False,
                              title="Win Probability Distribution")
            st.plotly_chart(fig, use_container_width=True)

            c4,c5,c6 = st.columns(3)
            with c4: st.metric(f"xG {flag(team_a)}", r['xg_a'])
            with c5: st.metric(f"xG {flag(team_b)}", r['xg_b'])
            with c6: st.metric("Confidence", f"{r['confidence']}%")

            # ── Save prediction to Firebase ──
            st.divider()
            col_save, col_share = st.columns(2)
            with col_save:
                if st.button("💾 Save Prediction", use_container_width=True):
                    uid      = st.session_state.user_data.get("uid","")
                    id_token = st.session_state.user_data.get("idToken","")
                    if uid and id_token:
                        ok = save_prediction(uid, id_token, team_a, team_b, stage, r)
                        if ok:
                            st.success("✅ Prediction saved to your history!")
                        else:
                            st.error("Could not save — check your connection.")
                    else:
                        st.error("Please log in again to save predictions.")

            with col_share:
                share_text = (
                    f"⚽ My WC 2026 Prediction\n"
                    f"{flag(team_a)} {team_a} vs {flag(team_b)} {team_b}\n"
                    f"Score: {r['predicted_score']}\n"
                    f"{team_a} Win: {r['prob_a_win']}% | "
                    f"Draw: {r['prob_draw']}% | "
                    f"{team_b} Win: {r['prob_b_win']}%\n"
                    f"Confidence: {r['confidence']}%\n"
                    f"🤖 World Cup 2026 AI Predictor"
                )
                st.download_button(
                    "📤 Export Prediction",
                    data=share_text,
                    file_name=f"{team_a}_vs_{team_b}_prediction.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            st.divider()
            st.subheader("⭐ Key Players")
            pc1,pc2 = st.columns(2)
            for col,team in [(pc1,team_a),(pc2,team_b)]:
                with col:
                    st.markdown(f"### {flag(team)} {team}")
                    for name,pos,club,stat in get_players(team):
                        st.markdown(f"""
                        <div class="player-card">
                            <div class="player-name">{name}</div>
                            <div class="player-pos">{pos}</div>
                            <div class="player-club">🏟️ {club}</div>
                            <div class="player-stat">{stat}</div>
                        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TOURNAMENT SIMULATOR  (BUG FIXED)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Tournament Simulator":
    st.title("🌍 World Cup 2026 Simulator")
    st.caption("Monte Carlo simulation across all 48 teams in 12 groups")

    if sim_results.empty:
        st.warning("Run simulations/simulator.py to generate results.")
    else:
        sim = sim_results.copy()
        # ── FIX: safely rename only the columns that exist ──
        team_col  = sim.columns[0]
        champ_col = sim.columns[1]
        sim["display"] = sim[team_col].apply(lambda t: ft(t))

        # Chart
        top12 = sim.head(12)
        fig = px.bar(
            top12, x=champ_col, y="display", orientation="h",
            color=champ_col, color_continuous_scale="Blues",
            labels={champ_col:"Champion Probability (%)", "display":""},
            title="🏆 World Cup 2026 Win Probability"
        )
        fig.update_layout(paper_bgcolor="#0a0e1a",plot_bgcolor="#1a2744",
                          font_color="white",height=500,coloraxis_showscale=False)
        fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        # Top 5
        st.subheader("🏅 Top Favorites")
        cols = st.columns(5)
        for i,(_,row) in enumerate(sim.head(5).iterrows()):
            with cols[i]:
                st.markdown(f"<div style='text-align:center;font-size:36px'>{flag(row[team_col])}</div>",
                            unsafe_allow_html=True)
                st.metric(row[team_col], f"{row[champ_col]}%")

        # Full table — safe rename
        st.subheader("📋 Full Table")
        tbl = sim[["display", champ_col]].copy()
        tbl.columns = ["Team", "Champion %"]
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        # Dark horses
        st.subheader("🐴 Dark Horse Watch")
        dark = sim[(sim[champ_col] > 0) & (sim[champ_col] < 3)].head(4)
        if not dark.empty:
            dcols = st.columns(min(4, len(dark)))
            for i,(_,row) in enumerate(dark.iterrows()):
                with dcols[i]:
                    st.markdown(f"<div style='text-align:center;font-size:28px'>{flag(row[team_col])}</div>",
                                unsafe_allow_html=True)
                    st.metric(row[team_col], f"{row[champ_col]}%", delta="Dark Horse 🐴")

        # Group breakdown
        st.subheader("📋 2026 World Cup Groups")
        gcols = st.columns(4)
        for i,(grp,teams) in enumerate(WC_2026_GROUPS.items()):
            with gcols[i % 4]:
                st.markdown(f"**Group {grp}**")
                for t in teams:
                    st.markdown(f"{flag(t)} {t}")
                st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ODDS ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Odds Analyzer":
    st.title("📊 Smart Odds Analyzer")
    st.caption("Educational analytics only — not a betting recommendation tool")
    st.info("⚠️ For analytical and educational purposes only.")

    c1,c2 = st.columns(2)
    with c1:
        sel_a  = st.selectbox("Team A", TEAMS_WITH_FLAGS, index=TEAMS_WITH_FLAGS.index(ft("Brazil")))
        team_a = strip_flag(sel_a)
    with c2:
        sel_b  = st.selectbox("Team B", TEAMS_WITH_FLAGS, index=TEAMS_WITH_FLAGS.index(ft("Germany")))
        team_b = strip_flag(sel_b)

    c3,c4,c5 = st.columns(3)
    with c3: odds_a    = st.number_input(f"{flag(team_a)} {team_a}", min_value=1.01, value=2.10, step=0.05)
    with c4: odds_draw = st.number_input("Draw", min_value=1.01, value=3.40, step=0.05)
    with c5: odds_b    = st.number_input(f"{flag(team_b)} {team_b}", min_value=1.01, value=3.20, step=0.05)

    if st.button("🔍 Analyze", use_container_width=True, type="primary"):
        if team_a == team_b:
            st.error("Select two different teams.")
        else:
            ri   = 1/odds_a + 1/odds_draw + 1/odds_b
            imp_a, imp_d, imp_b = round(1/odds_a/ri*100,1), round(1/odds_draw/ri*100,1), round(1/odds_b/ri*100,1)
            r    = safe_predict(team_a, team_b)
            mod_a, mod_d, mod_b = r["prob_a_win"], r["prob_draw"], r["prob_b_win"]

            fig = go.Figure()
            labels = [f"{flag(team_a)} {team_a}", "Draw", f"{flag(team_b)} {team_b}"]
            fig.add_trace(go.Bar(name="Bookmaker", x=labels, y=[imp_a,imp_d,imp_b], marker_color="#e74c3c"))
            fig.add_trace(go.Bar(name="AI Model",  x=labels, y=[mod_a,mod_d,mod_b], marker_color="#4a9eff"))
            fig.update_layout(barmode="group", paper_bgcolor="#0a0e1a",
                              plot_bgcolor="#1a2744", font_color="white",
                              height=380, title="Model vs Market")
            st.plotly_chart(fig, use_container_width=True)

            c6,c7,c8 = st.columns(3)
            for col,lbl,disc in zip([c6,c7,c8], labels,
                                     [mod_a-imp_a, mod_d-imp_d, mod_b-imp_b]):
                with col:
                    if abs(disc) > 10:
                        st.success(f"**{lbl}**\n\n🟢 Discrepancy detected\n\nDiff: **{disc:+.1f}%**")
                    else:
                        st.info(f"**{lbl}**\n\nNo major discrepancy\n\nDiff: {disc:+.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TEAM INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Team Intelligence":
    st.title("🧠 Team Intelligence")

    sel  = st.selectbox("Select a Team", TEAMS_WITH_FLAGS, index=TEAMS_WITH_FLAGS.index(ft("Brazil")))
    team = strip_flag(sel)

    st.markdown(f"<div style='font-size:80px;text-align:center'>{flag(team)}</div>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align:center'>{team}</h2>", unsafe_allow_html=True)

    rating = TEAM_RATINGS_48.get(team, 1500)
    rank   = sorted(TEAM_RATINGS_48, key=TEAM_RATINGS_48.get, reverse=True).index(team)+1

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Elo Rating",  rating)
    with c2: st.metric("WC Ranking",  f"#{rank}")
    with c3: st.metric("Group", [g for g,t in WC_2026_GROUPS.items() if team in t][0] if any(team in t for t in WC_2026_GROUPS.values()) else "—")
    with c4: st.metric("Confederation",
        "UEFA" if team in ["England","France","Germany","Spain","Portugal","Netherlands",
                            "Belgium","Croatia","Switzerland","Norway","Scotland","Austria",
                            "Turkey","Czechia","Sweden","Bosnia and Herzegovina"] else
        "CONMEBOL" if team in ["Argentina","Brazil","Ecuador","Uruguay","Colombia","Paraguay","Bolivia"] else
        "CONCACAF" if team in ["USA","Mexico","Canada","Jamaica","Suriname","Panama","Honduras","Costa Rica"] else
        "CAF" if team in ["Morocco","Senegal","Egypt","Algeria","Tunisia","Ghana","Cameroon","DR Congo","South Africa"] else
        "AFC" if team in ["Japan","South Korea","Australia","Iran","Jordan","Uzbekistan"] else "OFC")

    st.divider()
    st.subheader("⭐ Star Players")
    players = get_players(team)
    pcols   = st.columns(3)
    for i,(name,pos,club,stat) in enumerate(players):
        with pcols[i % 3]:
            st.markdown(f"""
            <div class="player-card">
                <div style="font-size:28px">{"🥇" if i==0 else "⭐"}</div>
                <div class="player-name">{name}</div>
                <div class="player-pos">{pos}</div>
                <div class="player-club">🏟️ {club}</div>
                <div class="player-stat">{stat}</div>
            </div>""", unsafe_allow_html=True)

    # Radar
    st.subheader("📊 Team Profile Radar")
    wc_titles = {"Brazil":5,"Germany":4,"Italy":4,"Argentina":3,"France":2,
                 "England":1,"Spain":1,"Uruguay":2}.get(team, 0)
    vals = [
        min(rating/1850*100, 100),
        min((2000-rating)/500*60+20, 100),
        min(rating/1850*95, 100),
        min(wc_titles/5*100+15, 100),
        max(100 - rank/48*100 + 10, 0),
    ]
    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]],
        theta=["Attack","Defense","Form","Experience","Ranking","Attack"],
        fill="toself", fillcolor="rgba(74,158,255,0.2)",
        line_color="#4a9eff"
    ))
    fig.update_layout(
        polar=dict(bgcolor="#1a2744", radialaxis=dict(visible=True,range=[0,100],color="#888")),
        paper_bgcolor="#0a0e1a", font_color="white", height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # H2H
    st.subheader(f"⚔️ {flag(team)} vs Top Nations")
    rivals = [t for t in ["Brazil","France","Argentina","England","Spain","Germany","Norway","Colombia"] if t!=team][:5]
    rows = []
    for opp in rivals:
        r = safe_predict(team, opp, "final")
        rows.append({
            "Opponent": f"{flag(opp)} {opp}",
            f"{flag(team)} Win%": f"{r['prob_a_win']}%",
            "Draw%": f"{r['prob_draw']}%",
            "Opp Win%": f"{r['prob_b_win']}%",
            "Score": r["predicted_score"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — AI NARRATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📰 AI Match Narrator":
    st.title("📰 AI Match Narrator")
    st.caption("Auto-generated match previews with tactical analysis")

    c1,c2 = st.columns(2)
    with c1:
        sel_a  = st.selectbox("Team A", TEAMS_WITH_FLAGS, index=TEAMS_WITH_FLAGS.index(ft("Spain")))
        team_a = strip_flag(sel_a)
    with c2:
        sel_b  = st.selectbox("Team B", TEAMS_WITH_FLAGS, index=TEAMS_WITH_FLAGS.index(ft("Portugal")))
        team_b = strip_flag(sel_b)

    stage = st.selectbox("Stage", ["group","round_of_16","quarter","semi","final"])

    if st.button("📝 Generate Preview", use_container_width=True, type="primary"):
        if team_a == team_b:
            st.error("Select two different teams.")
        else:
            r = safe_predict(team_a, team_b, stage)
            fav = team_a if r["prob_a_win"] > r["prob_b_win"] else team_b
            und = team_b if fav == team_a else team_a
            fav_prob = max(r["prob_a_win"], r["prob_b_win"])
            pa = get_players(team_a); pb = get_players(team_b)
            stage_str = {"final":"THE WORLD CUP FINAL ⭐","semi":"a blockbuster semifinal",
                         "quarter":"a high-stakes quarterfinal","round_of_16":"a Round of 16 clash",
                         "group":"a crucial group stage match"}.get(stage,"this match")

            st.markdown(f"""
## {flag(team_a)} {team_a} vs {flag(team_b)} {team_b}
### {stage_str.upper()}
---
**{flag(fav)} {fav}** is the statistical favorite at **{fav_prob:.1f}%** win probability,
while **{flag(und)} {und}** will look to cause an upset.

---
### 📊 Head-to-Head Stats
| Metric | {flag(team_a)} {team_a} | {flag(team_b)} {team_b} |
|--------|----------|----------|
| Elo Rating | {r['rating_a']} | {r['rating_b']} |
| World Ranking | #{r['ranking_a']} | #{r['ranking_b']} |
| xG Estimate | {r['xg_a']} | {r['xg_b']} |

---
### 🔮 AI Prediction
- **{flag(team_a)} {team_a} Win:** {r['prob_a_win']}%
- **🤝 Draw:** {r['prob_draw']}%
- **{flag(team_b)} {team_b} Win:** {r['prob_b_win']}%
- **Predicted Score:** {r['predicted_score']}
- **Model Confidence:** {r['confidence']}%

---
### ⭐ Players to Watch
- **{flag(team_a)} {team_a}:** {pa[0][0]} — {pa[0][3]}
- **{flag(team_b)} {team_b}:** {pb[0][0]} — {pb[0][3]}
""")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — MY PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 My Predictions":
    st.title("📋 My Predictions")
    st.caption("All your saved match predictions — powered by Firebase")

    uid      = st.session_state.user_data.get("uid", "")
    id_token = st.session_state.user_data.get("idToken", "")

    if not uid or not id_token:
        st.error("Please log out and log back in to view predictions.")
    else:
        with st.spinner("Loading your predictions..."):
            preds = get_user_predictions(uid, id_token)

        if not preds:
            st.info("You haven't saved any predictions yet. Go to Match Predictor and click 💾 Save Prediction!")
        else:
            st.success(f"You have **{len(preds)}** saved predictions!")
            st.divider()

            for i, p in enumerate(preds):
                ta = p.get("team_a", "?")
                tb = p.get("team_b", "?")
                pa_win = float(p.get("prob_a_win", 0))
                pb_win = float(p.get("prob_b_win", 0))
                draw   = float(p.get("prob_draw", 0))
                score  = p.get("score", "?–?")
                conf   = float(p.get("confidence", 0))
                stage  = p.get("stage", "group").replace("_"," ").title()
                fav    = ta if pa_win > pb_win else tb

                with st.expander(f"{flag(ta)} {ta} vs {flag(tb)} {tb} · {stage} · Score: {score}"):
                    c1,c2,c3,c4 = st.columns(4)
                    with c1: st.metric(f"{flag(ta)} {ta}", f"{pa_win}%")
                    with c2: st.metric("Draw", f"{draw}%")
                    with c3: st.metric(f"{flag(tb)} {tb}", f"{pb_win}%")
                    with c4: st.metric("Predicted Score", score)

                    st.markdown(f"""
                    <div style='background:#1a2744;border:1px solid #2a4080;
                                border-radius:8px;padding:12px;margin-top:8px'>
                        <span style='color:#4a9eff;font-weight:600'>Favorite:</span>
                        {flag(fav)} {fav} &nbsp;·&nbsp;
                        <span style='color:#4a9eff;font-weight:600'>Confidence:</span>
                        {conf}% &nbsp;·&nbsp;
                        <span style='color:#4a9eff;font-weight:600'>Stage:</span>
                        {stage}
                    </div>
                    """, unsafe_allow_html=True)

                    # Shareable card text
                    share_text = (
                        f"⚽ My WC 2026 Prediction\n"
                        f"{flag(ta)} {ta} vs {flag(tb)} {tb}\n"
                        f"Score: {score}\n"
                        f"{ta} Win: {pa_win}% | Draw: {draw}% | {tb} Win: {pb_win}%\n"
                        f"Confidence: {conf}%\n"
                        f"🤖 World Cup 2026 AI Predictor"
                    )
                    st.download_button(
                        "📤 Export this prediction",
                        data=share_text,
                        file_name=f"{ta}_vs_{tb}.txt",
                        mime="text/plain",
                        key=f"dl_{i}"
                    )
