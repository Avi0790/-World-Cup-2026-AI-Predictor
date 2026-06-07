"""World Cup 2026 — XGBoost Model trained on real data"""
import pandas as pd, numpy as np, json, pickle, os, urllib.request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings; warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data/raw")
os.makedirs(DATA_DIR, exist_ok=True)

RESULTS_URL  = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
RESULTS_PATH = os.path.join(DATA_DIR, "results.csv")

TEAM_RATINGS = {
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

WC_TITLES = {"Brazil":5,"Germany":4,"Argentina":3,"France":2,"England":1,"Spain":1,"Uruguay":2}
FEATURE_COLS = ["elo_diff","elo_a","elo_b","wc_titles_diff","ranking_diff","squad_value_diff","neutral","stage"]

def load_data():
    if not os.path.exists(RESULTS_PATH):
        print("  Downloading real match data...")
        urllib.request.urlretrieve(RESULTS_URL, RESULTS_PATH)
    results = pd.read_csv(RESULTS_PATH)
    ranks = sorted(TEAM_RATINGS, key=TEAM_RATINGS.get, reverse=True)
    features = pd.DataFrame([{
        "team":t,"elo_rating":r,"form_points":1.5,
        "avg_goals_scored":1.4,"avg_goals_conceded":1.0,
        "wc_titles":WC_TITLES.get(t,0),"squad_value_m":r/3,
        "world_ranking":ranks.index(t)+1
    } for t,r in TEAM_RATINGS.items()])
    return results, features, TEAM_RATINGS

def build_training_data(results):
    ranks = sorted(TEAM_RATINGS, key=TEAM_RATINGS.get, reverse=True)
    def rank(t): return ranks.index(t)+1 if t in ranks else 35
    df = results[results["date"] >= "2000-01-01"].copy()
    df["neutral"] = df["neutral"].map({"TRUE":True,"FALSE":False,True:True,False:False})
    stage_map = {"FIFA World Cup":"final","UEFA Euro":"semi","Copa América":"semi",
                 "Africa Cup of Nations":"semi","Friendly":"friendly",
                 "UEFA Nations League":"group","CONCACAF Gold Cup":"group","Asian Cup":"group"}
    def get_stage(t):
        for k,v in stage_map.items():
            if k in str(t): return v
        return "group"
    stage_num = {"friendly":0,"group":1,"round_of_16":2,"quarter":3,"semi":4,"final":5}
    df["stage_n"] = df["tournament"].apply(get_stage).map(stage_num).fillna(1)
    df["outcome"] = df.apply(
        lambda r: "home_win" if r["home_score"]>r["away_score"]
        else ("away_win" if r["home_score"]<r["away_score"] else "draw"), axis=1)
    wc_teams = set(TEAM_RATINGS.keys())
    df = df[df["home_team"].isin(wc_teams) & df["away_team"].isin(wc_teams)]
    rows = []
    for _,r in df.iterrows():
        ta,tb = r["home_team"],r["away_team"]
        ra=TEAM_RATINGS.get(ta,1500); rb=TEAM_RATINGS.get(tb,1500)
        rows.append({"elo_diff":ra-rb,"elo_a":ra,"elo_b":rb,
            "wc_titles_diff":WC_TITLES.get(ta,0)-WC_TITLES.get(tb,0),
            "ranking_diff":rank(tb)-rank(ta),"squad_value_diff":(ra-rb)/3,
            "neutral":int(r.get("neutral",True)),"stage":r["stage_n"],"outcome":r["outcome"]})
    return pd.DataFrame(rows)

def predict_match(ta, tb, features, model, le, feature_cols, neutral=True, stage="group"):
    ranks = sorted(TEAM_RATINGS, key=TEAM_RATINGS.get, reverse=True)
    def rank(t): return ranks.index(t)+1 if t in ranks else 35
    stage_num = {"friendly":0,"group":1,"round_of_16":2,"quarter":3,"semi":4,"final":5}
    ra=TEAM_RATINGS.get(ta,1500); rb=TEAM_RATINGS.get(tb,1500)
    feat = pd.DataFrame([{"elo_diff":ra-rb,"elo_a":ra,"elo_b":rb,
        "wc_titles_diff":WC_TITLES.get(ta,0)-WC_TITLES.get(tb,0),
        "ranking_diff":rank(tb)-rank(ta),"squad_value_diff":(ra-rb)/3,
        "neutral":int(neutral),"stage":stage_num.get(stage,1)}])[FEATURE_COLS]
    probs = model.predict_proba(feat)[0]
    pd_ = dict(zip(le.classes_, [float(p) for p in probs]))
    pa=round(pd_.get("home_win",0)*100,1)
    pdr=round(pd_.get("draw",0)*100,1)
    pb=round(pd_.get("away_win",0)*100,1)
    xg_a=max(0.3,round(ra/1400*(1+(ra-rb)/2000),2))
    xg_b=max(0.3,round(rb/1400*(1+(rb-ra)/2000),2))
    sc_a=int(round(xg_a)); sc_b=int(round(xg_b))
    if pa>pb and sc_a<=sc_b: sc_a=sc_b+1
    if pb>pa and sc_b<=sc_a: sc_b=sc_a+1
    return {"team_a":ta,"team_b":tb,"prob_a_win":pa,"prob_draw":pdr,"prob_b_win":pb,
            "predicted_score":f"{sc_a}–{sc_b}","xg_a":xg_a,"xg_b":xg_b,
            "confidence":round(max(pa,pdr,pb),1),"rating_a":ra,"rating_b":rb,
            "ranking_a":rank(ta),"ranking_b":rank(tb),"stage":stage}

if __name__ == "__main__":
    print("Training on real match data...")
    results,features,ratings = load_data()
    print(f"Loaded {len(results):,} real matches")
    df = build_training_data(results)
    print(f"Training samples: {len(df):,}")
    le = LabelEncoder()
    y = le.fit_transform(df["outcome"])
    X = df[FEATURE_COLS]
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42)
    model = xgb.XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.08,
        subsample=0.8,colsample_bytree=0.8,eval_metric='mlogloss',random_state=42,verbosity=0)
    model.fit(X_tr,y_tr)
    acc = accuracy_score(y_te,model.predict(X_te))
    print(f"Accuracy: {acc:.2%} | Features: {FEATURE_COLS}")
    with open(os.path.join(BASE_DIR,"match_predictor.pkl"),"wb") as f:
        pickle.dump({"model":model,"le":le,"feature_cols":FEATURE_COLS,"accuracy":acc},f)
    print("Model saved!")
    for ta,tb,st in [("France","Spain","final"),("Argentina","England","semi"),("Brazil","Germany","quarter")]:
        r=predict_match(ta,tb,features,model,le,FEATURE_COLS,True,st)
        print(f"{ta} vs {tb}: {ta} {r['prob_a_win']}% | Draw {r['prob_draw']}% | {tb} {r['prob_b_win']}%")
