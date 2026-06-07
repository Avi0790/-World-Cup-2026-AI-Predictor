import pandas as pd, numpy as np, os
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

def ko(ta, tb):
    ra=RATINGS.get(ta,1500); rb=RATINGS.get(tb,1500)
    pa=1/(1+10**(-(ra-rb)/400))
    return ta if np.random.random()<pa else tb

def sim_group(groups):
    q={}
    for g,teams in groups.items():
        pts=defaultdict(int); gd=defaultdict(int)
        for i in range(len(teams)):
            for j in range(i+1,len(teams)):
                ta,tb=teams[i],teams[j]
                ra=RATINGS.get(ta,1500); rb=RATINGS.get(tb,1500)
                pa=1/(1+10**(-(ra-rb)/400))*0.72
                pd_=0.27
                r=np.random.random()
                if r<pa: pts[ta]+=3;gd[ta]+=1;gd[tb]-=1
                elif r<pa+pd_: pts[ta]+=1;pts[tb]+=1
                else: pts[tb]+=3;gd[tb]+=1;gd[ta]-=1
        q[g]=sorted(teams,key=lambda t:(pts[t],gd[t]),reverse=True)[:2]
    return q

def run_simulation(n=500):
    print(f"Running {n} simulations with correct 2026 teams...")
    cc=defaultdict(int)
    for _ in range(n):
        q=sim_group(WC_2026_GROUPS)
        gl=list(q.keys())
        r16=[]
        for i in range(0,len(gl),2):
            g1,g2=gl[i],gl[i+1]
            r16+=[(q[g1][0],q[g2][1]),(q[g2][0],q[g1][1])]
        qf=[ko(a,b) for a,b in r16]
        sf=[ko(qf[i],qf[i+1]) for i in range(0,len(qf),2)]
        fi=[ko(sf[i],sf[i+1]) for i in range(0,len(sf),2)]
        cc[ko(fi[0],fi[1])]+=1
    all_t=[t for g in WC_2026_GROUPS.values() for t in g]
    rows=[{"team":t,"champion_pct":round(cc.get(t,0)/n*100,1)} for t in all_t]
    df=pd.DataFrame(rows).sort_values("champion_pct",ascending=False)
    os.makedirs(os.path.join(BASE_DIR,"../data/processed"),exist_ok=True)
    df.to_csv(os.path.join(BASE_DIR,"../data/processed/simulation_results.csv"),index=False)
    print("Top 10 champion probabilities:")
    for _,r in df.head(10).iterrows():
        print(f"  {r['team']:<30}{r['champion_pct']}%")
    return df

if __name__=="__main__":
    run_simulation(500)
