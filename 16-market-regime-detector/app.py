import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="RegimeScope",page_icon="📉",layout="wide")
st.title("RegimeScope — model-defined market conditions")
st.warning("Regimes are retrospective statistical clusters, not objectively existing states or reliable forecasts.")


@st.cache_data
def demo_prices():
    rng=np.random.default_rng(47);dates=pd.bdate_range("2008-01-01",periods=4200);states=[];s=0
    transition=np.array([[.975,.018,.007],[.05,.93,.02],[.035,.025,.94]])
    mus=[.00045,-.00085,.0002];vols=[.007,.026,.013];r=[]
    for _ in dates:
        s=rng.choice(3,p=transition[s]);states.append(s);r.append(rng.normal(mus[s],vols[s]))
    return pd.Series(100*np.exp(np.cumsum(r)),index=dates,name="price")


upload=st.sidebar.file_uploader("Optional date/price CSV",type="csv")
try:
    if upload:
        d=pd.read_csv(upload);date=next((c for c in d if c.lower() in {"date","timestamp"}),d.columns[0]);price=next((c for c in d if c.lower() in {"price","close","adj_close"}),None)
        if not price:raise ValueError("CSV needs a price/close column.")
        d[date]=pd.to_datetime(d[date],errors="coerce");prices=d.set_index(date)[price].dropna().sort_index();source=f"Uploaded prices: {upload.name}"
    else:prices=demo_prices();source="Synthetic market path with changing volatility"
    if len(prices)<250:raise ValueError("At least 250 prices are required.")
except Exception as exc:st.error(str(exc));st.stop()
lookback=st.sidebar.slider("Momentum lookback",5,120,20);vol_window=st.sidebar.slider("Volatility window",5,120,20);k=st.sidebar.slider("Number of regimes",2,6,4)
f=pd.DataFrame({"return":prices.pct_change(),"volatility":prices.pct_change().rolling(vol_window).std()*np.sqrt(252),"momentum":prices.pct_change(lookback),"drawdown":prices/prices.rolling(max(lookback*3,60)).max()-1,"ma_gap":prices/prices.rolling(lookback).mean()-1}).dropna()
scaler=StandardScaler();labels=KMeans(k,random_state=42,n_init=30).fit_predict(scaler.fit_transform(f));f["regime_id"]=labels
summary=f.groupby("regime_id").agg(average_return=("return",lambda x:x.mean()*252),volatility=("return",lambda x:x.std()*np.sqrt(252)),average_drawdown=("drawdown","mean"),frequency=("return","size"))
order=summary.sort_values(["volatility","average_return"]).index;names={rid:f"Regime {i+1}" for i,rid in enumerate(order)};f["regime"]=f.regime_id.map(names);summary.index=summary.index.map(names);summary.index.name="regime"
runs=(f.regime!=f.regime.shift()).cumsum().rename("run_id");dur=f.groupby(runs).agg(regime=("regime","first"),duration=("regime","size")).reset_index(drop=True);summary["mean_duration_days"]=dur.groupby("regime").duration.mean()
trans=pd.crosstab(f.regime.shift(),f.regime,normalize="index").reindex(index=names.values(),columns=names.values()).fillna(0)
tabs=st.tabs(["Timeline","Regime profiles","Transitions","Stability test","Method & limits"])
with tabs[0]:
    st.info(source);chart=pd.DataFrame({"date":f.index,"price":prices.reindex(f.index),"regime":f.regime})
    st.plotly_chart(px.scatter(chart,x="date",y="price",color="regime",log_y=True,title="Historical prices colored by contemporaneous cluster"),width="stretch")
with tabs[1]:
    st.dataframe(summary.style.format({"average_return":"{:.1%}","volatility":"{:.1%}","average_drawdown":"{:.1%}","mean_duration_days":"{:.1f}"}),width="stretch")
    st.plotly_chart(px.scatter(summary.reset_index(),x="volatility",y="average_return",size="frequency",hover_name="regime",title="Model-defined profile"),width="stretch")
with tabs[2]:
    st.plotly_chart(px.imshow(trans,text_auto=".1%",color_continuous_scale="Blues",title="One-day transition matrix"),width="stretch")
with tabs[3]:
    cut=int(len(f)*.65);train=f.iloc[:cut].copy();later=f.iloc[cut:].copy();cols=["return","volatility","momentum","drawdown","ma_gap"]
    scale=StandardScaler().fit(train[cols]);cluster=KMeans(k,random_state=3,n_init=30).fit(scale.transform(train[cols]));train_lab=cluster.labels_;later_lab=cluster.predict(scale.transform(later[cols]))
    st.write("Centroids are learned only on the first 65%, then fixed and applied to the later 35%.")
    compare=pd.DataFrame({"Period":["Earlier"]*k+["Later"]*k,"Cluster":list(range(k))*2,"Share":list(pd.Series(train_lab).value_counts(normalize=True).reindex(range(k),fill_value=0))+list(pd.Series(later_lab).value_counts(normalize=True).reindex(range(k),fill_value=0))})
    st.plotly_chart(px.bar(compare,x="Cluster",y="Share",color="Period",barmode="group",title="Cluster prevalence stability"),width="stretch")
with tabs[4]:
    st.markdown("""Features use current/past returns, rolling volatility, momentum, drawdown, and moving-average distance. They are standardized before K-means. The timeline is descriptive because each point is assigned using a model fitted on the full selected history; the stability tab offers a stricter frozen-model check. Cluster identities can change with windows, scaling, history, and random initialization. Transition frequencies are not promises about the next state, and the system does not predict future regimes.""")
