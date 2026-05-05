import os, warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title = "HeartLens · Cardiac Screener",
    page_icon  = "🩺",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, .stApp {
    background: #f5f4f0 !important;
    font-family: 'Outfit', sans-serif;
    color: #1a1a1a;
}

[data-testid="stSidebar"] {
    background: #1a1a1a !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #f5f4f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label { color: #888 !important; font-size: 0.75rem !important; }

div[data-baseweb="select"] > div {
    background: #2a2a2a !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    color: #f5f4f0 !important;
}
div[data-baseweb="select"] * { color: #f5f4f0 !important; }
.stNumberInput input {
    background: #2a2a2a !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    color: #f5f4f0 !important;
    font-family: 'Space Mono', monospace !important;
}

.stButton > button {
    width: 100%;
    background: #c8f542 !important;
    color: #1a1a1a !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 12px !important;
    letter-spacing: 0.04em !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #b8e030 !important;
    transform: translateY(-1px);
}

.stDataFrame { border-radius: 10px !important; overflow: hidden; }
.stDataFrame * { font-family: 'Space Mono', monospace !important; font-size: 0.76rem !important; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_artefacts():
    BASE = os.path.dirname(os.path.abspath(__file__))
    return (
        joblib.load(os.path.join(BASE, "heart_model.pkl")),
        joblib.load(os.path.join(BASE, "scaler.pkl")),
        joblib.load(os.path.join(BASE, "feature_names.pkl")),
        joblib.load(os.path.join(BASE, "feature_importances.pkl")),
    )

model, scaler, FEATURE_NAMES, IMPORTANCES = load_artefacts()
CONTINUOUS_COLS = ['age','trestbps','chol','thalach','oldpeak','ca']


def build_feature_vector(raw):
    CP      = {0:"cp_0",1:"cp_1",2:"cp_2",3:"cp_3"}
    RECG    = {0:"restecg_0",1:"restecg_1",2:"restecg_2"}
    SLOPE   = {0:"slope_0",1:"slope_1",2:"slope_2"}
    THAL    = {1:"thal_1.0",2:"thal_2.0",3:"thal_3.0"}
    row = {f:0 for f in FEATURE_NAMES}
    for k in ['age','trestbps','chol','thalach','oldpeak','ca','sex','fbs','exang']:
        row[k] = raw[k]
    for mapping, key in [(CP,'cp'),(RECG,'restecg'),(SLOPE,'slope'),(THAL,'thal')]:
        col = mapping.get(raw[key])
        if col and col in row: row[col] = 1
    df = pd.DataFrame([row], columns=FEATURE_NAMES)
    cont = [c for c in CONTINUOUS_COLS if c in df.columns]
    df[cont] = scaler.transform(df[cont])
    return df


def top3_features(raw):
    labels = {'thalach':'Max Heart Rate','oldpeak':'ST Depression','ca':'Major Vessels',
               'cp_3':'Chest Pain','age':'Age','sex':'Sex','exang':'Exercise Angina',
               'slope_2':'ST Slope','trestbps':'Resting BP','chol':'Cholesterol',
               'thal_3.0':'Thalassemia','cp_0':'Chest Pain (Typ.)'}
    vals   = {'thalach':raw['thalach'],'oldpeak':raw['oldpeak'],'ca':raw['ca'],
               'age':raw['age'],'trestbps':raw['trestbps'],'chol':raw['chol']}
    res = []
    for feat, imp in sorted(IMPORTANCES.items(), key=lambda x:-x[1]):
        if len(res)==3: break
        res.append((labels.get(feat,feat), imp, vals.get(feat.split('_')[0],'—')))
    return res


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:24px 4px 8px'>
        <div style='font-family:Outfit,sans-serif;font-weight:700;font-size:1.5rem;color:#f5f4f0;'>
            Heart<span style='color:#c8f542;'>Lens</span>
        </div>
        <div style='font-size:0.7rem;color:#555;letter-spacing:0.12em;text-transform:uppercase;margin-top:2px;'>
            Cardiac Risk Screener
        </div>
    </div>
    <hr style='border:none;border-top:1px solid #2a2a2a;margin:16px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.7rem;color:#555;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;'>Vitals</p>", unsafe_allow_html=True)
    age      = st.number_input("Age",                 20, 80,  57)
    trestbps = st.number_input("Resting BP (mmHg)",   90, 200, 150)
    chol     = st.number_input("Cholesterol (mg/dl)", 100,600, 168)
    thalach  = st.number_input("Max Heart Rate",       70, 210, 174)
    oldpeak  = st.number_input("ST Depression",        0.0,6.2,1.6,step=0.1)
    ca       = st.number_input("Major Vessels",        0,  3,   0)

    st.markdown("<hr style='border:none;border-top:1px solid #2a2a2a;margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.7rem;color:#555;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:12px;'>Clinical Flags</p>", unsafe_allow_html=True)

    sex     = st.selectbox("Sex",              [0,1], format_func=lambda x:"Female" if x==0 else "Male", index=1)
    cp      = st.selectbox("Chest Pain",       [0,1,2,3], format_func=lambda x:{0:"Typical",1:"Atypical",2:"Non-Anginal",3:"Asymptomatic"}[x], index=3)
    fbs     = st.selectbox("Fasting BS >120",  [0,1], format_func=lambda x:"No" if x==0 else "Yes")
    restecg = st.selectbox("Resting ECG",      [0,1,2], format_func=lambda x:{0:"Normal",1:"ST-T Abnormal",2:"LV Hypertrophy"}[x])
    exang   = st.selectbox("Exercise Angina",  [0,1], format_func=lambda x:"No" if x==0 else "Yes")
    slope   = st.selectbox("ST Slope",         [0,1,2], format_func=lambda x:{0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
    thal    = st.selectbox("Thalassemia",      [1,2,3], format_func=lambda x:{1:"Normal",2:"Fixed Defect",3:"Reversible Defect"}[x])

    st.markdown("<hr style='border:none;border-top:1px solid #2a2a2a;margin:16px 0;'>", unsafe_allow_html=True)
    predict_btn = st.button("Run Screening →")


# ── MAIN ──────────────────────────────────────────────────────────────────────
top_l, top_r = st.columns([1, 2.2])

with top_l:
    rows = [
        ("Age",         f"{age} yrs"),
        ("Sex",         "Male" if sex==1 else "Female"),
        ("BP",          f"{trestbps} mmHg"),
        ("Cholesterol", f"{chol} mg/dl"),
        ("Max HR",      f"{thalach} bpm"),
        ("ST Depress.", f"{oldpeak}"),
        ("Vessels",     str(ca)),
        ("Fasting BS",  "High" if fbs==1 else "Normal"),
        ("Exer.Angina", "Yes" if exang==1 else "No"),
    ]
    row_items = "".join([
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"padding:9px 0;border-bottom:1px solid #2a2a2a;'>"
        f"<span style='font-size:0.76rem;color:#666;font-family:Outfit,sans-serif;'>{k}</span>"
        f"<span style='font-size:0.76rem;color:#f5f4f0;font-family:Space Mono,monospace;"
        f"font-weight:700;'>{v}</span></div>"
        for k, v in rows
    ])
    st.markdown(f"""
    <div style='padding:28px 26px;background:#1a1a1a;border-radius:16px;'>
        <div style='font-size:0.68rem;color:#555;text-transform:uppercase;
                    letter-spacing:0.14em;margin-bottom:18px;font-family:Outfit,sans-serif;'>
            Patient Summary
        </div>
        {row_items}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:12px;padding:14px 16px;background:white;border-radius:10px;'>
        <div style='font-size:0.68rem;color:#aaa;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:6px;'>Model</div>
        <div style='font-size:0.78rem;color:#1a1a1a;font-family:Outfit,sans-serif;line-height:1.7;'>
            Random Forest<br>UCI Cleveland (n=303)<br>
            <span style='color:#1b8a4a;font-weight:600;'>AUC-ROC &gt; 0.88</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


with top_r:
    if predict_btn:
        raw_inputs = dict(age=age,sex=sex,cp=cp,trestbps=trestbps,chol=chol,
                          fbs=fbs,restecg=restecg,thalach=thalach,exang=exang,
                          oldpeak=oldpeak,slope=slope,ca=ca,thal=thal)

        X     = build_feature_vector(raw_inputs)
        prob  = float(model.predict_proba(X)[0][1])
        cls   = int(prob >= 0.5)
        conf  = prob if cls==1 else (1-prob)
        cpct  = round(conf*100,1)

        if cls == 1:
            acc     = "#e53935"
            verdict = "Risk Detected"
            sub     = "Elevated cardiac indicators — clinical review recommended"
            bg_card = "#fff0f2"
        else:
            acc     = "#1b8a4a"
            verdict = "Low Risk"
            sub     = "Indicators within normal range — routine follow-up advised"
            bg_card = "#f0fff6"

        # Verdict card
        st.markdown(f"""
        <div style='background:{bg_card};border-radius:16px;padding:28px 32px;
                    border:1.5px solid {acc}33;margin-bottom:18px;'>
            <div style='display:flex;align-items:center;gap:10px;margin-bottom:6px;'>
                <div style='width:9px;height:9px;border-radius:50%;background:{acc};'></div>
                <span style='font-size:0.68rem;color:{acc};text-transform:uppercase;
                             letter-spacing:0.14em;font-family:Outfit,sans-serif;'>
                    Screening Result
                </span>
            </div>
            <div style='font-family:Outfit,sans-serif;font-size:2.2rem;font-weight:700;
                        color:{acc};line-height:1.1;margin-bottom:8px;'>
                {verdict}
            </div>
            <div style='font-size:0.82rem;color:#555;font-family:Outfit,sans-serif;'>
                {sub}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence + Probability tiles
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div style='background:white;border-radius:12px;padding:20px 22px;'>
                <div style='font-size:0.68rem;color:#aaa;text-transform:uppercase;
                            letter-spacing:0.12em;margin-bottom:6px;'>Confidence</div>
                <div style='font-size:2rem;font-weight:700;color:{acc};
                            font-family:Outfit,sans-serif;'>{cpct}%</div>
                <div style='height:5px;background:#eee;border-radius:99px;margin-top:10px;'>
                    <div style='height:5px;background:{acc};border-radius:99px;width:{cpct}%;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style='background:white;border-radius:12px;padding:20px 22px;'>
                <div style='font-size:0.68rem;color:#aaa;text-transform:uppercase;
                            letter-spacing:0.12em;margin-bottom:6px;'>Disease Probability</div>
                <div style='font-size:2rem;font-weight:700;color:#1a1a1a;
                            font-family:Outfit,sans-serif;'>{prob*100:.1f}%</div>
                <div style='font-size:0.74rem;color:#aaa;margin-top:10px;'>Threshold: 50%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # Feature chart
        top3  = top3_features(raw_inputs)
        lbls  = [t[0] for t in top3]
        vals  = [t[1] for t in top3]
        rvals = [t[2] for t in top3]

        fig, ax = plt.subplots(figsize=(6, 2.0))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.barh([2,1,0], [max(vals)*1.5]*3, color='#f0f0f0', height=0.42, zorder=1)
        ax.barh([2,1,0], vals, color=acc, height=0.42, zorder=2, alpha=0.9)
        for i,(v,rv) in enumerate(zip(vals,rvals)):
            ax.text(v+0.003,[2,1,0][i],f'{v:.3f}',va='center',
                    fontsize=8.5,color='#888',fontfamily='monospace')
        ax.set_yticks([0,1,2])
        ax.set_yticklabels(lbls[::-1], fontsize=9.5, color='#333')
        ax.set_xlabel("Feature Importance (MDI)", fontsize=8, color='#aaa')
        ax.set_xlim(0, max(vals)*1.65)
        ax.set_title("Top 3 Predictive Features", fontsize=10, fontweight='600',
                     color='#1a1a1a', pad=10)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(colors='#aaa', labelsize=8.5)
        ax.xaxis.label.set_color('#aaa')
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=False)

        chips = "".join([
            f"<span style='display:inline-block;background:white;border:1px solid #e0e0e0;"
            f"border-radius:20px;padding:4px 14px;font-size:0.74rem;color:#333;"
            f"font-family:Space Mono,monospace;margin:3px 4px 3px 0;'>{l}: {v}</span>"
            for l,_,v in top3
        ])
        st.markdown(chips, unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # Clinical note
        t1,t2,t3 = top3[0][0],top3[1][0],top3[2][0]
        v1,v2,v3 = top3[0][2],top3[1][2],top3[2][2]

        if cls==1:
            note = (f"Profile shows elevated cardiac risk indicators. Primary drivers: "
                    f"<b>{t1}</b> ({v1}), <b>{t2}</b> ({v2}), <b>{t3}</b> ({v3}). "
                    f"Reduced max HR and elevated ST depression suggest possible myocardial ischaemia. "
                    f"<b>Refer to cardiologist</b> for stress ECG and further evaluation.")
        else:
            note = (f"No high-risk indicators found in this profile. Key measurements: "
                    f"<b>{t1}</b> ({v1}), <b>{t2}</b> ({v2}), <b>{t3}</b> ({v3}). "
                    f"Values are within lower-risk ranges. "
                    f"<b>Continue routine monitoring</b> and reassess if symptoms develop.")

        st.markdown(f"""
        <div style='background:white;border-radius:12px;padding:20px 24px;
                    border-left:4px solid {acc};'>
            <div style='font-size:0.68rem;color:#aaa;text-transform:uppercase;
                        letter-spacing:0.12em;margin-bottom:8px;'>Clinical Interpretation</div>
            <div style='font-size:0.82rem;color:#333;line-height:1.8;
                        font-family:Outfit,sans-serif;'>{note}</div>
        </div>
        <p style='font-size:0.68rem;color:#ccc;margin-top:10px;'>
            ⚠ For educational use only. Not a substitute for professional medical diagnosis.
        </p>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='background:white;border-radius:16px;padding:60px 40px;
                    text-align:center;border:1.5px dashed #ddd;'>
            <div style='font-size:3rem;margin-bottom:16px;'>🩺</div>
            <div style='font-family:Outfit,sans-serif;font-size:1.25rem;
                        font-weight:700;color:#1a1a1a;margin-bottom:10px;'>
                Ready to Screen
            </div>
            <div style='font-size:0.82rem;color:#aaa;line-height:1.8;
                        max-width:320px;margin:0 auto;'>
                Patient form is pre-loaded with a real test case.<br>
                Adjust parameters and press
                <b style='color:#1a1a1a;'>Run Screening →</b>
            </div>
        </div>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px;'>
            <div style='background:white;border-radius:12px;padding:18px 20px;'>
                <div style='font-size:1.5rem;font-weight:700;color:#1a1a1a;
                            font-family:Outfit,sans-serif;'>303</div>
                <div style='font-size:0.72rem;color:#aaa;margin-top:2px;'>Training Patients</div>
            </div>
            <div style='background:white;border-radius:12px;padding:18px 20px;'>
                <div style='font-size:1.5rem;font-weight:700;color:#1b8a4a;
                            font-family:Outfit,sans-serif;'>88%+</div>
                <div style='font-size:0.72rem;color:#aaa;margin-top:2px;'>AUC-ROC Score</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# Footer
st.markdown("""
<hr style='border:none;border-top:1px solid #e0e0e0;margin:28px 0 14px;'>
<div style='display:flex;justify-content:space-between;font-size:0.7rem;
            color:#bbb;font-family:Outfit,sans-serif;'>
    <span>DS-3002 · Assignment #4 · FAST-NUCES BSDS Spring 2026</span>
    <span>Random Forest · UCI Cleveland Heart Disease Dataset</span>
</div>
""", unsafe_allow_html=True)