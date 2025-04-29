# app.py

import warnings, io, tempfile, os
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.metrics         import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.keras.models  import Model
from tensorflow.keras.layers  import Input, Dense

import pyshark
from pyshark.tshark.tshark import TSharkNotFoundException

sns.set_theme(style="ticks")

# --- 1) Features + Label names (41 features + 'label') ---
COLUMN_NAMES = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
    "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label"
]

# --- 2) Load & split NSL-KDD once ---
@st.cache_data(show_spinner=False)
def load_and_split():
    train = pd.read_csv("NSL_KDD_Train.csv", names=COLUMN_NAMES)
    test  = pd.read_csv("NSL_KDD_Test.csv",  names=COLUMN_NAMES)
    df = pd.concat([train, test], ignore_index=True)
    tr, te = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
    return tr.reset_index(drop=True), te.reset_index(drop=True)

# --- 3) Preprocess function ---
@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame):
    df = df.copy()
    for c in ["protocol_type","service","flag"]:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
    df["label"] = (
        df["label"].astype(str).str.strip().str.lower()
        .map(lambda s: 0 if s == "normal" else 1)
    )
    X = df.drop("label", axis=1).apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["label"].astype("int32")
    return X.to_numpy(), y.to_numpy()

# --- 4) Train models once ---
@st.cache_resource(show_spinner=False)
def train_models():
    tr_df, te_df = load_and_split()
    X_tr, y_tr = preprocess(tr_df)
    X_te, y_te = preprocess(te_df)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    svm = SVC(kernel="rbf", gamma="scale")
    svm.fit(X_tr_s, y_tr)

    ae_scaler = StandardScaler()
    X_ae_tr = ae_scaler.fit_transform(X_tr[y_tr==0])
    X_ae_te = ae_scaler.transform(X_te)

    inp = Input(shape=(X_ae_tr.shape[1],))
    e = Dense(32, activation="relu")(inp)
    e = Dense(16, activation="relu")(e)
    d = Dense(32, activation="relu")(e)
    out = Dense(X_ae_tr.shape[1], activation="linear")(d)
    ae = Model(inp, out); ae.compile(optimizer="adam", loss="mse")
    ae.fit(X_ae_tr, X_ae_tr, epochs=20, batch_size=256,
           shuffle=True, validation_split=0.1, verbose=0)

    recon = ae.predict(X_ae_te)
    mse = np.mean((X_ae_te - recon)**2, axis=1)
    thresh = np.percentile(mse, 95)

    return rf, svm, scaler, ae, ae_scaler, thresh

# build models
rf_model, svm_model, svm_scaler, ae_model, ae_scaler, ae_thresh = train_models()

# --- 5) UI ---
st.title("🚦 Network Traffic Inspector")

st.markdown(
    "Upload a **CSV** or **PCAP/PCAPNG** and click **Analyze My Traffic**."
)

uploaded = st.file_uploader("Choose CSV or PCAP", type=["csv","pcap","pcapng"])
if uploaded:
    ext = uploaded.name.rsplit('.',1)[-1].lower()
    user_df = None

    if ext == 'csv':
        user_df = pd.read_csv(uploaded, names=COLUMN_NAMES)

    else:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix='.'+ext, delete=False)
            tmp.write(uploaded.read()); tmp.close()
            caps = pyshark.FileCapture(tmp.name, keep_packets=False)
            rows = []
            for pkt in caps:
                try:
                    row = {**{f:0 for f in COLUMN_NAMES},
                           'duration': float(pkt.frame_info.time_epoch),
                           'protocol_type': pkt.transport_layer or '',
                           'service': pkt.highest_layer or '',
                           'flag': pkt.tcp.flags if hasattr(pkt,'tcp') else '',
                           'src_bytes': int(pkt.length),
                           'label':'normal'}
                    rows.append(row)
                except: pass
            caps.close(); os.remove(tmp.name)
            user_df = pd.DataFrame(rows, columns=COLUMN_NAMES)
        except TSharkNotFoundException:
            st.error("⚠️ `tshark` not found. Please upload CSV instead.")

    if user_df is not None and st.button("Analyze My Traffic 🚀"):
        X_u, _ = preprocess(user_df)
        y_rf = rf_model.predict(X_u)
        cnt = np.bincount(y_rf, minlength=2)
        pct = cnt[1]/cnt.sum()*100

        # threshold scale
        st.metric("% Attack Traffic", f"{pct:.2f}%")
        if pct < 5:
            st.success("✅ Your network is mostly safe (<5% attacks)")
        elif pct < 20:
            st.warning("⚠️ Moderate suspicion (5–20%). Review your logs!")
        else:
            st.error("🚨 High attack rate (>20%)! Immediate action recommended.")
            st.subheader("Possible Reasons:")
            st.write("- Malware beaconing to external servers")
            st.write("- Port scans or brute-force attempts")
            st.write("- Compromised devices sending spam or DDoS traffic")
            st.subheader("Recommended Actions:")
            st.write("1. Isolate suspicious hosts from the network.")
            st.write("2. Run antivirus/anti-malware scans on affected machines.")
            st.write("3. Check firewall logs for port-scanning patterns.")
            st.write("4. Update software/OS to patch known vulnerabilities.")

        # breakdown
        df_out = pd.DataFrame({
            'Model':['RF','SVM','AE'],
            'Normal':[cnt[0], *np.bincount(svm_model.predict(svm_scaler.transform(X_u)), minlength=2)[0:1]],
            'Attack':[cnt[1], *np.bincount(svm_model.predict(svm_scaler.transform(X_u)), minlength=2)[1:2]]
        }).set_index('Model')
        st.table(df_out)

        # confusion matrix (RF only)
        cm = confusion_matrix(y_rf, y_rf)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set(xlabel='Predicted', ylabel='Actual')
        st.pyplot(fig)
