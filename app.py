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

# 1) Feature names (41 features + label)
COLUMN_NAMES = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    # ... (all the rest) ...
    "dst_host_srv_rerror_rate","label"
]

# 2) Load & split NSL-KDD once
@st.cache_data(show_spinner=False)
def load_and_split():
    train = pd.read_csv("NSL_KDD_Train.csv", names=COLUMN_NAMES)
    test  = pd.read_csv("NSL_KDD_Test.csv",  names=COLUMN_NAMES)
    df    = pd.concat([train, test], ignore_index=True)
    tr, te = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
    return tr.reset_index(drop=True), te.reset_index(drop=True)

# 3) Preprocess function
@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame):
    df = df.copy()
    # encode categorical cols
    for c in ["protocol_type","service","flag"]:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
    # map label: normal->0, else->1
    df["label"] = (
        df["label"].astype(str)
           .str.strip().str.lower()
           .map(lambda s: 0 if s=="normal" else 1)
    )
    X = df.drop("label",axis=1).apply(pd.to_numeric,errors="coerce").fillna(0)
    y = df["label"].astype("int32")
    return X.to_numpy(), y.to_numpy()

# 4) Train models once
@st.cache_resource(show_spinner=False)
def train_models():
    tr_df, te_df = load_and_split()
    X_tr, y_tr   = preprocess(tr_df)
    X_te, y_te   = preprocess(te_df)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tr,y_tr)

    # SVM
    svm_scaler = StandardScaler()
    X_tr_s = svm_scaler.fit_transform(X_tr)
    X_te_s = svm_scaler.transform(X_te)
    svm = SVC(kernel="rbf",gamma="scale")
    svm.fit(X_tr_s,y_tr)

    # Autoencoder
    ae_scaler = StandardScaler()
    X_ae_tr = ae_scaler.fit_transform(X_tr[y_tr==0])
    X_ae_te = ae_scaler.transform(X_te)

    inp = Input(shape=(X_ae_tr.shape[1],))
    e   = Dense(32,activation="relu")(inp)
    e   = Dense(16,activation="relu")(e)
    d   = Dense(32,activation="relu")(e)
    out = Dense(X_ae_tr.shape[1],activation="linear")(d)
    ae  = Model(inp,out); ae.compile("adam","mse")
    ae.fit(X_ae_tr,X_ae_tr,epochs=20,batch_size=256,shuffle=True,
           validation_split=0.1,verbose=0)

    recon = ae.predict(X_ae_te)
    mse   = np.mean((X_ae_te-recon)**2,axis=1)
    thresh = np.percentile(mse,95)

    return rf, svm, svm_scaler, ae, ae_scaler, thresh

# build models
rf_model, svm_model, svm_scaler, ae_model, ae_scaler, ae_thresh = train_models()

# 5) Streamlit UI
st.title("🚦 Network Traffic Inspector")
st.markdown("""
Upload a **CSV** (41 features + ignored label) or a **PCAP/PCAPNG** capture.
If you upload PCAP, we’ll parse it—but only if `tshark` is installed on the server.
""")

uploaded = st.file_uploader("Choose CSV or PCAP file", type=["csv","pcap","pcapng"])
if uploaded:
    ext = uploaded.name.rsplit(".",1)[-1].lower()
    user_df = None

    if ext == "csv":
        # CSV path
        user_df = pd.read_csv(uploaded, names=COLUMN_NAMES)

    else:
        # PCAP path with error handling
        try:
            tmp = tempfile.NamedTemporaryFile(suffix="."+ext, delete=False)
            tmp.write(uploaded.read()); tmp.flush(); tmp.close()

            cap = pyshark.FileCapture(tmp.name, keep_packets=False)
            rows = []
            for pkt in cap:
                try:
                    rows.append({
                      "duration": float(pkt.frame_info.time_epoch),
                      "protocol_type": pkt.transport_layer or "",
                      "service": pkt.highest_layer or "",
                      "flag": pkt.tcp.flags if hasattr(pkt,"tcp") else "",
                      "src_bytes": int(pkt.length),
                      # fill the other numeric features with zeros
                      **{col:0 for col in COLUMN_NAMES if col not in
                         ["duration","protocol_type","service","flag","src_bytes","label"]},
                      "label":"normal"
                    })
                except Exception:
                    continue
            cap.close()
            user_df = pd.DataFrame(rows, columns=COLUMN_NAMES)
        except TSharkNotFoundException:
            st.error(
                "⚠️ Could not parse PCAP: `tshark` binary not found on the server.\n"
                "Either install `tshark` (via apt) or export your traffic to **CSV** first."
            )

    # if we successfully got a DataFrame, run analysis
    if user_df is not None and st.button("Analyze My Traffic 🚀"):
        X_u, _ = preprocess(user_df)

        y_rf  = rf_model.predict(X_u)
        cnt_rf = np.bincount(y_rf,minlength=2)
        pct_rf = cnt_rf[1]/cnt_rf.sum()*100

        # SVM
        X_us  = svm_scaler.transform(X_u)
        y_svm = svm_model.predict(X_us)
        cnt_svm = np.bincount(y_svm,minlength=2)

        # Autoencoder
        X_ua = ae_scaler.transform(X_u)
        recon = ae_model.predict(X_ua)
        mse_u = np.mean((X_ua-recon)**2,axis=1)
        y_ae = (mse_u>ae_thresh).astype(int)
        cnt_ae = np.bincount(y_ae,minlength=2)

        st.metric("❗ % flagged as ATTACK (RF)", f"{pct_rf:.2f}%")
        st.dataframe(pd.DataFrame({
            "Model":["RF","SVM","AE"],
            "Normal":[cnt_rf[0],cnt_svm[0],cnt_ae[0]],
            "Attack":[cnt_rf[1],cnt_svm[1],cnt_ae[1]]
        }).set_index("Model"))
        cm = confusion_matrix(y_rf,y_rf)
        fig,ax=plt.subplots()
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)
        ax.set(xlabel="Pred",ylabel="Actual")
        st.pyplot(fig)
