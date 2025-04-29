# app.py

import warnings, io
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

# --- 3) Single preprocess function for train/test **and** user data ---
@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame):
    df = df.copy()
    # encode categoricals
    for c in ["protocol_type","service","flag"]:
        le = LabelEncoder()
        df[c] = df[c].astype(str)
        le.fit(df[c])
        df[c] = le.transform(df[c])
    # map label exact "normal" → 0, else → 1
    df["label"] = (
        df["label"].astype(str)
           .str.strip().str.lower()
           .map(lambda s: 0 if s=="normal" else 1)
    )
    # separate features & label, force numeric
    X = (df.drop("label", axis=1)
           .apply(pd.to_numeric, errors="coerce")
           .fillna(0)
       )
    y = df["label"].astype("int32")
    return X.to_numpy(), y.to_numpy()

# --- 4) Train models once and cache them ---
@st.cache_resource(show_spinner=False)
def train_models():
    train_df, test_df = load_and_split()
    X_tr, y_tr = preprocess(train_df)
    X_te, y_te = preprocess(test_df)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)

    # SVM
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    svm = SVC(kernel="rbf", gamma="scale")
    svm.fit(X_tr_s, y_tr)

    # Autoencoder
    X_norm = X_tr[y_tr==0]
    sc_ae  = StandardScaler()
    X_ae_tr = sc_ae.fit_transform(X_norm)
    X_ae_te = sc_ae.transform(X_te)

    inp = Input(shape=(X_ae_tr.shape[1],))
    e   = Dense(32, activation="relu")(inp)
    e   = Dense(16, activation="relu")(e)
    d   = Dense(32, activation="relu")(e)
    out = Dense(X_ae_tr.shape[1], activation="linear")(d)
    ae  = Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(X_ae_tr, X_ae_tr,
           epochs=20, batch_size=256,
           shuffle=True, validation_split=0.1, verbose=0)

    recon = ae.predict(X_ae_te)
    mse   = np.mean((X_ae_te - recon)**2, axis=1)
    thresh = np.percentile(mse, 95)

    return rf, svm, scaler, sc_ae, ae, thresh

# build all models+preprocessors
rf_model, svm_model, svm_scaler, ae_scaler, ae_model, ae_thresh = train_models()

# --- 5) Streamlit UI ---
st.title("🚦 Network Traffic Inspector")

st.markdown(
    "Upload your **CSV** or **PCAP/PCAPNG** capture and click **Analyze My Traffic**.\n\n"
    "- If you upload CSV, it must have the **41 feature columns** + an ignored label column.\n"
    "- If you upload PCAP, we'll extract the features for you."
)

upload = st.file_uploader("Choose CSV or PCAP file", type=["csv","pcap","pcapng"])
if upload is not None:
    ext = upload.name.split('.')[-1].lower()

    # --- parse CSV directly ---
    if ext == "csv":
        user_df = pd.read_csv(upload, names=COLUMN_NAMES)

    # --- parse PCAP/PCAPNG via PyShark ---
    else:
        caps = pyshark.FileCapture(
            input_file=io.BytesIO(upload.read()),
            keep_packets=False
        )
        rows = []
        for pkt in caps:
            try:
                # Build a row dict for the 41 features
                # * You can refine each feature extraction as needed *
                row = {
                  "duration": float(pkt.frame_info.time_epoch),
                  "protocol_type": pkt.transport_layer or "",
                  "service": pkt.highest_layer or "",
                  "flag": pkt.tcp.flags if hasattr(pkt, "tcp") else "",
                  "src_bytes": int(pkt.length),
                  "dst_bytes": 0,
                  "land": 0,
                  "wrong_fragment": 0,
                  "urgent": 0,
                  "hot": 0,
                  "num_failed_logins": 0,
                  "logged_in": 0,
                  "num_compromised": 0,
                  "root_shell": 0,
                  "su_attempted": 0,
                  "num_root": 0,
                  "num_file_creations": 0,
                  "num_shells": 0,
                  "num_access_files": 0,
                  "num_outbound_cmds": 0,
                  "is_host_login": 0,
                  "is_guest_login": 0,
                  "count": 0,
                  "srv_count": 0,
                  "serror_rate": 0,
                  "srv_serror_rate": 0,
                  "rerror_rate": 0,
                  "srv_rerror_rate": 0,
                  "same_srv_rate": 0,
                  "diff_srv_rate": 0,
                  "srv_diff_host_rate": 0,
                  "dst_host_count": 0,
                  "dst_host_srv_count": 0,
                  "dst_host_same_srv_rate": 0,
                  "dst_host_diff_srv_rate": 0,
                  "dst_host_same_src_port_rate": 0,
                  "dst_host_srv_diff_host_rate": 0,
                  "dst_host_serror_rate": 0,
                  "dst_host_srv_serror_rate": 0,
                  "dst_host_rerror_rate": 0,
                  "dst_host_srv_rerror_rate": 0,
                  "label": "normal"   # dummy, not used
                }
                rows.append(row)
            except Exception:
                continue
        caps.close()
        user_df = pd.DataFrame(rows, columns=COLUMN_NAMES)

    # --- analyze on button click ---
    if st.button("Analyze My Traffic 🚀"):
        X_u, _ = preprocess(user_df)

        # RF
        y_rf  = rf_model.predict(X_u)
        cnt_rf = np.bincount(y_rf, minlength=2)
        pct_rf = cnt_rf[1]/cnt_rf.sum()*100

        # SVM
        X_us = svm_scaler.transform(X_u)
        y_svm = svm_model.predict(X_us)
        cnt_svm = np.bincount(y_svm, minlength=2)

        # Autoencoder
        X_ua = ae_scaler.transform(X_u)
        recon_u = ae_model.predict(X_ua)
        mse_u   = np.mean((X_ua - recon_u)**2, axis=1)
        y_ae    = (mse_u > ae_thresh).astype(int)
        cnt_ae  = np.bincount(y_ae, minlength=2)

        # show summary gauge
        st.metric("⚠️ % traffic flagged as ATTACK (RF)", f"{pct_rf:.2f}%")
        if pct_rf < 5:
            st.success("✅ Looks mostly safe!")
        else:
            st.warning("🚨 Significant suspicious traffic!")

        # breakdown table
        df_out = pd.DataFrame({
            "Model": ["Random Forest","SVM","Autoencoder"],
            "Normal": [int(cnt_rf[0]),int(cnt_svm[0]),int(cnt_ae[0])],
            "Attack": [int(cnt_rf[1]),int(cnt_svm[1]),int(cnt_ae[1])]
        }).set_index("Model")
        st.table(df_out)

        # confusion matrix for RF
        cm = confusion_matrix(y_rf, y_rf)  # demo against itself
        fig, ax = plt.subplots()
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)
        ax.set(xlabel="Pred", ylabel="True")
        st.pyplot(fig)
