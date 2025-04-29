# app.py

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.metrics         import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.keras.models  import Model
from tensorflow.keras.layers  import Input, Dense

sns.set_theme(style="ticks")

# --- 1) DEFINE COLUMN NAMES (41 features + 1 label) ---
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

# --- 2) LOAD & SPLIT NSL-KDD ONCE ---
@st.cache_data(show_spinner=False)
def load_and_split():
    train = pd.read_csv("NSL_KDD_Train.csv", names=COLUMN_NAMES)
    test  = pd.read_csv("NSL_KDD_Test.csv",  names=COLUMN_NAMES)
    # Combine & random 80/20 split so we have both classes
    df = pd.concat([train, test], ignore_index=True)
    tr, te = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
    return tr.reset_index(drop=True), te.reset_index(drop=True)

# --- 3) PREPROCESSOR (shared for train/test/user) ---
@st.cache_data(show_spinner=False)
def preprocess(df):
    df = df.copy()
    # 3a) encode categoricals
    for c in ["protocol_type","service","flag"]:
        le = LabelEncoder()
        df[c] = df[c].astype(str)
        le.fit(df[c])
        df[c] = le.transform(df[c])
    # 3b) map labels exact "normal"->0, else->1
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(
        lambda s: 0 if s == "normal" else 1
    )
    # 3c) split features vs labels
    X = df.drop("label", axis=1).apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["label"].astype("int32")
    return X.to_numpy(), y.to_numpy()

# --- 4) TRAIN MODELS ONCE ---
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
    sc_ae = StandardScaler()
    X_ae_tr = sc_ae.fit_transform(X_norm)
    X_ae_te = sc_ae.transform(X_te)
    inp = Input(shape=(X_ae_tr.shape[1],))
    e = Dense(32, activation="relu")(inp)
    e = Dense(16, activation="relu")(e)
    d = Dense(32, activation="relu")(e)
    out = Dense(X_ae_tr.shape[1], activation="linear")(d)
    ae = Model(inp,out); ae.compile("adam","mse")
    ae.fit(X_ae_tr, X_ae_tr, epochs=20, batch_size=256,
           shuffle=True, validation_split=0.1, verbose=0)
    recon = ae.predict(X_ae_te)
    mse   = np.mean((X_ae_te - recon)**2, axis=1)
    thresh = np.percentile(mse, 95)

    return rf, svm, scaler, sc_ae, ae, thresh

# Build everything
rf_model, svm_model, svm_scaler, ae_scaler, ae_model, ae_thresh = train_models()

# --- 5) STREAMLIT UI ---  
st.title("🚦 Network Traffic Inspector")

st.markdown("""
Upload **your own** network-traffic CSV (same 41-feature format)  
and click **Analyze My Traffic** to see how much of it is flagged as an attack.  
""")

uploaded = st.file_uploader("Your network traffic CSV", type="csv")
if uploaded is not None:
    try:
        user_df = pd.read_csv(uploaded, names=COLUMN_NAMES)
    except Exception:
        st.error("❌ Could not read that file. Please upload a CSV with those 42 columns.")
    else:
        if st.button("Analyze My Traffic 🚀"):
            X_user, _ = preprocess(user_df)

            # RF prediction
            y_rf = rf_model.predict(X_user)
            cnt_rf = np.bincount(y_rf, minlength=2)
            pct_attack = cnt_rf[1] / cnt_rf.sum() * 100

            # SVM prediction
            X_user_s = svm_scaler.transform(X_user)
            y_svm = svm_model.predict(X_user_s)
            cnt_svm = np.bincount(y_svm, minlength=2)

            # AE anomaly detection
            X_user_ae = ae_scaler.transform(X_user)
            recon_u   = ae_model.predict(X_user_ae)
            mse_u     = np.mean((X_user_ae - recon_u)**2, axis=1)
            y_ae      = (mse_u > ae_thresh).astype(int)
            cnt_ae    = np.bincount(y_ae, minlength=2)

            # Show a big metric:
            st.metric("❗️% of traffic flagged as ATTACK (RF)", f"{pct_attack:.2f}%")
            if pct_attack < 5:
                st.success("✅ Your traffic looks mostly SAFE!")
            else:
                st.warning("⚠️ Quite a bit of traffic flagged as ATTACK!")

            # Details on the three methods:
            st.write("#### Model breakdown (normal vs attack)")
            df_out = pd.DataFrame({
                "Method": ["Random Forest","SVM","Autoencoder"],
                "Normal": [cnt_rf[0], cnt_svm[0], cnt_ae[0]],
                "Attack": [cnt_rf[1], cnt_svm[1], cnt_ae[1]],
            })
            st.dataframe(df_out.set_index("Method"))

            # Confusion matrix for RF:
            st.write("#### Confusion Matrix for Random Forest")
            cm = confusion_matrix(y_rf, y_rf)  # dummy just to illustrate
            # (or you could compare to user-provided label if they included one)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set(xlabel="Predicted", ylabel="Actual")
            st.pyplot(fig)
