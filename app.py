import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
import io

from sklearn.ensemble      import RandomForestClassifier
from sklearn.svm           import SVC
from sklearn.decomposition import PCA
from sklearn.metrics       import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler

from tensorflow.keras.models  import Model
from tensorflow.keras.layers  import Input, Dense

sns.set_theme(style="ticks")

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

def load_data():
    train = pd.read_csv("NSL_KDD_Train.csv", names=COLUMN_NAMES)
    test  = pd.read_csv("NSL_KDD_Test.csv",  names=COLUMN_NAMES)
    df = pd.concat([train, test], ignore_index=True)
    return train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

def preprocess(df: pd.DataFrame):
    df = df.copy()
    for col in ["protocol_type","service","flag"]:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df["label"] = (
        df["label"].astype(str).str.strip().str.lower()
           .map(lambda s: 0 if s=="normal" else 1)
    )
    X = df.drop("label", axis=1).apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["label"].astype("int32")
    return X.values, y.values

def train_models():
    # 1) load & split
    tr_df, te_df = load_data()
    X_tr, y_tr = preprocess(tr_df)
    X_te, y_te = preprocess(te_df)

    # 2) Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)

    # 3) SVM w/ scaling
    svm_scaler = StandardScaler().fit(X_tr)
    svm = SVC(kernel="rbf", gamma="scale")
    svm.fit(svm_scaler.transform(X_tr), y_tr)

    # 4) Autoencoder on normals
    ae_scaler = StandardScaler().fit(X_tr[y_tr==0])
    X_ae_tr = ae_scaler.transform(X_tr[y_tr==0])
    X_ae_te = ae_scaler.transform(X_te)
    inp = Input(shape=(X_ae_tr.shape[1],))
    e = Dense(32, activation="relu")(inp)
    e = Dense(16, activation="relu")(e)
    d = Dense(32, activation="relu")(e)
    out = Dense(X_ae_tr.shape[1], activation="linear")(d)
    ae = Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(X_ae_tr, X_ae_tr, epochs=20, batch_size=256,
           shuffle=True, validation_split=0.1, verbose=0)
    recon_te = ae.predict(X_ae_te)
    ae_thresh = np.percentile(((X_ae_te - recon_te)**2).mean(axis=1), 95)

    # 5) PCA baseline (2D)
    pca = PCA(n_components=2).fit(X_tr)
    Xp_tr = pca.transform(X_tr)
    centroid = Xp_tr[y_tr==0].mean(axis=0)
    dists_tr = np.linalg.norm(Xp_tr - centroid, axis=1)
    pca_thresh = np.percentile(dists_tr[y_tr==0], 95)

    # 6) Evaluate all on the test set
    # RF
    y_rf_te  = rf.predict(X_te)
    # SVM
    y_svm_te = svm.predict(svm_scaler.transform(X_te))
    # AE
    y_ae_te  = ( ((ae_scaler.transform(X_te) - ae.predict(ae_scaler.transform(X_te)))**2).mean(axis=1) 
                 > ae_thresh ).astype(int)
    # PCA
    dists_te = np.linalg.norm(pca.transform(X_te) - centroid, axis=1)
    y_pca_te = (dists_te > pca_thresh).astype(int)

    # 7) Build performance table
    rows = []
    for name, y_pred in [
        ("RF",  y_rf_te),
        ("SVM", y_svm_te),
        ("AE",  y_ae_te),
        ("PCA", y_pca_te)
    ]:
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, y_pred),
            "Precision": precision_score(y_te, y_pred),
            "Recall":    recall_score(y_te, y_pred)
        })
    perf_df = pd.DataFrame(rows).set_index("Model")

    return rf, svm, svm_scaler, ae, ae_scaler, ae_thresh, pca, centroid, pca_thresh, perf_df

# train once
(rf_model, svm_model, svm_scaler,
 ae_model, ae_scaler, ae_thresh,
 pca_model, pca_centroid, pca_thresh,
 perf_df) = train_models()

def analyze(uploaded_file):
    # read raw upload, always headerless parsing into 42-cols
    raw = open(uploaded_file.name,"r").read()
    df  = pd.read_csv(io.StringIO(raw), header=None, names=COLUMN_NAMES)
    X_u, _ = preprocess(df)

    # RF
    y_rf    = rf_model.predict(X_u)
    cnt_rf  = np.bincount(y_rf, minlength=2)
    pct     = cnt_rf[1]/cnt_rf.sum()*100

    # SVM
    y_svm   = svm_model.predict(svm_scaler.transform(X_u))
    cnt_svm = np.bincount(y_svm, minlength=2)

    # AE
    recon_u = ae_model.predict(ae_scaler.transform(X_u))
    y_ae    = ( ((ae_scaler.transform(X_u)-recon_u)**2).mean(axis=1) > ae_thresh ).astype(int)
    cnt_ae  = np.bincount(y_ae, minlength=2)

    # PCA
    dists   = np.linalg.norm(pca_model.transform(X_u) - pca_centroid, axis=1)
    y_pca   = (dists > pca_thresh).astype(int)
    cnt_pca = np.bincount(y_pca, minlength=2)

    # risk message
    if pct < 5:
        risk = f"✅ Mostly safe ({pct:.2f}% attack)"
    elif pct < 20:
        risk = f"⚠️ Moderate suspicion ({pct:.2f}% attack)"
    else:
        risk = f"🚨 High attack rate ({pct:.2f}% attack!)"

    # breakdown table (including PCA)
    breakdown = pd.DataFrame({
        "Model":  ["RF","SVM","AE","PCA"],
        "Normal": [cnt_rf[0], cnt_svm[0], cnt_ae[0], cnt_pca[0]],
        "Attack": [cnt_rf[1], cnt_svm[1], cnt_ae[1], cnt_pca[1]],
    }).set_index("Model")

    # confusion matrix for RF
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_rf, y_rf)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual")
    plt.tight_layout()

    # static deployment discussion
    discussion = """
**Deployment Challenges & False-Positive Handling**
- **tshark & System Packages**  
  Hosted platforms often disallow low-level packet tools—consider Docker or self-hosted VMs for full PCAP support.
- **Threshold Tuning**  
  AE/PCA thresholds set at the 95th percentile may need adjustment per network baseline.  
  Integrate a feedback loop to re-calibrate thresholds over time.
- **False Positives**  
  Even a 5–10% false-positive rate can overwhelm ops teams.  
  *Mitigation:*  
    1. Correlate alerts with other signals (firewall logs, IDS signatures).  
    2. Implement a whitelist of known benign hosts/services.  
    3. Add a human-in-the-loop review step before automated quarantines.
    """

    return risk, breakdown, fig, perf_df, discussion

iface = gr.Interface(
    fn=analyze,
    inputs=gr.File(file_types=[".csv"], label="Upload CSV"),
    outputs=[
        gr.Text(label="🚦 Attack Percentage"),
        gr.Dataframe(label="Sample Breakdown (incl. PCA)"),
        gr.Plot(label="RF Confusion Matrix"),
        gr.Dataframe(label="Test Set Performance"),
        gr.Markdown(label="Deployment & FP Handling")
    ],
    title="Network Traffic Inspector + PCA Baseline & Metrics",
    description="Upload your NSL-KDD CSV (42-column format). Shows RF/SVM/AE/PCA results plus test performance."
)

if __name__ == "__main__":
    iface.launch()
