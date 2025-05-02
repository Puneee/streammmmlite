import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

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
    test = pd.read_csv("NSL_KDD_Test.csv", names=COLUMN_NAMES)
    df = pd.concat([train, test], ignore_index=True)
    tr, te = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
    return tr, te

def preprocess(df):
    df = df.copy()
    for col in ["protocol_type", "service", "flag"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(lambda s: 0 if s == "normal" else 1)
    X = df.drop("label", axis=1).apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["label"].astype("int32")
    return X.to_numpy(), y.to_numpy()

# Train models
def train_models():
    tr_df, te_df = load_data()
    X_tr, y_tr = preprocess(tr_df)
    X_te, y_te = preprocess(te_df)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)

    svm_scaler = StandardScaler()
    X_tr_s = svm_scaler.fit_transform(X_tr)
    X_te_s = svm_scaler.transform(X_te)
    svm = SVC(kernel="rbf", gamma="scale")
    svm.fit(X_tr_s, y_tr)

    ae_scaler = StandardScaler()
    X_ae_tr = ae_scaler.fit_transform(X_tr[y_tr == 0])
    X_ae_te = ae_scaler.transform(X_te)
    inp = Input(shape=(X_ae_tr.shape[1],))
    e = Dense(32, activation="relu")(inp)
    e = Dense(16, activation="relu")(e)
    d = Dense(32, activation="relu")(e)
    out = Dense(X_ae_tr.shape[1], activation="linear")(d)
    ae = Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(X_ae_tr, X_ae_tr, epochs=20, batch_size=256, shuffle=True, validation_split=0.1, verbose=0)
    recon = ae.predict(X_ae_te)
    thresh = np.percentile(np.mean((X_ae_te - recon) ** 2, axis=1), 95)

    return rf, svm, svm_scaler, ae, ae_scaler, thresh

rf_model, svm_model, svm_scaler, ae_model, ae_scaler, ae_thresh = train_models()

def analyze(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file.name, names=COLUMN_NAMES)
        X_u, _ = preprocess(df)

        y_rf = rf_model.predict(X_u)
        cnt_rf = np.bincount(y_rf, minlength=2)
        pct = cnt_rf[1] / cnt_rf.sum() * 100

        X_us = svm_scaler.transform(X_u)
        y_svm = svm_model.predict(X_us)
        cnt_svm = np.bincount(y_svm, minlength=2)

        X_ae = ae_scaler.transform(X_u)
        recon_u = ae_model.predict(X_ae)
        mse_u = np.mean((X_ae - recon_u)**2, axis=1)
        y_ae = (mse_u > ae_thresh).astype(int)
        cnt_ae = np.bincount(y_ae, minlength=2)

        risk_msg = (
            "✅ Mostly safe (<5%)" if pct < 5 else
            "⚠️ Moderate suspicion (5–20%). Check logs." if pct < 20 else
            "🚨 High attack rate (>20%)! Take immediate action!"
        )

        fig, ax = plt.subplots()
        cm = confusion_matrix(y_rf, y_rf)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set(xlabel='Predicted', ylabel='Actual')
        plt.tight_layout()

        breakdown = pd.DataFrame({
            'Model': ['RF', 'SVM', 'AE'],
            'Normal': [cnt_rf[0], cnt_svm[0], cnt_ae[0]],
            'Attack': [cnt_rf[1], cnt_svm[1], cnt_ae[1]]
        })

        return f"{pct:.2f}% attack traffic\n\n{risk_msg}", breakdown, fig

    except Exception as e:
        return str(e), None, None

iface = gr.Interface(
    fn=analyze,
    inputs=gr.File(file_types=[".csv"], label="Upload CSV File"),
    outputs=[
        gr.Text(label="Analysis Result"),
        gr.Dataframe(label="Model Breakdown"),
        gr.Plot(label="Confusion Matrix (RF)")
    ],
    title="🚦 Network Traffic Inspector (CSV Only)",
    description="Upload a CSV file of network traffic. The system will analyze it using ML models and report attack levels."
)

if __name__ == "__main__":
    iface.launch()
