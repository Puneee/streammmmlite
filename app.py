# app.py  –  Streamlit Network-Traffic Inspector
import warnings, streamlit as st, pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt, seaborn as sns

warnings.filterwarnings("ignore")          # keep Streamlit log tidy
sns.set_theme(style="ticks")

# ----------------------- DATA LOADER -----------------------
@st.cache_data
def load_data():
    cols = [
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
    train_df = pd.read_csv("NSL_KDD_Train.csv", names=cols)
    test_df  = pd.read_csv("NSL_KDD_Test.csv",  names=cols)
    return train_df, test_df

# -------------------- PRE-PROCESSING -----------------------
def preprocess_data(train_df, test_df):
    cat_cols = ['protocol_type', 'service', 'flag']

    for col in cat_cols:
        le = LabelEncoder()
        # cast first so strip()/encoder see strings
        train_df[col] = train_df[col].astype(str)
        test_df[col]  = test_df[col].astype(str)

        le.fit(pd.concat([train_df[col], test_df[col]]))
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])

    # binary label 0 = normal, 1 = attack  (strip handles "normal ", etc.)
    train_df['label'] = (train_df['label'].astype(str).str.strip()
                         .eq('normal').map({True: 0, False: 1}))
    test_df['label']  = (test_df['label'].astype(str).str.strip()
                         .eq('normal').map({True: 0, False: 1}))

    # guarantee numeric feature matrix
    X_train = train_df.drop('label', axis=1).apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
    X_test  = test_df.drop('label',  axis=1).apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()

    y_train = train_df['label'].astype('int32').to_numpy()
    y_test  = test_df['label'].astype('int32').to_numpy()

    return X_train, y_train, X_test, y_test

# ----------------- AUTO-ENCODER FACTORY --------------------
def build_autoencoder(input_dim):
    x_in = Input(shape=(input_dim,))
    enc  = Dense(32, activation='relu')(x_in)
    enc  = Dense(16, activation='relu')(enc)
    dec  = Dense(32, activation='relu')(enc)
    x_out= Dense(input_dim, activation='linear')(dec)
    model= Model(x_in, x_out)
    model.compile(optimizer='adam', loss='mse')
    return model

# --------------------- STREAMLIT UI ------------------------
st.title("🚦 Network Traffic Inspector – Streamlit Demo")

train_df, test_df = load_data()
X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)

if st.button("Run Models 🚀"):
    with st.spinner("Training…"):

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_rf = rf.predict(X_test)

        # SVM
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        svm = SVC(kernel='rbf', gamma='scale')
        svm.fit(X_tr_s, y_train)
        y_svm = svm.predict(X_te_s)

        # Auto-encoder anomaly detector
        X_train_norm = X_train[y_train == 0]
        sc_ae = StandardScaler()
        X_train_ae = sc_ae.fit_transform(X_train_norm)
        X_test_ae  = sc_ae.transform(X_test)
        ae = build_autoencoder(X_train_ae.shape[1])
        ae.fit(X_train_ae, X_train_ae, epochs=20,
               batch_size=256, shuffle=True, validation_split=0.1, verbose=0)
        recon = ae.predict(X_test_ae)
        mse   = np.mean(np.square(X_test_ae - recon), axis=1)
        thr   = np.percentile(mse, 95)
        y_ae  = (mse > thr).astype(int)

    # ----------------- DISPLAY RESULTS -----------------
    st.header("✅ Metrics")
    st.write(f"**Random Forest Accuracy:** {accuracy_score(y_test, y_rf):.4f}")
    st.write(f"**SVM Accuracy:** {accuracy_score(y_test, y_svm):.4f}")
    st.write(f"**Auto-encoder Accuracy:** {accuracy_score(y_test, y_ae):.4f}")

    st.subheader("Confusion Matrix – Random Forest")
    cm = confusion_matrix(y_test, y_rf)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='d',
                xticklabels=['Normal','Attack'],
                yticklabels=['Normal','Attack'])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.success("All models ran successfully 🎉")
