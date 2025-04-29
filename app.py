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

# ——— LOAD & COMBINE CSVs ———
@st.cache_data
def load_and_split():
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
    df1 = pd.read_csv("NSL_KDD_Train.csv", names=cols)
    df2 = pd.read_csv("NSL_KDD_Test.csv",  names=cols)
    df  = pd.concat([df1, df2], ignore_index=True)
    # simple random split
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        shuffle=True,
        random_state=42
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# ——— PREPROCESSOR ———
def preprocess_data(train_df, test_df):
    cat_cols = ['protocol_type','service','flag']

    # encode categoricals
    for c in cat_cols:
        le = LabelEncoder()
        train_df[c] = train_df[c].astype(str)
        test_df[c]  = test_df[c].astype(str)
        le.fit(pd.concat([train_df[c], test_df[c]]))
        train_df[c] = le.transform(train_df[c])
        test_df[c]  = le.transform(test_df[c])

    # map labels: exact "normal" → 0, else 1
    def map_label(x):
        return 0 if str(x).strip().lower() == "normal" else 1

    train_df['label'] = train_df['label'].apply(map_label)
    test_df ['label'] = test_df ['label'].apply(map_label)

    # force numeric features + fill NaN → 0
    X_tr = (train_df.drop('label', axis=1)
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0)
            .to_numpy())
    X_te = (test_df .drop('label', axis=1)
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0)
            .to_numpy())

    y_tr = train_df['label'].astype('int32').to_numpy()
    y_te = test_df ['label'].astype('int32').to_numpy()

    return X_tr, y_tr, X_te, y_te

# ——— AUTOENCODER BUILDER ———
def build_autoencoder(dim):
    inp = Input(shape=(dim,))
    e   = Dense(32, activation='relu')(inp)
    e   = Dense(16, activation='relu')(e)
    d   = Dense(32, activation='relu')(e)
    out = Dense(dim, activation='linear')(d)
    m   = Model(inp, out)
    m.compile(optimizer='adam', loss='mse')
    return m

# ——— STREAMLIT APP ———
st.title("🚦 Network Traffic Inspector")

train_df, test_df = load_and_split()
X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)

if st.button("Run Models 🚀"):
    with st.spinner("Training…"):

        # Random Forest
        rf   = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_rf = rf.predict(X_test)

        # show train class counts
        counts = np.bincount(y_train)
        st.write("🔢 y_train class counts:", {0:int(counts[0]), 1:int(counts[1])})

        # SVM
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        svm = SVC(kernel='rbf', gamma='scale')
        svm.fit(X_tr_s, y_train)
        y_svm = svm.predict(X_te_s)

        # Autoencoder (normals only)
        X_norm = X_train[y_train==0]
        sc_ae  = StandardScaler()
        X_ae_tr= sc_ae.fit_transform(X_norm)
        X_ae_te= sc_ae.transform(X_test)
        ae     = build_autoencoder(X_ae_tr.shape[1])
        ae.fit(X_ae_tr, X_ae_tr,
               epochs=20, batch_size=256,
               shuffle=True, validation_split=0.1, verbose=0)
        recon  = ae.predict(X_ae_te)
        mse    = np.mean((X_ae_te - recon)**2, axis=1)
        thr    = np.percentile(mse, 95)
        y_ae   = (mse>thr).astype(int)

    # Display results
    st.header("✅ Results")
    st.write(f"Random Forest Acc:   {accuracy_score(y_test,y_rf):.4f}")
    st.write(f"SVM Acc:             {accuracy_score(y_test,y_svm):.4f}")
    st.write(f"Autoencoder Acc:     {accuracy_score(y_test,y_ae):.4f}")

    cm = confusion_matrix(y_test, y_rf)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=["Normal","Attack"],
                yticklabels=["Normal","Attack"])
    ax.set(xlabel="Predicted", ylabel="Actual")
    st.pyplot(fig)

    st.success("Done! 🎉")
