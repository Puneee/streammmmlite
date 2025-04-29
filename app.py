# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
@st.cache_data
def load_data():
    train_data_path = "NSL_KDD_Train.csv"
    test_data_path = "NSL_KDD_Test.csv"
    
    column_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
        "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
        "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
        "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label"
    ]
    
    train_df = pd.read_csv(train_data_path, names=column_names)
    test_df = pd.read_csv(test_data_path, names=column_names)
    
    return train_df, test_df

# Preprocess dataset
def preprocess_data(train_df, test_df):
    categorical_cols = ['protocol_type', 'service', 'flag']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        
        train_df[col] = le.transform(train_df[col]).astype(str)
        test_df[col] = le.transform(test_df[col]).astype(str)
        
        combined_data = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
        le.fit(combined_data)

        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        
        
        encoders[col] = le
    train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    X_train = train_df.drop('label', axis=1)
    y_train = train_df['label']
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']
    
    return X_train, y_train, X_test, y_test

# Autoencoder model
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation='relu')(input_layer)
    encoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# App title
st.title("🚦 Network Traffic Inspector using ML (Streamlit Version)")

# Load and preprocess
train_df, test_df = load_data()
X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df)

if st.button("Run Models 🚀"):
    with st.spinner('Training models...'):
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        
        # SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm_model.fit(X_train_scaled, y_train)
        y_pred_svm = svm_model.predict(X_test_scaled)
        
        # Autoencoder
        X_train_ae = X_train[y_train == 0]  # Train only on normal traffic
        scaler_ae = StandardScaler()
        X_train_ae_scaled = scaler_ae.fit_transform(X_train_ae)
        X_test_ae_scaled = scaler_ae.transform(X_test)
        
        autoencoder = build_autoencoder(X_train_ae_scaled.shape[1])
        autoencoder.fit(X_train_ae_scaled, X_train_ae_scaled, epochs=20, batch_size=256, shuffle=True, validation_split=0.1, verbose=0)
        
        reconstructions = autoencoder.predict(X_test_ae_scaled)
        mse = np.mean(np.power(X_test_ae_scaled - reconstructions, 2), axis=1)
        threshold = np.percentile(mse, 95)
        y_pred_ae = (mse > threshold).astype(int)
    
    # Display Results
    st.subheader("✅ Results")
    
    st.write("### Random Forest Accuracy:", round(accuracy_score(y_test, y_pred_rf), 4))
    st.write("### SVM Accuracy:", round(accuracy_score(y_test, y_pred_svm), 4))
    st.write("### Autoencoder Accuracy:", round(accuracy_score(y_test, y_pred_ae), 4))

    st.subheader("📊 Confusion Matrix - Random Forest")
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    fig, ax = plt.subplots()
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap="Blues")
    st.pyplot(fig)

    st.success('All models ran successfully! 🚀')

