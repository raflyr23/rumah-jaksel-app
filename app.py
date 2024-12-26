# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

# Set page config
st.set_page_config(
    page_title="Rumah Jaksel Analysis",
    page_icon="🏠",
    layout="wide"
)

# Function to load and prepare data
@st.cache_data
def load_data():
    data = pd.read_csv("data/DATA RUMAH JAKSEL.csv")
    return data.dropna()

# Function to create KNN regression model
def create_knn_regression_model(X_train, X_test, y_train, y_test, n_neighbors=5):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_train_scaled, X_test_scaled

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select a page",
    ["Home", "Data Analysis", "Price Prediction", "Model Performance"]
)

# Load data
try:
    df = load_data()
except FileNotFoundError:
    st.error("Please make sure the data file 'DATA RUMAH JAKSEL.csv' is in the same directory as this script.")
    st.stop()

# Home Page
if page == "Home":
    st.title("🏠 Analisis Harga Rumah Jakarta Selatan")
    st.write("""
    Selamat datang di aplikasi analisis harga rumah Jakarta Selatan. 
    Aplikasi ini menggunakan algoritma K-Nearest Neighbors untuk menganalisis 
    dan memprediksi harga rumah berdasarkan berbagai fitur.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Overview")
        st.write(f"Total data points: {len(df)}")
        st.write(f"Features available: {', '.join(df.columns.tolist())}")
        
    with col2:
        st.subheader("Quick Statistics")
        st.write("Ringkasan statistik harga rumah:")
        st.dataframe(df['HARGA'].describe())

# Data Analysis Page
elif page == "Data Analysis":
    st.title("📊 Analisis Data")
    
    # Data preview
    st.subheader("Preview Dataset")
    st.dataframe(df.head())
    
    # Distribution plots
    st.subheader("Distribusi Harga Rumah")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x='HARGA', bins=30)
    plt.title("Distribusi Harga Rumah")
    st.pyplot(fig)
    
    # Correlation analysis
    st.subheader("Correlation Matrix")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    st.pyplot(fig)

# Price Prediction Page
elif page == "Price Prediction":
    st.title("💰 Prediksi Harga Rumah")
    
    # Feature input
    st.subheader("Masukkan Karakteristik Rumah")
    
    col1, col2 = st.columns(2)
    
    with col1:
        lb = st.number_input("Luas Bangunan (m²)", min_value=0)
        lt = st.number_input("Luas Tanah (m²)", min_value=0)
        kt = st.number_input("Jumlah Kamar Tidur", min_value=0)
    
    with col2:
        km = st.number_input("Jumlah Kamar Mandi", min_value=0)
        grs = st.number_input("Jumlah Garasi", min_value=0)
    
    # Prepare model
    X = df[["LB", "LT", "KT", "KM", "GRS"]]
    y = df["HARGA"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model, scaler, _, _ = create_knn_regression_model(X_train, X_test, y_train, y_test)
    
    if st.button("Prediksi Harga"):
        # Prepare input data
        input_data = np.array([[lb, lt, kt, km, grs]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        st.success(f"Prediksi Harga Rumah: Rp {prediction:,.2f}")

# Model Performance Page
elif page == "Model Performance":
    st.title("📈 Performa Model")
    
    # Prepare data and model
    X = df[["LB", "LT", "KT", "KM", "GRS"]]
    y = df["HARGA"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training and evaluation
    model, scaler, X_train_scaled, X_test_scaled = create_knn_regression_model(X_train, X_test, y_train, y_test)
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Mean Squared Error", f"{mse:,.2f}")
    
    with col2:
        st.metric("R² Score", f"{r2:.4f}")
    
    # Actual vs Predicted Plot
    st.subheader("Actual vs Predicted Values")
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted House Prices")
    st.pyplot(fig)