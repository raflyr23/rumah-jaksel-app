import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

# Custom styling
st.set_page_config(
    page_title="Analisis Rumah Jakarta Selatan",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main > div {
            padding: 2rem 3rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #FF4B4B;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
        }
        .stButton>button:hover {
            background-color: #FF3333;
        }
        .stat-box {
           
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        h1, h2, h3 {
            color: #FF4B4B;
        }
        .stAlert {
            background-color: rgba(255, 75, 75, 0.1);
            border-left-color: #FF4B4B;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
    </style>
""", unsafe_allow_html=True)

# Function to load and prepare data
@st.cache_data
def load_data():
    data = pd.read_csv("data/DATA RUMAH JAKSEL.csv")
    data['KATEGORI'] = pd.cut(
        data['HARGA'],
        bins=[0, 1e9, 5e9, float('inf')],
        labels=['Murah', 'Sedang', 'Mahal']
    )
    return data.dropna()

# Function to format currency
def format_currency(value):
    return f"Rp {value:,.0f}"

# Function to get price category with emoji
def get_price_category(price):
    if price <= 1e9:
        return '💚 Murah'
    elif price <= 5e9:
        return '💛 Sedang'
    else:
        return '❤️ Mahal'

# Function to create KNN regression model
def create_knn_regression_model(X_train, X_test, y_train, y_test, n_neighbors=5):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_train_scaled, X_test_scaled

# Sidebar styling and navigation
with st.sidebar:
    st.title("🏠 Navigation")
    st.markdown("---")
    page = st.radio(
        "Select Page",
        ["Home", "Data Analysis", "Price Prediction", "Model Performance"],
        format_func=lambda x: f"📍 {x}"
    )
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h4>About</h4>
            <p>Aplikasi analisis harga rumah Jakarta Selatan menggunakan algoritma KNN</p>
        </div>
    """, unsafe_allow_html=True)

# Load data
try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ Error: File 'DATA RUMAH JAKSEL.csv' tidak ditemukan!")
    st.stop()

# Home Page
if page == "Home":
    st.title("🏠 Analisis Harga Rumah Jakarta Selatan")
    st.markdown("---")
    
    # Welcome message with card-like styling
    st.markdown("""
        <div style= padding: 2rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
            <h3 style='margin-top: 0;'>👋 Selamat Datang!</h3>
            <p>Aplikasi ini membantu Anda menganalisis dan memprediksi harga rumah di Jakarta Selatan 
            menggunakan algoritma K-Nearest Neighbors (KNN).</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='stat-box'>
                <h4>📊 Total Data</h4>
                <h2>{:,}</h2>
                <p>properti</p>
            </div>
        """.format(len(df)), unsafe_allow_html=True)
        
    with col2:
        avg_price = df['HARGA'].mean()
        st.markdown("""
            <div class='stat-box'>
                <h4>💰 Rata-rata Harga</h4>
                <h2>{}</h2>
                <p>rupiah</p>
            </div>
        """.format(format_currency(avg_price)), unsafe_allow_html=True)
        
    with col3:
        most_common_category = df['KATEGORI'].mode()[0]
        st.markdown("""
            <div class='stat-box'>
                <h4>🏷️ Kategori Terbanyak</h4>
                <h2>{}</h2>
                <p>properti</p>
            </div>
        """.format(most_common_category), unsafe_allow_html=True)

# Data Analysis Page
elif page == "Data Analysis":
    st.title("📊 Analisis Data")
    st.markdown("---")
    
    # Interactive data explorer
    st.subheader("🔍 Eksplorasi Data")
    
    # Add tabs for different views
    tab1, tab2, tab3 = st.tabs(["Preview Data", "Statistik", "Visualisasi"])
    
    with tab1:
        st.dataframe(
            df[['HARGA', 'LB', 'LT', 'KT', 'KM', 'GRS', 'KATEGORI']].style.format({
                'HARGA': format_currency
            }),
            use_container_width=True
        )
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📈 Statistik Harga")
            stats_df = df['HARGA'].describe()
            stats_df = pd.DataFrame(stats_df).style.format(format_currency)
            st.dataframe(stats_df, use_container_width=True)
            
        with col2:
            st.markdown("##### 📊 Distribusi Kategori")
            category_counts = df['KATEGORI'].value_counts()
            st.bar_chart(category_counts)
    
    with tab3:
        # Enhanced visualizations
        st.markdown("##### 📈 Visualisasi Data")
        
        # Correlation heatmap with improved styling
        fig, ax = plt.subplots(figsize=(10, 8))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='RdYlBu',
            center=0,
            fmt='.2f',
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .5}
        )
        
        plt.title("Correlation Matrix", pad=20)
        st.pyplot(fig)

# Price Prediction Page
elif page == "Price Prediction":
    st.title("💰 Prediksi Harga Rumah")
    st.markdown("---")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style= padding: 1rem; border-radius: 0.5rem;'>
                <h4>📐 Dimensi Properti</h4>
            </div>
        """, unsafe_allow_html=True)
        
        lb = st.number_input("Luas Bangunan (m²)", min_value=0, value=100)
        lt = st.number_input("Luas Tanah (m²)", min_value=0, value=120)
    
    with col2:
        st.markdown("""
            <div style= padding: 1rem; border-radius: 0.5rem;'>
                <h4>🏠 Fasilitas</h4>
            </div>
        """, unsafe_allow_html=True)
        
        kt = st.number_input("Jumlah Kamar Tidur", min_value=0, value=3)
        km = st.number_input("Jumlah Kamar Mandi", min_value=0, value=2)
        grs = st.number_input("Jumlah Garasi", min_value=0, value=1)
    
    # Prepare model
    X = df[["LB", "LT", "KT", "KM", "GRS"]]
    y = df["HARGA"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model, scaler, _, _ = create_knn_regression_model(X_train, X_test, y_train, y_test)
    
    st.markdown("---")
    
    if st.button("🎯 Prediksi Harga"):
        # Show loading spinner
        with st.spinner('Calculating prediction...'):
            input_data = np.array([[lb, lt, kt, km, grs]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            category = get_price_category(prediction)
            
            # Display results in an attractive format
            st.markdown("""
                <div style= padding: 2rem; border-radius: 0.5rem; text-align: center;'>
                    <h3 style='color: #FF4B4B;'>Hasil Prediksi</h3>
                    <h2>{}</h2>
                    <h4>{}</h4>
                </div>
            """.format(format_currency(prediction), category), unsafe_allow_html=True)

# Model Performance Page
elif page == "Model Performance":
    st.title("📈 Performa Model")
    st.markdown("---")
    
    # Prepare data and model
    X = df[["LB", "LT", "KT", "KM", "GRS"]]
    y = df["HARGA"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model, scaler, X_train_scaled, X_test_scaled = create_knn_regression_model(X_train, X_test, y_train, y_test)
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Display metrics in an attractive grid
    col1, col2, col3 = st.columns(3)
    
    metrics = [
        ("Mean Squared Error", mse, "📉"),
        ("R² Score", r2, "📊"),
        ("Root Mean Squared Error", rmse, "📈")
    ]
    
   for col, (metric_name, value, emoji) in zip([col1, col2, col3], metrics):
        with col:
            st.markdown(f"""
            <div style='background-color: #21212b; padding: 1rem; border-radius: 0.5rem; text-align: center;'>
                <h4>{emoji} {metric_name}</h4>
                <h2>{value}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced visualization of actual vs predicted values
    st.subheader("📊 Actual vs Predicted Values")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, c='#FF4B4B')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='gray')
    
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted House Prices")
    
    # Add grid and style
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    st.pyplot(fig)
