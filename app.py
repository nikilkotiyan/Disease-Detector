import os
import pickle
import json
import streamlit as st
from streamlit_option_menu import option_menu
import shap
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# -------------------------
# AUTHENTICATION FUNCTIONS
# -------------------------
def load_users():
    try:
        with open('users.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users):
    with open('users.json', 'w') as f:
        json.dump(users, f)

def validate_login(username, password):
    users = load_users()
    return users.get(username) == password

def register_user(username, password):
    users = load_users()
    if username in users:
        return False  # User already exists
    users[username] = password
    save_users(users)
    return True

# -------------------------
# SESSION STATE INITIALIZATION
# -------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'username' not in st.session_state:
    st.session_state.username = ''

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Health AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------
# CUSTOM CSS FOR ADVANCED UI
# -------------------------
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0 0 20px 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #fff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Card Styles */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.12);
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Input Styles */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 0.75rem !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Button Styles */
    .stButton > button {
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        border: none !important;
    }
    
    .primary-button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    .secondary-button {
        background: linear-gradient(90deg, #4CAF50 0%, #2E7D32 100%) !important;
        color: white !important;
    }
    
    .danger-button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
        color: white !important;
    }
    
    /* Sidebar Styles */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #1a1a2e 100%) !important;
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        padding: 2rem 1rem;
    }
    
    /* Risk Meter */
    .risk-meter {
        width: 100%;
        height: 30px;
        background: linear-gradient(90deg, #4CAF50, #FFC107, #FF5722);
        border-radius: 15px;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .risk-indicator {
        position: absolute;
        height: 100%;
        width: 4px;
        background: #2c3e50;
        transform: translateX(-50%);
    }
    
    /* Prediction Result */
    .prediction-result {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .positive-result {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
        color: white;
    }
    
    .negative-result {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
    }
    
    /* Auth Pages */
    .auth-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 80vh;
        padding: 2rem;
    }
    
    .auth-card {
        background: white;
        padding: 3rem;
        border-radius: 25px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        width: 100%;
        max-width: 500px;
    }
    
    .auth-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 600;
    }
    
    /* SHAP Styles */
    .shap-container {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .shap-feature {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        border-radius: 12px;
        border-left: 5px solid;
        transition: transform 0.2s ease;
    }
    
    .shap-feature:hover {
        transform: translateX(10px);
    }
    
    .shap-positive {
        border-left-color: #FF416C;
    }
    
    .shap-negative {
        border-left-color: #4CAF50;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .card {
            padding: 1.5rem;
        }
        .header-title {
            font-size: 2rem;
        }
    }
    
    /* Loading Animation */
    .loader {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-top: 5px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def create_risk_meter(risk_percentage):
    """Create a visual risk meter"""
    colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#FF5722']
    thresholds = [20, 40, 60, 80, 100]
    
    fig = go.Figure()
    
    # Add gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=risk_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Level", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': colors[0]},
                {'range': [20, 40], 'color': colors[1]},
                {'range': [40, 60], 'color': colors[2]},
                {'range': [60, 80], 'color': colors[3]},
                {'range': [80, 100], 'color': colors[4]},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_percentage
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def display_prediction_result(result, prob, disease):
    """Display prediction result with advanced styling"""
    if result == 1:
        st.markdown(f"""
        <div class="prediction-result positive-result">
            <h2 style="margin: 0; font-size: 2.5rem;">⚠️ HIGH RISK DETECTED</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0;">
                The person is likely to have <strong>{disease}</strong>
            </p>
            <div style="font-size: 3rem; font-weight: bold; margin: 1rem 0;">
                {prob:.1f}% Risk
            </div>
            <p style="font-size: 1rem; opacity: 0.9;">
                Please consult with a healthcare professional
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-result negative-result">
            <h2 style="margin: 0; font-size: 2.5rem;">✅ LOW RISK</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0;">
                The person is unlikely to have <strong>{disease}</strong>
            </p>
            <div style="font-size: 3rem; font-weight: bold; margin: 1rem 0;">
                {prob:.1f}% Risk
            </div>
            <p style="font-size: 1rem; opacity: 0.9;">
                Continue maintaining a healthy lifestyle
            </p>
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# PAGE LOGIC - AUTHENTICATION
# -------------------------
if not st.session_state.logged_in:
    if st.session_state.page == 'login':
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="auth-container">
                <div class="auth-card">
                    <div class="auth-title">
                        🏥 Health AI Assistant<br>
                        <span style="font-size: 1rem; color: #666;">Secure Login</span>
                    </div>
            """, unsafe_allow_html=True)
            
            with st.container():
                username = st.text_input("👤 Username", key="login_username")
                password = st.text_input("🔒 Password", type="password", key="login_password")
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("🔓 Login", key="login_btn", use_container_width=True):
                        if validate_login(username, password):
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.success("Login successful! Redirecting...")
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password")
                
                with col_btn2:
                    if st.button("📝 Register", key="goto_register", use_container_width=True):
                        st.session_state.page = 'register'
                        st.rerun()
            
            st.markdown("""
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    elif st.session_state.page == 'register':
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="auth-container">
                <div class="auth-card">
                    <div class="auth-title">
                        🚀 Create Account<br>
                        <span style="font-size: 1rem; color: #666;">Join Health AI Assistant</span>
                    </div>
            """, unsafe_allow_html=True)
            
            with st.container():
                username = st.text_input("👤 Choose Username", key="register_username")
                password = st.text_input("🔒 Create Password", type="password", key="register_password")
                confirm_password = st.text_input("🔐 Confirm Password", type="password", key="register_confirm_password")
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("✅ Register", key="register_btn", use_container_width=True):
                        if not username or not password:
                            st.error("Please fill all fields")
                        elif password != confirm_password:
                            st.error("Passwords do not match")
                        elif register_user(username, password):
                            st.success("🎉 Registration successful! Please login.")
                            st.session_state.page = 'login'
                            st.rerun()
                        else:
                            st.error("Username already exists")
                
                with col_btn2:
                    if st.button("🔙 Back to Login", key="goto_login", use_container_width=True):
                        st.session_state.page = 'login'
                        st.rerun()
            
            st.markdown("""
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.stop()

# -------------------------
# MAIN APPLICATION
# -------------------------
# Load models and data
working_dir = os.path.dirname(os.path.abspath(__file__))

# Try loading models with error handling
@st.cache_resource
def load_models():
    try:
        diabetes_model = pickle.load(open(f'{working_dir}/diabetes_model_advanced.sav', 'rb'))
        heart_disease_model = pickle.load(open(f'{working_dir}/heart_disease_model_advanced.sav', 'rb'))
        parkinsons_model = pickle.load(open(f'{working_dir}/parkinsons_model_advanced.sav', 'rb'))
        return diabetes_model, heart_disease_model, parkinsons_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

@st.cache_data
def load_datasets():
    try:
        diabetes_data = pd.read_csv(f'{working_dir}/diabetes.csv')
        heart_data = pd.read_csv(f'{working_dir}/heart.csv')
        parkinsons_data = pd.read_csv(f'{working_dir}/parkinsons.csv')
        
        # Feature engineering
        X_diabetes = diabetes_data.drop(columns='Outcome', axis=1)
        X_diabetes['BMI_Age'] = X_diabetes['BMI'] * X_diabetes['Age']
        X_diabetes['Glucose_Insulin'] = X_diabetes['Glucose'] * X_diabetes['Insulin']
        
        X_heart = heart_data.drop(columns='target', axis=1)
        X_heart['Age_Chol'] = X_heart['age'] * X_heart['chol']
        X_heart['Thalach_Oldpeak'] = X_heart['thalach'] * X_heart['oldpeak']
        
        X_parkinsons = parkinsons_data.drop(columns=['name', 'status'], axis=1)
        X_parkinsons['jitter_ratio'] = X_parkinsons['MDVP:Jitter(%)'] / (X_parkinsons['MDVP:Jitter(Abs)'] + 1e-6)
        X_parkinsons['vocal_instability'] = X_parkinsons['Jitter:DDP'] + X_parkinsons['Shimmer:DDA']
        X_parkinsons['log_PPE'] = np.log(X_parkinsons['PPE'] + 1e-6)
        
        return X_diabetes, X_heart, X_parkinsons
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return None, None, None

# Load models and datasets
diabetes_model, heart_disease_model, parkinsons_model = load_models()
X_diabetes, X_heart, X_parkinsons = load_datasets()

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0;">
        <h2 style="color: white; margin-bottom: 0.5rem;">👋 Welcome,</h2>
        <h3 style="color: #667eea; margin-top: 0;">{st.session_state.username}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Diabetes", "Heart Disease", "Parkinson's", "History", "Dashboard", "Disclaimer"],
        icons=["activity", "heart-pulse", "brain", "clock-history", "speedometer2", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#667eea", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "8px 0",
                "border-radius": "8px",
                "padding": "12px 20px",
                "color": "white"
            },
            "nav-link-selected": {
                "background-color": "#667eea",
                "color": "white"
            },
        }
    )
    
    st.markdown("---")
    
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.page = 'login'
        st.rerun()
    
    st.markdown("""
    <div style="margin-top: 3rem; color: #888; font-size: 0.8rem; text-align: center;">
        <p>Health AI Assistant v1.0</p>
        <p>© 2024 AI Healthcare Solutions</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.markdown(f"""
<div class="main-header">
    <h1 class="header-title">🏥 Health AI Assistant</h1>
    <p class="header-subtitle">Advanced Disease Prediction with Explainable AI • Welcome, {st.session_state.username}</p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# PREDICTION HISTORY FUNCTIONS
# -------------------------
def load_predictions():
    try:
        with open('predictions.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_predictions(predictions):
    # Convert numpy types to Python types for JSON serialization
    serializable_predictions = []
    for pred in predictions:
        serializable_pred = {}
        for key, value in pred.items():
            if isinstance(value, (np.float32, np.float64)):
                serializable_pred[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                serializable_pred[key] = int(value)
            else:
                serializable_pred[key] = value
        serializable_predictions.append(serializable_pred)

    with open('predictions.json', 'w') as f:
        json.dump(serializable_predictions, f)

# -------------------------
# HELPER: RUN PREDICTION
# -------------------------
def run_prediction(model, inputs, X_background, disease):
    try:
        float_inputs = [float(x) for x in inputs]
        prediction = model.predict([float_inputs])[0]
        prob = model.predict_proba([float_inputs])[0][1] * 100
        
        # SHAP Explanation
        explainer = shap.Explainer(model)
        shap_values = explainer(X_background)
        input_shap = explainer([float_inputs])
        feature_importance = dict(zip(X_background.columns, input_shap.values[0]))
        
        return prediction, float(prob), feature_importance
    except ValueError:
        st.error("❌ Invalid input! Please ensure all fields are filled with numbers.")
        return None, None, None
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
        return None, None, None

# -------------------------
# DISPLAY SHAP RESULTS
# -------------------------
def display_shap_results(feature_importance):
    st.markdown("""
    <div class="shap-container">
        <h3 style="color: #2c3e50; margin-bottom: 1.5rem;">🔍 Feature Impact Analysis</h3>
    """, unsafe_allow_html=True)
    
    if feature_importance:
        # Sort by absolute value
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        
        for feature, value in sorted_features:
            impact_type = "positive" if value > 0 else "negative"
            impact_class = "shap-positive" if value > 0 else "shap-negative"
            icon = "📈" if value > 0 else "📉"
            
            st.markdown(f"""
            <div class="shap-feature {impact_class}">
                <span style="font-weight: 600; color: #2c3e50;">{icon} {feature}</span>
                <span style="font-weight: 700; color: {'#FF416C' if value > 0 else '#4CAF50'}">
                    {value:.4f}
                </span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# DASHBOARD PAGE
# -------------------------
if selected == "Dashboard":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="card-title">📊 Health Dashboard</h2>', unsafe_allow_html=True)
    
    predictions = load_predictions()
    user_predictions = [p for p in predictions if p.get('user') == st.session_state.username]
    
    if user_predictions:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(user_predictions))
        with col2:
            positive = sum(1 for p in user_predictions if p['result'] == 'Positive')
            st.metric("High Risk Cases", positive)
        with col3:
            avg_risk = sum(p['risk'] for p in user_predictions) / len(user_predictions)
            st.metric("Avg Risk %", f"{avg_risk:.1f}%")
        with col4:
            diseases = set(p['disease'] for p in user_predictions)
            st.metric("Diseases Monitored", len(diseases))
        
        # Visualization
        df = pd.DataFrame(user_predictions)
        df['date'] = pd.to_datetime(df['date'])
        
        tab1, tab2, tab3 = st.tabs(["📈 Risk Trend", "🫀 Disease Distribution", "📅 Recent Activity"])
        
        with tab1:
            if len(df) > 1:
                df_sorted = df.sort_values('date')
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_sorted['date'],
                    y=df_sorted['risk'],
                    mode='lines+markers',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=8),
                    name='Risk Percentage'
                ))
                fig.update_layout(
                    title="Risk Percentage Over Time",
                    xaxis_title="Date",
                    yaxis_title="Risk %",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            disease_counts = df['disease'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=disease_counts.index,
                values=disease_counts.values,
                hole=.3,
                marker=dict(colors=['#667eea', '#764ba2', '#4CAF50'])
            )])
            fig.update_layout(title="Predictions by Disease")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("📝 No predictions yet. Make your first prediction to see analytics here!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# DISCLAIMER PAGE
# -------------------------
elif selected == "Disclaimer":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="card-title">⚠️ Disclaimer</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div style="line-height:1.6">
        <p><strong>Important:</strong> This application is provided for educational and awareness purposes only. It uses machine learning models to estimate risk levels but is <strong>not</strong> a substitute for professional medical diagnosis, advice, or treatment.</p>
        <ul>
            <li>Do not make medical decisions based solely on the results from this app.</li>
            <li>If you have concerns about your health or receive a high-risk result, consult a qualified healthcare professional immediately.</li>
            <li>Models can produce false positives and false negatives; accuracy depends on the input data and model limitations.</li>
            <li>Data entered into this app may be stored locally in the project files for demonstration; avoid entering personally identifiable or sensitive information.</li>
            <li>The authors assume no liability for actions taken based on information provided by this application.</li>
        </ul>
        <p>By using this app you acknowledge that it is for informational purposes only.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# DIABETES PREDICTION PAGE
# -------------------------
elif selected == "Diabetes":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="card-title">🩺 Diabetes Risk Assessment</h2>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ About Diabetes Prediction", expanded=False):
        st.markdown("""
        This model predicts the likelihood of diabetes based on medical parameters. 
        Higher glucose levels, BMI, and age increase risk, while regular exercise and healthy diet reduce risk.
        """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 📋 Personal Details")
        Pregnancies = st.number_input("Pregnancies", min_value=0.0, max_value=20.0, value=0.0, step=1.0)
        Age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
        BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    
    with col2:
        st.markdown("##### 🩸 Blood Parameters")
        Glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=100.0, step=1.0)
        BloodPressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=80.0, step=1.0)
        Insulin = st.number_input("Insulin (μU/mL)", min_value=0.0, max_value=300.0, value=50.0, step=1.0)
    
    with col3:
        st.markdown("##### 📊 Medical Metrics")
        SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.1)
    
    if st.button("🔍 Predict Diabetes Risk", use_container_width=True, type="primary"):
        with st.spinner("Analyzing health parameters..."):
            inputs = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            BMI_Age = BMI * Age
            Glucose_Insulin = Glucose * Insulin
            inputs_extended = inputs + [BMI_Age, Glucose_Insulin]
            
            result, prob, feature_importance = run_prediction(diabetes_model, inputs_extended, X_diabetes, "Diabetes")
            
            if result is not None:
                # Display risk meter
                fig = create_risk_meter(prob)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display result
                display_prediction_result(result, prob, "Diabetes")
                
                # SHAP Analysis
                display_shap_results(feature_importance)
                
                # Recommendations
                with st.expander("💡 Health Recommendations", expanded=True):
                    if result == 1:
                        st.warning("""
                        **Immediate Actions Recommended:**
                        1. Consult a doctor for proper diagnosis
                        2. Monitor blood sugar levels regularly
                        3. Start a balanced diet with low glycemic index foods
                        4. Engage in 30 minutes of daily exercise
                        5. Reduce stress through meditation or yoga
                        """)
                    else:
                        st.success("""
                        **Maintain Healthy Lifestyle:**
                        1. Continue regular health checkups
                        2. Maintain balanced diet and exercise
                        3. Monitor glucose levels annually
                        4. Stay hydrated and manage stress
                        """)
                
                # Save prediction
                predictions = load_predictions()
                predictions.append({
                    "user": st.session_state.username,
                    "disease": "Diabetes",
                    "result": "Positive" if result == 1 else "Negative",
                    "risk": float(prob),
                    "date": datetime.now().isoformat()
                })
                save_predictions(predictions)
    
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# HEART DISEASE PREDICTION PAGE
# -------------------------
elif selected == "Heart Disease":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="card-title">❤️ Heart Disease Risk Assessment</h2>', unsafe_allow_html=True)
    
    with st.expander("ℹ️ About Heart Disease Prediction", expanded=False):
        st.markdown("""
        This model assesses cardiovascular risk based on clinical parameters. 
        Factors like cholesterol, blood pressure, and exercise-induced angina contribute to risk assessment.
        """)
    
    tabs = st.tabs(["📊 Basic Info", "🩺 Clinical Data", "❤️‍🩹 Heart Metrics"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
            sex = st.selectbox("Sex", ["Male", "Female"])
        with col2:
            cp = st.selectbox("Chest Pain Type", 
                            ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            trestbps = st.number_input("Resting BP (mmHg)", min_value=0.0, max_value=250.0, value=120.0, step=1.0)
            chol = st.number_input("Cholesterol (mg/dL)", min_value=0.0, max_value=600.0, value=200.0, step=1.0)
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120", ["Yes", "No"])
            restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
    
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            thalach = st.number_input("Max Heart Rate", min_value=0.0, max_value=250.0, value=150.0, step=1.0)
            exang = st.selectbox("Exercise Angina", ["Yes", "No"])
        with col2:
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("ST Segment Slope", ["Upsloping", "Flat", "Downsloping"])
        
        ca = st.slider("Major Vessels (0-3)", 0, 3, 0)
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    
    if st.button("🔍 Predict Heart Disease Risk", use_container_width=True, type="primary"):
        with st.spinner("Analyzing cardiovascular health..."):
            # Convert inputs
            sex_num = 1 if sex == "Male" else 0
            cp_num = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}[cp]
            fbs_num = 1 if fbs == "Yes" else 0
            restecg_num = {"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2}[restecg]
            exang_num = 1 if exang == "Yes" else 0
            slope_num = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]
            thal_num = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2}[thal]
            
            inputs = [age, sex_num, cp_num, trestbps, chol, fbs_num, restecg_num,
                     thalach, exang_num, oldpeak, slope_num, ca, thal_num]
            
            Age_Chol = age * chol
            Thalach_Oldpeak = thalach * oldpeak
            inputs_extended = inputs + [Age_Chol, Thalach_Oldpeak]
            
            result, prob, feature_importance = run_prediction(heart_disease_model, inputs_extended, X_heart, "Heart Disease")
            
            if result is not None:
                fig = create_risk_meter(prob)
                st.plotly_chart(fig, use_container_width=True)
                display_prediction_result(result, prob, "Heart Disease")
                display_shap_results(feature_importance)
                
                # Recommendations
                with st.expander("💡 Health Recommendations", expanded=True):
                    if result == 1:
                        st.warning("""
                        **Immediate Actions Recommended:**
                        1. Schedule an appointment with a cardiologist immediately
                        2. Undergo comprehensive cardiac evaluation (ECG, stress test, angiogram)
                        3. Monitor blood pressure daily and keep a record
                        4. Reduce sodium intake to less than 2.3g per day
                        5. Follow a heart-healthy diet (Mediterranean or DASH diet)
                        6. Exercise moderately for 150 minutes per week as approved by doctor
                        7. Avoid smoking and secondhand smoke exposure
                        8. Reduce stress through meditation, yoga, or counseling
                        9. Limit alcohol consumption
                        10. Take prescribed medications regularly and don't skip doses
                        """)
                    else:
                        st.success("""
                        **Maintain Heart Health:**
                        1. Continue regular cardiovascular checkups (annually)
                        2. Maintain a balanced, heart-healthy diet rich in fruits and vegetables
                        3. Exercise regularly - aim for 150 minutes of moderate activity per week
                        4. Keep weight within healthy range (BMI 18.5-24.9)
                        5. Monitor and control blood pressure (optimal: <120/80 mmHg)
                        6. Maintain healthy cholesterol levels through diet and exercise
                        7. Avoid smoking and exposure to secondhand smoke
                        8. Limit alcohol consumption (men: ≤2 drinks/day, women: ≤1 drink/day)
                        9. Manage stress through relaxation techniques
                        10. Get adequate sleep (7-9 hours per night)
                        """)
                
                # Save prediction
                predictions = load_predictions()
                predictions.append({
                    "user": st.session_state.username,
                    "disease": "Heart Disease",
                    "result": "Positive" if result == 1 else "Negative",
                    "risk": float(prob),
                    "date": datetime.now().isoformat()
                })
                save_predictions(predictions)

# -------------------------
# PARKINSON'S PREDICTION PAGE
# -------------------------
elif selected == "Parkinson's":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="card-title">🧠 Parkinson\'s Disease Risk Assessment</h2>', unsafe_allow_html=True)
    
    st.info("Enter voice measurement parameters for analysis:")
    
    # Create 5 columns for inputs
    cols = st.columns(5)
    
    # Voice measurement parameters
    voice_params = [
        ("MDVP:Fo(Hz)", 150.0, 50.0, 250.0),
        ("MDVP:Fhi(Hz)", 200.0, 100.0, 300.0),
        ("MDVP:Flo(Hz)", 100.0, 50.0, 200.0),
        ("MDVP:Jitter(%)", 0.005, 0.0, 0.1),
        ("MDVP:Jitter(Abs)", 0.00003, 0.0, 0.001),
        ("MDVP:RAP", 0.003, 0.0, 0.1),
        ("MDVP:PPQ", 0.003, 0.0, 0.1),
        ("Jitter:DDP", 0.009, 0.0, 0.1),
        ("MDVP:Shimmer", 0.02, 0.0, 0.5),
        ("MDVP:Shimmer(dB)", 0.2, 0.0, 2.0),
        ("Shimmer:APQ3", 0.01, 0.0, 0.2),
        ("Shimmer:APQ5", 0.01, 0.0, 0.2),
        ("MDVP:APQ", 0.02, 0.0, 0.3),
        ("Shimmer:DDA", 0.03, 0.0, 0.3),
        ("NHR", 0.01, 0.0, 0.5),
        ("HNR", 25.0, 0.0, 40.0),
        ("RPDE", 0.5, 0.0, 1.0),
        ("DFA", 0.7, 0.0, 1.0),
        ("spread1", -5.0, -10.0, 0.0),
        ("spread2", 0.2, 0.0, 1.0),
        ("D2", 2.0, 0.0, 5.0),
        ("PPE", 0.2, 0.0, 1.0)
    ]
    
    # Create input fields
    values = []
    for i, (label, default, min_val, max_val) in enumerate(voice_params):
        with cols[i % 5]:
            val = st.number_input(
                label,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default),
                step=0.001 if default < 1 else 1.0
            )
            values.append(val)
    
    if st.button("🔍 Predict Parkinson's Risk", use_container_width=True, type="primary"):
        with st.spinner("Analyzing voice parameters..."):
            # Feature engineering
            jitter_ratio = values[3] / (values[4] + 1e-6)
            vocal_instability = values[7] + values[13]
            log_PPE = np.log(values[21] + 1e-6)
            inputs_extended = values + [jitter_ratio, vocal_instability, log_PPE]
            
            result, prob, feature_importance = run_prediction(parkinsons_model, inputs_extended, X_parkinsons, "Parkinson's")
            
            if result is not None:
                fig = create_risk_meter(prob)
                st.plotly_chart(fig, use_container_width=True)
                display_prediction_result(result, prob, "Parkinson's Disease")
                display_shap_results(feature_importance)
                
                # Recommendations
                with st.expander("💡 Health Recommendations", expanded=True):
                    if result == 1:
                        st.warning("""
                        **Immediate Actions Recommended:**
                        1. Consult a neurologist specializing in movement disorders
                        2. Get comprehensive neurological assessment and MRI/PET scan if needed
                        3. Start early intervention therapy and medication as prescribed
                        4. Participate in speech and occupational therapy
                        5. Maintain regular physical therapy to preserve mobility
                        6. Engage in daily exercise (walking, swimming, tai chi) for 30+ minutes
                        7. Keep brain active with cognitive exercises (puzzles, reading, learning)
                        8. Maintain balanced diet rich in antioxidants and omega-3 fatty acids
                        9. Get adequate sleep (7-9 hours) with proper sleep hygiene
                        10. Join support groups for emotional and psychological support
                        11. Avoid alcohol and limit caffeine intake
                        12. Stay connected with family, friends, and community
                        """)
                    else:
                        st.success("""
                        **Maintain Neurological Health:**
                        1. Continue regular neurological checkups (annually)
                        2. Maintain healthy lifestyle with regular physical activity
                        3. Engage in 150 minutes of moderate exercise per week (walking, dancing, swimming)
                        4. Keep mind active with mental exercises and learning new skills
                        5. Eat a Mediterranean diet rich in antioxidants and healthy fats
                        6. Get 7-9 hours of quality sleep per night
                        7. Manage stress through meditation, yoga, or mindfulness
                        8. Stay socially connected and maintain strong relationships
                        9. Limit alcohol consumption to moderate levels
                        10. Avoid smoking and secondhand smoke exposure
                        11. Monitor any changes in voice, movement, or coordination
                        12. Keep blood pressure and cholesterol under control
                        """)
                
                # Save prediction
                predictions = load_predictions()
                predictions.append({
                    "user": st.session_state.username,
                    "disease": "Parkinson's",
                    "result": "Positive" if result == 1 else "Negative",
                    "risk": float(prob),
                    "date": datetime.now().isoformat()
                })
                save_predictions(predictions)
    
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# PREDICTION HISTORY PAGE
# -------------------------
elif selected == "History":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="card-title">📋 Prediction History</h2>', unsafe_allow_html=True)
    
    predictions = load_predictions()
    user_predictions = [p for p in predictions if p.get('user') == st.session_state.username]
    
    if user_predictions:
        # Create DataFrame
        df = pd.DataFrame(user_predictions)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=False)
        
        # Search and filter
        col1, col2, col3 = st.columns(3)
        with col1:
            search_disease = st.multiselect("Filter by Disease", df['disease'].unique(), default=df['disease'].unique())
        with col2:
            search_result = st.multiselect("Filter by Result", df['result'].unique(), default=df['result'].unique())
        
        # Apply filters
        filtered_df = df[
            (df['disease'].isin(search_disease)) & 
            (df['result'].isin(search_result))
        ]
        
        if not filtered_df.empty:
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_risk = filtered_df['risk'].mean()
                st.metric("Average Risk", f"{avg_risk:.1f}%")
            with col2:
                st.metric("Total Records", len(filtered_df))
            with col3:
                positive = (filtered_df['result'] == 'Positive').sum()
                st.metric("High Risk", positive)
            with col4:
                recent = filtered_df.iloc[0]['date'].strftime('%b %d')
                st.metric("Most Recent", recent)
            
            # Display table
            display_df = filtered_df.copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Color coding for risk (high-contrast for readability)
            def color_risk(val):
                try:
                    v = float(val)
                except Exception:
                    return ''

                # High risk: dark red background, white bold text
                if v > 70:
                    return 'background-color: #b71c1c; color: white; font-weight: 700;'
                # Medium risk: orange background, white bold text
                elif v > 40:
                    return 'background-color: #ff9800; color: white; font-weight: 700;'
                # Low risk: dark green background, white bold text
                else:
                    return 'background-color: #2e7d32; color: white; font-weight: 700;'
            
            styled_df = display_df[['disease', 'result', 'risk', 'date']].style.map(
                color_risk, subset=['risk']
            )
            
            st.dataframe(
                styled_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "disease": "Disease",
                    "result": "Result",
                    "risk": st.column_config.NumberColumn("Risk %", format="%.1f%%"),
                    "date": "Date"
                }
            )
            
            # Export option
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download History (CSV)",
                data=csv,
                file_name=f"health_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("No predictions match the selected filters.")
    else:
        st.info("No predictions found. Make some predictions to see your history here.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    