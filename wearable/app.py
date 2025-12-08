import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Stress Detection Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress potential warnings
warnings.filterwarnings('ignore')

# --- MODEL AND DATA LOADING ---
@st.cache_resource
def load_model():
    """Loads the trained machine learning model from a file."""
    try:
        model = joblib.load("trained_random_forest_model.pkl")
        return model
    except FileNotFoundError:
        st.error("🚨 Model file 'trained_random_forest_model.pkl' not found. Please ensure it's in the correct directory.")
        st.stop()

model = load_model()

# --- STATIC DATA & CONFIGURATION ---
ACCURACY = 0.97
CONFUSION_MATRIX_DATA = [
    [7285, 28, 0],
    [2176, 78453, 935],
    [0, 26, 4897]
]
CLASSIFICATION_REPORT_TEXT = """
              precision    recall  f1-score   support

         1.0       0.77      1.00      0.87      7313
         2.0       1.00      0.96      0.98     81564
         3.0       0.84      0.99      0.91      4923

    accuracy                           0.97     93800
   macro avg       0.87      0.98      0.92     93800
weighted avg       0.97      0.97      0.97     93800
"""
FEATURE_NAMES = ['ECG', 'EMG', 'EDA', 'TEMP', 'RESP', 'ACC_X', 'ACC_Y', 'ACC_Z']
STRESS_LEVEL_MAPPING = {
    1.0: "Low Stress",
    2.0: "Medium Stress",
    3.0: "High Stress"
}
STRESS_LEVEL_COLORS = {
    "Low Stress": "#28a745",    # Green
    "Medium Stress": "#ffc107", # Yellow
    "High Stress": "#dc3545"    # Red
}

# --- PAGE 1: WELCOME PAGE ---
def show_welcome_page():
    """Displays the welcome and project overview page."""
    st.title("🧠 Welcome to the AI-Powered Stress Detection Dashboard")
    st.markdown("---")
    st.image("https://images.unsplash.com/photo-1598972175923-50f09a5c8624?q=80&w=2070", caption="Monitoring physiological signals for wellness insights.")
    st.header("About This Project")
    st.write("This interactive dashboard utilizes a **Random Forest classifier** to predict human stress levels in real-time based on physiological data.")
    st.header("How to Use This Dashboard")
    st.info("""
    - **🧠 Predict Stress**: Navigate to this page to use the interactive sliders. Adjust the values to simulate different physiological readings and get an instant stress level prediction from the AI.
    - **📈 Model Performance**: Go here to see a detailed breakdown of the model's accuracy, a confusion matrix, classification report, and a chart of which features the model finds most important.
    """)
    st.header("Understanding the Prediction Chart")
    st.write("When you get a prediction, you will see a bar chart that shows the model's confidence. This chart is key to understanding **how close** your input values are to each stress category.")
    st.markdown("---")

# --- PAGE 2: MODEL PERFORMANCE DASHBOARD ---
def show_model_statistics():
    """Displays the model's performance metrics and visualizations."""
    st.title("📈 Model Performance Dashboard")
    st.write("Here's a detailed summary of the Random Forest model's performance on the test dataset.")
    st.markdown("---")
    st.header("Overall Accuracy")
    st.info(f"The model achieved an overall accuracy of **{ACCURACY:.2%}** on the test data.")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Confusion Matrix")
        cm = np.array(CONFUSION_MATRIX_DATA)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=STRESS_LEVEL_MAPPING.values(), yticklabels=STRESS_LEVEL_MAPPING.values(), ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        st.pyplot(fig)
    with col2:
        st.header("Feature Importance")
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': FEATURE_NAMES, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', ax=ax)
        ax.set_title('Which signals matter most?')
        st.pyplot(fig)
    st.markdown("---")
    st.header("Classification Report")
    st.code(CLASSIFICATION_REPORT_TEXT, language='text')

# --- PAGE 3: REAL-TIME PREDICTION INTERFACE ---
def show_prediction_page():
    """Renders the interactive page for users to input data and get a stress prediction."""
    st.title("🧠 Real-Time Stress Prediction")
    st.write("Adjust the sliders below or use the buttons to see how different physiological signals affect stress predictions.")
    
    with st.expander("📖 Guide to Signal Values for Stress States"):
        st.markdown("""
        | Signal                    | Low Stress (Relaxed) 😌          | High Stress (Fight-or-Flight) 😫          |
        |---------------------------|-----------------------------------|------------------------------------------|
        | **💧 EDA (Sweating)** | **Low values** (e.g., < 2.0 µS)   | **High values** (e.g., > 5.0 µS)         |
        | **💪 EMG (Muscle Tension)** | Very low values near 0            | Higher values (e.g., > 0.5 or < -0.5)    |
        | **🫁 Resp (Breathing)** | Slow, regular rhythm (near 0)     | Fast, erratic rhythm (far from 0)        |
        | **🌡️ Temp (Skin Temp)** | Warmer skin (e.g., > 33.8 °C)     | Cooler skin (e.g., < 33.5 °C)            |
        """)
    
    st.markdown("---")

    # Initialize session state for inputs and history
    if 'input_data' not in st.session_state:
        st.session_state.input_data = {'ce': -0.004, 'em': -0.0001, 'ed': 1.75, 'tp': 33.8, 'rs': 0.04, 'ax': 0.20, 'ay': -0.22, 'az': 0.94}
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

    st.subheader("Load Example Data")
    c1, c2, c3 = st.columns(3)
    if c1.button("Example: Low Stress 😌"):
        st.session_state.input_data = {'ce': -0.02, 'em': 0.01, 'ed': 0.8, 'tp': 33.9, 'rs': 2.1, 'ax': -0.05, 'ay': -0.15, 'az': 0.98}
    if c2.button("Example: Medium Stress 🤔"):
        st.session_state.input_data = {'ce': 0.15, 'em': -0.1, 'ed': 3.5, 'tp': 33.7, 'rs': -3.0, 'ax': 0.10, 'ay': -0.25, 'az': 0.90}
    if c3.button("Example: High Stress 😫"):
        st.session_state.input_data = {'ce': -0.8, 'em': 0.4, 'ed': 6.2, 'tp': 33.5, 'rs': 8.0, 'ax': 0.35, 'ay': 0.05, 'az': 0.75}

    st.markdown("---")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.header("Core & Autonomic Signals")
            st.session_state.input_data['ed'] = st.slider("💧 EDA (µS)", 0.1, 10.0, st.session_state.input_data['ed'], 0.01)
            st.session_state.input_data['tp'] = st.slider("🌡️ Temperature (°C)", 32.5, 35.0, st.session_state.input_data['tp'], 0.01)
            st.session_state.input_data['em'] = st.slider("💪 EMG (mV)", -1.5, 1.5, st.session_state.input_data['em'], 0.001)
            st.session_state.input_data['ce'] = st.slider("⚡ ECG (mV)", -2.0, 2.0, st.session_state.input_data['ce'], 0.01)
        with col2:
            st.header("Respiration & Motion")
            st.session_state.input_data['rs'] = st.slider("🫁 Respiration (a.u.)", -25.0, 25.0, st.session_state.input_data['rs'], 0.1)
            st.subheader("Motion Data (Accelerometer)")
            st.session_state.input_data['ax'] = st.slider("➡️ X-axis (g)", -2.0, 2.0, st.session_state.input_data['ax'], 0.01)
            st.session_state.input_data['ay'] = st.slider("⬆️ Y-axis (g)", -2.0, 2.0, st.session_state.input_data['ay'], 0.01)
            st.session_state.input_data['az'] = st.slider("↗️ Z-axis (g)", -2.0, 2.0, st.session_state.input_data['az'], 0.01)
        submitted = st.form_submit_button("Analyze Stress Level", type="primary", use_container_width=True)

    if submitted:
        input_data_values = list(st.session_state.input_data.values())
        input_array = np.array(input_data_values).reshape(1, -1)
        prediction = model.predict(input_array)
        prediction_proba = model.predict_proba(input_array)
        predicted_class = prediction[0]
        stress_level = STRESS_LEVEL_MAPPING.get(predicted_class, "Unknown")
        
        # --- MODIFIED: Append result to history with probabilities ---
        new_entry = {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Predicted Stress': stress_level,
            'Low Stress %': f"{prediction_proba[0][0]:.1%}",
            'Medium Stress %': f"{prediction_proba[0][1]:.1%}",
            'High Stress %': f"{prediction_proba[0][2]:.1%}",
            'ECG': input_data_values[0],
            'EMG': input_data_values[1],
            'EDA': input_data_values[2],
            'TEMP': input_data_values[3],
            'RESP': input_data_values[4],
            'ACC_X': input_data_values[5],
            'ACC_Y': input_data_values[6],
            'ACC_Z': input_data_values[7],
        }
        st.session_state.prediction_history.append(new_entry)
        
        st.markdown("---")
        st.header("Prediction Result")
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.subheader("Final Prediction")
            if stress_level == "Low Stress": st.success(f"**{stress_level}** 😌")
            elif stress_level == "Medium Stress": st.warning(f"**{stress_level}** 🤔")
            elif stress_level == "High Stress": st.error(f"**{stress_level}** 😫")
        with res_col2:
            st.subheader("Confidence Score")
            fig, ax = plt.subplots()
            bars = ax.bar(STRESS_LEVEL_MAPPING.values(), prediction_proba[0], color=[STRESS_LEVEL_COLORS.get(label) for label in STRESS_LEVEL_MAPPING.values()])
            ax.set_ylabel('Probability')
            ax.set_ylim(0, 1)
            ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
            st.pyplot(fig)
            
    if st.session_state.prediction_history:
        st.markdown("---")
        st.header("Prediction Log")
        st.write("All predictions from this session are logged here. You can download this log as a CSV file.")
        
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # --- MODIFIED: Reorder columns for better readability ---
        column_order = [
            'Timestamp', 'Predicted Stress', 'Low Stress %', 'Medium Stress %', 'High Stress %', 
            'EDA', 'TEMP', 'EMG', 'RESP', 'ECG', 'ACC_X', 'ACC_Y', 'ACC_Z'
        ]
        st.dataframe(history_df[column_order])

        csv = history_df[column_order].to_csv(index=False).encode('utf-8')
        
        dl_col, clr_col = st.columns(2)
        with dl_col:
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name='stress_prediction_log.csv',
                mime='text/csv',
                use_container_width=True
            )
        with clr_col:
            if st.button("🗑️ Clear Prediction Log", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()

# --- MAIN APP ROUTING LOGIC ---
def main():
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    page_options = {
        "🏠 Welcome": show_welcome_page,
        "🧠 Predict Stress": show_prediction_page,
        "📈 Model Performance": show_model_statistics
    }
    selection = st.sidebar.radio("Go to", list(page_options.keys()))
    page_function = page_options[selection]
    page_function()
    st.sidebar.markdown("---")
    st.sidebar.info("This is an educational demo and not a medical diagnostic tool.")

if __name__ == "__main__":
    main()