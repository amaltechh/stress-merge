import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import time

# --- Configuration and Professional Setup ---
st.set_page_config(
    page_title="Stress Fusion AI Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "A professional application for multi-modal stress analysis."}
)

# Custom CSS for a clean, dark-mode look and hiding warnings/caching info
st.markdown("""
<style>
/* 1. HIDE the deprecation warning you saw */
.css-1uixxvy {
    display: none;
}
/* 2. Hide caching status for a cleaner look */
div[data-testid="stStatusWidget"] {
    display: none;
}
/* 3. Custom Animated Card for Metrics */
@keyframes fadeIn {
  0% {opacity: 0; transform: translateY(20px);}
  100% {opacity: 1; transform: translateY(0);}
}

div.stCard {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5); /* Stronger shadow for dark theme */
    border-radius: 12px;
    border-left: 5px solid #4a90e2; /* Highlight bar (Primary Color) */
    transition: all 0.3s;
    animation: fadeIn 0.5s ease-out;
    background-color: #1a1e27 !important; /* Ensure card background is dark */
}
div.stCard:hover {
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.7);
    transform: translateY(-3px);
}

/* 4. Agreement Score Big Font - Ensure high visibility on dark theme */
.agreement-score {
    font-size: 72px !important;
    font-weight: 900;
    color: #10b981; /* Success color (Vibrant Green) */
    text-align: center;
    padding-top: 10px;
}
.metric-title {
    font-size: 18px;
    font-weight: 600;
    color: #9ca3af; /* Light gray text for dark contrast */
    text-align: center;
    margin-bottom: 0px;
}
/* 5. Clean up Matplotlib figures for dark theme (plots) */
.stPlotlyChart {
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
}

</style>
""", unsafe_allow_html=True)

# --- Core Functions (Data Processing and Plotting remain unchanged for stability) ---

@st.cache_data(show_spinner=False)
def _find_column(df, options):
    for col in options:
        lower_cols = [c.lower() for c in df.columns]
        if col.lower() in lower_cols:
            return df.columns[lower_cols.index(col.lower())]
    return None

@st.cache_data(show_spinner=False)
def merge_and_process_data(survey_file, wearable_file):
    """Implements the core Stress Fusion Alignment Algorithm (SFAA)."""
    try:
        survey_df = pd.read_csv(survey_file)
        wearable_df = pd.read_csv(wearable_file)
    except Exception as e:
        raise ValueError(f"Error reading files: {e}")

    original_survey_df = survey_df.copy()
    original_wearable_df = wearable_df.copy()

    if 'Timestamp' in survey_df.columns: survey_df = survey_df.drop(columns=['Timestamp'])
    if 'Timestamp' in wearable_df.columns: wearable_df = wearable_df.drop(columns=['Timestamp'])
    
    n_survey, n_wearable = len(survey_df), len(wearable_df)

    if n_survey == 0 or n_wearable == 0:
        raise ValueError("One of the uploaded files is empty.")

    # --- SFAA Step 2: Dynamic Aggregation/Resampling ---
    if n_wearable > n_survey:
        chunk_size = n_wearable // n_survey
        grouping_key = np.arange(n_wearable) // chunk_size
        wearable_df = wearable_df.iloc[:len(grouping_key)].copy()
        wearable_df['group'] = grouping_key
        wearable_df = wearable_df[wearable_df['group'] < n_survey]
        
        aggregation_rules = {
            'EDA':'mean', 'TEMP':'mean', 'EMG':'mean', 'RESP':'mean', 'ECG':'mean', 
            'Predicted Stress':lambda x: x.mode()[0] if not x.empty and not x.mode().empty else np.nan
        }
        valid_rules = {k: v for k, v in aggregation_rules.items() if _find_column(wearable_df, [k])}
        
        wearable_agg_df = wearable_df.groupby('group').agg(valid_rules)
        merged_df = pd.concat([survey_df.reset_index(drop=True), wearable_agg_df.reset_index(drop=True)], axis=1)
    else:
        min_rows = min(n_survey, n_wearable)
        merged_df = pd.concat([survey_df.iloc[:min_rows].reset_index(drop=True), wearable_df.iloc[:min_rows].reset_index(drop=True)], axis=1)

    if merged_df.empty:
        raise ValueError("Could not merge files after alignment.")
    
    # --- SFAA Step 3: Label Normalization ---
    survey_col = _find_column(merged_df, ['Stress_Level', 'Stress Level'])
    wearable_col = _find_column(merged_df, ['Predicted Stress'])

    if not survey_col or not wearable_col:
        raise ValueError("Missing 'Stress_Level' (Survey) or 'Predicted Stress' (Wearable) column. Please check file headers.")

    stress_mapping = {'No stress': 'Low', 'Low Stress': 'Low', 'Medium stress': 'Medium', 'Medium Stress': 'Medium', 'High stress': 'High', 'High Stress': 'High'}
    merged_df['Survey Stress'] = merged_df[survey_col].astype(str).map(stress_mapping)
    merged_df['Wearable Stress'] = merged_df[wearable_col].astype(str).map(stress_mapping)
    
    return merged_df, original_survey_df, original_wearable_df

# (Plotting functions remain the same)
@st.cache_data
def _plot_confusion_matrix(data):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 5))
    order = ['Low', 'Medium', 'High']
    valid_comp = data[['Survey Stress', 'Wearable Stress']].dropna()
    cm = confusion_matrix(valid_comp['Survey Stress'], valid_comp['Wearable Stress'], labels=order)
    sns.heatmap(cm, annot=True, fmt='d', cmap='flare', xticklabels=order, yticklabels=order, ax=ax, annot_kws={"size": 14})
    ax.set_title('Classification Agreement (Survey vs. Wearable)', fontsize=14, weight='bold', color='white')
    ax.set_xlabel('Predicted (Wearable)', fontsize=12, color='white')
    ax.set_ylabel('Actual (Survey)', fontsize=12, color='white')
    plt.tight_layout()
    return fig

@st.cache_data
def _plot_distributions(data):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    order = ['Low', 'Medium', 'High']
    colors = ['#10b981', '#f59e0b', '#ef4444']
    
    for i, col_name in enumerate(['Survey Stress', 'Wearable Stress']):
        counts = data[col_name].value_counts().reindex(order, fill_value=0)
        axes[i].pie(
            counts, labels=counts.index, autopct='%1.1f%%', startangle=90, 
            colors=[colors[order.index(key)] for key in counts.index], 
            wedgeprops=dict(width=0.4, edgecolor='#1a202c'), pctdistance=0.8, textprops={'color': 'white'}
        )
        axes[i].set_title(f'{col_name} Distribution', fontsize=14, weight='bold', color='white')
    
    plt.tight_layout()
    return fig

@st.cache_data
def _plot_biometric_boxplots(data, columns):
    plt.style.use('dark_background')
    valid_cols = [c for c in columns if c in data.columns]
    if not valid_cols: return None

    fig, axes = plt.subplots(1, len(valid_cols), figsize=(5 * len(valid_cols), 5), squeeze=False)
    order = ['Low', 'Medium', 'High']
    palette = ['#10b981', '#f59e0b', '#ef4444']

    for i, col in enumerate(valid_cols):
        ax = axes[0, i]
        sns.boxplot(x='Survey Stress', y=col, data=data.dropna(subset=['Survey Stress', col]), order=order, ax=ax, palette=palette, flierprops=dict(markerfacecolor='r', marker='o', markersize=5))
        ax.set_title(f'{col} vs. Survey Stress Level', fontsize=14, weight='bold', color='white')
        ax.set_xlabel('Stress Level (Self-Reported)', fontsize=12, color='white')
        ax.set_ylabel(col, fontsize=12, color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    return fig

@st.cache_data
def _plot_correlation_heatmap(data):
    plt.style.use('dark_background')
    total_col = _find_column(data, ['Total', 'total_score'])
    corr_cols = ([total_col] if total_col else []) + ['EDA', 'TEMP', 'EMG', 'RESP', 'ECG']
    valid_cols = [c for c in corr_cols if c in data.columns]
    
    if len(valid_cols) < 2: return None

    fig, ax = plt.subplots(figsize=(7, 6))
    corr_matrix = data[valid_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, linewidths=0.5, linecolor='gray', annot_kws={"size": 10})
    ax.set_title('Biometric & Survey Score Correlation Matrix', fontsize=14, weight='bold', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.tight_layout()
    return fig

# --- Model and SFAA Explanation Functions ---

def display_rf_model_explanation():
    st.subheader("Random Forest Classifier (The Prediction Model) :chart_line:")
    st.caption("This model is assumed to have generated the 'Predicted Stress' column in your Wearable Data.")
    
    col_rf1, col_rf2 = st.columns(2)
    
    with col_rf1:
        st.markdown(
            """
            The **Random Forest Classifier** is an **ensemble learning** method that builds multiple Decision Trees and combines their outputs for highly accurate and stable predictions. It is highly effective for stress classification due to its resilience to noisy data (common in wearables) and its ability to handle multiple feature types simultaneously (biometrics, context).
            
            **How it works:**
            1.  **Bagging (Bootstrap Aggregation):** It creates individual trees from random subsets of the input data.
            2.  **Feature Randomness:** At each node split, it only considers a random subset of the total features, increasing diversity among the trees.
            3.  **Majority Vote:** The final stress prediction (Low, Medium, or High) is determined by the class chosen by the majority of the individual trees.
            """
        )
    with col_rf2:
        st.info("#### Why Random Forest for Stress?")
        st.markdown(
            """
            * **High Accuracy:** Aggregating many trees provides more precise results than any single tree.
            * **Robustness:** Performs well even with missing or noisy values, which is typical for real-world wearable sensor data.
            * **Feature Importance:** It can rank which physiological signals (e.g., EDA vs. TEMP) are most influential in determining the stress level.
            """
        )
        

def display_sfaa_explanation():
    st.subheader("Stress Fusion Alignment Algorithm (SFAA) :dna:")
    st.markdown("""
        The **SFAA** is the core engine that synchronizes your two datasets (low-frequency surveys and high-frequency wearables) 
        to enable meaningful comparison and analysis.
    """)
    # (Rest of SFAA explanation remains the same)
    
    col_alg1, col_alg2 = st.columns([1, 1])
    
    with col_alg1:
        st.info("#### 1. Data Cleansing & Preparation")
        st.markdown(
            """
            * **Goal:** Ensure data integrity and remove non-essential columns (like ambiguous Timestamps).
            * **Action:** Verify non-empty dataframes and drop auxiliary columns to prepare for alignment.
            """
        )
    
    with col_alg2:
        st.info("#### 3. Feature Fusion & Normalization")
        st.markdown(
            """
            * **Goal:** Create a consistent labeling system for comparison.
            * **Action:** Map all variant stress labels (e.g., 'Low Stress', 'No stress') to a single normalized set ('Low', 'Medium', 'High').
            """
        )

    st.markdown("---")
    st.markdown("### 2. Dynamic Aggregation (The Core Alignment Step)")
    st.markdown("""
        This step handles the critical issue where Wearable data has many more readings ($N_{Wearable}$) than Survey records ($N_{Survey}$).
    """)

    st.markdown(
        r"""
        1.  **Calculate Chunk Size ($\Delta t$):** The wearable data is divided into $N_{Survey}$ equal-sized chunks: $\Delta t = \text{floor}(\frac{N_{Wearable}}{N_{Survey}})$.
        2.  **Aggregation Rules Applied per Chunk:**
            * **Continuous Biometrics (EDA, TEMP, etc.):** The **mean** is calculated for all values in that $\Delta t$ chunk. This smooths noise and represents the average physiological state.
            * **Discrete Labels (Predicted Stress):** The **statistical mode** (most frequent value) is calculated. This represents the most likely predicted stress state during that survey period.
        3.  **Output:** A new Wearable dataset with $N_{Survey}$ rows is created, which is then merged row-wise with the Survey data.
        """
    )
    
def display_analysis_breakdown(merged_data):
    st.subheader("How Each Data Type is Analyzed :chart_with_upwards_trend:")
    st.divider()

    col_survey, col_wearable = st.columns(2)

    with col_survey:
        st.markdown("#### 1. Survey Data Analysis (The Ground Truth)")
        st.caption("Survey data (e.g., self-reported stress, total scores) provides the subjective baseline for stress.")
        st.markdown(
            """
            * **Distribution Analysis:** We calculate the frequency of 'Low', 'Medium', and 'High' stress levels based on self-reports (used in the Pie Chart).
            * **Baseline for Comparison:** Survey Stress is used as the **Actual/True Label** on the Y-axis of the Confusion Matrix.
            * **Biometric Correlation:** The numerical **Total Score** (if available) is correlated directly with aggregated biometrics to find physiological links (used in the Correlation Heatmap).
            """
        )

    with col_wearable:
        st.markdown("#### 2. Wearable Data Analysis (The Predictive Signal)")
        st.caption("Wearable data (aggregated biometrics and predicted labels) provides the objective, physiological data.")
        st.markdown(
            """
            * **Biometric Response (EDA, TEMP, etc.):** Box plots show how the distribution of each biometric changes across the three Survey Stress levels, indicating physiological response patterns.
            * **Correlation Analysis:** Aggregated biometrics are correlated with each other and the Survey Total Score to assess feature redundancy and relevance.
            * **Prediction Validation:** The aggregated **Wearable Stress** label is used as the **Predicted Label** on the X-axis of the Confusion Matrix to validate the **Random Forest Model**'s performance.
            """
        )
    
    st.divider()
    st.markdown("### Outputs of the Analysis")
    st.markdown(
        """
        * **Agreement Score:** The primary metric—the percentage of records where the Survey Stress and Wearable Stress labels match (the sum of the diagonal in the Confusion Matrix).
        * **Confusion Matrix:** Shows the granular performance, revealing where the wearable device tends to **misclassify** stress (e.g., predicting 'Low' when the user reported 'Medium').
        """
    )


# --- Streamlit UI Layout & Execution ---

# --- Improved Sidebar UI (unchanged) ---
st.sidebar.markdown('<h2 style="color:white; margin-bottom: 0px;">Stress Fusion AI</h2>', unsafe_allow_html=True)
st.sidebar.caption("High-Fidelity Multi-Modal Analysis Engine")
st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True) 

st.sidebar.markdown("### :floppy_disk: Data Sources")
with st.sidebar.container(border=True):
    survey_file = st.file_uploader(":notebook: **Survey Data (CSV)**", type="csv")
    st.caption("Low-Frequency Ground Truth (e.g., Stress_Level)")
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    wearable_file = st.file_uploader(":watch: **Wearable Data (CSV)**", type="csv")
    st.caption("High-Frequency Biometrics (e.g., EDA, Predicted Stress)")
    
st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

analyze_button = st.sidebar.button("Run SFAA & Generate Report", type="primary", use_container_width=True, disabled=(survey_file is None or wearable_file is None))
st.sidebar.caption("Click to start the synchronization and report generation.")

# --- Main Content Area ---
st.title("Stress Fusion AI Analyzer 🚀")

if not (survey_file and wearable_file):
    # Initial message
    st.caption("A specialized tool for synchronizing low-frequency subjective data with high-frequency objective data.")
    st.divider()
    
    st.markdown("""
        ### Multi-Modal Data Alignment for Holistic Stress Measurement
        
        Welcome to the **Stress Fusion AI Analyzer**. This tool utilizes the **Stress Fusion Alignment Algorithm (SFAA)** to synchronize disparate data frequencies for a holistic stress report.
        
        1.  Upload your structured **Survey Data** (low-frequency, labeled stress).
        2.  Upload your **Wearable Data** (high-frequency, predicted stress/biometrics).
    """)
    
    st.divider()
    
    with st.expander("Click to view: Model and Algorithm Details", expanded=False):
        display_rf_model_explanation()
        st.divider()
        display_sfaa_explanation()


elif analyze_button:
    
    # ----------------------------------------------------
    # --- ANIMATED LOADING SEQUENCE ---
    # ----------------------------------------------------
    
    status_container = st.empty()
    progress_bar = st.progress(0, text="Initializing SFAA...")
    
    total_steps = 5
    step_delay = 1.0 

    try:
        # Step 1: Loading and Validation (0% to 20%)
        status_container.info("Step 1/5: Reading and Validating Data Files...")
        progress_bar.progress(20, text="Reading and Validating Data Files...")
        time.sleep(step_delay)
        
        merged_data, original_survey_df, original_wearable_df = merge_and_process_data(survey_file, wearable_file)

        # Step 2: Dynamic Aggregation (20% to 40%)
        status_container.info("Step 2/5: Executing Dynamic Aggregation (Time-Series Alignment)...")
        progress_bar.progress(40, text="Executing Dynamic Aggregation (Time-Series Alignment)...")
        time.sleep(step_delay)
        
        # Step 3: Normalization and Fusion (40% to 60%)
        status_container.info("Step 3/5: Normalizing Stress Labels and Fusing Datasets...")
        progress_bar.progress(60, text="Normalizing Stress Labels and Fusing Datasets...")
        time.sleep(step_delay)

        # Step 4: Generating Metrics (60% to 80%)
        status_container.info("Step 4/5: Calculating Agreement Score and Classification Metrics...")
        progress_bar.progress(80, text="Calculating Agreement Score and Classification Metrics...")
        time.sleep(step_delay)

        # Step 5: Rendering Visualizations (80% to 100%)
        status_container.info("Step 5/5: Generating Professional Visualizations (Plots)...")
        progress_bar.progress(100, text="Generating Professional Visualizations (Plots)...")
        time.sleep(step_delay)
        
        # Clean up loading elements
        status_container.empty()
        progress_bar.empty()
        
        # ----------------------------------------------------
        # --- DISPLAY RESULTS (The entire report) ---
        # ----------------------------------------------------
        
        if merged_data is not None:
            # Calculate metrics
            agreement_score = (merged_data['Survey Stress'] == merged_data['Wearable Stress']).mean() * 100
            
            st.success("✅ Analysis Complete! Report Generated.")
            
            # --- Report Header ---
            st.header("Analysis Report :trophy:")
            st.divider()
            
            # --- Metric Cards (Animated Professional UI) ---
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.container(border=True).markdown(
                    f'<p class="metric-title">Overall Agreement</p><p class="agreement-score">{agreement_score:.1f}%</p>', 
                    unsafe_allow_html=True
                )
            
            with col2:
                st.metric(label="Synchronized Records", value=f"{len(merged_data):,}")
            
            with col3:
                missing_count = merged_data[['Survey Stress', 'Wearable Stress']].isnull().any(axis=1).sum()
                st.metric(label="Alignment Discrepancies", value=f"{missing_count:,}", delta=f"{missing_count / len(merged_data) * 100:.1f}% Missing", delta_color="inverse")

            with col4:
                st.metric(label="Survey Stress Levels", value=merged_data['Survey Stress'].nunique())


            st.divider()
            
            # --- Tabs for organization ---
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Agreement Score", "Data Distributions", "Biometric Response", "Correlation Map", "Data Preview", "Algorithm & Model"])

            with tab1:
                st.subheader("Classification Agreement :clipboard:")
                st.caption("The matrix shows how often the Wearable model's **Predicted** stress level matches the Survey's **Actual** (Ground Truth). Diagonal values are correct predictions.")
                st.pyplot(_plot_confusion_matrix(merged_data), use_container_width=True) 

            with tab2:
                st.subheader("Stress Level Distribution Comparison :chart_bar:")
                st.pyplot(_plot_distributions(merged_data), use_container_width=True)
            
            with tab3:
                box_fig = _plot_biometric_boxplots(merged_data, ['EDA', 'TEMP'])
                if box_fig:
                    st.subheader("Biometric Response by Stress Level :heartpulse:")
                    st.pyplot(box_fig, use_container_width=True)
                else:
                    st.warning("Biometric Boxplots require 'EDA' and/or 'TEMP' columns.")
            
            with tab4:
                corr_fig = _plot_correlation_heatmap(merged_data)
                if corr_fig:
                    st.subheader("Inter-feature Correlation Map :link:")
                    st.pyplot(corr_fig, use_container_width=True)
                else:
                    st.info("Correlation Heatmap requires 'Total'/'total_score' column and at least one biometric column.")
                    
            with tab5:
                st.subheader("Raw Data Input Preview (First 5 Rows) :page_facing_up:")
                st.caption("This shows the state of the data immediately after upload, before the SFAA merges them.")
                
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    st.markdown("#### Survey Data Input")
                    st.dataframe(original_survey_df.head(), use_container_width=True)
                    st.caption(f"Total Rows: {len(original_survey_df):,}")

                with col_p2:
                    st.markdown("#### Wearable Data Input")
                    st.dataframe(original_wearable_df.head(), use_container_width=True)
                    st.caption(f"Total Rows: {len(original_wearable_df):,}")

                st.divider()
                st.markdown("### Final Synchronized Dataset (First 5 Rows)")
                st.caption("This DataFrame is the output of the SFAA, used for all subsequent analysis.")
                st.dataframe(merged_data.head(), use_container_width=True)

            with tab6:
                display_rf_model_explanation()
                st.divider()
                display_sfaa_explanation()
                st.divider()
                display_analysis_breakdown(merged_data)
        
        else:
            status_container.error("Analysis Failed during processing.")

    except Exception as e:
        progress_bar.empty()
        status_container.error(f"Analysis Error: {e}")
        st.warning("Please ensure your CSV files contain required columns like 'Stress_Level' (Survey) and 'Predicted Stress' (Wearable).")