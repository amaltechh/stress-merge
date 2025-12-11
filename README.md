# 🚀 Stress Fusion AI Analyzer: Multi-Modal Data Synchronization

**4th Year Main Project**

| Team Member | Role |
| :--- | :--- |
| **Amal Benny Joseph** | **Team Lead , streamlit and model developer** |
| **Adarsh K Sundaresan** | SFAA Implementation, Core Development |
| **Alstin Gloria Chacko** | Data Pre-processing, Model Validation |
| **Akhil S Nambiar** | Documentation, Visualization Logic |

## 🌟 Project Goal

The primary objective of the **Stress Fusion AI Analyzer** is to bridge the gap between subjective stress data (self-reported surveys) and objective stress data (high-frequency wearable biometrics). The project implements a robust framework to synchronize these disparate data streams and provide quantitative validation of external stress prediction models.

## 💡 The Problem Solved

Traditional stress analysis is challenged by two factors:
1.  **Frequency Mismatch:** Surveys provide one data point over a long period, while wearables generate hundreds of sensor readings per minute. 
2.  **Validation Gap:** There is no easy way to formally assess if a machine learning model's "predicted stress" accurately reflects a human's "actual reported stress" during the same timeframe.

Our solution, built as an intuitive Streamlit web application, solves this through the **Stress Fusion Alignment Algorithm (SFAA)**.

## 🧠 Core Technical Architecture

The project is structured around three interconnected components: the External Prediction Model (validated), the Custom Alignment Algorithm, and the Interactive UI.

### 1. The Predictive Model (Assumed)

We validate the performance of an assumed external classification model, typically the **Random Forest Classifier**, which generates the `Predicted Stress` labels in the input wearable file.

#### Random Forest Classifier Explanation
The Random Forest is an **ensemble learning** method that is highly effective for stress classification due to its robustness against noisy physiological data. 

* **Mechanism (Majority Vote):** It builds multiple independent Decision Trees (a 'forest') from random subsets of the input features (EDA, TEMP, etc.). The final prediction (Low, Medium, or High Stress) is determined by the **class chosen by the majority of the trees**.
* **Suitability:** It provides stable, accurate predictions and is resilient to the outliers commonly found in raw sensor data.

### 2. The Synchronization Engine: Stress Fusion Alignment Algorithm (SFAA)

The SFAA is the original contribution of the project, handling the synchronization and fusion of data.

| SFAA Step | Purpose | Core Action |
| :--- | :--- | :--- |
| **1. Data Cleansing & Preparation** | Input validation. | Reads uploaded CSV files and drops redundant columns (e.g., Timestamps) to rely on implicit time ordering. |
| **2. Dynamic Aggregation (Core)** | **Frequency Synchronization.** | Divides the highly dense $N_{Wearable}$ data into $N_{Survey}$ equal time chunks ($\Delta t$). It applies aggregation rules to each chunk:
    * **Biometrics (EDA, TEMP):** Calculates the **Mean** to find the average physiological state.
    * **Labels (Predicted Stress):** Calculates the **Mode** (most frequent prediction) to identify the consensus stress level. |
| **3. Fusion & Normalization** | Create Unified Dataset. | Merges the aggregated Wearable data with the Survey data and standardizes all stress labels (e.g., 'Medium Stress' $\to$ 'Medium'). |

### 3. The Validation Interface (Streamlit)

The web application presents the analysis through several intuitive reports:

* **Agreement Score:** The central metric—the percentage of records where the **Wearable Prediction** matches the **Survey Report** (the true subjective state).
* **Confusion Matrix:** Provides granular validation, showing areas where the model tends to **misclassify** stress (e.g., predicting 'Low' when the user reports 'Medium'). 
* **Biometric Box Plots:** Visually maps the distribution of physiological signals (EDA, TEMP) across the three self-reported stress categories (Low, Medium, High). 
* **Correlation Heatmap:** Assesses the link between numerical survey scores and aggregated biometrics. 

## 🛠️ Setup and Installation

### Prerequisites

You need Python 3.8+ and the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit