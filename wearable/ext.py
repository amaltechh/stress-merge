import pandas as pd
import numpy as np
import joblib
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

# Import visualization and advanced metrics libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# --- Step 1: Professional Configuration & Logging Setup ---

# Suppress warnings for a cleaner console output
warnings.filterwarnings('ignore')

def setup_logging():
    """Configures the root logger for the project."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

logger = setup_logging()

@dataclass(frozen=True)
class ModelConfig:
    """A dataclass to hold all model configuration parameters."""
    pickle_path: Path = Path("S8.pkl")
    model_path: Path = Path("trained_random_forest_model.pkl")
    results_dir: Path = Path("evaluation_results") # Directory for charts and reports
    features: tuple = ('cecg', 'cemg', 'ceda', 'ctemp', 'cresp', 'cax', 'cay', 'caz')
    target: str = 'label'
    class_labels: dict = {1.0: 'Low Stress', 2.0: 'Medium Stress', 3.0: 'High Stress'}
    random_state: int = 42
    test_size: float = 0.2

# --- Step 2: Object-Oriented Model Trainer ---

class StressModelTrainer:
    """
    A class to encapsulate the entire model training and evaluation pipeline.
    """
    def __init__(self, config: ModelConfig):
        """Initializes the trainer with a configuration object."""
        self.config = config
        self.data = None
        self.model = None
        # Create the results directory if it doesn't exist
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info("StressModelTrainer initialized and results directory prepared.")

    def _load_data(self):
        """Loads and structures data from the source pickle file."""
        logger.info(f"Loading data from '{self.config.pickle_path}'...")
        # ... (loading logic remains the same)
        if not self.config.pickle_path.exists():
            logger.error(f"Data file not found at '{self.config.pickle_path}'!")
            raise FileNotFoundError
        
        raw_data = pd.read_pickle(self.config.pickle_path)
        
        signal_data = {
            'cax': raw_data['signal']['chest']['ACC'][:, 0],
            'cay': raw_data['signal']['chest']['ACC'][:, 1],
            'caz': raw_data['signal']['chest']['ACC'][:, 2],
            'cecg': raw_data['signal']['chest']['ECG'][:, 0],
            'cemg': raw_data['signal']['chest']['EMG'][:, 0],
            'ceda': raw_data['signal']['chest']['EDA'][:, 0],
            'ctemp': raw_data['signal']['chest']['Temp'][:, 0],
            'cresp': raw_data['signal']['chest']['Resp'][:, 0],
            'label': raw_data['label']
        }
        self.data = pd.DataFrame(signal_data)
        logger.info("Data loaded and structured successfully.")

    def _preprocess_data(self):
        """Filters and re-categorizes stress levels using signal percentiles."""
        logger.info("Preprocessing data to create Low/Medium/High stress labels...")
        # ... (preprocessing logic remains the same)
        stress_df = self.data[self.data['label'] == 2].copy()
        
        percentiles = {feat: np.percentile(stress_df[feat], [35, 60]) for feat in self.config.features}
        
        for feature in self.config.features:
            lower, upper = percentiles[feature]
            stress_df[f'{feature}_cat'] = np.select(
                [stress_df[feature] <= lower, stress_df[feature] <= upper],
                [1, 2], default=3
            )
            
        category_cols = [f'{feature}_cat' for feature in self.config.features]
        stress_df['avg_cat'] = stress_df[category_cols].mean(axis=1)
        
        stress_df[self.config.target] = np.select(
            [stress_df['avg_cat'] <= 1.5, stress_df['avg_cat'] <= 2.5],
            [1.0, 2.0], default=3.0
        )
        
        self.data = stress_df[list(self.config.features) + [self.config.target]].copy()
        logger.info("Preprocessing complete.")
        logger.info(f"New label distribution:\n{self.data[self.config.target].value_counts(normalize=True)}")

    def _train_and_tune_model(self):
        """Splits data, defines a pipeline, and finds the best model using GridSearchCV."""
        logger.info("Starting model training and hyperparameter tuning...")
        X = self.data[list(self.config.features)]
        y = self.data[self.config.target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state, stratify=y
        )
        
        pipeline = ImbPipeline([
            ('undersampler', RandomUnderSampler(random_state=self.config.random_state)),
            ('classifier', RandomForestClassifier(random_state=self.config.random_state, class_weight='balanced'))
        ])
        
        param_grid = {
            'classifier__n_estimators': [100, 150],
            'classifier__max_depth': [20, None],
            'classifier__min_samples_leaf': [1, 2]
        }
        
        search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1, scoring='f1_weighted')
        logger.info("Fitting GridSearchCV... This may take a while.")
        search.fit(X_train, y_train)
        
        self.model = search.best_estimator_
        logger.info(f"Best parameters found: {search.best_params_}")
        
        self._generate_evaluation_report(X_train, y_train, X_test, y_test)

    def _generate_evaluation_report(self, X_train, y_train, X_test, y_test):
        """Generates a complete report with metrics, charts, and properties."""
        logger.info("Generating full evaluation report...")
        y_pred = self.model.predict(X_test)
        
        # --- Metrics Calculation ---
        report_str = classification_report(y_test, y_pred, target_names=self.config.class_labels.values())
        kappa = cohen_kappa_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # --- Save Text Report ---
        report_path = self.config.results_dir / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write("--- Model Evaluation Report ---\n\n")
            f.write(f"Best Parameters Found: {self.model.named_steps['classifier'].get_params()}\n\n")
            f.write("--- Classification Report ---\n")
            f.write(report_str)
            f.write("\n\n--- Additional Metrics ---\n")
            f.write(f"Cohen's Kappa Score: {kappa:.4f}\n")
            f.write(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}\n")
        logger.info(f"Text report saved to {report_path}")
            
        # --- Generate and Save Visualizations ---
        self._plot_confusion_matrix(y_test, y_pred)
        self._plot_feature_importance()
        self._plot_learning_curves(X_train, y_train)

    def _plot_confusion_matrix(self, y_true, y_pred):
        """Creates and saves a heatmap of the confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.config.class_labels.values(), 
                    yticklabels=self.config.class_labels.values())
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        save_path = self.config.results_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Confusion matrix plot saved to {save_path}")

    def _plot_feature_importance(self):
        """Creates and saves a bar chart of model feature importances."""
        importances = self.model.named_steps['classifier'].feature_importances_
        feature_df = pd.DataFrame({
            'Feature': self.config.features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        save_path = self.config.results_dir / "feature_importance.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Feature importance plot saved to {save_path}")

    def _plot_learning_curves(self, X_train, y_train):
        """Generates and saves learning curves for the model."""
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=self.model, X=X_train, y=y_train, cv=3, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5), scoring="f1_weighted"
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.title('Learning Curves')
        plt.xlabel('Training Examples')
        plt.ylabel('F1 Score (Weighted)')
        plt.legend(loc="best")
        plt.grid()
        save_path = self.config.results_dir / "learning_curve.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Learning curves plot saved to {save_path}")

    def _save_model(self):
        """Saves the final trained model to a file."""
        logger.info(f"Saving final model to '{self.config.model_path}'...")
        joblib.dump(self.model, self.config.model_path)
        logger.info("Model saved successfully. ✨")

    def run_pipeline(self):
        """Executes the full training pipeline in sequence."""
        try:
            self._load_data()
            self._preprocess_data()
            self._train_and_tune_model()
            self._save_model()
            logger.info("--- Training Pipeline Completed Successfully ---")
        except Exception as e:
            logger.exception(f"An error occurred during the pipeline execution: {e}")

# --- Step 3: Clean Main Execution Block ---

if __name__ == "__main__":
    config = ModelConfig()
    trainer = StressModelTrainer(config=config)
    trainer.run_pipeline()