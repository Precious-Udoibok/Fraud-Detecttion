import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

# Configuration
class ModelConfig:
    random_state = 42
    test_size = 0.2
    sample_fraction = 0.1
    n_jobs = -1
    cv_folds = 2
    lof_neighbors = 20
    model_output_dir = "models"

class FraudDetectionSystem:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.setup_logging()
        self.setup_output_directory()

    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )

    def setup_output_directory(self):
        """Create output directory for models"""
        Path(self.config.model_output_dir).mkdir(parents=True, exist_ok=True)

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and validate the dataset"""
        df = pd.read_csv(filepath)
        if not all(col in df.columns for col in ['Class', 'Amount']):
            raise ValueError("Missing required columns: 'Class' or 'Amount'")
        df = df.sample(frac=self.config.sample_fraction, random_state=self.config.random_state)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df

    def explore_data(self, df: pd.DataFrame):
        """Generate data exploration visualizations"""
        logging.info("Generating data exploration visualizations...")
        
        plt.figure(figsize=(15, 10))
        
        # Class Distribution
        plt.subplot(2, 2, 1)
        sns.countplot(x='Class', data=df)
        plt.title('Class Distribution')
        
        # Amount Distribution by Class
        plt.subplot(2, 2, 2)
        for label, color in zip([0, 1], ['green', 'red']):
            mask = df['Class'] == label
            plt.hist(df[mask]['Amount'], bins=50, alpha=0.5, label=f"{'Normal' if label == 0 else 'Fraud'}", color=color)
        plt.yscale('log')
        plt.title('Amount Distribution by Class')
        plt.legend()
        
        # Correlation Matrix Heatmap
        plt.subplot(2, 2, 3)
        sns.heatmap(df.corr(), cmap='coolwarm', center=0, xticklabels=False, yticklabels=False)
        plt.title('Feature Correlation Matrix')
        
        # Transaction Patterns Over ID
        plt.subplot(2, 2, 4)
        sns.scatterplot(data=df, x='id', y='Amount', hue='Class', alpha=0.5, size='Class')
        plt.title('Transaction Patterns Over ID')
        
        plt.tight_layout()
        plt.savefig('data_exploration.png')
        plt.close()

    def preprocess_data(self, df: pd.DataFrame):
        """Preprocess the data with feature engineering"""
        logging.info("Preprocessing data and engineering features...")
        
        # Feature engineering using 'id' instead of 'Time'
        df['Amount_Log'] = np.log1p(df['Amount'])
        df['Amount_Squared'] = df['Amount'] ** 2
        df['ID_Hour'] = df['id'] % 24  # Use 'id' for time-based features
        df['ID_Day'] = (df['id'] // 24) % 7
        
        X = df.drop(['Class', 'Time'] if 'Time' in df.columns else ['Class'], axis=1)
        y = df['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test, X.columns

    def train_supervised_model(self, X_train, X_test, y_train, y_test, feature_names):
        """Train supervised models"""
        logging.info("Training supervised models...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=self.config.random_state)),
            ('classifier', RandomForestClassifier(random_state=self.config.random_state))
        ])
        
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [4, 6, None],
            'classifier__max_features': ['sqrt', 'log2']
        }
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=self.config.cv_folds, scoring='f1', n_jobs=self.config.n_jobs, verbose=1
        )
        grid_search.fit(X_train, y_train)
        self._evaluate_and_save_model(grid_search, X_test, y_test, feature_names, 'supervised')

    def train_lof_model(self, X_train, X_test, y_train, y_test):
        """Train LOF model"""
        logging.info("Training LOF model...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        contamination = max(0.001, min(0.1, y_train.mean()))
        lof = LocalOutlierFactor(
            n_neighbors=self.config.lof_neighbors, contamination=contamination, novelty=True, n_jobs=self.config.n_jobs
        )
        lof.fit(X_train_scaled)
        self._evaluate_and_save_model(lof, X_test_scaled, y_test, None, 'lof', scaler=scaler)

    def _evaluate_and_save_model(self, model, X_test, y_test, feature_names, model_type, scaler=None):
        """Evaluate and save the model"""
        if model_type == 'lof':
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == -1, 1, 0)
            y_scores = -model.decision_function(X_test)
        else:
            y_pred = model.predict(X_test)
            y_scores = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = Path(self.config.model_output_dir) / f"{model_type}_model_{timestamp}"
        model_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, model_path / f"{model_type}_model.pkl")
        if scaler:
            joblib.dump(scaler, model_path / "scaler.pkl")
        if feature_names is not None:
            joblib.dump(feature_names.tolist(), model_path / "feature_names.pkl")
        
        self._save_model_visualizations(y_test, y_pred, y_scores, model_path, model_type)
        logging.info(f"{model_type.upper()} Model Evaluation:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")

    def _save_model_visualizations(self, y_test, y_pred, y_scores, model_path, model_type):
        """Save model evaluation visualizations"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_type.upper()} Model - Confusion Matrix')
        plt.savefig(model_path / "confusion_matrix.png")
        plt.close()
        
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_type.upper()} Model - ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(model_path / "roc_curve.png")
        plt.close()

def main():
    config = ModelConfig()
    fraud_detection = FraudDetectionSystem(config)
    
    df = fraud_detection.load_data('creditcard.csv')
    fraud_detection.explore_data(df)
    
    X_train, X_test, y_train, y_test, feature_names = fraud_detection.preprocess_data(df)
    fraud_detection.train_supervised_model(X_train, X_test, y_train, y_test, feature_names)
    fraud_detection.train_lof_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()