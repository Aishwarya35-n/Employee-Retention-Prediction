"""
Machine Learning Models Module
Implements Logistic Regression, Decision Tree, Random Forest, and XGBoost
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)
import joblib
import os


class EmployeeRetentionModels:
    """Class to train and manage multiple ML models"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.predictions = {}
        self.performance = {}
        
    def initialize_models(self):
        """Initialize all models with default parameters"""
        print("\nInitializing models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=6,
                learning_rate=0.1,
                eval_metric='logloss',
                use_label_encoder=False
            )
        }
        
        print(f"✓ Initialized {len(self.models)} models")
        
    def train_models(self, X_train, y_train):
        """
        Train all models
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            print(f"✓ {name} trained successfully!")
        
        print("\n" + "="*60)
        print("ALL MODELS TRAINED!")
        print("="*60 + "\n")
    
    def predict(self, X_test):
        """
        Generate predictions for all models
        
        Args:
            X_test: Test features
        """
        print("\nGenerating predictions...")
        
        for name, model in self.models.items():
            self.predictions[name] = {
                'y_pred': model.predict(X_test),
                'y_pred_proba': model.predict_proba(X_test)[:, 1]
            }
        
        print(f"✓ Predictions generated for {len(self.models)} models")
    
    def evaluate_models(self, y_test):
        """
        Evaluate all models
        
        Args:
            y_test: True labels
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        for name in self.models.keys():
            y_pred = self.predictions[name]['y_pred']
            y_pred_proba = self.predictions[name]['y_pred_proba']
            
            # Calculate metrics
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
            }
            
            self.performance[name] = metrics
            
            # Print results
            print(f"\n{name}:")
            print("-" * 40)
            for metric, value in metrics.items():
                print(f"  {metric:12s}: {value:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\n  Confusion Matrix:")
            print(f"    TN: {cm[0, 0]:4d}  FP: {cm[0, 1]:4d}")
            print(f"    FN: {cm[1, 0]:4d}  TP: {cm[1, 1]:4d}")
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print("="*60 + "\n")
    
    def get_performance_summary(self) -> pd.DataFrame:
        """
        Get performance summary as DataFrame
        
        Returns:
            DataFrame with all model performances
        """
        return pd.DataFrame(self.performance).T
    
    def get_best_model(self, metric: str = 'F1-Score') -> tuple:
        """
        Get the best performing model
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, model_object, score)
        """
        if not self.performance:
            raise ValueError("Models haven't been evaluated yet!")
        
        best_name = max(self.performance.keys(), 
                       key=lambda x: self.performance[x][metric])
        best_score = self.performance[best_name][metric]
        best_model = self.models[best_name]
        
        print(f"\n🏆 Best Model: {best_name}")
        print(f"   {metric}: {best_score:.4f}")
        
        return best_name, best_model, best_score
    
    def save_models(self, output_dir: str = 'outputs/models'):
        """
        Save all trained models
        
        Args:
            output_dir: Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving models to {output_dir}...")
        
        for name, model in self.models.items():
            filename = f"{name.replace(' ', '_').lower()}.pkl"
            filepath = os.path.join(output_dir, filename)
            joblib.dump(model, filepath)
            print(f"  ✓ Saved: {filename}")
        
        # Save performance metrics
        perf_df = self.get_performance_summary()
        perf_df.to_csv(os.path.join(output_dir, 'model_performance.csv'))
        print(f"  ✓ Saved: model_performance.csv")
        
        print("✓ All models saved!")
    
    def load_model(self, model_path: str):
        """
        Load a saved model
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        return joblib.load(model_path)
    
    def print_classification_report(self, y_test, model_name: str = None):
        """
        Print detailed classification report
        
        Args:
            y_test: True labels
            model_name: Name of specific model (if None, print all)
        """
        if model_name:
            models_to_report = {model_name: self.models[model_name]}
        else:
            models_to_report = self.models
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORTS")
        print("="*60)
        
        for name in models_to_report.keys():
            y_pred = self.predictions[name]['y_pred']
            
            print(f"\n{name}:")
            print("-" * 60)
            print(classification_report(y_test, y_pred, 
                                       target_names=['Retained', 'Left']))
        
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # This would normally be called from main.py with actual data
    print("Model training module ready!")
    print("Import this module and use EmployeeRetentionModels class")