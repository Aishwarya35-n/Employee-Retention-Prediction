"""
Explainability Module
Implements SHAP analysis and feature importance visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from typing import List


class ModelExplainer:
    """Class to explain model predictions using SHAP and feature importance"""
    
    def __init__(self, model, model_name: str, X_train, X_test, 
                 feature_names: List[str]):
        """
        Initialize explainer
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_train: Training features
            X_test: Test features
            feature_names: List of feature names
        """
        self.model = model
        self.model_name = model_name
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.shap_values = None
        self.explainer = None
        
    def calculate_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Calculate feature importance
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        print(f"\nCalculating feature importance for {self.model_name}...")
        
        # Get feature importance based on model type
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            print("⚠ Model doesn't support feature importance")
            return None
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print(f"✓ Feature importance calculated!")
        
        return importance_df.head(top_n)
    
    def plot_feature_importance(self, top_n: int = 15, save_path: str = None):
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to plot
            save_path: Path to save the plot
        """
        importance_df = self.calculate_feature_importance(top_n)
        
        if importance_df is None:
            return
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, y='Feature', x='Importance', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - {self.model_name}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved: {save_path}")
        
        plt.show()
    
    def calculate_shap_values(self, background_samples: int = 100):
        """
        Calculate SHAP values
        
        Args:
            background_samples: Number of background samples for SHAP
        """
        print(f"\nCalculating SHAP values for {self.model_name}...")
        print("⏳ This may take a few minutes...")
        
        # Use a subset of training data as background
        if len(self.X_train) > background_samples:
            background = shap.sample(self.X_train, background_samples)
        else:
            background = self.X_train
        
        # Create explainer based on model type
        model_type = type(self.model).__name__
        
        if 'Tree' in model_type or 'Forest' in model_type or 'XGB' in model_type:
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(self.X_test)
        else:
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, background
            )
            self.shap_values = self.explainer.shap_values(self.X_test)
        
        # Handle different SHAP value formats
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Use positive class
        
        print("✓ SHAP values calculated!")
    
    def plot_shap_summary(self, save_path: str = None):
        """
        Plot SHAP summary plot
        
        Args:
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            print("⚠ Calculate SHAP values first!")
            return
        
        print("\nGenerating SHAP summary plot...")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            self.X_test,
            feature_names=self.feature_names,
            show=False
        )
        plt.title(f'SHAP Feature Importance - {self.model_name}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved: {save_path}")
        
        plt.show()
    
    def plot_shap_waterfall(self, instance_idx: int = 0, save_path: str = None):
        """
        Plot SHAP waterfall plot for a single prediction
        
        Args:
            instance_idx: Index of the instance to explain
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            print("⚠ Calculate SHAP values first!")
            return
        
        print(f"\nGenerating SHAP waterfall plot for instance {instance_idx}...")
        
        # Create explanation object
        if isinstance(self.X_test, pd.DataFrame):
            X_test_array = self.X_test.values
        else:
            X_test_array = self.X_test
        
        explanation = shap.Explanation(
            values=self.shap_values[instance_idx],
            base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            data=X_test_array[instance_idx],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False)
        plt.title(f'SHAP Explanation - Instance {instance_idx}', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved: {save_path}")
        
        plt.show()
    
    def get_top_shap_features(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get top features by mean absolute SHAP value
        
        Args:
            top_n: Number of top features
            
        Returns:
            DataFrame with top SHAP features
        """
        if self.shap_values is None:
            print("⚠ Calculate SHAP values first!")
            return None
        
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        shap_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_|SHAP|': mean_shap
        }).sort_values('Mean_|SHAP|', ascending=False)
        
        return shap_df.head(top_n)
    
    def generate_managerial_insights(self, top_n: int = 10):
        """
        Generate managerial insights from SHAP values
        
        Args:
            top_n: Number of top features to analyze
        """
        print("\n" + "="*60)
        print("MANAGERIAL INSIGHTS FROM AI MODEL")
        print("="*60)
        
        # Get feature importance
        importance_df = self.calculate_feature_importance(top_n)
        
        if importance_df is not None:
            print(f"\n📊 Top {top_n} Factors Influencing Employee Turnover:\n")
            
            for idx, row in importance_df.iterrows():
                feature = row['Feature']
                importance = row['Importance']
                print(f"  {idx+1}. {feature:30s} (Importance: {importance:.4f})")
        
        # Get SHAP insights if available
        if self.shap_values is not None:
            shap_df = self.get_top_shap_features(top_n)
            print(f"\n🎯 Top {top_n} Features by SHAP Impact:\n")
            
            for idx, row in shap_df.iterrows():
                feature = row['Feature']
                impact = row['Mean_|SHAP|']
                print(f"  {idx+1}. {feature:30s} (Impact: {impact:.4f})")
        
        print("\n💡 ACTIONABLE RECOMMENDATIONS:")
        print("-" * 60)
        print("""
  1. COMPENSATION & BENEFITS
     → Review salary structures for competitive positioning
     → Consider performance-based incentives
     → Improve stock option and benefits packages

  2. WORK-LIFE BALANCE
     → Implement flexible working arrangements
     → Monitor and reduce overtime requirements
     → Promote work-life balance initiatives

  3. JOB SATISFACTION & ENGAGEMENT
     → Conduct regular satisfaction surveys
     → Address workplace environment concerns
     → Improve manager-employee relationships

  4. CAREER DEVELOPMENT
     → Create clear career progression paths
     → Increase training and development opportunities
     → Implement mentorship programs

  5. EARLY WARNING SYSTEM
     → Monitor employees in high-risk categories
     → Conduct stay interviews with valuable employees
     → Implement proactive retention strategies
        """)
        
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    print("Explainability module ready!")
    print("Import this module and use ModelExplainer class")