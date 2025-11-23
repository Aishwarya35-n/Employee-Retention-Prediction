"""
Main Script for Employee Retention Prediction Project
Orchestrates the complete pipeline from data loading to model interpretation
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from data_loader import DataLoader
from preprocessing import DataPreprocessor
from models import EmployeeRetentionModels
from explainability import ModelExplainer

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print(" "*10 + "EMPLOYEE RETENTION PREDICTION PROJECT")
    print(" "*15 + "Using Artificial Intelligence")
    print("="*70 + "\n")
    
    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    print("\n📁 STEP 1: LOADING DATA")
    print("-" * 70)
    
    try:
        loader = DataLoader()
        df = loader.load_data()
        loader.print_summary()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset")
        print("And place it in the data/ folder\n")
        return
    
    # =========================================================================
    # STEP 2: PREPROCESS DATA
    # =========================================================================
    print("\n🔧 STEP 2: PREPROCESSING DATA")
    print("-" * 70)
    
    preprocessor = DataPreprocessor(df)
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline(
        test_size=0.2,
        handle_imbalance=True
    )
    
    # =========================================================================
    # STEP 3: TRAIN MODELS
    # =========================================================================
    print("\n🤖 STEP 3: TRAINING MACHINE LEARNING MODELS")
    print("-" * 70)
    
    model_trainer = EmployeeRetentionModels(random_state=42)
    model_trainer.initialize_models()
    model_trainer.train_models(X_train, y_train)
    
    # =========================================================================
    # STEP 4: EVALUATE MODELS
    # =========================================================================
    print("\n📊 STEP 4: EVALUATING MODELS")
    print("-" * 70)
    
    model_trainer.predict(X_test)
    model_trainer.evaluate_models(y_test)
    
    # Get performance summary
    performance_df = model_trainer.get_performance_summary()
    print("\n📈 Performance Summary:")
    print(performance_df.round(4))
    
    # Get best model
    best_name, best_model, best_score = model_trainer.get_best_model(metric='F1-Score')
    
    # Print detailed classification report
    model_trainer.print_classification_report(y_test)
    
    # Save models
    model_trainer.save_models()
    
    # =========================================================================
    # STEP 5: MODEL INTERPRETATION (EXPLAINABLE AI)
    # =========================================================================
    print("\n🔍 STEP 5: INTERPRETING MODEL WITH EXPLAINABLE AI")
    print("-" * 70)
    
    # Create explainer for best model
    explainer = ModelExplainer(
        model=best_model,
        model_name=best_name,
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names
    )
    
    # Feature Importance
    print("\n📌 Feature Importance Analysis")
    explainer.plot_feature_importance(
        top_n=15,
        save_path='outputs/figures/feature_importance.png'
    )
    
    # SHAP Analysis
    print("\n🎯 SHAP Analysis")
    explainer.calculate_shap_values(background_samples=100)
    
    explainer.plot_shap_summary(
        save_path='outputs/figures/shap_summary.png'
    )
    
    # SHAP Waterfall for sample predictions
    print("\n💧 SHAP Waterfall Plots (Individual Predictions)")
    for i in [0, 10, 20]:
        explainer.plot_shap_waterfall(
            instance_idx=i,
            save_path=f'outputs/figures/shap_waterfall_instance_{i}.png'
        )
    
    # =========================================================================
    # STEP 6: GENERATE MANAGERIAL INSIGHTS
    # =========================================================================
    print("\n💼 STEP 6: GENERATING MANAGERIAL INSIGHTS")
    print("-" * 70)
    
    explainer.generate_managerial_insights(top_n=10)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print(" "*20 + "PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📂 Outputs saved in:")
    print("   • outputs/models/ - Trained models")
    print("   • outputs/figures/ - Visualizations")
    print("   • outputs/reports/ - Analysis reports")
    
    print("\n🏆 Best Model:", best_name)
    print(f"   F1-Score: {best_score:.4f}")
    
    print("\n💡 Next Steps:")
    print("   1. Review the visualizations in outputs/figures/")
    print("   2. Implement recommended retention strategies")
    print("   3. Monitor employees using the predictive model")
    print("   4. Conduct regular model retraining with new data")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Create output directories
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)
    
    # Run main pipeline
    main()