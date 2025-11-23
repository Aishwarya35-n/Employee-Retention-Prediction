"""
Data Preprocessing Module
Handles data cleaning, encoding, normalization, and class balancing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple


class DataPreprocessor:
    """Class to handle all preprocessing operations"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize preprocessor
        
        Args:
            df: Input DataFrame
        """
        self.df = df.copy()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Check and handle missing values
        
        Returns:
            DataFrame with no missing values
        """
        print("\nChecking for missing values...")
        missing = self.df.isnull().sum()
        
        if missing.sum() == 0:
            print("✓ No missing values found!")
        else:
            print(f"Missing values found:\n{missing[missing > 0]}")
            # Fill numerical with median, categorical with mode
            for col in self.df.columns:
                if self.df[col].isnull().sum() > 0:
                    if self.df[col].dtype in ['int64', 'float64']:
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    else:
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            print("✓ Missing values handled!")
        
        return self.df
    
    def remove_irrelevant_features(self) -> pd.DataFrame:
        """
        Remove features that don't contribute to prediction
        
        Returns:
            DataFrame with relevant features only
        """
        print("\nRemoving irrelevant features...")
        
        # Features to remove (as per methodology)
        irrelevant_features = ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
        
        features_to_drop = [f for f in irrelevant_features if f in self.df.columns]
        
        if features_to_drop:
            self.df.drop(columns=features_to_drop, inplace=True)
            print(f"✓ Removed features: {features_to_drop}")
        else:
            print("✓ No irrelevant features to remove")
        
        return self.df
    
    def encode_target_variable(self) -> pd.DataFrame:
        """
        Encode the target variable 'Attrition' to binary (0/1)
        
        Returns:
            DataFrame with encoded target
        """
        print("\nEncoding target variable...")
        
        if 'Attrition' in self.df.columns:
            # Convert Yes -> 1, No -> 0
            self.df['Attrition'] = self.df['Attrition'].map({'Yes': 1, 'No': 0})
            print(f"✓ Target encoded: No=0, Yes=1")
            print(f"  Class distribution: {self.df['Attrition'].value_counts().to_dict()}")
        
        return self.df
    
    def encode_categorical_features(self) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding and One-Hot Encoding
        
        Returns:
            DataFrame with encoded categorical features
        """
        print("\nEncoding categorical features...")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        if not categorical_cols:
            print("✓ No categorical features to encode")
            return self.df
        
        # Use Label Encoding for binary features
        binary_features = [col for col in categorical_cols 
                          if self.df[col].nunique() == 2]
        
        # Use One-Hot Encoding for multi-class features
        multi_class_features = [col for col in categorical_cols 
                               if self.df[col].nunique() > 2]
        
        # Label Encoding
        for col in binary_features:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
            print(f"  Label encoded: {col}")
        
        # One-Hot Encoding
        if multi_class_features:
            self.df = pd.get_dummies(self.df, columns=multi_class_features, 
                                     drop_first=True, dtype=int)
            print(f"  One-hot encoded: {multi_class_features}")
        
        print(f"✓ Encoding complete! New shape: {self.df.shape}")
        
        return self.df
    
    def normalize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Normalized DataFrame
        """
        print("\nNormalizing numerical features...")
        
        X_scaled = self.scaler.fit_transform(X)
        X_normalized = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        print("✓ Features normalized!")
        
        return X_normalized
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split data into train and test sets
        
        Args:
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print(f"\nSplitting data (test_size={test_size})...")
        
        X = self.df.drop('Attrition', axis=1)
        y = self.df['Attrition']
        
        self.feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✓ Train set: {X_train.shape[0]} samples")
        print(f"✓ Test set: {X_test.shape[0]} samples")
        print(f"  Train class distribution: {y_train.value_counts().to_dict()}")
        print(f"  Test class distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def handle_class_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series,
                               method: str = 'smote') -> Tuple:
        """
        Handle class imbalance using SMOTE or random oversampling
        
        Args:
            X_train: Training features
            y_train: Training labels
            method: 'smote' or 'random'
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        print(f"\nHandling class imbalance using {method.upper()}...")
        print(f"  Before: {y_train.value_counts().to_dict()}")
        
        if method == 'smote':
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        else:
            raise ValueError("Only 'smote' method is currently supported")
        
        print(f"  After: {pd.Series(y_resampled).value_counts().to_dict()}")
        print("✓ Class imbalance handled!")
        
        return X_resampled, y_resampled
    
    def preprocess_pipeline(self, test_size: float = 0.2, 
                           handle_imbalance: bool = True) -> Tuple:
        """
        Complete preprocessing pipeline
        
        Args:
            test_size: Test set proportion
            handle_imbalance: Whether to apply SMOTE
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        print("\n" + "="*60)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Handle missing values
        self.handle_missing_values()
        
        # Step 2: Remove irrelevant features
        self.remove_irrelevant_features()
        
        # Step 3: Encode target variable
        self.encode_target_variable()
        
        # Step 4: Encode categorical features
        self.encode_categorical_features()
        
        # Step 5: Split data
        X_train, X_test, y_train, y_test = self.split_data(test_size=test_size)
        
        # Step 6: Normalize features
        X_train = self.normalize_features(X_train)
        X_test = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Step 7: Handle class imbalance (only on training data)
        if handle_imbalance:
            X_train, y_train = self.handle_class_imbalance(X_train, y_train)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE!")
        print("="*60)
        print(f"Final training set shape: {X_train.shape}")
        print(f"Final test set shape: {X_test.shape}")
        print(f"Number of features: {len(self.feature_names)}\n")
        
        return X_train, X_test, y_train, y_test, self.feature_names


# Example usage
if __name__ == "__main__":
    from data_loader import DataLoader
    
    loader = DataLoader()
    df = loader.load_data()
    
    preprocessor = DataPreprocessor(df)
    X_train, X_test, y_train, y_test, features = preprocessor.preprocess_pipeline()
    
    print(f"Features ({len(features)}): {features[:10]}...")