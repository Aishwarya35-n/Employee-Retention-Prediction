"""
Data Loader Module
Handles loading and initial validation of the IBM HR Analytics dataset
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple


class DataLoader:
    """Class to handle data loading operations"""
    
    def __init__(self, data_path: str = 'data/WA_Fn-UseC_-HR-Employee-Attrition.csv'):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to the CSV file
        """
        self.data_path = data_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from CSV file
        
        Returns:
            DataFrame containing the employee data
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Please download it from Kaggle and place it in the data/ folder."
            )
        
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully! Shape: {self.df.shape}")
        
        return self.df
    
    def get_basic_info(self) -> dict:
        """
        Get basic information about the dataset
        
        Returns:
            Dictionary containing dataset statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates': self.df.duplicated().sum(),
            'attrition_distribution': self.df['Attrition'].value_counts().to_dict()
        }
        
        return info
    
    def print_summary(self):
        """Print a summary of the dataset"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        
        info = self.get_basic_info()
        
        print(f"\nDataset Shape: {info['shape'][0]} rows × {info['shape'][1]} columns")
        print(f"Missing Values: {info['missing_values']}")
        print(f"Duplicate Rows: {info['duplicates']}")
        
        print("\nAttrition Distribution:")
        for key, value in info['attrition_distribution'].items():
            percentage = (value / info['shape'][0]) * 100
            print(f"  {key}: {value} ({percentage:.2f}%)")
        
        print("\nData Types:")
        print(self.df.dtypes.value_counts())
        
        print("\nFirst few rows:")
        print(self.df.head())
        
        print("\nBasic Statistics:")
        print(self.df.describe())
        
        print("="*60 + "\n")
    
    def get_feature_types(self) -> Tuple[list, list]:
        """
        Identify numerical and categorical features
        
        Returns:
            Tuple of (numerical_features, categorical_features)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        numerical_features = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable from features if present
        if 'Attrition' in categorical_features:
            categorical_features.remove('Attrition')
        
        return numerical_features, categorical_features


# Example usage
if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_data()
    loader.print_summary()
    
    num_features, cat_features = loader.get_feature_types()
    print(f"\nNumerical features ({len(num_features)}): {num_features[:5]}...")
    print(f"Categorical features ({len(cat_features)}): {cat_features}")