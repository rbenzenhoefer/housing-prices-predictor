"""
Data preprocessing functions for House Prices dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def remove_outliers(df):
    """
    Entferne bekannte Outliers aus dem Training Set
    """
    df = df.copy()
    
    # Bekannte Outliers: Häuser mit sehr großer GrLivArea aber niedrigem Preis
    # IDs: 523 und 1298
    outlier_ids = [523, 1298]
    
    if 'Id' in df.columns:
        df = df[~df['Id'].isin(outlier_ids)]
        print(f"Deleted: {len(outlier_ids)} Outliers")
    
    return df


def handle_missing_values(df):
    """
    Behandle fehlende Werte basierend auf Feature-Bedeutung
    
    NA kann bedeuten:
    - "None" = Feature nicht vorhanden (Pool, Garage, etc.)
    - 0 = Numerischer Wert wenn Feature nicht vorhanden
    - Median/Mode = Echte fehlende Werte
    """
    df = df.copy()
    
    # 1. Features wo NA = "None" bedeutet (kategorisch)
    none_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]
    
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    
    # 2. Features wo NA = 0 bedeutet (numerisch)
    zero_cols = [
        'GarageYrBlt', 'GarageArea', 'GarageCars',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath',
        'MasVnrArea'
    ]
    
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 3. LotFrontage: Impute mit Median per Neighborhood
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
    
    # 4. Electrical: Most frequent (nur 1 fehlender Wert im Training Set)
    if 'Electrical' in df.columns:
        df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    
    # 5. Restliche kategorische Features: Most frequent
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # 6. Restliche numerische Features: Median
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    print(f"Missing Values handeled. Remaining NAs: {df.isnull().sum().sum()}")
    
    return df


def transform_target(y):
    """
    Log-Transformation der Target Variable (SalePrice)
    
    Warum? SalePrice ist rechtsschief (Skewness: 1.88)
    Log-Transformation macht es normalverteilt
    """
    return np.log1p(y)


def inverse_transform_target(y_log):
    """
    Rücktransformation: log(y) -> y
    """
    return np.expm1(y_log)


def encode_ordinal_features(df):
    """
    Encode ordinale Features mit natürlicher Reihenfolge
    """
    df = df.copy()
    
    # Quality/Condition Mappings
    quality_map = {
        'None': 0,
        'Po': 1,    # Poor
        'Fa': 2,    # Fair
        'TA': 3,    # Average/Typical
        'Gd': 4,    # Good
        'Ex': 5     # Excellent
    }
    
    quality_cols = [
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
        'HeatingQC', 'KitchenQual', 'FireplaceQu',
        'GarageQual', 'GarageCond', 'PoolQC'
    ]
    
    for col in quality_cols:
        if col in df.columns:
            df[col] = df[col].map(quality_map)
    
    # Basement Exposure
    bsmt_exposure_map = {
        'None': 0,
        'No': 1,
        'Mn': 2,    # Minimum
        'Av': 3,    # Average
        'Gd': 4     # Good
    }
    
    if 'BsmtExposure' in df.columns:
        df['BsmtExposure'] = df['BsmtExposure'].map(bsmt_exposure_map)
    
    # Basement Finish
    bsmt_finish_map = {
        'None': 0,
        'Unf': 1,   # Unfinished
        'LwQ': 2,   # Low Quality
        'Rec': 3,   # Average Rec Room
        'BLQ': 4,   # Below Average Living Quarters
        'ALQ': 5,   # Average Living Quarters
        'GLQ': 6    # Good Living Quarters
    }
    
    for col in ['BsmtFinType1', 'BsmtFinType2']:
        if col in df.columns:
            df[col] = df[col].map(bsmt_finish_map)
    
    # Garage Finish
    garage_finish_map = {
        'None': 0,
        'Unf': 1,   # Unfinished
        'RFn': 2,   # Rough Finished
        'Fin': 3    # Finished
    }
    
    if 'GarageFinish' in df.columns:
        df['GarageFinish'] = df['GarageFinish'].map(garage_finish_map)
    
    # Fence Quality
    fence_map = {
        'None': 0,
        'MnWw': 1,  # Minimum Wood/Wire
        'GdWo': 2,  # Good Wood
        'MnPrv': 3, # Minimum Privacy
        'GdPrv': 4  # Good Privacy
    }
    
    if 'Fence' in df.columns:
        df['Fence'] = df['Fence'].map(fence_map)
    
    return df


def preprocess_data(train_df, test_df, remove_outliers_flag=True):
    """
    Komplettes Preprocessing für Training und Test Sets
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        remove_outliers_flag: Ob Outliers entfernt werden sollen
        
    Returns:
        train_processed, test_processed, y_log (log-transformierter Target)
    """
    print("=" * 50)
    print("Start Preprocessing...")
    print("=" * 50)
    
    # 1. Outliers entfernen (nur Training)
    if remove_outliers_flag:
        train_df = remove_outliers(train_df)
    
    # 2. Target Variable separieren und log-transformieren
    y = train_df['SalePrice']
    y_log = transform_target(y)
    
    # IDs speichern
    train_ids = train_df['Id']
    test_ids = test_df['Id']
    
    # 3. Features vorbereiten (ohne SalePrice und Id)
    train_features = train_df.drop(['SalePrice', 'Id'], axis=1)
    test_features = test_df.drop(['Id'], axis=1)
    
    # 4. Fehlende Werte behandeln
    print("\n1. Handeling Missing Values...")
    train_features = handle_missing_values(train_features)
    test_features = handle_missing_values(test_features)
    
    # 5. Ordinale Features encoden
    print("\n2. Encoding Ordinal Features...")
    train_features = encode_ordinal_features(train_features)
    test_features = encode_ordinal_features(test_features)
    
    print("\n" + "=" * 50)
    print("Preprocessing Done!")
    print(f"Training Shape: {train_features.shape}")
    print(f"Test Shape: {test_features.shape}")
    print("=" * 50)
    
    return train_features, test_features, y_log, train_ids, test_ids