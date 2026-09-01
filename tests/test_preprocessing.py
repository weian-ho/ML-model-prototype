import pandas as pd
import numpy as np
from src.preprocessing import engineer_features

def test_engineer_features_creates_expected_columns():
    """Verify that derived features are generated and TotalCharges is dropped"""
    raw_df = pd.DataFrame([{
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.0,
        'TotalCharges': 840.0
    }])
    
    transformed_df = engineer_features(raw_df)
    
    # 1. Verify that target derived columns are created
    assert 'TotalServices' in transformed_df.columns
    assert 'AvgMonthlyCharges' in transformed_df.columns
    assert 'ChargeRatio' in transformed_df.columns
    
    # 2. Verify that highly collinear columns are dropped
    assert 'TotalCharges' not in transformed_df.columns
    
    # 3. Verify numerical computation accuracy
    # OnlineBackup='Yes' -> TotalServices should equal 1
    assert transformed_df['TotalServices'].iloc[0] == 1
    # AvgMonthlyCharges = 840.0 / (12 + 1) ≈ 64.615
    assert np.isclose(transformed_df['AvgMonthlyCharges'].iloc[0], 840.0 / 13.0)

def test_engineer_features_zero_tenure_handling():
    """Verify that tenure=0 does not trigger ZeroDivisionError, Inf, or NaN"""
    raw_df = pd.DataFrame([{
        'MultipleLines': 'No',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'tenure': 0,
        'MonthlyCharges': 50.0,
        'TotalCharges': 0.0
    }])
    
    transformed_df = engineer_features(raw_df)
    assert not np.isnan(transformed_df['AvgMonthlyCharges'].iloc[0])
    assert not np.isinf(transformed_df['AvgMonthlyCharges'].iloc[0])
    assert transformed_df['AvgMonthlyCharges'].iloc[0] == 0.0