import pandas as pd

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """"Core feature engineering function: automatically generates aggregate and ratio features, and drops collinear features"""
    df_out = data.copy()
    
# Ensure TotalCharges is numeric
    if 'TotalCharges' in df_out.columns:
        df_out['TotalCharges'] = pd.to_numeric(df_out['TotalCharges'], errors='coerce').fillna(0)
    
    # 1. Count total value-added services -> Generate TotalServices
    service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        if col in df_out.columns:
            df_out[col + '_flag'] = (df_out[col] == 'Yes').astype(int)
    
    flag_cols = [c + '_flag' for c in service_cols if c in df_out.columns]
    df_out['TotalServices'] = df_out[flag_cols].sum(axis=1)
    df_out = df_out.drop(columns=flag_cols)
    
    # 2. Calculate charge ratios -> Generate AvgMonthlyCharges and ChargeRatio
    df_out['AvgMonthlyCharges'] = df_out['TotalCharges'] / (df_out['tenure'] + 1)
    df_out['ChargeRatio'] = df_out['MonthlyCharges'] / (df_out['AvgMonthlyCharges'] + 1e-5)
    
    # 3. Drop highly collinear TotalCharges
    if 'TotalCharges' in df_out.columns:
        df_out = df_out.drop(columns=['TotalCharges'])
        
    return df_out