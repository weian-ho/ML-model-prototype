import pandas as pd

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """特征工程核心函数：自动生成聚合与比率特征，并剔除共线性特征"""
    df_out = data.copy()
    
# 确保 TotalCharges 为数值型
    if 'TotalCharges' in df_out.columns:
        df_out['TotalCharges'] = pd.to_numeric(df_out['TotalCharges'], errors='coerce').fillna(0)
    
    # 1. 增值服务总数统计 -> 生成 TotalServices
    service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        if col in df_out.columns:
            df_out[col + '_flag'] = (df_out[col] == 'Yes').astype(int)
    
    flag_cols = [c + '_flag' for c in service_cols if c in df_out.columns]
    df_out['TotalServices'] = df_out[flag_cols].sum(axis=1)
    df_out = df_out.drop(columns=flag_cols)
    
    # 2. 费用比率计算 -> 生成 AvgMonthlyCharges 和 ChargeRatio
    df_out['AvgMonthlyCharges'] = df_out['TotalCharges'] / (df_out['tenure'] + 1)
    df_out['ChargeRatio'] = df_out['MonthlyCharges'] / (df_out['AvgMonthlyCharges'] + 1e-5)
    
    # 3. 剔除高共线性的 TotalCharges
    if 'TotalCharges' in df_out.columns:
        df_out = df_out.drop(columns=['TotalCharges'])
        
    return df_out