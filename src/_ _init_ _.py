import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# 从共享模块导入
from src.preprocessing import engineer_features

# 1. 加载数据
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

y = (df['Churn'] == 'Yes').astype(int)
X = df.drop(columns=['Churn'])

# 2. 确定特征列表
sample_transformed = engineer_features(X)
numeric_features = ['tenure', 'MonthlyCharges', 'TotalServices', 'AvgMonthlyCharges', 'ChargeRatio']
categorical_features = [col for col in sample_transformed.columns if col not in numeric_features]

# 3. 构造预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# 4. 组装 Pipeline
full_pipeline = Pipeline(steps=[
    ('feature_engineering', FunctionTransformer(engineer_features)),
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200, 
        max_depth=8, 
        min_samples_split=10, 
        class_weight='balanced', 
        random_state=42
    ))
])

# 5. 训练与持久化
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

full_pipeline.fit(X_train, y_train)

os.makedirs('model', exist_ok=True)
joblib.dump(full_pipeline, 'model/churn_pipeline.joblib')
print("Model re-exported successfully with modularized src.preprocessing!")