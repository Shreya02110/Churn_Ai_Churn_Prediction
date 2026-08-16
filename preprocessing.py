import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────
df = pd.read_csv('C:/Users/LENOVO/Desktop/Customer_Churn_Project/data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print("Dataset loaded! Shape:", df.shape)

# ══════════════════════════════════════════════════════
# STEP 1 — Basic Cleaning
# ══════════════════════════════════════════════════════
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)

print("\n=== After Cleaning ===")
print("Missing values:", df.isnull().sum().sum())
print("Shape:", df.shape)

# ══════════════════════════════════════════════════════
# STEP 2 — Feature Engineering
# ══════════════════════════════════════════════════════
df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
df['IsHighValue'] = (df['MonthlyCharges'] > 65).astype(int)
df['SeniorNoSupport'] = (
    (df['SeniorCitizen'] == 1) &
    (df['TechSupport'] == 'No')
).astype(int)

services = ['PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']

def count_services(row):
    count = 0
    for col in services:
        if row[col] not in ['No', 'No internet service', 'No phone service']:
            count += 1
    return count

df['TotalServices'] = df.apply(count_services, axis=1)

print("\n=== New Features Added ===")
print(df[['AvgMonthlySpend','IsHighValue',
          'SeniorNoSupport','TotalServices']].head())

# ══════════════════════════════════════════════════════
# STEP 3 — Encode Categorical Columns
# ══════════════════════════════════════════════════════
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

binary_cols = ['gender', 'Partner', 'Dependents',
               'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0,
                           'Male': 1, 'Female': 0})

multi_cols = ['MultipleLines', 'InternetService',
              'OnlineSecurity', 'OnlineBackup',
              'DeviceProtection', 'TechSupport',
              'StreamingTV', 'StreamingMovies',
              'Contract', 'PaymentMethod']

le = LabelEncoder()
for col in multi_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Check karo koi NaN toh nahi
print("\n=== After Encoding ===")
print("Any NaN?", df.isnull().sum().sum())
print(df.dtypes)

# ══════════════════════════════════════════════════════
# STEP 4 — Train / Test Split
# ══════════════════════════════════════════════════════
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n=== Train/Test Split ===")
print(f"Training set   : {X_train.shape}")
print(f"Testing set    : {X_test.shape}")
print(f"Churn in train : {y_train.mean()*100:.1f}%")
print(f"Churn in test  : {y_test.mean()*100:.1f}%")

# ══════════════════════════════════════════════════════
# STEP 5 — Feature Scaling
# ══════════════════════════════════════════════════════
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n=== Scaling Done ===")
print("Any NaN after scaling?", np.isnan(X_train_scaled).sum())
print("Mean of scaled train:", round(X_train_scaled.mean(), 4))

# ══════════════════════════════════════════════════════
# STEP 6 — Save Everything
# ══════════════════════════════════════════════════════
save_path  = 'C:/Users/LENOVO/Desktop/Customer_Churn_Project/data/'
model_path = 'C:/Users/LENOVO/Desktop/Customer_Churn_Project/models/'

os.makedirs(save_path,  exist_ok=True)
os.makedirs(model_path, exist_ok=True)

pd.DataFrame(X_train_scaled,
             columns=X_train.columns).to_csv(
    save_path + 'X_train.csv', index=False)
pd.DataFrame(X_test_scaled,
             columns=X_test.columns).to_csv(
    save_path + 'X_test.csv', index=False)
y_train.to_csv(save_path + 'y_train.csv', index=False)
y_test.to_csv(save_path + 'y_test.csv',  index=False)

joblib.dump(scaler,               model_path + 'scaler.pkl')
joblib.dump(X_train.columns.tolist(), model_path + 'feature_names.pkl')

print("\n" + "="*45)
print("="*45)
print("  Files saved:")
print("  ✓ X_train.csv")
print("  ✓ X_test.csv")
print("  ✓ y_train.csv")
print("  ✓ y_test.csv")
print("  ✓ scaler.pkl")
print("  ✓ feature_names.pkl")
