from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import base64
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Paths ──────────────────────────────────────────────
BASE_PATH  = 'C:/Users/LENOVO/Desktop/Customer_Churn_Project/'
MODEL_PATH = BASE_PATH + 'models/'
DATA_PATH  = BASE_PATH + 'data/'

# ── Load Model & Scaler ────────────────────────────────
model         = joblib.load(MODEL_PATH + 'best_model.pkl')
scaler        = joblib.load(MODEL_PATH + 'scaler.pkl')
feature_names = joblib.load(MODEL_PATH + 'feature_names.pkl')

print("✓ Model loaded successfully!")
print(f"✓ Features: {len(feature_names)}")

# ══════════════════════════════════════════════════════
# HELPER — Preprocess uploaded CSV
# ══════════════════════════════════════════════════════
def preprocess(df):
    from sklearn.preprocessing import LabelEncoder

    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    if 'Churn' in df.columns:
        df.drop('Churn', axis=1, inplace=True)

    # Feature Engineering
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['IsHighValue']     = (df['MonthlyCharges'] > 65).astype(int)
    df['SeniorNoSupport'] = (
        (df['SeniorCitizen'] == 1) &
        (df['TechSupport'] == 'No')
    ).astype(int)

    services = ['PhoneService','MultipleLines','InternetService',
                'OnlineSecurity','OnlineBackup','DeviceProtection',
                'TechSupport','StreamingTV','StreamingMovies']

    def count_services(row):
        count = 0
        for col in services:
            if row[col] not in ['No','No internet service',
                                 'No phone service']:
                count += 1
        return count

    df['TotalServices'] = df.apply(count_services, axis=1)

    # Binary encoding
    binary_cols = ['gender','Partner','Dependents',
                   'PhoneService','PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes':1,'No':0,
                               'Male':1,'Female':0})

    # Multi-category encoding
    le = LabelEncoder()
    multi_cols = ['MultipleLines','InternetService',
                  'OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport',
                  'StreamingTV','StreamingMovies',
                  'Contract','PaymentMethod']
    for col in multi_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    df = df.reindex(columns=feature_names, fill_value=0)
    df_scaled = scaler.transform(df)
    return df_scaled, df

# ══════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# ── EDA Graphs Route ───────────────────────────────────
@app.route('/eda-graphs')
def eda_graphs():
    graph_files = [
        ('plot1_churn_distribution.png', 'Churn Distribution'),
        ('plot2_contract_churn.png',     'Contract Type vs Churn'),
        ('plot3_tenure_charges.png',     'Tenure vs Monthly Charges'),
        ('plot4_monthly_dist.png',       'Monthly Charges Distribution'),
        ('plot5_internet_churn.png',     'Internet Service vs Churn'),
        ('plot6_model_comparison.png',   'Model Accuracy Comparison'),
        ('plot7_confusion_matrix.png',   'Confusion Matrix'),
        ('plot8_roc_curve.png',          'ROC Curve — All Models'),
        ('plot9_feature_importance.png', 'Top 15 Feature Importances'),
    ]
    graphs = []
    for filename, title in graph_files:
        path = DATA_PATH + filename
        if os.path.exists(path):
            with open(path, 'rb') as f:
                encoded = base64.b64encode(
                    f.read()).decode('utf-8')
            graphs.append({
                'title': title,
                'data':  encoded,
                'file':  filename,
            })
    return jsonify({'graphs': graphs})

# ── Predict Route ──────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        df_original = pd.read_csv(file)
        df_proc, _  = preprocess(df_original.copy())

        predictions   = model.predict(df_proc)
        probabilities = model.predict_proba(df_proc)[:, 1]

        total       = len(predictions)
        churned     = int(predictions.sum())
        not_churned = total - churned
        churn_rate  = round((churned / total) * 100, 2)

        # Per customer results
        id_col = df_original.get(
            'customerID',
            pd.Series(range(1, total + 1))
        )

        results = []
        for i in range(total):
            prob = float(probabilities[i])
            results.append({
                'id':          str(id_col.iloc[i])
                               if hasattr(id_col, 'iloc')
                               else str(i + 1),
                'prediction':  int(predictions[i]),
                'probability': round(prob * 100, 2),
                'risk':        'High'   if prob > 0.7
                          else 'Medium' if prob > 0.4
                          else 'Low',
            })

        # Contract distribution
        contract_data = {}
        if 'Contract' in df_original.columns:
            contract_data = df_original[
                'Contract'].value_counts().to_dict()

        # Tenure distribution
        tenure_data = {}
        if 'tenure' in df_original.columns:
            df_original['tenure_group'] = pd.cut(
                df_original['tenure'],
                bins=[0, 12, 24, 48, 72],
                labels=['0-12','13-24','25-48','49-72'])
            tenure_data = df_original[
                'tenure_group'].value_counts(
                sort=False).to_dict()

        return jsonify({
            'success':       True,
            'total':         total,
            'churned':       churned,
            'not_churned':   not_churned,
            'churn_rate':    churn_rate,
            'results':       results[:100],
            'contract_data': contract_data,
            'tenure_data':   {str(k): v
                              for k, v in tenure_data.items()},
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Stats Route ────────────────────────────────────────
@app.route('/stats')
def stats():
    try:
        df = pd.read_csv(
            DATA_PATH +
            'WA_Fn-UseC_-Telco-Customer-Churn.csv')
        df['TotalCharges'] = pd.to_numeric(
            df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)

        total       = len(df)
        churned     = int((df['Churn'] == 'Yes').sum())
        retained    = total - churned
        churn_rate  = round(churned / total * 100, 2)
        avg_tenure  = round(df['tenure'].mean(), 1)
        avg_charges = round(df['MonthlyCharges'].mean(), 2)

        # Churn rate by contract
        contract = df.groupby('Contract')['Churn'].apply(
            lambda x: round(
                (x == 'Yes').mean() * 100, 2)
        ).to_dict()

        # Churn rate by internet service
        internet = df.groupby('InternetService')['Churn'].apply(
            lambda x: round(
                (x == 'Yes').mean() * 100, 2)
        ).to_dict()

        # Churn rate by tenure group
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 48, 72],
            labels=['0-12m','13-24m','25-48m','49-72m'])
        tenure = df.groupby(
            'tenure_group',
            observed=True)['Churn'].apply(
            lambda x: round(
                (x == 'Yes').mean() * 100, 2)
        ).to_dict()

        # Monthly charges distribution
        bins   = [0, 20, 40, 60, 80, 100, 120]
        labels = ['0-20','20-40','40-60',
                  '60-80','80-100','100+']
        df['charge_group'] = pd.cut(
            df['MonthlyCharges'],
            bins=bins, labels=labels)
        charges_dist = df.groupby(
            'charge_group',
            observed=True)['Churn'].apply(
            lambda x: round(
                (x == 'Yes').mean() * 100, 2)
        ).to_dict()

        # Payment method churn
        payment = df.groupby('PaymentMethod')['Churn'].apply(
            lambda x: round(
                (x == 'Yes').mean() * 100, 2)
        ).to_dict()

        return jsonify({
            'total':        total,
            'churned':      churned,
            'retained':     retained,
            'churn_rate':   churn_rate,
            'avg_tenure':   avg_tenure,
            'avg_charges':  avg_charges,
            'contract':     contract,
            'internet':     internet,
            'tenure':       {str(k): v
                             for k, v in tenure.items()},
            'charges_dist': {str(k): v
                             for k, v in charges_dist.items()},
            'payment':      payment,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ══════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True, port=5000) 