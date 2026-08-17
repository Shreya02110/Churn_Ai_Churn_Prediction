# Churn_Ai_Churn_PredictionChurnAI – Customer Churn Prediction & Retention Recommendation System

ChurnAI is a Machine Learning-based Customer Churn Prediction and Retention Recommendation System developed during a Data Science internship at Pratinik Infotech.

The project analyzes telecom customer data and predicts whether a customer is likely to churn. Along with the prediction, the system calculates churn probability, categorizes customers into Low, Medium, and High Risk, and provides analytical insights through an interactive web dashboard.

The system is built using Python, Pandas, NumPy, Scikit-learn, Random Forest, XGBoost, Flask, HTML, CSS, JavaScript, Chart.js, and Matplotlib/Seaborn.

 Key Features

- Customer churn prediction using Machine Learning
- Random Forest-based final prediction model
- Churn probability calculation
- Low, Medium, and High risk categorization
- CSV file upload for prediction
- Automated data preprocessing
- Feature engineering
- Exploratory Data Analysis (EDA)
- Interactive charts and analytics dashboard
- Model performance visualization
- Customer-level prediction results
- CSV export of prediction results
- Business-oriented retention insights

🔄 Project Workflow

Customer Data
↓
Data Preprocessing
↓
Feature Engineering
↓
Data Encoding & Scaling
↓
SMOTE Balancing
↓
Machine Learning Models
↓
Model Evaluation
↓
Best Model Selection
↓
Flask Deployment
↓
Churn Prediction & Risk Categorization
↓
Dashboard Visualization
↓
Business Decision Support

📊 Dataset

The project uses the IBM Telco Customer Churn Dataset containing telecom customer information such as:

- Customer demographics
- Tenure
- Contract type
- Internet service
- Payment method
- Monthly charges
- Total charges
- Additional services
- Churn status

The dataset is cleaned and transformed before being provided to the Machine Learning models.

🧹 Data Preprocessing

The preprocessing pipeline includes:

- Conversion of "TotalCharges" into numeric format
- Handling of missing values
- Removal of unnecessary "customerID"
- Binary encoding of categorical variables
- Label encoding of multi-category variables
- Feature engineering
- Train-test splitting
- StandardScaler normalization

Additional engineered features include:

- "AvgMonthlySpend"
- "IsHighValue"
- "SeniorNoSupport"
- "TotalServices"

🤖 Machine Learning Models

Three classification models were evaluated:

1. Logistic Regression

Used as a baseline classification model.

2. Random Forest

An ensemble learning algorithm based on multiple decision trees. It was selected as the final model based on the evaluation results.

3. XGBoost

A gradient boosting algorithm used for comparison with the other models.

SMOTE was applied to the training data to address class imbalance.

📈 Model Performance

The selected Random Forest model achieved:

Metric| Score
Accuracy| 76.62%
ROC-AUC| 82.82%

The model evaluation includes accuracy comparison, confusion matrix, and ROC curve analysis.

🎯 Prediction Output

For each customer, the system provides:

- Customer ID
- Churn Prediction
- Churn Probability
- Risk Level

Risk levels are determined from the predicted churn probability:

- High Risk: Probability > 70%
- Medium Risk: Probability > 40% and ≤ 70%
- Low Risk: Probability ≤ 40%

📊 Dashboard

The Flask dashboard presents important business insights such as:

- Total customers
- Churn rate
- Retained customers
- Average tenure
- Average monthly charges
- Churn rate by contract type
- Churn rate by internet service
- Churn rate by tenure group
- Model performance metrics

🖼️ Dashboard Preview

Add your dashboard screenshot here.
file:///C:/Users/LENOVO/Desktop/Customer_Churn_Project/Results_Output/ChurnAI%20-%20Customer%20Churn%20Prediction%20Final.html

📈 Exploratory Data Analysis

The project includes multiple visualizations for understanding customer behavior, including:

- Customer churn distribution
- Contract type vs churn
- Tenure vs monthly charges
- Monthly charges distribution
- Internet service vs churn
- Model accuracy comparison
- Confusion matrix
- ROC curve
- Feature important.

Project Architecture

The application follows a layered architecture:

Presentation Layer

- HTML
- CSS
- JavaScript
- Chart.js

Application Layer

- Flask
- Python
- Preprocessing pipeline
- Prediction APIs

Data / Model Layer

- IBM Telco Dataset
- Random Forest model
- Scaler
- Feature names
- EDA outputs

📁 Project Structure

Customer_Churn_Project/
│
├── app/
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── data/
│   ├── Dataset
│   ├── Training/Test Data
│   └── EDA Graphs
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
├── notebooks/
│   ├── preprocessing.py
│   ├── model.py
│   └── eda.py
│
├── Results_Output/
│
└── README.md

Technologies Used

Technology| Purpose
Python| Core programming
Pandas| Data manipulation
NumPy| Numerical operations
Scikit-learn| Machine Learning
Random Forest| Churn prediction
XGBoost| Model comparison
SMOTE| Class balancing
Flask| Web application
HTML/CSS| Frontend
JavaScript| Interactivity
Chart.js| Dashboard visualization
Matplotlib & Seaborn| EDA and model visualization
Joblib| Model/scaler serialization

▶️ How to Run

1. Clone the repository

git clone <your-github-repository-url>
cd Customer_Churn_Project

2. Create and activate virtual environment

python -m venv venv

Windows:

venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Run the Flask application

Run the Flask "app.py" from the folder where the file is located:

python app.py

5. Open the application

Open the local Flask URL shown in the terminal.

Future Scope

The project can be further enhanced by adding:

- Real-time customer activity integration
- CRM integration
- Cloud deployment
- Automated email/SMS alerts
- Mobile application
- Advanced Machine Learning and Deep Learning models
- Continuous model retraining

 Internship Project

Project: ChurnAI – Customer Churn Prediction & Retention Recommendation System
Domain: Data Science & Machine Learning
Organization: Pratinik Infotech
Technology: Python, Machine Learning & Flask

Author

Shreya Saini

B.Sc. Computer Science
