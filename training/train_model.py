import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Ensure the project root is the current directory or handle paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'edited_job_stress_productivity_dataset.csv')

def train():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("Error: Dataset not found. Please ensure 'edited_job_stress_productivity_dataset.csv' is in the project root.")
        return

    # Construct Columns based on CSV header
    # Stress constructs mapping
    df['Workload'] = df[['Workload_TargetTime', 'Workload_ExtraWork']].mean(axis=1)
    df['Role_Ambiguity'] = df['RoleAmbiguity_ClearInfo']
    df['Job_Security'] = df['JobSecurity_Secure']
    df['Gender_Discrimination'] = df['GenderDiscrimination_EqualGrowth']
    df['Interpersonal_Relationships'] = df['Interpersonal_GoodRelations']
    df['Resource_Constraints'] = df['Resources_EnoughTime']
    df['Job_Satisfaction'] = df['JobSatisfaction_WorkConditions']
    df['Organizational_Support'] = df[['OrgSupport_Training', 'OrgSupport_CareerGrowth']].mean(axis=1)
    
    stress_cols = [
        'Workload', 'Role_Ambiguity', 'Job_Security', 'Gender_Discrimination',
        'Interpersonal_Relationships', 'Resource_Constraints', 'Job_Satisfaction', 'Organizational_Support'
    ]
    
    # Productivity constructs mapping
    df['Timings'] = df['Productivity_TimeUtilization']
    df['Supervisor_Competence'] = df[['Supervisor_Motivation', 'Supervisor_Communication']].mean(axis=1)
    df['Compensation'] = df['Compensation_Salary']
    df['Systems_Procedures'] = df['Systems_QualityProcedures']
    
    prod_cols = [
        'Timings', 'Supervisor_Competence', 'Compensation', 'Systems_Procedures'
    ]
    
    # Calculate Composite Scores
    # Job_Stress = mean(8 stress constructs)
    df['Job_Stress_Score'] = df[stress_cols].mean(axis=1)
    
    # Productivity = mean(4 productivity constructs)
    df['Productivity_Score'] = df[prod_cols].mean(axis=1)
    
    print("Data processed. Sample:")
    print(df[['Job_Stress_Score', 'Productivity_Score']].head())
    
    # Features & Target
    # We use Job_Stress_Score to predict Productivity_Score as per app.py logic and SEM formula
    X = df[['Job_Stress_Score']]
    y = df['Productivity_Score']
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Linear Regression (The main model used in app)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    print(f"Linear Regression R2 Score: {lr.score(X_test_scaled, y_test)}")
    print(f"Coefficients: {lr.coef_}")
    print(f"Intercept: {lr.intercept_}")
    
    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    print(f"Random Forest R2 Score: {rf.score(X_test_scaled, y_test)}")
    
    # 3. Gradient Boosting Regressor
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train_scaled, y_train)
    print(f"Gradient Boosting R2 Score: {gb.score(X_test_scaled, y_test)}")
    
    # 4. Logistic Regression (High vs Low Productivity)
    # Threshold = median
    threshold = y.median()
    y_class = (y > threshold).astype(int)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    
    log_reg = LogisticRegression()
    log_reg.fit(scaler.transform(X_train_c), y_train_c)
    print(f"Logistic Regression Accuracy: {log_reg.score(scaler.transform(X_test_c), y_test_c)}")
    
    # Save Artifacts
    joblib.dump(lr, os.path.join(BASE_DIR, 'model_lr.pkl'))
    joblib.dump(rf, os.path.join(BASE_DIR, 'model_rf.pkl'))
    joblib.dump(gb, os.path.join(BASE_DIR, 'model_gb.pkl'))
    joblib.dump(log_reg, os.path.join(BASE_DIR, 'model_log.pkl'))
    joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.pkl'))
    # Keeping model.pkl pointing to lr for backward compatibility
    joblib.dump(lr, os.path.join(BASE_DIR, 'model.pkl'))
    
    # Store the median threshold for the logistic model
    with open(os.path.join(BASE_DIR, 'threshold.txt'), 'w') as f:
        f.write(str(threshold))
        
    print("Multi-models updated and saved to project root.")

if __name__ == '__main__':
    train()
