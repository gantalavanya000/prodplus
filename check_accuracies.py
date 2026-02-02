import joblib
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'SEM_JobStress_Productivity_5000.csv')

def check():
    df = pd.read_csv(DATA_PATH)
    stress_cols = ['Workload', 'Role_Ambiguity', 'Job_Security', 'Gender_Discrimination',
                   'Interpersonal_Relationships', 'Resource_Constraints', 'Job_Satisfaction', 'Organizational_Support']
    prod_cols = ['Timings', 'Supervisor_Competence', 'Compensation', 'Systems_Procedures']
    
    df['Job_Stress_Score'] = df[stress_cols].mean(axis=1)
    df['Productivity_Score'] = df[prod_cols].mean(axis=1)
    
    corr = df['Job_Stress_Score'].corr(df['Productivity_Score'])
    print(f"Correlation between Stress and Productivity: {corr}")
    
    scaler = joblib.load('scaler.pkl')
    lr = joblib.load('model_lr.pkl')
    rf = joblib.load('model_rf.pkl')
    gb = joblib.load('model_gb.pkl')
    log_reg = joblib.load('model_log.pkl')
    
    X = df[['Job_Stress_Score']]
    X_scaled = scaler.transform(X)
    y = df['Productivity_Score']
    
    print(f"LR R2: {lr.score(X_scaled, y)}")
    print(f"RF R2: {rf.score(X_scaled, y)}")
    print(f"GB R2: {gb.score(X_scaled, y)}")

if __name__ == '__main__':
    check()
