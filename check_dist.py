import pandas as pd
import numpy as np

def check_distribution():
    df = pd.read_csv('edited_job_stress_productivity_dataset.csv')
    
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
    
    df['Job_Stress_Score'] = df[stress_cols].mean(axis=1)
    
    def label(s):
        if s < 2: return 'Low'
        if s <= 3: return 'Medium'
        return 'High'
    
    df['Level'] = df['Job_Stress_Score'].apply(label)
    print(df['Level'].value_counts())
    print(df['Job_Stress_Score'].describe())

if __name__ == '__main__':
    check_distribution()
