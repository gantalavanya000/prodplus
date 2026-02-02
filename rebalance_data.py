import pandas as pd
import numpy as np

def balance_dataset():
    input_file = 'edited_job_stress_productivity_dataset.csv'
    df = pd.read_csv(input_file)
    n = len(df)
    
    # We want 1/3 Low, 1/3 Medium, 1/3 High
    # Targets: 33,333 each
    
    stress_cols_map = {
        'Workload': ['Workload_TargetTime', 'Workload_ExtraWork'],
        'Role_Ambiguity': ['RoleAmbiguity_ClearInfo'],
        'Job_Security': ['JobSecurity_Secure'],
        'Gender_Discrimination': ['GenderDiscrimination_EqualGrowth'],
        'Interpersonal_Relationships': ['Interpersonal_GoodRelations'],
        'Resource_Constraints': ['Resources_EnoughTime'],
        'Job_Satisfaction': ['JobSatisfaction_WorkConditions'],
        'Organizational_Support': ['OrgSupport_Training', 'OrgSupport_CareerGrowth']
    }
    
    # Flatten cols for easier manipulation
    all_stress_raw_cols = [c for sublist in stress_cols_map.values() for c in sublist]
    prod_cols_raw = ['Productivity_TimeUtilization', 'Supervisor_Motivation', 'Supervisor_Communication', 'Compensation_Salary', 'Systems_QualityProcedures']
    
    # Split dataframe into three chunks
    chunk_size = n // 3
    
    # Chunk 1: Low Stress (Score 1-2) -> Productivity High (4-5)
    for col in all_stress_raw_cols:
        df.loc[:chunk_size-1, col] = np.random.randint(1, 3, size=chunk_size)
    for col in prod_cols_raw:
        df.loc[:chunk_size-1, col] = np.random.randint(4, 6, size=chunk_size)
        
    # Chunk 2: Medium Stress (Score 2.5-3.5) -> Productivity Med (2.5-3.5)
    for col in all_stress_raw_cols:
        df.loc[chunk_size:2*chunk_size-1, col] = np.random.randint(2, 5, size=chunk_size)
    for col in prod_cols_raw:
        df.loc[chunk_size:2*chunk_size-1, col] = np.random.randint(2, 5, size=chunk_size)
    
    # Chunk 3: High Stress (Score 4-5) -> Productivity Low (1-2)
    for col in all_stress_raw_cols:
        start = 2*chunk_size
        size = n - start
        df.loc[start:, col] = np.random.randint(4, 6, size=size)
    for col in prod_cols_raw:
        start = 2*chunk_size
        size = n - start
        df.loc[start:, col] = np.random.randint(1, 3, size=size)

    # Add Random Gender and Department
    genders = ['Male', 'Female']
    departments = ['HR', 'Finance', 'IT', 'Sales', 'Operations', 'Marketing']
    
    df['Gender'] = np.random.choice(genders, size=n)
    df['Department'] = np.random.choice(departments, size=n)

    # Add differences in productivity based on departments
    # For example: IT and Operations have slightly higher productivity boost, HR and Sales are average
    dept_shifts = {
        'IT': 0.5,
        'Operations': 0.3,
        'Finance': 0.1,
        'Marketing': 0.0,
        'HR': -0.2,
        'Sales': -0.4
    }
    
    for dept, shift in dept_shifts.items():
        mask = df['Department'] == dept
        for col in prod_cols_raw:
            df.loc[mask, col] = (df.loc[mask, col] + shift).clip(1, 5)

    df.to_csv(input_file, index=False)
    print(f"Dataset refined with binary Gender and Dept shifts, saved to {input_file}")

if __name__ == '__main__':
    balance_dataset()
