import sqlite3
import pandas as pd
import numpy as np
import json
import os

DB_NAME = 'database.db'
CSV_PATH = 'edited_job_stress_productivity_dataset.csv'

def import_data():
    if not os.path.exists(CSV_PATH):
        print("CSV not found.")
        return

    df = pd.read_csv(CSV_PATH)
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    inserted = 0
    for i, row in df.iterrows():
        username = f'csv_user_{i+1}'
        
        # Check if user exists
        c.execute("SELECT id FROM users WHERE username = ?", (username,))
        if c.fetchone():
            continue
            
        gender = row.get('Gender', 'Unknown')
        department = row.get('Department', 'Unknown')
        
        try:
            c.execute('INSERT INTO users (username, password, role, position, gender, department) VALUES (?, ?, ?, ?, ?, ?)',
                      (username, 'csvimport', 'employee', 'Staff', gender, department))
            user_id = c.lastrowid
            
            workload = np.mean([float(row['Workload_TargetTime']), float(row['Workload_ExtraWork'])])
            role_ambiguity = float(row['RoleAmbiguity_ClearInfo'])
            job_security = float(row['JobSecurity_Secure'])
            gender_discrim = float(row['GenderDiscrimination_EqualGrowth'])
            interpersonal = float(row['Interpersonal_GoodRelations'])
            resources = float(row['Resources_EnoughTime'])
            satisfaction = float(row['JobSatisfaction_WorkConditions'])
            support = np.mean([float(row['OrgSupport_Training']), float(row['OrgSupport_CareerGrowth'])])

            job_stress_score = np.mean([workload, role_ambiguity, job_security,
                                        gender_discrim, interpersonal, resources,
                                        satisfaction, support])
            
            timings = float(row['Productivity_TimeUtilization'])
            supervisor = np.mean([float(row['Supervisor_Motivation']), float(row['Supervisor_Communication'])])
            compensation = float(row['Compensation_Salary'])
            systems = float(row['Systems_QualityProcedures'])
            productivity_score = np.mean([timings, supervisor, compensation, systems])
            
            c.execute('''
                INSERT INTO responses (
                    user_id, job_stress_score, productivity_score,
                    workload, role_ambiguity, job_security, gender_discrim,
                    interpersonal, resources, satisfaction, support,
                    timings, supervisor, compensation, systems, raw_answers
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, job_stress_score, productivity_score,
                workload, role_ambiguity, job_security, gender_discrim,
                interpersonal, resources, satisfaction, support,
                timings, supervisor, compensation, systems, json.dumps({})
            ))
            inserted += 1
        except Exception as e:
            print(f"Error at row {i}: {e}")
            continue
            
    conn.commit()
    conn.close()
    print(f"Imported {inserted} new records.")

if __name__ == '__main__':
    import_data()
