import sqlite3
import os
import csv
import numpy as np

DB_NAME = 'database.db'
CSV_PATH = os.path.join(os.path.dirname(__file__), 'SEM_JobStress_Productivity_5000.csv')

if not os.path.exists(CSV_PATH):
    print('CSV file not found:', CSV_PATH)
    exit(1)

conn = sqlite3.connect(DB_NAME)
c = conn.cursor()

inserted = 0
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, start=1):
        username = f'csv_user_{i}'
        exists = c.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
        if exists:
            continue

        gender = row.get('Gender')
        department = row.get('Department')
        try:
            c.execute('INSERT INTO users (username, password, role, gender, department) VALUES (?, ?, ?, ?, ?)',
                      (username, 'csvimport', 'employee', gender, department))
            user_id = c.lastrowid
        except Exception as e:
            print('Failed insert user', username, e)
            continue

        try:
            workload = float(row.get('Workload') or 0)
            role_ambiguity = float(row.get('Role_Ambiguity') or 0)
            job_security = float(row.get('Job_Security') or 0)
            gender_discrim = float(row.get('Gender_Discrimination') or 0)
            interpersonal = float(row.get('Interpersonal_Relationships') or 0)
            resources = float(row.get('Resource_Constraints') or 0)
            satisfaction = float(row.get('Job_Satisfaction') or 0)
            support = float(row.get('Organizational_Support') or 0)

            job_stress_score = np.mean([workload, role_ambiguity, job_security,
                                        gender_discrim, interpersonal, resources,
                                        satisfaction, support])

            timings = float(row.get('Timings') or 0)
            supervisor = float(row.get('Supervisor_Competence') or 0)
            compensation = float(row.get('Compensation') or 0)
            systems = float(row.get('Systems_Procedures') or 0)
            productivity_score = np.mean([timings, supervisor, compensation, systems])
        except Exception as e:
            print('Malformed row', i, e)
            continue

        c.execute('''
            INSERT INTO responses (
                user_id, job_stress_score, productivity_score,
                workload, role_ambiguity, job_security, gender_discrim,
                interpersonal, resources, satisfaction, support
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, job_stress_score, productivity_score,
            workload, role_ambiguity, job_security, gender_discrim,
            interpersonal, resources, satisfaction, support
        ))
        inserted += 1

conn.commit()
conn.close()
print(f'Imported {inserted} rows into database.')
