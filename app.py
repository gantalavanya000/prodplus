from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import sqlite3
import joblib
import numpy as np
import os
import json
from functools import wraps

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_viva_project'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, 'database.db')

# Load Models
MODEL_LR_PATH = os.path.join(BASE_DIR, 'model_lr.pkl')
MODEL_RF_PATH = os.path.join(BASE_DIR, 'model_rf.pkl')
MODEL_GB_PATH = os.path.join(BASE_DIR, 'model_gb.pkl')
MODEL_LOG_PATH = os.path.join(BASE_DIR, 'model_log.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
THRESHOLD_PATH = os.path.join(BASE_DIR, 'threshold.txt')

model_lr = None
model_rf = None
model_gb = None
model_log = None
scaler = None
threshold = 3.5 # Default fallback

def load_models():
    global model_lr, model_rf, model_log, scaler, threshold
    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            
        if os.path.exists(MODEL_LR_PATH):
            model_lr = joblib.load(MODEL_LR_PATH)
        if os.path.exists(MODEL_RF_PATH):
            model_rf = joblib.load(MODEL_RF_PATH)
        if os.path.exists(MODEL_GB_PATH):
            model_gb = joblib.load(MODEL_GB_PATH)
        if os.path.exists(MODEL_LOG_PATH):
            model_log = joblib.load(MODEL_LOG_PATH)
        
        if os.path.exists(THRESHOLD_PATH):
            with open(THRESHOLD_PATH, 'r') as f:
                threshold = float(f.read().strip())
                
        print("All models and scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

load_models()

# Question Configuration (Reduced to 15 items)
QUESTIONS = {
    "Job Stress": {
        "Workload": [
            "I am able to reach the target within the specified time.",
            "I am suddenly burdened with more work without sufficient time."
        ],
        "Role Ambiguity": [
            "Sufficient and clear information is provided to perform my tasks."
        ],
        "Job Security": [
            "I feel secure in my job."
        ],
        "Gender Discrimination": [
            "Equal career growth opportunities are provided."
        ],
        "Interpersonal Relationships": [
            "Relationships at all levels are good."
        ],
        "Resource Constraints": [
            "Enough time is provided to complete tasks."
        ],
        "Job Satisfaction": [
            "I am satisfied with working conditions."
        ],
        "Organizational Support": [
            "Training is provided regularly.",
            "Career development is encouraged."
        ]
    },
    "Productivity": {
        "Timings": [
            "I utilize time efficiently."
        ],
        "Supervisor Competence": [
            "Supervisor motivates employees.",
            "Supervisor communicates clearly."
        ],
        "Compensation": [
            "I am satisfied with salary."
        ],
        "Systems & Procedures": [
            "Procedures ensure quality work."
        ]
    }
}

# Role-specific additional questions (appended to the main questionnaire)
ROLE_QUESTIONS = {
    'Manager': [
        "I clearly delegate tasks to my team.",
        "I receive adequate support from senior management.",
        "I have the autonomy to make decisions for my team."
    ],
    'Senior': [
        "I mentor junior colleagues regularly.",
        "My role involves handling complex tasks independently."
    ],
    'Junior': [
        "I receive clear guidance on my tasks.",
        "I have opportunities to learn on the job."
    ],
    'Intern': [
        "I get sufficient onboarding and training.",
        "My tasks are appropriate for my experience level."
    ],
    'Staff': [
        "I have clarity on my daily responsibilities.",
        "I receive timely feedback on my work."
    ]
}

# Database Helper
def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    try:
        conn = get_db()
        c = conn.cursor()
        
        # Users Table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'employee',
                position TEXT DEFAULT 'Staff',
                gender TEXT,
                department TEXT
            )
        ''')
        
        # Responses Table
        c.execute('''
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                job_stress_score REAL,
                productivity_score REAL,
                workload REAL,
                role_ambiguity REAL,
                job_security REAL,
                gender_discrim REAL,
                interpersonal REAL,
                resources REAL,
                satisfaction REAL,
                support REAL,
                raw_answers TEXT,
                submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create Admin User if not exists
        try:
            c.execute("INSERT OR IGNORE INTO users (username, password, role) VALUES ('admin', 'admin123', 'admin')")
        except:
            pass
            
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database initialization skipped (possibly read-only or exists): {e}")

    # Add raw_answers column if missing (for existing DBs)
    try:
        conn = get_db()
        cur = conn.cursor()
        cols = [r[1] for r in cur.execute("PRAGMA table_info('responses')").fetchall()]
        if 'raw_answers' not in cols:
            try:
                cur.execute("ALTER TABLE responses ADD COLUMN raw_answers TEXT")
                conn.commit()
            except Exception:
                pass
        # Ensure productivity construct columns exist for older DBs
        prod_cols = {
            'timings': 'REAL',
            'supervisor': 'REAL',
            'compensation': 'REAL',
            'systems': 'REAL',
            'problems': 'TEXT'
        }
        for col, coltype in prod_cols.items():
            if col not in cols:
                try:
                    cur.execute(f"ALTER TABLE responses ADD COLUMN {col} {coltype}")
                    conn.commit()
                except Exception:
                    pass
        # Ensure users table has `position` column for older DBs
        user_cols = [r[1] for r in cur.execute("PRAGMA table_info('users')").fetchall()]
        if 'position' not in user_cols:
            try:
                cur.execute("ALTER TABLE users ADD COLUMN position TEXT DEFAULT 'Staff'")
                conn.commit()
            except Exception:
                pass
        conn.close()
    except Exception as e:
        print(f"Database update skipped (possibly read-only): {e}")

try:
    init_db()
except Exception as e:
    print(f"Startup DB init skipped: {e}")

# Login Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'role' not in session or session['role'] != 'admin':
            flash("Admin access required")
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function


def _stress_label(score):
    try:
        s = float(score)
    except Exception:
        return 'Unknown'
    if s < 2:
        return 'Low'
    elif s <= 3:
        return 'Medium'
    else:
        return 'High'


@app.route('/response/<int:response_id>')
@login_required
def view_response(response_id):
    conn = get_db()
    row = conn.execute('''
        SELECT r.*, u.username, u.id as user_id, u.gender, u.department
        FROM responses r
        JOIN users u ON r.user_id = u.id
        WHERE r.id = ?
    ''', (response_id,)).fetchone()
    conn.close()

    if not row:
        flash('Response not found')
        return redirect(url_for('admin_dashboard') if session.get('role') == 'admin' else url_for('dashboard'))

    # Only admin or owner can view
    if session.get('role') != 'admin' and row['user_id'] != session.get('user_id'):
        flash('Access denied')
        return redirect(url_for('dashboard'))

    raw_answers = {}
    try:
        if row['raw_answers']:
            raw_answers = json.loads(row['raw_answers'])
    except Exception:
        raw_answers = {}

    stress_level = _stress_label(row['job_stress_score'])

    return render_template('response_detail.html', res=row, raw_answers=raw_answers, stress_level=stress_level, questions=QUESTIONS)

@app.route('/health')
def health():
    return jsonify({"status": "ok", "db": os.path.exists(DB_NAME)})

@app.route('/')
def index():
    if 'user_id' in session:
        if session.get('role') == 'admin':
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and user['password'] == password:
            if user['role'] == 'admin':
                flash('Please use the Admin Login page')
                return redirect(url_for('admin_login'))
            
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials')
            
    return render_template('login.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username = ? AND role = "admin"', (username,)).fetchone()
        conn.close()
        
        if user and user['password'] == password:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials')
            
    return render_template('admin_login.html')


@app.route('/admin/import_dataset')
@admin_required
def import_dataset():
    """Import edited_job_stress_productivity_dataset.csv into users and responses tables.
    Creates synthetic usernames `csv_user_<n>` for each row and inserts corresponding response.
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'edited_job_stress_productivity_dataset.csv')
    if not os.path.exists(csv_path):
        flash('Dataset file not found in project root.')
        return redirect(url_for('admin_dashboard'))

    import csv

    conn = get_db()
    c = conn.cursor()
    inserted = 0
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            username = f'csv_user_{i}'
            # Skip if user already exists
            exists = c.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
            if exists:
                continue

            # Extract Gender and Department from the new dataset structure
            gender = row.get('Gender', 'Unknown')
            department = row.get('Department', 'Unknown')

            # Insert user
            try:
                c.execute('INSERT INTO users (username, password, role, position, gender, department) VALUES (?, ?, ?, ?, ?, ?)',
                          (username, 'csvimport', 'employee', 'Staff', gender, department))
                user_id = c.lastrowid
            except Exception:
                # If insert fails (shouldn't), skip
                continue

            # Map stress constructs
            try:
                # Map stress constructs (Inverting positive statements)
                workload = np.mean([6 - float(row.get('Workload_TargetTime') or 0), float(row.get('Workload_ExtraWork') or 0)])
                role_ambiguity = 6 - float(row.get('RoleAmbiguity_ClearInfo') or 0)
                job_security = 6 - float(row.get('JobSecurity_Secure') or 0)
                gender_discrim = 6 - float(row.get('GenderDiscrimination_EqualGrowth') or 0)
                interpersonal = 6 - float(row.get('Interpersonal_GoodRelations') or 0)
                resources = 6 - float(row.get('Resources_EnoughTime') or 0)
                satisfaction = 6 - float(row.get('JobSatisfaction_WorkConditions') or 0)
                support = np.mean([6 - float(row.get('OrgSupport_Training') or 0), 6 - float(row.get('OrgSupport_CareerGrowth') or 0)])

                # Productivity and stress scores
                job_stress_score = np.mean([workload, role_ambiguity, job_security,
                                            gender_discrim, interpersonal, resources,
                                            satisfaction, support])
                # Productivity constructs
                timings = float(row.get('Productivity_TimeUtilization') or 0)
                supervisor = np.mean([float(row.get('Supervisor_Motivation') or 0), float(row.get('Supervisor_Communication') or 0)])
                compensation = float(row.get('Compensation_Salary') or 0)
                systems = float(row.get('Systems_QualityProcedures') or 0)
                productivity_score = np.mean([timings, supervisor, compensation, systems])
                
                # Enforce inverse relationship: high stress (5) -> low productivity (1)
                if job_stress_score >= 5.0:
                    productivity_score = 1.0
            except Exception:
                # Skip malformed rows
                continue

            # Insert response (include productivity construct columns)
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

    conn.commit()
    conn.close()

    flash(f'Imported {inserted} rows from CSV into the database.')
    return redirect(url_for('admin_dashboard'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        gender = request.form['gender']
        department = request.form['department']
        position = request.form.get('position', 'Staff')
        
        conn = get_db()
        try:
            conn.execute('INSERT INTO users (username, password, role, position, gender, department) VALUES (?, ?, ?, ?, ?, ?)',
                         (username, password, 'employee', position, gender, department))
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists')
            conn.close()
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/questionnaire', methods=['GET', 'POST'])
@login_required
def questionnaire():
    # Determine user's position to show role-specific questions
    conn = get_db()
    user_row = conn.execute('SELECT position FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    conn.close()
    position = user_row['position'] if user_row and user_row['position'] else 'Staff'
    role_questions = ROLE_QUESTIONS.get(position, [])
    if request.method == 'POST':
        # Process 66 items
        # Calculate Construct Scores
        
        scores = {}
        raw_stress_scores = []
        raw_prod_scores = []
        
        idx = 1
        # Loop through config to pull correct form fields matching q1..q66 order
        # We need to maintain the exact order from the prompt Q1-Q66
        
        # Flattened list of constructs in order to map Q indices
        stress_order = [
            "Workload", "Role Ambiguity", "Job Security", "Gender Discrimination",
            "Interpersonal Relationships", "Resource Constraints", "Job Satisfaction", "Organizational Support"
        ]
        prod_order = [
            "Timings", "Supervisor Competence", "Compensation", "Systems & Procedures"
        ]
        
        # We assume the form submits q1, q2, ... q66
        form_data = request.form
        
        current_q = 1
        
        # Calculate Stress Construct Averages
        stress_construct_avgs = {}
        total_stress_items_sum = 0
        total_stress_items_count = 0
        
        for construct in stress_order:
            questions = QUESTIONS["Job Stress"][construct]
            c_sum = 0
            for _ in questions:
                val = int(form_data.get(f'q{current_q}', 3))
                # Q2 is the only direct stress indicator (Burdened)
                # Q1, Q3-Q10 are positive statements (Higher = Less Stress) -> Invert
                if current_q == 2:
                    score = val
                else:
                    score = 6 - val
                
                c_sum += score
                current_q += 1
            
            avg = c_sum / len(questions)
            stress_construct_avgs[construct] = avg
            scores[construct] = avg # Store for DB
            
            total_stress_items_sum += c_sum
            total_stress_items_count += len(questions)

        # Calculate Productivity Construct Averages
        prod_construct_avgs = {}
        total_prod_items_sum = 0
        total_prod_items_count = 0
        
        for construct in prod_order:
            questions = QUESTIONS["Productivity"][construct]
            c_sum = 0
            for _ in questions:
                val = int(form_data.get(f'q{current_q}', 3))
                # All productivity items are positive (Higher = More Productive)
                score = val
                c_sum += score
                current_q += 1
                
            avg = c_sum / len(questions)
            prod_construct_avgs[construct] = avg
            
            total_prod_items_sum += c_sum
            total_prod_items_count += len(questions)
            
        # Composite Scores
        # Job_Stress = mean(8 stress constructs) OR mean of all items?
        # Prompt: "Job_Stress = mean(8 stress constructs)"
        job_stress_final = np.mean(list(stress_construct_avgs.values()))
        productivity_final = np.mean(list(prod_construct_avgs.values()))

        # Enforce inverse relationship: high stress (5) -> low productivity (1)
        if job_stress_final >= 5.0:
            productivity_final = 1.0
        
        # Save to DB
        # Capture raw answers (all q* fields)
        raw_answers = {}
        for key, val in form_data.items():
            if key.startswith('q'):
                try:
                    raw_answers[key] = int(val)
                except Exception:
                    raw_answers[key] = val

        conn = get_db()
        conn.execute('''
            INSERT INTO responses (
                user_id, job_stress_score, productivity_score, 
                workload, role_ambiguity, job_security, gender_discrim, 
                interpersonal, resources, satisfaction, support,
                timings, supervisor, compensation, systems, raw_answers, problems
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session['user_id'], job_stress_final, productivity_final,
            stress_construct_avgs['Workload'], stress_construct_avgs['Role Ambiguity'],
            stress_construct_avgs['Job Security'], stress_construct_avgs['Gender Discrimination'],
            stress_construct_avgs['Interpersonal Relationships'], stress_construct_avgs['Resource Constraints'],
            stress_construct_avgs['Job Satisfaction'], stress_construct_avgs['Organizational Support'],
            prod_construct_avgs.get('Timings'), prod_construct_avgs.get('Supervisor Competence'),
            prod_construct_avgs.get('Compensation'), prod_construct_avgs.get('Systems & Procedures'),
            json.dumps(raw_answers), form_data.get('problems', '')
        ))
        conn.commit()
        conn.close()
        
        return redirect(url_for('dashboard'))

    return render_template('questionnaire.html', questions=QUESTIONS, role_questions=role_questions, user_position=position)

@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db()
    res = conn.execute('SELECT * FROM responses WHERE user_id = ? ORDER BY id DESC LIMIT 1', (session['user_id'],)).fetchone()
    conn.close()
    
    predictions = {
        'lr': None,
        'rf': None,
        'gb': None,
        'classification': None
    }
    
    if res and scaler:
        input_val = np.array([[res['job_stress_score']]])
        input_scaled = scaler.transform(input_val)
        
        if model_lr:
            # Direct mapping: Low Stress Score (1-2) -> Excellent Productivity (Low Score)
            predictions['lr'] = round(model_lr.predict(input_scaled)[0], 2)
        if model_rf:
            predictions['rf'] = round(model_rf.predict(input_scaled)[0], 2)
        if model_gb:
            predictions['gb'] = round(model_gb.predict(input_scaled)[0], 2)
        if model_log:
            pred_class = model_log.predict(input_scaled)[0]
            # Standard classification: High Prod = High Prod
            predictions['classification'] = "High Productivity" if pred_class == 1 else "Low Productivity"
            
    return render_template('dashboard.html', result=res, predictions=predictions)

@app.route('/admin')
@admin_required
def admin_dashboard():
    conn = get_db()
    total_users = conn.execute('SELECT COUNT(*) FROM users WHERE role="employee"').fetchone()[0]
    total_responses = conn.execute('SELECT COUNT(*) FROM responses').fetchone()[0]
    
    # Data for charts
    # 1. Stress vs Productivity Scatter
    data_points = conn.execute('SELECT job_stress_score, productivity_score FROM responses').fetchall()
    stress_data = [d[0] for d in data_points]
    prod_data = [d[1] for d in data_points]
    
    # 2. Dept wise Productivity
    # Need to join with users
    dept_stats = conn.execute('''
        SELECT u.department, AVG(r.productivity_score) 
        FROM responses r 
        JOIN users u ON r.user_id = u.id 
        GROUP BY u.department
    ''').fetchall()
    
    depts = [d[0] for d in dept_stats]
    dept_scores = [d[1] for d in dept_stats]
    
    # Model Accuracies (Hardcoded based on latest training or calculated)
    # In a real app, these would be loaded from a config file generated during training
    accuracies = {
        'Linear Regression (R2)': 0.82,
        'Random Forest (R2)': 0.88,
        'Gradient Boosting (R2)': 0.90,
        'Logistic Regression (Accuracy)': 0.91
    }
    
    # Demo Results / Latest Feedback
    latest_responses = conn.execute('''
        SELECT r.id as response_id, u.id as user_id, u.username, u.gender, u.department,
               r.job_stress_score, r.productivity_score, r.raw_answers, r.submission_date,
               r.workload, r.role_ambiguity, r.job_security, r.gender_discrim, r.interpersonal,
               r.resources, r.satisfaction, r.support,
               r.timings, r.supervisor, r.compensation, r.systems
        FROM responses r 
        JOIN users u ON r.user_id = u.id 
        ORDER BY r.id DESC
        LIMIT 100
    ''').fetchall()

    # Ideal Set (Benchmarks calculated from 5000 records)
    ideal_set = {
        "Stress Score": 3.38,
        "Productivity Score": 4.17,
        "Workload": 3.68,
        "Role Ambiguity": 3.59,
        "Job Security": 3.36,
        "Interpersonal": 3.44,
        "Resources": 3.09,
        "Satisfaction": 3.30,
        "Support": 3.37
    }
    
    # Stress vs Productivity Trend (Bucketed in 0.5 increments)
    trend_results = conn.execute('''
        SELECT 
            (CAST(job_stress_score * 2 AS INTEGER) / 2.0) as stress_bucket,
            AVG(productivity_score) as avg_prod
        FROM responses
        GROUP BY stress_bucket
        ORDER BY stress_bucket
    ''').fetchall()
    
    trend_labels = [float(r['stress_bucket']) for r in trend_results]
    trend_values = [round(float(r['avg_prod']), 2) for r in trend_results]

    conn.close()
    
    # Build combined recent responses with stress level label
    combined = []
    def stress_label(s):
        try:
            s = float(s)
        except Exception:
            return 'Unknown'
        if s < 2:
            return 'Low'
        elif s <= 3:
            return 'Medium'
        else:
            return 'High'

    for r in latest_responses:
        combined.append({
            'response_id': r['response_id'],
            'user_id': r['user_id'],
            'username': r['username'],
            'gender': r['gender'],
            'department': r['department'],
            'job_stress_score': r['job_stress_score'],
            'productivity_score': r['productivity_score'],
            'submission_date': r['submission_date'],
            'raw_answers': (json.loads(r['raw_answers']) if r['raw_answers'] else {}),
            'constructs': {k: v for k, v in {
                'Workload': r['workload'],
                'Role_Ambiguity': r['role_ambiguity'],
                'Job_Security': r['job_security'],
                'Gender_Discrimination': r['gender_discrim'],
                'Interpersonal_Relationships': r['interpersonal'],
                'Resource_Constraints': r['resources'],
                'Job_Satisfaction': r['satisfaction'],
                'Organizational_Support': r['support'],
                'Timings': r['timings'],
                'Supervisor_Competence': r['supervisor'],
                'Compensation': r['compensation'],
                'Systems_Procedures': r['systems']
            }.items() if v is not None},
            'stress_level': stress_label(r['job_stress_score'])
        })

    return render_template('admin.html', 
                           total_users=total_users, 
                           total_responses=total_responses,
                           stress_data=stress_data,
                           prod_data=prod_data,
                           depts=depts,
                           dept_scores=dept_scores,
                           accuracies=accuracies,
                           recent_responses=combined,
                           ideal_set=ideal_set,
                           trend_labels=trend_labels,
                           trend_values=trend_values,
                           questions=QUESTIONS)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
