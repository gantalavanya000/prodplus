import pandas as pd
df = pd.read_csv('edited_job_stress_productivity_dataset.csv', nrows=0)
with open('cols_list.txt', 'w') as f:
    for col in df.columns:
        f.write(col + '\n')
