import pandas as pd

df = pd.read_csv('data/processed_data/segment_level_data.csv')
firms = df['gvkey'].unique().tolist()

with open('data/processed_data/firms.txt', 'w') as f:
    for firm in firms:
        f.write(str(firm) + '\n')
    f.close()
    

