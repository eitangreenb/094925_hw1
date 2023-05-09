import pandas as pd
from scipy.stats import mannwhitneyu

demographics = ["Age", "Gender", "HospAdmTime"] #  "Unit1", "Unit2", "ICULOS"
label = ["SepsisLabel"]

train_data = pd.read_csv('train_df.csv')
all_patients_stats = train_data.fillna(-1)

# Mann Whitney U:
same_dist, diff_dist = [], []
for key in all_patients_stats.keys():
    if key in demographics+label+['id']:
        continue
    temp_data = all_patients_stats[[key, 'SepsisLabel']]
    true_data = temp_data[temp_data['SepsisLabel'] == 1][key].values
    false_data = temp_data[temp_data['SepsisLabel'] == 0][key].values
    stat, p = mannwhitneyu(true_data, false_data)

    if p > 0.05: # can't reject null hypothesis
        same_dist.append(key)
    else:   # reject null hypothesis
        diff_dist.append(key)
print(len(same_dist))
print(same_dist)
train_data.drop(same_dist, axis=1).to_csv('final_train_df.csv', index=False)
pd.read_csv('test_df.csv').drop(same_dist, axis=1).to_csv('final_test_df.csv', index=False)
print('done')