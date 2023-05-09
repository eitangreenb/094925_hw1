import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

sample = pd.read_csv("../data/train/patient_1.psv", sep="|")
all_patients = pd.DataFrame(columns=sample.keys())
for file in tqdm(os.listdir('../data/train/')):
    df = pd.read_csv('../data/train/' + file, sep="|")
    all_patients = all_patients.append(df)

all_patients.to_csv('all_patiens.csv')

# all_patients = pd.read_csv('all_patiens.csv')
all_patients_stats = pd.read_csv('train_df.csv')

# histogram of Vital signs:
vital_signs = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 12))
axes = axes.flatten()  # Convert axes to 1d array of length 9
for i, key in zip(axes, vital_signs):
    sns.histplot(all_patients[key], bins=20, kde=True, ax=i)
    # plt.title(key + 'Histogram')
fig.suptitle('Histograms of Vital Signs')
plt.show()


vital_signs = ['HR', 'HR_mean', 'O2Sat', 'O2Sat_mean', 'Temp', 'Temp_mean', 'SBP', 'SBP_mean', 'MAP', 'MAP_mean', 'DBP',
               'DBP_mean', 'Resp', 'Resp_mean', 'EtCO2', 'EtCO2_mean']
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
axes = axes.flatten()  # Convert axes to 1d array of length 9
for i, key in zip(axes, vital_signs):
    if 'mean' in key:
        sns.histplot(all_patients_stats[key ], bins=20, kde=True, ax=i, color='red')
    else:
        sns.histplot(all_patients[key], bins=20, kde=True, ax=i)

    # plt.title(key + 'Histogram')
fig.suptitle('Histograms of Vital Signs - original compared to mean')
plt.show()

vital_signs = ['HR', 'HR_median', 'O2Sat', 'O2Sat_median', 'Temp', 'Temp_median', 'SBP', 'SBP_median', 'MAP', 'MAP_median', 'DBP',
               'DBP_median', 'Resp', 'Resp_median', 'EtCO2', 'EtCO2_median']
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
axes = axes.flatten()  # Convert axes to 1d array of length 9
for i, key in zip(axes, vital_signs):
    if 'median' in key:
        sns.histplot(all_patients_stats[key], bins=20, kde=True, ax=i, color='red')
    else:
        sns.histplot(all_patients[key], bins=20, kde=True, ax=i)
fig.suptitle('Histograms of Vital Signs - original compared to median')
plt.show()

vital_signs = ['HR', 'HR_max', 'O2Sat', 'O2Sat_max', 'Temp', 'Temp_max', 'SBP', 'SBP_max', 'MAP', 'MAP_max', 'DBP',
               'DBP_max', 'Resp', 'Resp_max', 'EtCO2', 'EtCO2_max']
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
axes = axes.flatten()  # Convert axes to 1d array of length 9
for i, key in zip(axes, vital_signs):
    if 'max' in key:
        sns.histplot(all_patients_stats[key], bins=20, kde=True, ax=i, color='red')
    else:
        sns.histplot(all_patients[key], bins=20, kde=True, ax=i)
fig.suptitle('Histograms of Vital Signs - original compared to max')
plt.show()

vital_signs = ['HR', 'HR_min', 'O2Sat', 'O2Sat_min', 'Temp', 'Temp_min', 'SBP', 'SBP_min', 'MAP', 'MAP_min', 'DBP',
               'DBP_min', 'Resp', 'Resp_min', 'EtCO2', 'EtCO2_min']
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
axes = axes.flatten()  # Convert axes to 1d array of length 9
for i, key in zip(axes, vital_signs):
    if 'min' in key:
        sns.histplot(all_patients_stats[key], bins=20, kde=True, ax=i, color='red')
    else:
        sns.histplot(all_patients[key], bins=20, kde=True, ax=i)
fig.suptitle('Histograms of Vital Signs - original compared to min')
plt.show()

# Sepsis label pie graph
results = [all_patients_stats['SepsisLabel'].sum(),
           all_patients_stats['SepsisLabel'].shape[0] - all_patients_stats['SepsisLabel'].sum()]
labels = ['sepsis', 'none']
plt.pie(results, labels=labels, autopct='%1.0f%%')
plt.title('Sepsis Labels')
plt.show()

# graphs of demographic signs:
results = [all_patients_stats['Gender'].sum(),
           all_patients_stats['Gender'].shape[0] - all_patients_stats['SepsisLabel'].sum()]
labels = ['Male', 'Female']
plt.pie(results, labels=labels, autopct='%1.0f%%')
plt.title('Gender')
plt.show()
sns.histplot(all_patients_stats['Age'], bins=20, kde=True)
plt.title('Age Histogram')
plt.show()
print(f'mean age: {all_patients_stats.Age.mean()}')
print(f'median age: {all_patients_stats.Age.median()}')

sns.histplot(all_patients_stats['time'], bins=20, kde=True)
plt.title('Time Histogram')
plt.show()
print(f'mean time: {all_patients_stats.time.mean()}')
print(f'median time: {all_patients_stats.time.median()}')

# Missing values:
percent_missing = all_patients.isnull().sum() * 100 / len(all_patients)
missing_value_df = pd.DataFrame({'column_name': all_patients.columns,
                                 'percent_missing': percent_missing})
missing_value_df.to_csv('missing_values.csv')
