import pandas as pd
import numpy as np
import warnings
import os
from tqdm import tqdm


vital = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
lab = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
         'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
            'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total',
                'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']
demographics = ["Age", "Gender", "HospAdmTime"] #  "Unit1", "Unit2", "ICULOS"
label = ["SepsisLabel"]


def get_statistics(df, col):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            mean_val = np.nanmean(df[col])
        except RuntimeWarning:
            mean_val = np.NaN
        try:
            median_val = np.nanmedian(df[col])
        except RuntimeWarning:
            median_val = np.NaN
        try:
            max_val = np.nanmax(df[col])
        except RuntimeWarning:
            max_val = np.NaN
        try:
            min_val = np.nanmin(df[col])
        except RuntimeWarning:
            min_val = np.NaN
        try:
            q0_25_val = np.nanquantile(df[col], 0.25)
        except RuntimeWarning:
            q0_25_val = np.NaN
        try:
            q0_75_val = np.nanquantile(df[col], 0.75)
        except RuntimeWarning:
            q0_75_val = np.NaN
        try:
            std_val = np.nanstd(df[col])
        except RuntimeWarning:
            std_val = np.NaN

    last_first_dif = df[col].iloc[-1] - df[col].iloc[0]
    return [mean_val, median_val, max_val, min_val, q0_25_val, q0_75_val, std_val, last_first_dif]
    

def take_until_sepsis_1(df):
    if df[df["SepsisLabel"] == 1].empty:
        return df
    idx = df[df["SepsisLabel"] == 1].index[0] + 1
    return df.iloc[:idx]


def preprocess(patient_file):
    patient = pd.read_csv(patient_file, sep="|")
    length = len(patient)
    nan_values = patient[vital+lab].isna().sum()/length
    patient = patient.ffill().bfill()
    patient = take_until_sepsis_1(patient)
    res_dict = {}
    res_dict['id'] = patient_file.split("/")[-1].split(".")[0]
    for col in vital+lab:
        vals = get_statistics(patient, col)
        res_dict[col + "_mean"] = vals[0]
        res_dict[col + "_median"] = vals[1]
        res_dict[col + "_max"] = vals[2]
        res_dict[col + "_min"] = vals[3]
        res_dict[col + "_q0.25"] = vals[4]
        res_dict[col + "_q0.75"] = vals[5]
        res_dict[col + "_std"] = vals[6]
        res_dict[col + "_last_first_dif"] = vals[7]
        res_dict[col + "_nan"] = nan_values[col]
    for col in demographics:
        res_dict[col] = patient[col].iloc[0]
    unit_value = 0
    if patient['Unit1'].iloc[0] == 1:
        unit_value = 1
    elif patient['Unit2'].iloc[0] == 1:
        unit_value = 2
    res_dict['Unit'] = unit_value
    res_dict['Time'] = patient['ICULOS'].iloc[-1]
    res_dict['SepsisLabel'] = patient['SepsisLabel'].iloc[-1]
    return pd.DataFrame(res_dict, index=[0])


def create_df(path, name_to_save=None):
    df = pd.DataFrame()
    for file in tqdm(os.listdir(path)):
        df = df.append(preprocess(path + file))
    if name_to_save is not None:
        df.to_csv(name_to_save, index=False)
    return df


if __name__ == '__main__':
    train_path = "../data/train/"
    test_path = "../data/test/"
    create_df(train_path, "train_df.csv")
    create_df(test_path, "test_df.csv")
