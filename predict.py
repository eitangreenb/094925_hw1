import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import numpy as np
import warnings
import os
from tqdm import tqdm
import sys
from preprocess import create_df


manwhitney_drop = [
    'HR_nan', 'Temp_last_first_dif', 'SBP_nan', 'MAP_min', 'DBP_mean', 'DBP_median', 'DBP_min', 'DBP_q0.25', 'DBP_nan', 'Resp_last_first_dif',
    'BaseExcess_min', 'BUN_std', 'Calcium_max', 'Calcium_std', 'Calcium_last_first_dif', 'Creatinine_last_first_dif', 'Magnesium_min',
    'Magnesium_std', 'Potassium_mean', 'Potassium_median', 'Potassium_q0.25', 'Potassium_q0.75', 'Potassium_last_first_dif', 'TroponinI_mean',
    'TroponinI_median', 'TroponinI_max', 'TroponinI_min', 'TroponinI_q0.25', 'TroponinI_q0.75', 'TroponinI_std', 'TroponinI_last_first_dif',
    'TroponinI_nan', 'Hct_last_first_dif', 'WBC_std', 'Platelets_max', 'Platelets_q0.75', 'Platelets_std'
    ]


if __name__ == "__main__":
    args = sys.argv
    path = args[1]
    print(f"preprocessing {path}")
    df = create_df(path)
    X = df.drop(['id', 'SepsisLabel'], axis=1)
    X = X.drop(manwhitney_drop, axis=1)
    with open('best_xgboost.pickle', 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(X)

    df['prediction'] = y_pred
    predictions = df[['id', 'prediction']]
    predictions['sort_id'] = predictions['id'].apply(lambda x: int(x.split('_')[1]))
    predictions = predictions.sort_values(by=['sort_id'])
    predictions = predictions.drop(['sort_id'], axis=1)
    predictions.to_csv('prediction.csv', index=False)

    try:
        y = df['SepsisLabel']
        print(f"f1 score: {f1_score(y, y_pred)}")
    except:
        pass
