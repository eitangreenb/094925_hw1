import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np
import warnings
import os
from tqdm import tqdm
import sys
from preprocess import create_df


def print_result(x, y, model, title):
    y_pred = model.predict(x)
    print(f"f1 score for {title}: {f1_score(y, y_pred)}")
    print(f"recall score for {title}: {recall_score(y, y_pred)}")
    print(f"precision score for {title}: {precision_score(y, y_pred)}\n")

def post_analysis_model(df, model):
    # analysing model result when splitting data based on gender:
    X_male = df[df['Gender'] == 1]
    y_male = X_male['SepsisLabel']
    X_male = X_male.drop(['id', 'SepsisLabel'], axis=1)

    print_result(X_male, y_male, model, 'male subgroup')

    X_female = df[df['Gender'] == 0]
    y_female = X_female['SepsisLabel']
    X_female = X_female.drop(['id', 'SepsisLabel'], axis=1)
    print_result(X_male, y_male, model, 'female subgroup')

    # subgroup of age:
    # young - 0-20, adult 20-40 40-60, old 60-80 80-100
    X_0_20 = df[df['Age'] <= 20]
    y_0_20 = X_0_20['SepsisLabel']
    X_0_20 = X_0_20.drop(['id', 'SepsisLabel'], axis=1)
    print_result(X_0_20, y_0_20, model, 'ages 0-20')

    X_21_40 = df[(df['Age'] > 20) & (df['Age'] <= 40)]
    y_21_40 = X_21_40['SepsisLabel']
    X_21_40 = X_21_40.drop(['id', 'SepsisLabel'], axis=1)
    print_result(X_21_40, y_21_40, model, 'ages 21-40')

    X_41_60 = df[(df['Age'] > 40) & (df['Age'] <= 60)]
    y_41_60 = X_41_60['SepsisLabel']
    X_41_60 = X_41_60.drop(['id', 'SepsisLabel'], axis=1)
    print_result(X_41_60, y_41_60, model, 'ages 41-60')

    X_61_80 = df[(df['Age'] > 60) & (df['Age'] <= 80)]
    y_61_80 = X_61_80['SepsisLabel']
    X_61_80 = X_61_80.drop(['id', 'SepsisLabel'], axis=1)
    print_result(X_61_80, y_61_80, model, 'ages 61-80')

    X_81_100 = df[df['Age'] > 80]
    y_81_100 = X_81_100['SepsisLabel']
    X_81_100 = X_81_100.drop(['id', 'SepsisLabel'], axis=1)
    print_result(X_81_100, y_81_100, model, 'ages 81 and older')

df = pd.read_csv("final_test_df.csv")

print('Post Analysis on XGBoost:')
with open('best_xgboost.pickle', 'rb') as f:
    model = pickle.load(f)
post_analysis_model(df, model)

df.fillna(-1, inplace=True)

print('Post Analysis on Random Forest:')
with open('best_random_forest.pickle', 'rb') as f:
    model = pickle.load(f)
post_analysis_model(df, model)

print('Post Analysis on Decision Tree:')
with open('best_decision_tree.pickle', 'rb') as f:
    model = pickle.load(f)
post_analysis_model(df, model)