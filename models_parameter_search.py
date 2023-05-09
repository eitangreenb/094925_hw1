import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


parameter_grid = dict(
    # Define the parameter grid for Decision Tree
    tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
    },

    # Define the parameter grid for Random Forest
    forest = {
    'n_estimators': [50, 100, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    },

    # Define the parameter grid for XGBoost
    xgboost = {
    'learning_rate': [0.1, 0.2, 0.3],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'subsample': [0.5, 0.7, 0.9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.5],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'scale_pos_weight': [1, 3, 5]
    }
    )


def get_best_model(X, y, model, model_name):
    print('Grid Search for ', model_name)
    # Define the cross-validation strategy
    cv = 5

    # Create a GridSearchCV object
    grid_search = GridSearchCV(model, parameter_grid[model_name], cv=cv, scoring='f1')

    # Fit the GridSearchCV object to the data
    grid_search.fit(X, y)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Print the best hyperparameters and the best score
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)

    return best_model


def compute_metrics(y, y_pred, y_proba):
    fpr, tpr, _ = roc_curve(y, y_proba.T[1])
    res_dict = dict(
        precision=precision_score(y, y_pred),
        recall=recall_score(y, y_pred),
        f1=f1_score(y, y_pred),
        accuracy=accuracy_score(y, y_pred),
        roc_auc=auc(fpr, tpr)
    )
    return res_dict, fpr, tpr


def plot_roc_curve(fpr, tpr, auc):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label=f'ROC curve (area = {auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    search_decision_tree = True
    search_random_forets = True
    search_xgboost = False

    idx = -1
    train_path = ["train_df.csv", "final_train_df.csv"][idx]
    test_path = ["test_df.csv", "final_test_df.csv"][idx]

    train = pd.read_csv(train_path)
    train_y = train['SepsisLabel']
    train_x = train.drop(['id', 'SepsisLabel'], axis=1)
    test = pd.read_csv(test_path)
    test_y = test['SepsisLabel']
    test_x = test.drop(['id', 'SepsisLabel'], axis=1)

    if search_decision_tree:
        print('Decision Tree')
        # Grid Search Model 1: Decision Tree:
        train_x.fillna(-1, inplace=True)
        test_x.fillna(-1, inplace=True)
        clf = tree.DecisionTreeClassifier()
        best_decision_tree = get_best_model(train_x, train_y, clf, 'tree')
        y_pred = best_decision_tree.predict(test_x)
        y_proba = best_decision_tree.predict_proba(test_x)

        res_dict, fpr, tpr = compute_metrics(test_y, y_pred, y_proba)
        print(res_dict)
        plot_roc_curve(fpr, tpr, res_dict['roc_auc'])
    
    if search_random_forets:
        print('Random Forest')
        # Grid Search Model 2: Random Forest:
        train_x.fillna(-1, inplace=True)
        test_x.fillna(-1, inplace=True)
        rf_model = RandomForestClassifier()
        best_random_forest = get_best_model(train_x, train_y, rf_model, 'forest')
        y_pred = best_random_forest.predict(test_x)
        y_proba = best_random_forest.predict_proba(test_x)

        res_dict, fpr, tpr = compute_metrics(test_y, y_pred, y_proba)
        print(res_dict)
        plot_roc_curve(fpr, tpr, res_dict['roc_auc'])

    if search_xgboost:
        # Grid Search Model 3: XGBoost:
        xgboost_model = XGBClassifier()
        best_xgboost = get_best_model(train_x, train_y, xgboost_model, 'xgboost')
        y_pred = best_xgboost.predict(test_x)
        y_proba = best_xgboost.predict_proba(test_x)
        
        res_dict, fpr, tpr = compute_metrics(test_y, y_pred, y_proba)
        print(res_dict)
        plot_roc_curve(fpr, tpr, res_dict['roc_auc'])
