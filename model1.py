#%% change dir to current dir
import os
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)
#%% import prepared data
from dataPreparation import dataPreparation
telcoChurn, xTrain, xTest , yTrain, yTest = dataPreparation(doUpSampling=False,criticalOutlierColsForARow=1)
#%% 
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

XGBC = XGBClassifier()
hyperParamRanges = {
    'learning_rate': [0.01, 0.02],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'n_estimators': [450, 500]}
scoring=['accuracy', 'precision', 'recall','f1','roc_auc']
grid_search = GridSearchCV(XGBC, param_grid=params, cv=5, scoring=scoring, refit= 'accuracy',n_jobs=-1)
grid_search.fit(xTrain, yTrain)
#%% 
import pandas as pd
cv_results = grid_search.cv_results_
results = []
for i in range(len(cv_results['params'])):
    param_set = cv_results['params'][i]
    for j in range(grid_search.cv):
        fold_scores = {}
        for metric in scoring:
            score_key = "split{}_test_{}".format(j, metric)
            score_value = cv_results[score_key][i]
            fold_scores[metric] = score_value
        fold_scores['Parameter Set'] = param_set
        fold_scores['Fold'] = j + 1
        results.append(fold_scores)

# Create the dataframe
resultsDf = pd.DataFrame(results, columns=['Parameter Set', 'Fold', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
#%% 

#%% 

#%% 

#%% 

#%% 

#%% 

#%% 

#%% 

#%% 

#%% 

#%% 

#%% 

