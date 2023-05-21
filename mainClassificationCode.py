#%% imports
import os
import pandas as pd
base_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_folder)

from dataPreparation import dataPreparation
from modelEvaluator import modelEvaluator,trainTestXY 

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
#%% 
telcoChurn, xTrain, xTest, yTrain, yTest = dataPreparation(doUpSampling=False, criticalOutlierColsForARow=1)

defaultCrossValidationNum = 5
allModels = [
    modelEvaluator(
        'xgboost',
        XGBClassifier,
        {
            'learning_rate': [0.01, 0.02],
            'subsample': [0.8, 1],
            'colsample_bytree': [0.8, 1],
            'n_estimators': [450, 500]
        },
        defaultCrossValidationNum
    ),
    
    # modelEvaluator('RandomForestClassifier', RandomForestClassifier, {
    # 'n_estimators': [50,100],
    # 'criterion': ['gini'],
    # 'max_depth': [None, 5, ],
    # 'min_samples_split': [2],
    # 'min_samples_leaf': [1, ],
    # 'max_features': ['auto']}, defaultCrossValidationNum),
    
    # modelEvaluator('RandomForestClassifier', RandomForestClassifier, {
    # 'n_estimators': [50, 100, 200, 500],
    # 'criterion': ['gini', 'entropy'],
    # 'max_depth': [None, 5, 10, 20],
    # 'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4],
    # 'max_features': ['auto', 'sqrt', 'log2']}, defaultCrossValidationNum),
    
    # modelEvaluator(name, modelFunc, hyperParamRanges, defaultCrossValidationNum),
]

#kkk add scaler
trainTestXY_ = trainTestXY(xTrain, xTest, yTrain, yTest)
totResultsDf = pd.DataFrame()

for m1 in allModels:
    totResultsDf = pd.concat([totResultsDf, m1.fitModelAndGetResults(trainTestXY_)])
#%% 

#%% 

#%% 

#%% 

#%% 

