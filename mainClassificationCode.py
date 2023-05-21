#%% imports
import os
import pandas as pd
from xgboost import XGBClassifier
base_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_folder)
from dataPreparation import dataPreparation
from modelEvaluator import modelEvaluator,trainTestXY 
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
    )
]

trainTestXY_ = trainTestXY(xTrain, xTest, yTrain, yTest)
totResultsDf = pd.DataFrame()

for m1 in allModels:
    totResultsDf = pd.concat([totResultsDf, m1.fitModelAndGetResults(trainTestXY_)])
#%% 

#%% 

#%% 

#%% 

#%% 

