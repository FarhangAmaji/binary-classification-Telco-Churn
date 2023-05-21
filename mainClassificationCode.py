#%% imports
import os
import pandas as pd
base_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_folder)

from dataPreparation import dataPreparation
from modelEvaluator import modelEvaluator,trainTestXY

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV #kkk gave error
# from sklearn.naive_bayes import CategoricalNB #kkk gave error
from sklearn.multioutput import ClassifierChain  #kkk gave lots of error
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.dummy import DummyClassifier
from sklearn.tree import ExtraTreeClassifier
#%% 
telcoChurn, xTrain, xTest, yTrain, yTest = dataPreparation(doUpSampling=False, criticalOutlierColsForARow=1)
#%%
defaultCrossValidationNum = 5
allModels = [
    # xgboost
    # modelEvaluator(
    #     'xgboost',
    #     XGBClassifier,
    #     {
    #         'learning_rate': [0.01, 0.02],
    #         'subsample': [0.8, 1],
    #         'colsample_bytree': [0.8, 1],
    #         'n_estimators': [450, 500]
    #     },
    #     defaultCrossValidationNum
    # ),
    
    #RandomForestClassifier
    # modelEvaluator('RandomForestClassifier', RandomForestClassifier, {
    # 'n_estimators': [50, 100, 200, 500],
    # 'criterion': ['gini', 'entropy'],
    # 'max_depth': [None, 5, 10, 20],
    # 'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4],
    # 'max_features': ['auto', 'sqrt', 'log2']}, defaultCrossValidationNum),
    
    
#     # DecisionTreeClassifier
#     modelEvaluator('DecisionTreeClassifier', DecisionTreeClassifier, {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [None, 5, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2']
# }, defaultCrossValidationNum),
    
    #AdaBoostClassifier
    # modelEvaluator('AdaBoostClassifier', AdaBoostClassifier,  {
#     'estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
#     'n_estimators': [50, 100, 200, 500],
#     'learning_rate': [0.1, 0.5, 1.0],
#     'algorithm': ['SAMME', 'SAMME.R']
# }, defaultCrossValidationNum),
    
#     # BaggingClassifier
#     modelEvaluator('BaggingClassifier', BaggingClassifier,  {
#     'estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
#     'n_estimators': [10, 50, 100, 200],
#     'max_samples': [0.5, 0.7, 1.0],
#     'max_features': [0.5, 0.7, 1.0],
#     'bootstrap': [True, False]
# }, defaultCrossValidationNum),
    
    
    # BernoulliNB
    # modelEvaluator('BernoulliNB', BernoulliNB,  {
    # 'alpha': [0.0, 0.5, 1.0, 2.0]}, defaultCrossValidationNum),

    # LogisticRegression
#     modelEvaluator('LogisticRegression', LogisticRegression, {
#     'penalty': ['l1', 'l2'],
#     'C': [0.001, 0.01, 0.1, 1, 10],
#     'solver': ['liblinear', 'saga'],
#     'max_iter': [100, 1000],
#     'class_weight': [None, 'balanced']
# }, defaultCrossValidationNum),

#     # ComplementNB
#     modelEvaluator('ComplementNB', ComplementNB, {
#     'alpha': [0.1, 0.5, 1.0],
#     'fit_prior': [True, False],
#     'norm': [True, False]
# }, defaultCrossValidationNum),

    # DummyClassifier
#     modelEvaluator('DummyClassifier', DummyClassifier, {
#     'strategy': ['stratified', 'most_frequent', 'prior', 'uniform'],
#     'random_state': [None, 42]
# }, defaultCrossValidationNum),

    # ExtraTreeClassifier
    modelEvaluator('ExtraTreeClassifier', ExtraTreeClassifier, {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'random_state': [None, 42]
}, defaultCrossValidationNum),

    # 
    # modelEvaluator('name', modelFunc, hyperParamRanges, defaultCrossValidationNum),
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

