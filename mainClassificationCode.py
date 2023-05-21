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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier #kkk took so long check it later
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct #kkk this is for GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegressionCV
# from sklearn.neural_network import MLPClassifier
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.neighbors import NearestCentroid
# from sklearn.svm import NuSVC
# from sklearn.multiclass import OneVsOneClassifier
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.multiclass import OutputCodeClassifier
# from sklearn.linear_model import PassiveAggressiveClassifier
# from sklearn.linear_model import Perceptron
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.neighbors import RadiusNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import RidgeClassifier
# from sklearn.linear_model import RidgeClassifierCV
# from sklearn.linear_model import SGDClassifier
# from sklearn.svm import SVC
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
#     modelEvaluator('ExtraTreeClassifier', ExtraTreeClassifier, {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [None, 5, 10],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'random_state': [None, 42]
# }, defaultCrossValidationNum),

    # ExtraTreesClassifier
#     modelEvaluator('ExtraTreesClassifier', ExtraTreesClassifier, {
#     'n_estimators': [100, 200, 300],
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [None, 5, 10],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'bootstrap': [True, False],
#     'random_state': [None, 42]
# }, defaultCrossValidationNum),

    # GaussianNB
    # modelEvaluator('GaussianNB', GaussianNB, {}, defaultCrossValidationNum),
    
    # GradientBoostingClassifier
#     # modelEvaluator('GradientBoostingClassifier', GradientBoostingClassifier, {
#     'loss': ['log_loss', 'exponential'],
#     'learning_rate': [0.1, 0.01, 0.001],
#     'n_estimators': [100, 200, 500],
#     'subsample': [0.5, 0.7, 1.0],
#     'max_depth': [3, 5, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2', None],
#     'random_state': [None, 42]
# }, defaultCrossValidationNum),

    # HistGradientBoostingClassifier
    # modelEvaluator('HistGradientBoostingClassifier', HistGradientBoostingClassifier, {
#     'loss': ['binary_crossentropy', 'log_loss'],
#     'learning_rate': [0.1, 0.01, 0.001],
#     'max_iter': [100, 200, 500],
#     'max_depth': [3, 5, None],
#     'min_samples_leaf': [1, 2, 4],
#     'max_bins': [64, 128, 255],
#     'l2_regularization': [0.0, 0.1, 0.01],
#     'early_stopping': [True, 'auto'],
#     'random_state': [None, 42]
# }, defaultCrossValidationNum),

    # KNeighborsClassifier
    # modelEvaluator('KNeighborsClassifier', KNeighborsClassifier, {
#     'n_neighbors': [3, 5, 10],
#     'weights': ['uniform', 'distance'],
#     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#     'leaf_size': [30, 50, 100],
#     'p': [1, 2],
#     'metric': ['euclidean', 'manhattan'],
# }, defaultCrossValidationNum),

    # LabelPropagation
    # modelEvaluator('LabelPropagation', LabelPropagation, {
#     'kernel': ['knn', 'rbf'],
#     'gamma': [None, 0.1, 1.0],
#     'n_neighbors': [3, 5, 10],
#     'max_iter': [100, 200, 500],
#     'tol': [1e-3, 1e-4],
# }, defaultCrossValidationNum),

    # LabelSpreading
    # modelEvaluator('LabelSpreading', LabelSpreading, {
#     'kernel': ['knn', 'rbf'],
#     'gamma': ['auto', 0.1, 1.0],
#     'n_neighbors': [3, 5, 10],
#     'alpha': [0.2, 0.5, 0.8],
#     'max_iter': [100, 200, 500],
#     'tol': [1e-3, 1e-4],
# }, defaultCrossValidationNum),
    
    
    # LinearDiscriminantAnalysis
    # modelEvaluator('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis, {
#     'solver': ['lsqr', 'eigen'],
#     'shrinkage': ['auto'],
#     'priors': [None, [0.1, 0.9], [0.3, 0.7]],
#     'n_components': [None, 1, 2],
#     'tol': [1e-4, 1e-5],
# }, defaultCrossValidationNum),
    
    # LinearSVC
    # modelEvaluator('LinearSVC', LinearSVC, {
#     'penalty': ['l2'],
#     'loss': ['hinge', 'squared_hinge'],
#     'dual': [True, False],
#     'tol': [1e-4, 1e-5],
#     'C': [1.0, 0.1, 0.01],
#     'multi_class': ['ovr', 'crammer_singer'],
#     'fit_intercept': [True, False],
#     'intercept_scaling': [1, 2, 5],
#     'class_weight': [None, 'balanced'],
#     'max_iter': [1000, 2000],
# }, defaultCrossValidationNum),
    
    # LogisticRegressionCV
    # modelEvaluator('LogisticRegressionCV', LogisticRegressionCV, {
#     'Cs': [10, 1, 0.1],
#     'fit_intercept': [True, False],
#     'cv': [3, 5],
#     'penalty': ['l2', 'l1'],
#     'scoring': ['accuracy', 'roc_auc'],
#     'solver': ['liblinear'],
#     'tol': [1e-4, 1e-5],
#     'max_iter': [100, 200],
# }, defaultCrossValidationNum),
    
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

