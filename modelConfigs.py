#%% imports
import os
import pandas as pd
base_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_folder)

from modelEvaluatorModule import modelEvaluator,trainTestXY
from dataPreparationModule import dataPreparation
from envVarsPreprocess import envVars

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV #kkk gave error
from sklearn.naive_bayes import CategoricalNB #kkk gave error
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
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier #kkk didnt understand how to apply hyperparameters; to be more precise the way of applying was different
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC #kkk took so long check it later
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC #kkk took so long check it later
from lightgbm import LGBMClassifier
#%% 
telcoChurn, trainTestXY_ = dataPreparation(criticalOutlierColsForARow=1)
#%%
#kkk the data preparation may be moved here
defaultCrossValidationNum = envVars['crossValNum']#kkk change it back to 5
allModelConfigs = [
    ## lgbm
    modelEvaluator('LGBMClassifier', LGBMClassifier, trainTestXY_,{
        'boosting_type': ['gbdt', 'dart', 'goss'],
        'num_leaves': [50, 100],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [150, 300],
        'max_depth': [-1, 5, 10],
        'min_child_samples': [5, 10],
        'subsample': [0.5, 0.8],
        'colsample_bytree': [0.5, 0.8],
        'reg_alpha': [0.1, 0.5],
    }, defaultCrossValidationNum),


    # # xgboost
    modelEvaluator(
        'xgboost',
        XGBClassifier,
        trainTestXY_,{
    'max_depth': [4, 8, 25],
    'learning_rate': [0.05, 0.01,0.3],
    'n_estimators': [150, 300,400],
    "eval_metric" : ['auc'],
    'use_label_encoder':[False],
    "random_state" : [42],
    "objective" : ['binary:logistic'],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'min_child_weight': [2, 5]
},
        defaultCrossValidationNum
    ),
    
    #RandomForestClassifier
    modelEvaluator('RandomForestClassifier', RandomForestClassifier, trainTestXY_,{
    'n_estimators': [50, 100, 200],#, 500
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 20],#None, 
    'min_samples_split': [2, 4,5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'random_state':[42]}, defaultCrossValidationNum),
    
    
    # DecisionTreeClassifier
    modelEvaluator('DecisionTreeClassifier', DecisionTreeClassifier, trainTestXY_,{
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}, defaultCrossValidationNum),
    
    #AdaBoostClassifier
    modelEvaluator('AdaBoostClassifier', AdaBoostClassifier,  trainTestXY_,{
    'estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [0.1, 0.5, 1.0],
    'algorithm': ['SAMME', 'SAMME.R']
}, defaultCrossValidationNum),
    
    ## BaggingClassifier
    modelEvaluator('BaggingClassifier', BaggingClassifier,  trainTestXY_,{
    'estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
    'n_estimators': [10, 50, 100, 200],
    'max_samples': [0.5, 0.7, 1.0],
    'max_features': [0.5, 0.7, 1.0],
    'bootstrap': [True, False]
}, defaultCrossValidationNum),
    
    
    # BernoulliNB
    modelEvaluator('BernoulliNB', BernoulliNB,  trainTestXY_,{
    'alpha': [0.0, 0.5, 1.0, 2.0]}, defaultCrossValidationNum),

    # LogisticRegression
    modelEvaluator('LogisticRegression', LogisticRegression, trainTestXY_,{
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 1000],
    'class_weight': [None, 'balanced']
}, defaultCrossValidationNum),

    # ComplementNB
    modelEvaluator('ComplementNB', ComplementNB, trainTestXY_,{
    'alpha': [0.1, 0.5, 1.0],
    'fit_prior': [True, False],
    'norm': [True, False]
}, defaultCrossValidationNum),

    # DummyClassifier
    modelEvaluator('DummyClassifier', DummyClassifier, trainTestXY_,{
    'strategy': ['stratified', 'most_frequent', 'prior', 'uniform'],
    'random_state': [None, 42]
}, defaultCrossValidationNum),

    # ExtraTreeClassifier
    modelEvaluator('ExtraTreeClassifier', ExtraTreeClassifier, trainTestXY_,{
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'random_state': [None, 42]
}, defaultCrossValidationNum),

    # ExtraTreesClassifier
    modelEvaluator('ExtraTreesClassifier', ExtraTreesClassifier, trainTestXY_,{
    'n_estimators': [100, 200],#, 300
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10],#None, 
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False],
    'random_state': [None]#, 42
}, defaultCrossValidationNum),

    # GaussianNB
    modelEvaluator('GaussianNB', GaussianNB, trainTestXY_,{}, defaultCrossValidationNum),
    
    # GradientBoostingClassifier
    modelEvaluator('GradientBoostingClassifier', GradientBoostingClassifier, trainTestXY_,{
    'loss': ['log_loss', 'exponential'],
    'learning_rate': [0.01, 0.05],
    'n_estimators': [100],
    'subsample': [0.5, 0.8],
    'max_depth': [5, None],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [4],
    'max_features': ['sqrt', 'log2'],
    'random_state': [None]#, 42
}, defaultCrossValidationNum),

    # # HistGradientBoostingClassifier
    modelEvaluator('HistGradientBoostingClassifier', HistGradientBoostingClassifier, trainTestXY_,{
    'loss': ['binary_crossentropy', 'log_loss'],
    'learning_rate': [0.01, 0.05],
    'max_iter': [100, 200],
    'max_depth': [5, None],
    'min_samples_leaf': [4],
    'max_bins': [64, 255],
    'l2_regularization': [0.0, 0.05],
    'early_stopping': [True],#, 'auto'
    'random_state': [None]#, 42
}, defaultCrossValidationNum),

    # KNeighborsClassifier
    modelEvaluator('KNeighborsClassifier', KNeighborsClassifier, trainTestXY_,{
    'n_neighbors': [3, 5, 10],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [30, 50, 100],
    'p': [1, 2],
    'metric': ['euclidean', 'manhattan'],
}, defaultCrossValidationNum),

    # LabelPropagation
    modelEvaluator('LabelPropagation', LabelPropagation, trainTestXY_,{
    'kernel': ['knn', 'rbf'],
    'gamma': [0.1, 1.0],
    'n_neighbors': [3, 5, 10],
    'max_iter': [100, 200, 500],
    'tol': [1e-3, 1e-4],
}, defaultCrossValidationNum),

    # LabelSpreading
    modelEvaluator('LabelSpreading', LabelSpreading, trainTestXY_,{
    'kernel': ['knn', 'rbf'],
    'gamma': [0.1, 1.0],
    'n_neighbors': [3, 5, 10],
    'alpha': [0.2, 0.5, 0.8],
    'max_iter': [100, 200, 500],
    'tol': [1e-3, 1e-4],
}, defaultCrossValidationNum),
    
    
    # LinearDiscriminantAnalysis
    modelEvaluator('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis, trainTestXY_,{
    'solver': ['lsqr', 'eigen'],
    'shrinkage': ['auto'],
    'priors': [None, [0.1, 0.9], [0.3, 0.7]],
    'n_components': [None, 1, 2],
    'tol': [1e-4, 1e-5],
}, defaultCrossValidationNum),
    
    # LinearSVC
    modelEvaluator('LinearSVC', LinearSVC, trainTestXY_,{
    'penalty': ['l2'],
    'loss': ['hinge', 'squared_hinge'],
    'dual': [True, False],
    'tol': [1e-4, 1e-5],
    'C': [1.0, 0.1, 0.01],
    'multi_class': ['ovr', 'crammer_singer'],
    'fit_intercept': [True, False],
    'intercept_scaling': [1, 2, 5],
    'class_weight': [None, 'balanced'],
    'max_iter': [1000, 2000],
}, defaultCrossValidationNum),
    
    # LogisticRegressionCV
    modelEvaluator('LogisticRegressionCV', LogisticRegressionCV, trainTestXY_,{
    'Cs': [10, 1],
    'fit_intercept': [True, False],
    'cv': [3, 5],
    'penalty': ['l2', 'l1'],
    'scoring': ['accuracy', 'roc_auc'],
    'solver': ['liblinear'],
    'tol': [1e-4, 1e-5],
    'max_iter': [100, 200],
}, defaultCrossValidationNum),
    
    # MLPClassifier
    modelEvaluator('MLPClassifier', MLPClassifier, trainTestXY_,{
    'hidden_layer_sizes': [(100,), (50, 50), (50, 50, 50)],
    'activation': ['relu', 'logistic'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'batch_size': ['auto'],
    'learning_rate': ['constant', 'adaptive'],
    'learning_rate_init': [0.001, 0.01],
    'max_iter': [200, 500],
    'shuffle': [True],
    'random_state': [42],
}, defaultCrossValidationNum),
    
    # MultinomialNB
    modelEvaluator('MultinomialNB', MultinomialNB, trainTestXY_,{
    'alpha': [0.1, 0.5, 1.0, 10.0],
}, defaultCrossValidationNum),
    
    # NearestCentroid
    modelEvaluator('NearestCentroid', NearestCentroid, trainTestXY_,{
    'metric': ['euclidean', 'manhattan', 'cosine'],
}, defaultCrossValidationNum),
    
    
    # OneVsOneClassifier
    modelEvaluator('OneVsOneClassifier', OneVsOneClassifier, trainTestXY_,{
    'estimator': [RandomForestClassifier()],
}, defaultCrossValidationNum),

    # OneVsRestClassifier
    modelEvaluator('OneVsRestClassifier', OneVsRestClassifier, trainTestXY_,{
    'estimator': [RandomForestClassifier()],
}, defaultCrossValidationNum),
    
    # OutputCodeClassifier
    modelEvaluator('OutputCodeClassifier', OutputCodeClassifier, trainTestXY_,{
    'estimator': [RandomForestClassifier()],
    'code_size': [0.5, 1.0, 1.5],  # Example values for the 'code_size' hyperparameter
}, defaultCrossValidationNum),
    
    # PassiveAggressiveClassifier
    modelEvaluator('PassiveAggressiveClassifier', PassiveAggressiveClassifier, trainTestXY_,{
    'C': [0.1, 1.0, 10.0],  # Regularization parameter
    'fit_intercept': [True, False],  # Whether to include an intercept term
    'max_iter': [1000, 2000, 3000],  # Maximum number of iterations
    'tol': [1e-3, 1e-4, 1e-5],  # Tolerance for stopping criterion
    'loss': ['hinge', 'squared_hinge'],  # Loss function ('hinge' or 'squared_hinge')
    'random_state': [42]  # Random state for reproducibility
}, defaultCrossValidationNum),
    
    # Perceptron
    modelEvaluator('Perceptron', Perceptron, trainTestXY_,{
    'alpha': [0.0001, 0.001, 0.01],  # Regularization strength
    'fit_intercept': [True, False],  # Whether to include an intercept term
    'max_iter': [1000, 2000, 3000],  # Maximum number of iterations
    'tol': [1e-3, 1e-4, 1e-5],  # Tolerance for stopping criterion
    'shuffle': [True, False],  # Whether to shuffle the training data in each epoch
    'random_state': [42]  # Random state for reproducibility
}, defaultCrossValidationNum),
    
    # QuadraticDiscriminantAnalysis
    modelEvaluator('QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis, trainTestXY_,{
    'priors': [None, [0.25, 0.75], [0.5, 0.5]],  # Class priors
    'reg_param': [0.0, 0.1, 0.2],  # Regularization parameter
    'store_covariance': [True, False],  # Whether to store covariance matrices
    'tol': [1e-4, 1e-5, 1e-6],  # Tolerance for stopping criterion
}, defaultCrossValidationNum),
    
    # RadiusNeighborsClassifier
    modelEvaluator('RadiusNeighborsClassifier', RadiusNeighborsClassifier, trainTestXY_,{
    'radius': [1.0, 1.5, 2.0],  # Radius of the neighborhood
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used for nearest neighbors search
    'leaf_size': [30, 40, 50],  # Leaf size for tree-based algorithms
    'p': [1, 2],  # Power parameter for the Minkowski metric
    'outlier_label': [None, 0, 1],  # Label assigned to outlier samples
}, defaultCrossValidationNum),
    
    # RidgeClassifier
    modelEvaluator('RidgeClassifier', RidgeClassifier, trainTestXY_,{
    'alpha': [1.0, 0.5, 0.1],  # Regularization strength
    'fit_intercept': [True, False],  # Whether to fit an intercept term
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],  # Solver for optimization problem
    'max_iter': [1000, 2000],  # Maximum number of iterations
    'tol': [1e-3, 1e-4, 1e-5],  # Tolerance for stopping criteria
}, defaultCrossValidationNum),
    
    # RidgeClassifierCV
    modelEvaluator('RidgeClassifierCV', RidgeClassifierCV, trainTestXY_,{
    'alphas': [[0.1, 1.0, 10.0]],  # List of alpha values to try
    'fit_intercept': [True, False],  # Whether to fit an intercept term
    'scoring': ['accuracy', 'f1_macro'],  # Scoring metric for cross-validation
    'cv': [None, 5, 10],  # Number of cross-validation folds or a specific cross-validation object
    'class_weight': [None, 'balanced'],  # Weights associated with classes
    'store_cv_values': [False, True],  # Whether to store the cross-validation values for each alpha
}, defaultCrossValidationNum),
    
#     # SGDClassifier
    modelEvaluator('SGDClassifier', SGDClassifier, trainTestXY_,{
    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],  # Loss function
    'penalty': ['l2', 'l1', 'elasticnet'],  # Regularization type
    'alpha': [0.01, 0.05],  # Regularization parameter
    'eta0': [0.01, 0.1],  # Initial learning rate
    'learning_rate': ['constant', 'optimal', 'invscaling'],  # Learning rate schedule
    'class_weight': [None, 'balanced'],  # Weights associated with classes
    'average': [False, True],  # Whether to compute the averaged SGD weights
}, defaultCrossValidationNum),
    
#     # SVC
    modelEvaluator('SVC', SVC, trainTestXY_,{
    'C': [0.1, 1.0],#, 10.0  # Penalty parameter C of the error term
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel function
    'degree': [2, 3],  # Degree of the polynomial kernel function
    'gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels
    'coef0': [0.0, 0.5],#, 1.0  # Independent term in kernel function
    'shrinking': [True, False],  # Whether to use the shrinking heuristic
    'probability': [False, True],  # Whether to enable probability estimates
    'tol': [1e-3, 1e-4], #, 1e-5 # Tolerance for stopping criterion
    'class_weight': [None, 'balanced'],  # Weights associated with classes
    'max_iter': [-1],  # Maximum number of iterations (-1 for no limit)
}, defaultCrossValidationNum),
]
#%% 

#%% 

#%% 

#%% 

#%% 

