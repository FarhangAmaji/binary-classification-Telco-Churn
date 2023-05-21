#%% change dir to current dir
import os
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)
#%% import prepared data
from dataPreparation import dataPreparation
telcoChurn, xTrain, xTest , yTrain, yTest = dataPreparation(doUpSampling=False,criticalOutlierColsForARow=1)
#%% imports
import pandas as pd
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
#%% 
class trainTestXY:
    def __init__(self, xTrain, xTest , yTrain, yTest):
        self.xTrain = xTrain
        self.xTest = xTest
        self.yTrain = yTrain
        self.yTest = yTest

class predictedY:
    def __init__(self, name, prediction,trueValue):
        self.name = name
        self.prediction = prediction
        self.trueValue = trueValue
    
    def getScores(self,scoringDictionary):
        scores={}
        for sdk,sdv in scoringDictionary.items():
            scores[self.name+sdk.capitalize()]=sdv(self.trueValue, self.prediction)
        return scores
        
class ModelInfo:
    def __init__(self, name, modelFunc, hyperParamRanges,crossValidationNum):
        self.name = name
        self.modelFunc = modelFunc
        self.hyperParamRanges = hyperParamRanges
        self.crossValidationNum = crossValidationNum
        self.stratifiedKFold = StratifiedKFold(n_splits=crossValidationNum)
        self.scoring={'accuracy':accuracy_score, 'precision':precision_score, 'recall':recall_score,'f1':f1_score,'roc_auc':roc_auc_score}
        self.gridSearch = None
        self.trainResultsDf = None
    
    def fitModelAndGetResults(self, data):
        #kkk parallelize steps
        param_combinations = list(itertools.product(*self.hyperParamRanges.values()))
        resultsDf= pd.DataFrame()
        for params in param_combinations:
            hyperparameters = dict(zip(self.hyperParamRanges.keys(), params))
            model = self.modelFunc(**hyperparameters)
            for fold, (train_index, test_index) in enumerate(self.stratifiedKFold.split(data.xTrain, data.yTrain)):
                fold_xTrain, fold_X_test = data.xTrain[train_index], data.xTrain[test_index]
                fold_yTrain, fold_y_test = data.yTrain[train_index], data.yTrain[test_index]
                
                fittedModel = model.fit(fold_xTrain, fold_yTrain)
                
                scores={'model':self.name, 'Parameter Set':str(hyperparameters), 'Fold':fold+1}
                
                predicteds=[predictedY('train',fittedModel.predict(fold_X_test),fold_y_test),
                            predictedY('test',fittedModel.predict(data.xTest),data.yTest)]
                for predicted in predicteds:
                    scores.update(predicted.getScores(self.scoring))#kkk do I need update
                colsOrder=['model', 'Parameter Set', 'Fold', 'testAccuracy', 'testPrecision',
                'testRecall', 'testF1', 'testRoc_auc', 'trainAccuracy', 'trainPrecision',
                       'trainRecall', 'trainF1', 'trainRoc_auc']
                dfRow=pd.DataFrame(scores,index=[0])[colsOrder]
                resultsDf=pd.concat([resultsDf,dfRow])
                
        return resultsDf
#%% 
from xgboost import XGBClassifier
defaultCrossValidationNum=5
allModels=[ModelInfo('xgboost', XGBClassifier, {
    'learning_rate': [0.01, 0.02],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'n_estimators': [450, 500]}, defaultCrossValidationNum),
    
    # ModelInfo(name, modelFunc, {hyperParamRanges},defaultCrossValidationNum)
    ]
trainTestXY_=trainTestXY(xTrain, xTest , yTrain, yTest)
totResultsDf= pd.DataFrame()
for am1 in allModels:
    totResultsDf = pd.concat([totResultsDf, am1.fitModelAndGetResults(trainTestXY_)])
#%% 

#%% 

#%% 

#%% 

#%% 

#%% 

#%% 

