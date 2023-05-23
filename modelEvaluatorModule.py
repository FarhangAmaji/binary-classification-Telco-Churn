#%% imports
import os
baseFolder = os.path.dirname(os.path.abspath(__file__))
os.chdir(baseFolder)
import itertools

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from utils import q,ti
from envVarsPreprocess import envVars
import traceback
#%%
class trainTestXY:
    def __init__(self, xTrain, xTest, yTrain, yTest, xScaler, yScaler):
        self.xTrain = xTrain
        self.xTest = xTest
        self.yTrain = yTrain
        self.yTest = yTest
        self.xScaler = xScaler
        self.yScaler = yScaler


class predictedY:
    def __init__(self, name, prediction, trueValue):
        self.name = name
        self.prediction = prediction
        self.trueValue = trueValue

    def getScores(self, scoringDictionary):
        scores = {}
        for sdk, sdv in scoringDictionary.items():
            scores[self.name + sdk.capitalize()] = sdv(self.trueValue, self.prediction)
        return scores


class modelEvaluator:
    def __init__(self, name, modelFunc, data, hyperParamRanges, crossValidationNum):
        self.name = name
        self.modelFunc = modelFunc
        self.data = data
        self.hyperParamRanges = hyperParamRanges
        self.crossValidationNum = crossValidationNum
        self.crossVal = self.crossValidationNum >= 2
        self.stratifiedKFold = StratifiedKFold(n_splits=crossValidationNum) if crossValidationNum>1 else None
        self.scoring = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'rocAuc': roc_auc_score,
            'cohenKappaScore': cohen_kappa_score
        }
        self.colsOrder = ['model', 'Parameter Set', 'Fold', 'testAccuracy', 'testPrecision', 'testRecall', 
            'testF1', 'testRocauc', 'testCohenkappascore']
        
        if self.crossVal:
            self.colsOrder.extend(['trainAccuracy', 'trainPrecision', 'trainRecall', 'trainF1', 'trainRocauc', 'trainCohenkappascore'])

    def shrinkParamRanges(self):
        if self.hyperParamRanges:
            for key, value in self.hyperParamRanges.items():
                if any(isinstance(item, (int, float)) for item in value):
                    self.hyperParamRanges[key] = [item for item in value if isinstance(item, (int, float))][:1]

    def updateCrossValidation(self):
        if self.crossVal:
            self.crossValidationNum = 2
            self.stratifiedKFold = StratifiedKFold(n_splits=self.crossValidationNum) if self.crossValidationNum > 1 else None

    def generateInputArgs(self):
        inputArgs = []

        if self.hyperParamRanges:
            paramCombinations = list(itertools.product(*self.hyperParamRanges.values()))
            for params in paramCombinations:
                hyperparameters = dict(zip(self.hyperParamRanges.keys(), params))
                model = self.modelFunc(**hyperparameters)

                if self.crossVal:
                    for fold, (trainIndex, testIndex) in enumerate(
                            self.stratifiedKFold.split(self.data.xTrain, self.data.yTrain)):
                        inputArgs.append([fold + 1, model, hyperparameters, trainIndex, testIndex])
                else:
                    inputArgs.append(['noCrossVal', model, hyperparameters, list(range(len(self.data.xTrain))), []])
        else:
            model = self.modelFunc()
            if self.crossVal:
                for fold, (trainIndex, testIndex) in enumerate(
                        self.stratifiedKFold.split(self.data.xTrain, self.data.yTrain)):
                    inputArgs.append([fold + 1, model, '', trainIndex, testIndex])
            else:
                inputArgs.append(['noCrossVal', model, '', list(range(len(self.data.xTrain))), []])

        return inputArgs

    def preventDuplicateInputArgs(self, inputArgs, totResultsDf):
        if not totResultsDf.empty:
            thisModelTotResultsDf = totResultsDf[(totResultsDf['model'] == self.name)]
            for irI in range(len(inputArgs) - 1, -1, -1):
                ir = inputArgs[irI]
                if not thisModelTotResultsDf[
                    (thisModelTotResultsDf['Parameter Set'] == str(ir[3])) & (thisModelTotResultsDf['Fold'] == ir[0])].empty:
                    inputArgs.remove(ir)

    def getInputArgs(self, totResultsDf=None, paramCheckMode=envVars['paramCheckMode']):
        if paramCheckMode:
            self.shrinkParamRanges()
            self.updateCrossValidation()

        inputArgs = self.generateInputArgs()
        self.preventDuplicateInputArgs(inputArgs, totResultsDf)

        return [self, inputArgs]
    def fitModelAndGetResults(self, fold, model, hyperparameters, trainIndex, testIndex):
        try:
            q('fitModelAndGetResults',fold, self.name, hyperparameters,ti(),filewrite=True)
            foldXTrain, foldXTest = self.data.xTrain[trainIndex], self.data.xTrain[testIndex]
            foldYTrain, foldYTest = self.data.yTrain[trainIndex], self.data.yTrain[testIndex]
    
            fitted_model = model.fit(foldXTrain, foldYTrain.ravel())
            scores = {'model': self.name, 'Parameter Set': str(hyperparameters), 'Fold': fold}
    
            predicteds = [predictedY('test', fitted_model.predict(self.data.xTest), self.data.yTest)]
            if self.crossVal:
                predicteds.append(predictedY('train', fitted_model.predict(foldXTest), foldYTest))
            
            for predicted in predicteds:
                scores.update(predicted.getScores(self.scoring))
            
            dfRow = pd.DataFrame(scores, index=[0])[self.colsOrder]
            return dfRow
        except Exception as e:
            q('fitModelAndGetResults Errrrrrr',fold, self.name, hyperparameters,'time:',ti(),'\nerrrr:',e,'\ntraceback:',traceback.format_exc(),'\n',filewrite=True,filename='fitModelAndGetResultsErrr')
            return pd.DataFrame()