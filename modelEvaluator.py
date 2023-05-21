#%% imports
import itertools
import multiprocessing
from joblib import delayed, Parallel
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
#%%
class trainTestXY:
    def __init__(self, xTrain, xTest, yTrain, yTest):
        self.xTrain = xTrain
        self.xTest = xTest
        self.yTrain = yTrain
        self.yTest = yTest


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
    def __init__(self, name, modelFunc, hyperParamRanges, crossValidationNum):
        self.name = name
        self.modelFunc = modelFunc
        self.hyperParamRanges = hyperParamRanges
        self.crossValidationNum = crossValidationNum
        self.stratifiedKFold = StratifiedKFold(n_splits=crossValidationNum)
        self.scoring = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'rocAuc': roc_auc_score,
            'cohenKappaScore': cohen_kappa_score
        }

    def fitModelAndGetResults(self, data):
        print(f'started fitting {self.name}')
        resultsDf = pd.DataFrame()
        
        if self.hyperParamRanges:
            paramCombinations = list(itertools.product(*self.hyperParamRanges.values()))
            for params in paramCombinations:
                hyperparameters = dict(zip(self.hyperParamRanges.keys(), params))
                model = self.modelFunc(**hyperparameters)
    
                resultRows = self.callParallel(data, model, hyperparameters)
                resultsDf = pd.concat([resultsDf, *resultRows])
        else:
            model = self.modelFunc()
            resultRows = self.callParallel(data, model, '')
            resultsDf = pd.concat([resultsDf, *resultRows])
        return resultsDf
    def callParallel(self,data, model, hyperparameters):
        resultRows = Parallel(n_jobs=max(multiprocessing.cpu_count() - 4, 1))(delayed(self.processFold)(
            fold, data, model, hyperparameters, trainIndex, testIndex)
                                for fold, (trainIndex, testIndex) in enumerate(
                                self.stratifiedKFold.split(data.xTrain, data.yTrain)))
        return resultRows
    def processFold(self, fold, data, model, hyperparameters, trainIndex, testIndex):
        foldXTrain, foldXTest = data.xTrain[trainIndex], data.xTrain[testIndex]
        foldYTrain, foldYTest = data.yTrain[trainIndex], data.yTrain[testIndex]

        fitted_model = model.fit(foldXTrain, foldYTrain)

        scores = {'model': self.name, 'Parameter Set': str(hyperparameters), 'Fold': fold + 1}

        predicteds = [
            predictedY('train', fitted_model.predict(foldXTest), foldYTest),
            predictedY('test', fitted_model.predict(data.xTest), data.yTest)
        ]
        for predicted in predicteds:
            scores.update(predicted.getScores(self.scoring))

        cols_order = [
            'model', 'Parameter Set', 'Fold', 'testAccuracy', 'testPrecision', 'testRecall', 
        'testF1', 'testRocauc','testCohenkappascore','trainAccuracy', 'trainPrecision',
        'trainRecall', 'trainF1', 'trainRocauc', 'trainCohenkappascore', ]
        df_row = pd.DataFrame(scores, index=[0])[cols_order]
        return df_row