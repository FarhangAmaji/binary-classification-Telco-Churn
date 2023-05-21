#%% imports
import os
base_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_folder)
import itertools
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from utils import q,ti
#kkk add option no to use cross validation
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

    def fitModelAndGetResults(self, data,totResultsDf=None parallel=True):
        print(f'started fitting {self.name}')
        resultsDf = pd.DataFrame()
        #kkk add an option if the params are in some fileSaved dont do the function
        #kkk add an option not to use cross validation
        
        inputArgs=[]
        
        if self.hyperParamRanges:
            paramCombinations = list(itertools.product(*self.hyperParamRanges.values()))
            for params in paramCombinations:
                hyperparameters = dict(zip(self.hyperParamRanges.keys(), params))
                model = self.modelFunc(**hyperparameters)
                
                for fold, (trainIndex, testIndex) in enumerate(self.stratifiedKFold.split(data.xTrain, data.yTrain)):
                    inputArgs.append([fold, data, model, hyperparameters, trainIndex, testIndex])
        else:
            model = self.modelFunc()
            for fold, (trainIndex, testIndex) in enumerate(self.stratifiedKFold.split(data.xTrain, data.yTrain)):
                inputArgs.append([fold, data, model, '', trainIndex, testIndex])
        
        # check if the inputArgs are not in the totResultsDf
        if totResultsDf:
            for ir in inputArgs:
                if ((totResultsDf['model'] == self.name) & (totResultsDf['Parameter Set'] == inputArgs[3]) & (totResultsDf['Fold'] == inputArgs[0])).any():
                    inputArgs.remove(ir)
        
        if parallel:
            with ThreadPoolExecutor(max_workers=max(multiprocessing.cpu_count() - 4, 1)) as executor:
                try:
                    resultRows = list(executor.map(self.processFold, *zip(*inputArgs)))
                    resultsDf = pd.concat([resultsDf, *resultRows]).reset_index(drop=True)
                except Exception as e:
                    print('ThreadPoolExecutor errrr', e)
        else:
            for ir in inputArgs:
                resultRows = self.processFold(*ir)
                resultsDf = pd.concat([resultsDf, resultRows]).reset_index(drop=True)
        return resultsDf
    def processFold(self, fold, data, model, hyperparameters, trainIndex, testIndex):
        try:
            q('processFold',fold, self.name, hyperparameters,ti(),filewrite=True)
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
        except Exception as e:
            q('processFold Errrrrrr',fold, self.name, hyperparameters,'errrr',e,ti(),filewrite=True,filename='processFoldErrr')
            return pd.DataFrame()